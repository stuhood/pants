// Copyright 2017 Pants project contributors (see CONTRIBUTORS.md).
// Licensed under the Apache License, Version 2.0 (see LICENSE).

#![deny(warnings)]
// Enable all clippy lints except for many of the pedantic ones. It's a shame this needs to be copied and pasted across crates, but there doesn't appear to be a way to include inner attributes from a common source.
#![deny(
  clippy::all,
  clippy::default_trait_access,
  clippy::expl_impl_clone_on_copy,
  clippy::if_not_else,
  clippy::needless_continue,
  clippy::single_match_else,
  clippy::unseparated_literal_suffix,
  clippy::used_underscore_binding
)]
// It is often more clear to show that nothing is being moved.
#![allow(clippy::match_ref_pats)]
// Subjective style.
#![allow(
  clippy::len_without_is_empty,
  clippy::redundant_field_names,
  clippy::too_many_arguments
)]
// Default isn't as big a deal as people seem to think it is.
#![allow(clippy::new_without_default, clippy::new_ret_no_self)]
// Arc<Mutex> can be more clear than needing to grok Orderings:
#![allow(clippy::mutex_atomic)]

use std::cmp::Ordering;
use std::collections::{hash_map, BTreeSet, HashMap, HashSet};

use crate::rules::{DependencyKey, Rule};
use crate::{
  entry_str, params_str, Diagnostic, Entry, EntryWithDeps, InnerEntry, ParamTypes, RootEntry,
  RuleEdges, RuleGraph, UnreachableError,
};

type ChosenDependency<R> = (<R as Rule>::DependencyKey, Vec<(bool, Entry<R>)>);

///
/// A polymorphic form of crate::RuleEdges. Each dep has multiple possible implementation rules.
///
#[derive(Eq, PartialEq, Clone, Debug)]
struct PolyRuleEdges<R: Rule> {
  dependencies: HashMap<R::DependencyKey, Vec<Entry<R>>>,
}

// TODO: We can't derive this due to https://github.com/rust-lang/rust/issues/26925, which
// unnecessarily requires `Rule: Default`.
impl<R: Rule> Default for PolyRuleEdges<R> {
  fn default() -> Self {
    PolyRuleEdges {
      dependencies: HashMap::default(),
    }
  }
}

// Given the task index and the root subjects, it produces a rule graph that allows dependency nodes
// to be found statically rather than dynamically.
pub struct Builder<'t, R: Rule> {
  tasks: &'t HashMap<R::TypeId, Vec<R>>,
  root_param_types: ParamTypes<R::TypeId>,
}

impl<'t, R: Rule> Builder<'t, R> {
  pub fn new(
    tasks: &'t HashMap<R::TypeId, Vec<R>>,
    root_param_types: Vec<R::TypeId>,
  ) -> Builder<'t, R> {
    let root_param_types = root_param_types.into_iter().collect();
    Builder {
      tasks,
      root_param_types,
    }
  }

  pub fn sub_graph(&self, param_type: R::TypeId, product_type: R::TypeId) -> RuleGraph<R> {
    // TODO: Update to support rendering a subgraph given a set of ParamTypes.
    let param_types = vec![param_type].into_iter().collect();

    if let Some(beginning_root) = self.gen_root_entry(&param_types, product_type) {
      self.construct_graph(vec![beginning_root])
    } else {
      RuleGraph::default()
    }
  }

  pub fn full_graph(&self) -> RuleGraph<R> {
    let roots = self
      .tasks
      .keys()
      .filter_map(|product_type| self.gen_root_entry(&self.root_param_types, *product_type))
      .collect();
    self.construct_graph(roots)
  }

  fn construct_graph(&self, roots: Vec<RootEntry<R>>) -> RuleGraph<R> {
    let mut unfulfillable_rules = HashMap::new();

    // First construct a polymorphic graph (where each dependency edge might have multiple
    // possible implementations).
    let dependency_edges = Construct::new(&self.tasks, roots).transform(&mut unfulfillable_rules);

    // Then monomorphize it, turning it into a graph where each dependency edge has exactly one
    // possible implementation.
    let rule_dependency_edges =
      Monomorphize::new(dependency_edges).transform(&mut unfulfillable_rules);

    // Finally, compute which rules are unreachable/dead post-monomorphization (which will have
    // chosen concrete implementations for each edge).
    let unreachable_rules = self.unreachable_rules(&rule_dependency_edges);

    RuleGraph {
      root_param_types: self.root_param_types.clone(),
      rule_dependency_edges,
      unfulfillable_rules,
      unreachable_rules,
    }
  }

  ///
  /// Compute input TaskRules that are unreachable from root entries.
  ///
  fn unreachable_rules(
    &self,
    full_dependency_edges: &HashMap<EntryWithDeps<R>, RuleEdges<R>>,
  ) -> Vec<UnreachableError<R>> {
    // Walk the graph, starting from root entries.
    let mut entry_stack: Vec<_> = full_dependency_edges
      .keys()
      .filter(|entry| match entry {
        EntryWithDeps::Root(_) => true,
        _ => false,
      })
      .collect();
    let mut visited = HashSet::new();
    while let Some(entry) = entry_stack.pop() {
      if visited.contains(&entry) {
        continue;
      }
      visited.insert(entry);

      if let Some(edges) = full_dependency_edges.get(entry) {
        entry_stack.extend(edges.all_dependencies().filter_map(|e| match e {
          Entry::WithDeps(ref e) => Some(e),
          _ => None,
        }));
      }
    }

    let reachable_rules: HashSet<_> = visited
      .into_iter()
      .filter_map(|entry| match entry {
        EntryWithDeps::Inner(InnerEntry { ref rule, .. }) if rule.require_reachable() => {
          Some(rule.clone())
        }
        _ => None,
      })
      .collect();

    self
      .tasks
      .values()
      .flat_map(|r| r.iter())
      .filter(|r| r.require_reachable() && !reachable_rules.contains(r))
      .map(|r| UnreachableError::new(r.clone()))
      .collect()
  }

  fn gen_root_entry(
    &self,
    param_types: &ParamTypes<R::TypeId>,
    product_type: R::TypeId,
  ) -> Option<RootEntry<R>> {
    let candidates = rhs(&self.tasks, param_types, product_type);
    if candidates.is_empty() {
      None
    } else {
      Some(RootEntry {
        params: param_types.clone(),
        dependency_key: R::DependencyKey::new_root(product_type),
      })
    }
  }
}

///
/// Select rules or parameters that can provide the given product type with the given parameters.
///
fn rhs<R: Rule>(
  tasks: &HashMap<R::TypeId, Vec<R>>,
  params: &ParamTypes<R::TypeId>,
  product_type: R::TypeId,
) -> Vec<Entry<R>> {
  let mut entries = Vec::new();
  // If the params can provide the type directly, add that.
  if let Some(type_id) = params.get(&product_type) {
    entries.push(Entry::Param(*type_id));
  }
  // If there are any rules which can produce the desired type, add them.
  if let Some(matching_rules) = tasks.get(&product_type) {
    entries.extend(matching_rules.iter().map(|rule| {
      Entry::WithDeps(EntryWithDeps::Inner(InnerEntry {
        params: params.clone(),
        rule: rule.clone(),
      }))
    }));
  }
  entries
}

enum GraphTransformResult<Node> {
  // The node was satisfiable without waiting for any additional nodes to be satisfied. The result
  // contains simplified copies of the input node.
  Fulfilled(Vec<Node>),
  // The node was not satisfiable.
  Unfulfillable,
  // The dependencies of a node might be satisfiable, but it is currently blocked waiting for the
  // results of the given other nodes.
  //
  // Holds partially-fulfilled Entries which do not yet contain their full set of used parameters.
  // These entries are only consumed the case when a caller is the source of a cycle, and in that
  // case they represent everything except the caller's own parameters (which provides enough
  // information for the caller to complete).
  CycledOn {
    cyclic_deps: HashSet<Node>,
    simplified_nodes: Vec<Node>,
  },
}

trait GraphTransform<R: Rule> {
  type OutputEdges: Clone + Default;

  ///
  /// Given a polymorphic graph, where each Rule might have multiple implementations of each dep,
  /// monomorphize it into a graph where each Rule has exactly one implementation per dep.
  ///
  fn transform(
    &self,
    unfulfillable_nodes: &mut HashMap<EntryWithDeps<R>, Vec<Diagnostic<R::TypeId>>>,
  ) -> HashMap<EntryWithDeps<R>, Self::OutputEdges> {
    let mut output_graph = HashMap::new();
    let mut memoized_outputs = HashMap::new();
    for node in self.roots() {
      self.transform_graph_helper(
        &node,
        &mut output_graph,
        &mut memoized_outputs,
        unfulfillable_nodes,
      );
    }
    output_graph
  }

  fn roots<'a>(&'a self) -> Box<dyn Iterator<Item = EntryWithDeps<R>> + 'a>;

  fn edges_for(&self, node: &EntryWithDeps<R>) -> Vec<(R::DependencyKey, Vec<Entry<R>>)>;

  #[allow(clippy::type_complexity)]
  fn select_edges_for(
    &self,
    node: &EntryWithDeps<R>,
    candidates_by_key: HashMap<R::DependencyKey, Vec<(bool, Entry<R>)>>,
  ) -> (
    HashMap<EntryWithDeps<R>, Self::OutputEdges>,
    Vec<Diagnostic<R::TypeId>>,
  );

  fn unsatisfiable_dependency(
    node: &EntryWithDeps<R>,
    edge: &R::DependencyKey,
  ) -> Diagnostic<R::TypeId>;

  ///
  /// Computes whether the given node is satisfiable, and if it is, returns a copy of the node for
  /// each set of input parameters that will satisfy it. Once computed, the simplified versions are
  /// memoized in memoized_outputs.
  ///
  /// When a node can be fulfilled it will end up stored in both the output_graph and
  /// memoized_outputs. If it can't be fulfilled, it is added to unfulfillable_nodes.
  ///
  fn transform_graph_helper(
    &self,
    node: &EntryWithDeps<R>,
    output_graph: &mut HashMap<EntryWithDeps<R>, Self::OutputEdges>,
    memoized_outputs: &mut HashMap<EntryWithDeps<R>, Vec<EntryWithDeps<R>>>,
    unfulfillable_nodes: &mut HashMap<EntryWithDeps<R>, Vec<Diagnostic<R::TypeId>>>,
  ) -> GraphTransformResult<EntryWithDeps<R>> {
    if let Some(simplified) = memoized_outputs.get(&node) {
      // The monomorphized entries have already been computed, return them.
      return GraphTransformResult::Fulfilled(simplified.clone());
    } else if unfulfillable_nodes.get(&node).is_some() {
      // The node is unfulfillable.
      return GraphTransformResult::Unfulfillable;
    }

    // Otherwise, store a placeholder in the output_graph map and then visit its children.
    //
    // This prevents infinite recursion by shortcircuiting when an node recursively depends on
    // itself. It's totally fine for nodes to be recursive: the recursive path just never
    // contributes to whether the node is satisfiable.
    match output_graph.entry(node.clone()) {
      hash_map::Entry::Vacant(re) => {
        // When a node has not been visited before, we start the visit by storing a placeholder in
        // the node dependencies map in order to detect node cycles.
        re.insert(Self::OutputEdges::default());
      }
      hash_map::Entry::Occupied(_) => {
        // We're currently recursively under this node, but its simplified equivalence has not yet
        // been computed (or we would have returned it above). The cyclic parent(s) will complete
        // before recursing to compute this node again.
        let mut cyclic_deps = HashSet::new();
        cyclic_deps.insert(node.clone());
        return GraphTransformResult::CycledOn {
          cyclic_deps,
          simplified_nodes: vec![node.simplified(BTreeSet::new())],
        };
      }
    };

    // For each dependency of the node, recurse for each potential match and collect RuleEdges and
    // used parameters.
    //
    // This is a `loop` because if we discover that this node needs to complete in order to break
    // a cycle on itself, it will re-compute dependencies after having partially-completed.
    loop {
      if let Ok(res) =
        self.transform_dependencies(node, output_graph, memoized_outputs, unfulfillable_nodes)
      {
        break res;
      }
    }
  }

  ///
  /// Given an node and a mapping of all legal sources of each of its dependencies, recursively
  /// generates a simplified node for each legal combination of parameters.
  ///
  fn transform_dependencies(
    &self,
    node: &EntryWithDeps<R>,
    output_graph: &mut HashMap<EntryWithDeps<R>, Self::OutputEdges>,
    memoized_outputs: &mut HashMap<EntryWithDeps<R>, Vec<EntryWithDeps<R>>>,
    unfulfillable_nodes: &mut HashMap<EntryWithDeps<R>, Vec<Diagnostic<R::TypeId>>>,
  ) -> Result<GraphTransformResult<EntryWithDeps<R>>, ()> {
    // Begin by recursively finding our monomorphized deps.
    let mut candidates_by_key = HashMap::new();
    let mut cycled_on = HashSet::new();
    let mut unfulfillable_diagnostics = Vec::new();

    for (edge, inputs) in self.edges_for(node) {
      let mut cycled = false;
      let candidates = candidates_by_key.entry(edge).or_insert_with(Vec::new);
      for input in inputs {
        match input {
          Entry::WithDeps(ref e) => {
            match self.transform_graph_helper(
              &e,
              output_graph,
              memoized_outputs,
              unfulfillable_nodes,
            ) {
              GraphTransformResult::Unfulfillable => {}
              GraphTransformResult::Fulfilled(simplified_nodes) => {
                candidates.extend(simplified_nodes.into_iter().map(|e| (false, e.into())));
              }
              GraphTransformResult::CycledOn {
                cyclic_deps,
                simplified_nodes,
              } => {
                cycled = true;
                cycled_on.extend(cyclic_deps);
                // NB: In the case of a cycle, we consider the dependency to be fulfillable, because
                // it is if we are.
                candidates.extend(simplified_nodes.into_iter().map(|e| (true, e.into())));
              }
            }
          }
          Entry::Param(_) => {
            candidates.push((false, input));
          }
        }
      }

      if cycled {
        // If any candidate triggered a cycle on a rule that has not yet completed, then we are not
        // yet fulfillable, and should finish gathering any other cyclic rule dependencies.
        continue;
      }

      if candidates.is_empty() {
        // If no candidates were fulfillable, this rule is not fulfillable.
        unfulfillable_diagnostics.push(Self::unsatisfiable_dependency(node, &edge));
      }
    }

    // If any dependencies were completely unfulfillable, then whether or not there were cyclic
    // dependencies isn't relevant.
    if !unfulfillable_diagnostics.is_empty() {
      // Was not fulfillable. Remove the placeholder: the unfulfillable entries we stored will
      // prevent us from attempting to expand this node again.
      unfulfillable_nodes
        .entry(node.clone())
        .or_insert_with(Vec::new)
        .extend(unfulfillable_diagnostics);
      output_graph.remove(&node);
      return Ok(GraphTransformResult::Unfulfillable);
    }

    let (selected_edges, diagnostics) = self.select_edges_for(&node, candidates_by_key);

    let simplified_nodes: Vec<_> = selected_edges.keys().cloned().collect();

    // If none of the selected_edges was satisfiable, store the generated diagnostics: otherwise,
    // store the memoized resulting entries.
    output_graph.remove(&node);
    if cycled_on.is_empty() {
      // No deps were blocked on cycles.
      if selected_edges.is_empty() {
        unfulfillable_nodes
          .entry(node.clone())
          .or_insert_with(Vec::new)
          .extend(diagnostics);
        Ok(GraphTransformResult::Unfulfillable)
      } else {
        output_graph.extend(selected_edges.clone());
        memoized_outputs.insert(node.clone(), simplified_nodes.clone());
        Ok(GraphTransformResult::Fulfilled(simplified_nodes))
      }
    } else {
      // The set of cycled dependencies can only contain call stack "parents" of the dependency: we
      // remove this entry from the set (if we're in it), until the top-most cyclic parent
      // (represented by an empty set) is the one that re-starts recursion.
      cycled_on.remove(node);
      if cycled_on.is_empty() {
        memoized_outputs.insert(node.clone(), simplified_nodes);
        Err(())
      } else {
        // This rule may be fulfillable, but we can't compute its complete set of dependencies until
        // parent rule entries complete.
        Ok(GraphTransformResult::CycledOn {
          cyclic_deps: cycled_on,
          simplified_nodes: simplified_nodes,
        })
      }
    }
  }
}

struct Construct<'t, R: Rule> {
  tasks: &'t HashMap<R::TypeId, Vec<R>>,
  roots: Vec<RootEntry<R>>,
}

impl<R: Rule> GraphTransform<R> for Construct<'_, R> {
  type OutputEdges = PolyRuleEdges<R>;

  fn roots<'a>(&'a self) -> Box<dyn Iterator<Item = EntryWithDeps<R>> + 'a> {
    Box::new(self.roots.iter().map(|r| EntryWithDeps::Root(r.clone())))
  }

  fn edges_for(&self, node: &EntryWithDeps<R>) -> Vec<(R::DependencyKey, Vec<Entry<R>>)> {
    node
      .dependency_keys()
      .into_iter()
      .map(|dependency_key| {
        let product = dependency_key.product();
        let provided_param = dependency_key.provided_param();
        let params = if let Some(provided_param) = provided_param {
          // The dependency key provides a parameter: include it in the Params that are already in
          // the context.
          let mut params = node.params().clone();
          params.insert(provided_param);
          params
        } else {
          node.params().clone()
        };
        (dependency_key, rhs(&self.tasks, &params, product))
      })
      .collect()
  }

  #[allow(clippy::type_complexity)]
  fn select_edges_for(
    &self,
    node: &EntryWithDeps<R>,
    candidates_by_key: HashMap<R::DependencyKey, Vec<(bool, Entry<R>)>>,
  ) -> (
    HashMap<EntryWithDeps<R>, Self::OutputEdges>,
    Vec<Diagnostic<R::TypeId>>,
  ) {
    let edges = PolyRuleEdges {
      dependencies: candidates_by_key
        .into_iter()
        .map(|(edge, candidates)| {
          (
            edge,
            candidates
              .into_iter()
              .map(|(_cyclic, candidate)| candidate)
              .collect(),
          )
        })
        .collect(),
    };
    let simplified_node = {
      // NB: The set of dependencies is further pruned by monomorphization, but we prune it here
      // since it results in a more accurate graph (and better error messages) earlier.
      let mut all_used_params = BTreeSet::new();
      for (key, inputs) in &edges.dependencies {
        let provided_param = key.provided_param();
        for input in inputs {
          all_used_params.extend(
            input
              .params()
              .into_iter()
              .filter(|p| Some(*p) != provided_param),
          );
        }
      }
      node.simplified(all_used_params)
    };

    let mut edges_by_node = HashMap::new();
    edges_by_node.insert(simplified_node, edges);
    (edges_by_node, vec![])
  }

  fn unsatisfiable_dependency(
    node: &EntryWithDeps<R>,
    edge: &R::DependencyKey,
  ) -> Diagnostic<R::TypeId> {
    let params = node.params();
    Diagnostic {
      params: params.clone(),
      reason: if params.is_empty() {
        format!(
          "No rule was available to compute {}. Maybe declare it as a RootRule({})?",
          edge,
          edge.product(),
        )
      } else {
        format!(
          "No rule was available to compute {} with parameter type{} {}",
          edge,
          if params.len() > 1 { "s" } else { "" },
          params_str(params),
        )
      },
      details: vec![],
    }
  }
}

impl<'t, R: Rule> Construct<'t, R> {
  pub fn new(tasks: &'t HashMap<R::TypeId, Vec<R>>, roots: Vec<RootEntry<R>>) -> Construct<R> {
    Construct { tasks, roots }
  }
}

struct Monomorphize<R: Rule> {
  input_graph: HashMap<EntryWithDeps<R>, PolyRuleEdges<R>>,
}

impl<R: Rule> GraphTransform<R> for Monomorphize<R> {
  type OutputEdges = RuleEdges<R>;

  fn roots<'a>(&'a self) -> Box<dyn Iterator<Item = EntryWithDeps<R>> + 'a> {
    Box::new(
      self
        .input_graph
        .keys()
        .filter(|node| match node {
          EntryWithDeps::Root(_) => true,
          EntryWithDeps::Inner(_) => false,
        })
        .cloned(),
    )
  }

  fn edges_for(&self, node: &EntryWithDeps<R>) -> Vec<(R::DependencyKey, Vec<Entry<R>>)> {
    self
      .input_graph
      .get(node)
      .unwrap()
      .dependencies
      .iter()
      .map(|(e, n)| (*e, n.clone()))
      .collect()
  }

  #[allow(clippy::type_complexity)]
  fn select_edges_for(
    &self,
    node: &EntryWithDeps<R>,
    candidates_by_key: HashMap<R::DependencyKey, Vec<(bool, Entry<R>)>>,
  ) -> (
    HashMap<EntryWithDeps<R>, Self::OutputEdges>,
    Vec<Diagnostic<R::TypeId>>,
  ) {
    let monomorphized_candidates: Vec<_> = candidates_by_key.into_iter().collect();

    // Collect the powerset of the union of used parameters, ordered by set size.
    let params_powerset: Vec<Vec<R::TypeId>> = {
      // Compute the powerset ordered by ascending set size.
      let mut all_used_params = BTreeSet::new();
      for (key, inputs) in &monomorphized_candidates {
        let provided_param = key.provided_param();
        for (_, input) in inputs {
          all_used_params.extend(
            input
              .params()
              .into_iter()
              .filter(|p| Some(*p) != provided_param),
          );
        }
      }
      let mut param_sets =
        Self::powerset(&all_used_params.into_iter().collect::<Vec<_>>()).collect::<Vec<_>>();
      param_sets.sort_by(|l, r| l.len().cmp(&r.len()));
      param_sets
    };

    // Then, for the powerset of used parameters, determine which dependency combinations are
    // satisfiable.
    let mut combinations: HashMap<EntryWithDeps<_>, _> = HashMap::new();
    let mut diagnostics = Vec::new();
    for available_params in params_powerset {
      let available_params = available_params.into_iter().collect();
      // If a subset of these parameters is already satisfied, skip. This has the effect of
      // selecting the smallest sets of parameters that will satisfy a rule.
      // NB: This scan over satisfied sets is linear, but should have a small N.
      if combinations
        .keys()
        .any(|satisfied_entry| satisfied_entry.params().is_subset(&available_params))
      {
        continue;
      }

      match Self::choose_dependencies(&available_params, &monomorphized_candidates) {
        Ok(Some(rule_edges)) => {
          combinations.insert(node.simplified(available_params), rule_edges);
        }
        Ok(None) => {}
        Err(diagnostic) => diagnostics.push(diagnostic),
      }
    }

    (combinations, diagnostics)
  }

  fn unsatisfiable_dependency(
    node: &EntryWithDeps<R>,
    edge: &R::DependencyKey,
  ) -> Diagnostic<R::TypeId> {
    let params = node.params();
    Diagnostic {
      params: params.clone(),
      reason: if params.is_empty() {
        format!(
          "No rule was available to compute {}. Maybe declare it as a RootRule({})?",
          edge,
          edge.product(),
        )
      } else {
        format!(
          "No rule was available to compute {} with parameter type{} {}",
          edge,
          if params.len() > 1 { "s" } else { "" },
          params_str(params),
        )
      },
      details: vec![],
    }
  }
}

impl<R: Rule> Monomorphize<R> {
  fn new(input_graph: HashMap<EntryWithDeps<R>, PolyRuleEdges<R>>) -> Monomorphize<R> {
    Monomorphize { input_graph }
  }

  ///
  /// Given a set of available Params, choose one combination of satisfiable Entry dependencies if
  /// it exists (it may not, because we're searching for sets of legal parameters in the powerset
  /// of all used params).
  ///
  /// If an ambiguity is detected in rule dependencies (ie, if multiple rules are satisfiable for
  /// a single dependency key), fail with a Diagnostic.
  ///
  fn choose_dependencies(
    available_params: &ParamTypes<R::TypeId>,
    deps: &[ChosenDependency<R>],
  ) -> Result<Option<RuleEdges<R>>, Diagnostic<R::TypeId>> {
    let mut combination = RuleEdges::default();
    for (key, input_entries) in deps {
      let provided_param = key.provided_param();
      let satisfiable_entries = input_entries
        .iter()
        .filter_map(|(cyclic, input_entry)| {
          let consumes_provided_param = if let Some(p) = provided_param {
            input_entry.params().contains(&p)
          } else {
            true
          };
          let accept = (*cyclic || consumes_provided_param)
            && input_entry
              .params()
              .iter()
              .all(|p| available_params.contains(p) || Some(*p) == provided_param);
          if accept {
            Some(input_entry)
          } else {
            None
          }
        })
        .collect::<Vec<_>>();

      let chosen_entries = Self::choose_dependency(satisfiable_entries);
      match chosen_entries.len() {
        0 => {
          return Ok(None);
        }
        1 => {
          combination.add_edge(key.clone(), chosen_entries[0].clone());
        }
        _ => {
          let params_clause = match available_params.len() {
            0 => "",
            1 => " with parameter type ",
            _ => " with parameter types ",
          };

          return Err(Diagnostic {
            params: available_params.clone(),
            reason: format!(
              "Ambiguous rules to compute {}{}{}",
              key,
              params_clause,
              params_str(&available_params),
            ),
            details: chosen_entries.into_iter().map(entry_str).collect(),
          });
        }
      }
    }

    Ok(Some(combination))
  }

  fn choose_dependency<'a>(satisfiable_entries: Vec<&'a Entry<R>>) -> Vec<&'a Entry<R>> {
    if satisfiable_entries.is_empty() {
      // No source of this dependency was satisfiable with these Params.
      return vec![];
    } else if satisfiable_entries.len() == 1 {
      return satisfiable_entries;
    }

    // We prefer the non-ambiguous entry with the smallest set of Params, as that minimizes Node
    // identities in the graph and biases toward receiving values from dependencies (which do not
    // affect our identity) rather than dependents.
    let mut minimum_param_set_size = ::std::usize::MAX;
    let mut rules = Vec::new();
    for satisfiable_entry in satisfiable_entries {
      let param_set_size = match satisfiable_entry {
        Entry::WithDeps(ref wd) => wd.params().len(),
        Entry::Param(_) => 1,
      };
      match param_set_size.cmp(&minimum_param_set_size) {
        Ordering::Less => {
          rules.clear();
          rules.push(satisfiable_entry);
          minimum_param_set_size = param_set_size;
        }
        Ordering::Equal => {
          rules.push(satisfiable_entry);
        }
        Ordering::Greater => {}
      }
    }

    rules
  }

  fn powerset<'a, T: Clone>(slice: &'a [T]) -> impl Iterator<Item = Vec<T>> + 'a {
    (0..(1 << slice.len())).map(move |mask| {
      let mut ss = Vec::new();
      let mut bitset = mask;
      while bitset > 0 {
        // isolate the rightmost bit to select one item
        let rightmost: u64 = bitset & !(bitset - 1);
        // turn the isolated bit into an array index
        let idx = rightmost.trailing_zeros();
        let item = &slice[idx as usize];
        ss.push(item.clone());
        // zero the trailing bit
        bitset &= bitset - 1;
      }
      ss
    })
  }
}
