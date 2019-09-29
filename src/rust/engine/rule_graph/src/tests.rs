use crate::{Palette, RuleGraph};
use std::fmt;

#[test]
fn valid() {
  let rules = vec![("a", vec![Rule("a_from_b", vec![DependencyKey("b", None)])])]
    .into_iter()
    .collect();
  let roots = vec!["b"];
  let graph = RuleGraph::new(&rules, roots);

  graph.validate().unwrap();
}

#[test]
fn no_root() {
  let rules = vec![("a", vec![Rule("a_from_b", vec![DependencyKey("b", None)])])]
    .into_iter()
    .collect();
  let roots = vec![];
  let graph = RuleGraph::new(&rules, roots);

  assert!(graph
    .validate()
    .err()
    .unwrap()
    .contains("No rule was available to compute DependencyKey(\"b\", None)."));
}

#[test]
fn recursion() {
  let rules = vec![(
    "Fib",
    vec![Rule(
      "fib",
      vec![
        DependencyKey("int", None),
        DependencyKey("Fib", Some("int")),
      ],
    )],
  )]
  .into_iter()
  .collect();
  let roots = vec!["Fib", "int", "nonsense"];
  let graph = RuleGraph::new(&rules, roots);

  graph.validate().unwrap();
  graph.find_exact_root_edges(vec!["int"], "Fib").unwrap();
  graph.find_exact_root_edges(vec!["Fib"], "Fib").unwrap();
}

#[test]
fn mutual_recursion_in_get() {
  let rules = vec![
    (
      "IsEven",
      vec![Rule(
        "is_even",
        vec![
          DependencyKey("int", None),
          DependencyKey("IsOdd", Some("int")),
        ],
      )],
    ),
    (
      "IsOdd",
      vec![Rule(
        "is_odd",
        vec![
          DependencyKey("int", None),
          DependencyKey("IsEven", Some("int")),
        ],
      )],
    ),
  ]
  .into_iter()
  .collect();
  let roots = vec!["IsEven", "IsOdd", "int", "nonsense"];
  let graph = RuleGraph::new(&rules, roots);

  graph.validate().unwrap();
  graph.find_exact_root_edges(vec!["int"], "IsEven").unwrap();
  graph.find_exact_root_edges(vec!["int"], "IsOdd").unwrap();
}

#[test]
fn mutual_recursion_in_select() {
  let rules = vec![
    (
      "Example",
      vec![Rule(
        "Example",
        vec![DependencyKey("Digest", Some("FilesContent"))],
      )],
    ),
    (
      "FilesContent",
      vec![Rule(
        "files_content_from_digest",
        vec![DependencyKey("Digest", None)],
      )],
    ),
    (
      "Digest",
      vec![Rule(
        "digest_from_files_content",
        vec![DependencyKey("FilesContent", None)],
      )],
    ),
  ]
  .into_iter()
  .collect();
  let roots = vec!["Digest", "FilesContent", "nonsense"];
  let graph = RuleGraph::new(&rules, roots);

  graph.validate().unwrap();
  graph
    .find_exact_root_edges(vec!["Digest"], "FilesContent")
    .unwrap();
  graph
    .find_exact_root_edges(vec!["FilesContent"], "Digest")
    .unwrap();
  graph.find_exact_root_edges(vec![], "Example").unwrap();
}

impl super::TypeId for &'static str {
  fn display<I>(type_ids: I) -> String
  where
    I: Iterator<Item = Self>,
  {
    type_ids.collect::<Vec<_>>().join("+")
  }
}

// A name and vec of DependencyKeys. Abbreviated for simpler construction and matching.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Rule(&'static str, Vec<DependencyKey>);

impl super::Rule for Rule {
  type TypeId = &'static str;
  type DependencyKey = DependencyKey;

  fn dependency_keys(&self) -> Vec<Self::DependencyKey> {
    self.1.clone()
  }

  fn require_reachable(&self) -> bool {
    true
  }

  fn color(&self) -> Option<Palette> {
    None
  }
}

impl super::DisplayForGraph for Rule {
  fn fmt_for_graph(&self) -> String {
    "???".to_string()
  }
}

impl fmt::Display for Rule {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    write!(f, "{:?}", self)
  }
}

// A product and a param. Abbreviated for simpler construction and matching.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
struct DependencyKey(&'static str, Option<&'static str>);

impl super::DependencyKey for DependencyKey {
  type TypeId = &'static str;

  fn new_root(product: Self::TypeId) -> Self {
    DependencyKey(product, None)
  }

  fn product(&self) -> Self::TypeId {
    self.0
  }

  fn provided_param(&self) -> Option<Self::TypeId> {
    self.1
  }
}

impl fmt::Display for DependencyKey {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    write!(f, "{:?}", self)
  }
}
