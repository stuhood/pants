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
  clippy::unseparated_literal_suffix,
  // TODO: Falsely triggers for async/await:
  //   see https://github.com/rust-lang/rust-clippy/issues/5360
  // clippy::used_underscore_binding
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
// We only use unsafe pointer dereferences in our no_mangle exposed API, but it is nicer to list
// just the one minor call as unsafe, than to mark the whole function as unsafe which may hide
// other unsafeness.
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![type_length_limit = "42187898"]

mod context;
mod core;
mod externs;
mod interning;
mod intrinsics;
mod nodes;
mod scheduler;
mod selectors;
mod tasks;
mod types;

// TODO: This is no longer an importable module, so these `pub use` statements should be
// removed/inlined-into-consumers-within-this-crate.
pub use crate::context::Core;
pub use crate::core::{Failure, Function, Key, Params, TypeId, Value};
pub use crate::intrinsics::Intrinsics;
pub use crate::scheduler::{ExecutionRequest, ExecutionTermination, Scheduler, Session};
pub use crate::tasks::{Rule, Tasks};
pub use crate::types::Types;

// Include the pyoxidizer config created by `build.rs`.
include!(env!("PYOXIDIZER_DEFAULT_PYTHON_CONFIG_RS"));

pub fn main() {
    // The following code is in a block so the MainPythonInterpreter is destroyed in an
    // orderly manner, before process exit.
    let code = {
        // Load the configuration defined by the include! above.
        let config = default_python_config();

        // Construct a new Python interpreter using that config, handling any errors
        // from construction.
        match pyembed::MainPythonInterpreter::new(config) {
            Ok(mut interp) => {
                // And run it using the default run configuration as specified by the
                // configuration. If an uncaught Python exception is raised, handle it.
                // This includes the special SystemExit, which is a request to terminate the
                // process.
                interp.run_as_main()
            }
            Err(msg) => {
                eprintln!("{}", msg);
                1
            }
        }
    };

    // And exit the process according to code execution results.
    std::process::exit(code);
}