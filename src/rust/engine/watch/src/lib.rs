// Copyright 2020 Pants project contributors (see CONTRIBUTORS.md).
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

#[cfg(test)]
mod tests;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Weak};
use std::thread;
use std::time::Duration;

use crossbeam_channel::{self, Receiver, RecvTimeoutError, TryRecvError};
use fs::GitignoreStyleExcludes;
use log::{debug, warn};
use logging;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use parking_lot::Mutex;
use task_executor::Executor;

///
/// An InvalidationWatcher maintains a Thread that receives events from a notify Watcher.
///
/// If the spawned Thread exits for any reason, InvalidationWatcher::running() will return False,
/// and the caller should create a new InvalidationWatcher (or shut down, in some cases). Generally
/// this will mean polling.
///
struct Inner {
  watcher: RecommendedWatcher,
  invalidatables: Vec<Weak<dyn Invalidatable>>,
  executor: Executor,
  liveness: Receiver<String>,
  enabled: bool,
  // Until the background task has started, contains the relevant inputs to launch it via
  // start_background_thread. The decoupling of creating the `InvalidationWatcher` and starting it
  // is to allow for testing of the background thread.
  background_task_inputs: Option<WatcherTaskInputs>,
}

type WatcherTaskInputs = (
  Arc<GitignoreStyleExcludes>,
  PathBuf,
  crossbeam_channel::Sender<String>,
  Receiver<notify::Result<notify::Event>>,
);

pub struct InvalidationWatcher(Mutex<Inner>);

impl InvalidationWatcher {
  pub fn new(
    executor: Executor,
    build_root: PathBuf,
    ignorer: Arc<GitignoreStyleExcludes>,
    enabled: bool,
  ) -> Result<Arc<InvalidationWatcher>, String> {
    // Inotify events contain canonical paths to the files being watched.
    // If the build_root contains a symlink the paths returned in notify events
    // wouldn't have the build_root as a prefix, and so we would miss invalidating certain nodes.
    // We canonicalize the build_root once so this isn't a problem.
    let canonical_build_root =
      std::fs::canonicalize(build_root.as_path()).map_err(|e| format!("{:?}", e))?;
    let (watch_sender, watch_receiver) = crossbeam_channel::unbounded();
    let mut watcher: RecommendedWatcher = Watcher::new(watch_sender, Duration::from_millis(50))
      .map_err(|e| format!("Failed to begin watching the filesystem: {}", e))?;

    let (liveness_sender, liveness_receiver) = crossbeam_channel::unbounded();
    if enabled {
      // On darwin the notify API is much more efficient if you watch the build root
      // recursively, so we set up that watch here and then return early when watch() is
      // called by nodes that are running. On Linux the notify crate handles adding paths to watch
      // much more efficiently so we do that instead on Linux.
      if cfg!(target_os = "macos") {
        watcher
          .watch(canonical_build_root.clone(), RecursiveMode::Recursive)
          .map_err(|e| {
            format!(
              "Failed to begin recursively watching files in the build root: {}",
              e
            )
          })?
      }
    }
    Ok(Arc::new(InvalidationWatcher(Mutex::new(Inner {
      watcher,
      invalidatables: Vec::new(),
      executor,
      liveness: liveness_receiver,
      enabled,
      background_task_inputs: Some((
        ignorer,
        canonical_build_root,
        liveness_sender,
        watch_receiver,
      )),
    }))))
  }

  ///
  /// Starts the background task that monitors watch events. Panics if called more than once.
  ///
  pub async fn start(self: &Arc<Self>) {
    let mut inner = self.0.lock();
    let (ignorer, canonical_build_root, liveness_sender, watch_receiver) = inner
      .background_task_inputs
      .take()
      .expect("An InvalidationWatcher can only be started once.");

    InvalidationWatcher::start_background_thread(
      Arc::downgrade(self),
      ignorer,
      canonical_build_root,
      liveness_sender,
      watch_receiver,
    );
  }

  // Public for testing purposes.
  pub(crate) fn start_background_thread(
    invalidation_watcher: Weak<InvalidationWatcher>,
    ignorer: Arc<GitignoreStyleExcludes>,
    canonical_build_root: PathBuf,
    liveness_sender: crossbeam_channel::Sender<String>,
    watch_receiver: Receiver<notify::Result<notify::Event>>,
  ) -> thread::JoinHandle<()> {
    thread::spawn(move || {
      logging::set_thread_destination(logging::Destination::Pantsd);
      let exit_msg = loop {
        let event_res = watch_receiver.recv_timeout(Duration::from_millis(100));
        let watcher = {
          if let Some(i) = invalidation_watcher.upgrade() {
            i
          } else {
            break "The watcher was shut down.".to_string();
          }
        };
        match event_res {
          Ok(Ok(ev)) => {
            let paths: HashSet<_> = ev
              .paths
              .into_iter()
              .filter_map(|path| {
                // relativize paths to build root.
                let path_relative_to_build_root = if path.starts_with(&canonical_build_root) {
                  // Unwrapping is fine because we check that the path starts with
                  // the build root above.
                  path.strip_prefix(&canonical_build_root).unwrap().into()
                } else {
                  path
                };
                // To avoid having to stat paths for events we will eventually ignore we "lie" to the ignorer
                // to say that no path is a directory, they could be if someone chmod's or creates a dir.
                // This maintains correctness by ensuring that at worst we have false negative events, where a directory
                // only glob (one that ends in `/` ) was supposed to ignore a directory path, but didn't because we claimed it was a file. That
                // directory path will be used to invalidate nodes, but won't invalidate anything because its path is somewhere
                // out of our purview.
                if ignorer.is_ignored_or_child_of_ignored_path(
                  &path_relative_to_build_root,
                  /* is_dir */ false,
                ) {
                  None
                } else {
                  Some(path_relative_to_build_root)
                }
              })
              .flat_map(|path_relative_to_build_root| {
                let mut paths_to_invalidate: Vec<PathBuf> = vec![];
                if let Some(parent_dir) = path_relative_to_build_root.parent() {
                  paths_to_invalidate.push(parent_dir.to_path_buf());
                }
                paths_to_invalidate.push(path_relative_to_build_root);
                paths_to_invalidate
              })
              .collect();

            // Only invalidate stuff if we have paths that weren't filtered out by gitignore.
            if paths.is_empty() {
              continue;
            };

            debug!("notify invalidating {:?} because of {:?}", paths, ev.kind);
            for invalidatable in watcher.collect_live_invalidatables() {
              invalidatable.invalidate(&paths, "notify");
            }
          }
          Ok(Err(err)) => {
            if let notify::ErrorKind::PathNotFound = err.kind {
              warn!("Path(s) did not exist: {:?}", err.paths);
              continue;
            } else {
              break format!("Watch error: {}", err);
            }
          }
          Err(RecvTimeoutError::Timeout) => continue,
          Err(RecvTimeoutError::Disconnected) => {
            break "The watch provider exited.".to_owned();
          }
        };
      };

      // Log and send the exit code.
      warn!("File watcher exiting with: {}", exit_msg);
      let _ = liveness_sender.send(exit_msg);
    })
  }

  ///
  /// Collect a copy of all Invalidatables that are still alive, while cleaning up any that are not.
  ///
  fn collect_live_invalidatables(&self) -> Vec<Arc<dyn Invalidatable>> {
    let mut watcher = self.0.lock();
    let mut invalidatables = Vec::new();
    watcher.invalidatables.retain(|invalidatable| {
      if let Some(i) = invalidatable.upgrade() {
        // Still alive: retain a copy.
        invalidatables.push(i);
        true
      } else {
        // No longer alive: remove.
        false
      }
    });
    invalidatables
  }

  ///
  /// Register an Invalidatable that will be called when any of the paths added via add_watched are
  /// changed. If all other copies of the `Arc<Invalidatable>` are dropped, the watch will be
  /// removed.
  ///
  /// It is legal to call `watch` before `start` or `is_valid`, but if the InvalidationWatcher has
  /// not been started or is no longer valid, no events will ever arrive on the Invalidatable.
  ///
  pub async fn add_invalidatable<I: Invalidatable>(&self, invalidatable: Arc<I>) {
    let invalidatable: Arc<dyn Invalidatable> = invalidatable;
    let mut watcher = self.0.lock();
    watcher.invalidatables.push(Arc::downgrade(&invalidatable));
  }

  ///
  /// An InvalidationWatcher will never restart on its own: a consumer should re-initialize if this
  /// method returns an error.
  ///
  /// NB: This is currently polled by pantsd, but it could be long-polled or a callback.
  ///
  pub async fn is_valid(&self) -> Result<(), String> {
    // Confirm that the Watcher itself is still alive.
    let watcher = self.0.lock();
    match watcher.liveness.try_recv() {
      Ok(msg) => {
        // The watcher background task set the exit condition.
        Err(msg)
      }
      Err(TryRecvError::Disconnected) => {
        // The watcher background task died (panic, possible?).
        Err(
          "The filesystem watcher exited abnormally: please see the log for more information."
            .to_owned(),
        )
      }
      Err(TryRecvError::Empty) => {
        // Still alive.
        Ok(())
      }
    }
  }

  ///
  /// Add a path to the set of paths being watched by this invalidation watcher, non-recursively.
  ///
  pub async fn watch(self: &Arc<Self>, path: PathBuf) -> Result<(), notify::Error> {
    let executor = {
      let inner = self.0.lock();
      if cfg!(target_os = "macos") || !inner.enabled {
        // Short circuit here if we are on a Darwin platform because we should be watching
        // the entire build root recursively already, or if we are not enabled.
        return Ok(());
      }
      inner.executor.clone()
    };

    let watcher = self.clone();
    executor
      .spawn_blocking(move || {
        let mut inner = watcher.0.lock();
        inner.watcher.watch(path, RecursiveMode::NonRecursive)
      })
      .await
  }
}

pub trait Invalidatable: Send + Sync + 'static {
  fn invalidate(&self, paths: &HashSet<PathBuf>, caller: &str) -> usize;
}
