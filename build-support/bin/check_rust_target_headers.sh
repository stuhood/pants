#!/usr/bin/env bash

REPO_ROOT="$(git rev-parse --show-toplevel)"

cargo="${REPO_ROOT}/build-support/bin/native/cargo"

if ! "${cargo}" install --list | grep -q "cargo-ensure-prefix v0.1.3"; then
  "${cargo}" install --force --version 0.1.3 cargo-ensure-prefix
fi

if ! out="$("${cargo}" ensure-prefix \
  --manifest-path="${REPO_ROOT}/src/rust/engine/Cargo.toml" \
  --prefix-path="${REPO_ROOT}/build-support/rust-target-prefix.txt" \
  --all --exclude=bazel_protos)"; then
  echo >&2 "Rust targets didn't have correct prefix:"
  echo >&2 "${out}"
  exit 1
fi
