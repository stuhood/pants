# Copyright 2019 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).

python_library(
  name = 'engine_test_base',
  sources = ['base_engine_test.py'],
  dependencies = [
    'src/python/pants/goal',
    'src/python/pants/goal:task_registrar',
    'src/python/pants/testutil:test_base',
  ],
  tags = {"partially_type_checked"},
)


python_library(
  name = 'util',
  sources = ['util.py'],
  dependencies = [
    '3rdparty/python:ansicolors',
    '3rdparty/python:dataclasses',
    'src/python/pants/base:project_tree',
    'src/python/pants/binaries',
    'src/python/pants/engine:addressable',
    'src/python/pants/engine:native',
    'src/python/pants/engine:parser',
    'src/python/pants/engine:rules',
    'src/python/pants/engine:scheduler',
    'src/python/pants/engine:struct',
    'src/python/pants/engine:unions',
    'src/python/pants/util:objects',
  ],
  tags = {"partially_type_checked"},
)
