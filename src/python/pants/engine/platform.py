# Copyright 2019 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from enum import Enum

from pants.engine.rules import RootRule
from pants.util.memo import memoized_classproperty
from pants.util.osutil import get_normalized_os_name


class Platform(Enum):
    darwin = "darwin"
    linux = "linux"

    # TODO: try to turn all of these accesses into v2 dependency injections!
    @memoized_classproperty
    def current(cls) -> "Platform":
        return Platform(get_normalized_os_name())


class PlatformConstraint(Enum):
    darwin = "darwin"
    linux = "linux"
    none = "none"

    @memoized_classproperty
    def local_platform(cls) -> "PlatformConstraint":
        return PlatformConstraint(Platform.current.value)


def create_platform_rules():
    return [RootRule(Platform)]
