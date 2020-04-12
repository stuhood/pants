# Copyright 2020 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).


# This is a temporary hack that allows us to note the fact that we're in v2-exclusive mode
# in a static location, as soon as we know it. This way code that cannot access options
# can still use this information to customize behavior. Again, this is a temporary hack
# to provide a better v2 experience to users who are not (and possibly never have been)
# running v1, and should go away ASAP.
class IsV2Exclusive:
    def __init__(self):
        self._value = False

    def set(self):
        self._value = True

    def __bool__(self):
        return self._value


is_v2_exclusive = IsV2Exclusive()
