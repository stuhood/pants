# Copyright 2020 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import typing
from dataclasses import dataclass
from typing import Dict, Type

from pants.util.meta import decorated_type_checkable
from pants.util.ordered_set import OrderedSet


@decorated_type_checkable
def union(cls):
    """A class decorator which other classes can specify that they can resolve to with `UnionRule`.

    Annotating a class with @union allows other classes to use a UnionRule() instance to indicate
    that they can be resolved to this base union class. This class will never be instantiated, and
    should have no members -- it is used as a tag only, and will be replaced with whatever object is passed
    in as the subject of a `await Get(...)`. See the following example:

    @union
    class UnionBase: pass

    @rule
    async def get_some_union_type(x: X) -> B:
      result = await Get(ResultType, UnionBase, x.f())
      # ...

    If there exists a single path from (whatever type the expression `x.f()` returns) -> `ResultType`
    in the rule graph, the engine will retrieve and execute that path to produce a `ResultType` from
    `x.f()`. This requires also that whatever type `x.f()` returns was registered as a union member of
    `UnionBase` with a `UnionRule`.

    Unions allow @rule bodies to be written without knowledge of what types may eventually be provided
    as input -- rather, they let the engine check that there is a valid path to the desired result.
    """
    # TODO: Check that the union base type is used as a tag and nothing else (e.g. no attributes)!
    assert isinstance(cls, type)

    def non_member_error_message(subject):
        if hasattr(cls, "non_member_error_message"):
            return cls.non_member_error_message(subject)
        desc = f' ("{cls.__doc__}")' if cls.__doc__ else ""
        return f"Type {type(subject).__name__} is not a member of the {cls.__name__} @union{desc}"

    return union.define_instance_of(
        cls, non_member_error_message=staticmethod(non_member_error_message)
    )


@dataclass(frozen=True)
class UnionMembership:
    union_rules: Dict[Type, OrderedSet[Type]]

    def is_member(self, union_type, putative_member):
        members = self.union_rules.get(union_type)
        if members is None:
            raise TypeError(f"Not a registered union type: {union_type}")
        return type(putative_member) in members

    def has_members(self, union_type: Type) -> bool:
        """Check whether the union has an implementation or not."""
        return bool(self.union_rules.get(union_type))

    def has_members_for_all(self, union_types: typing.Iterable[Type]) -> bool:
        """Check whether every union given has an implementation or not."""
        return all(self.has_members(union_type) for union_type in union_types)


@dataclass(frozen=True)
class UnionRule:
    """Specify that an instance of `union_member` can be substituted wherever `union_base` is
    used."""

    union_base: Type
    union_member: Type

    def __post_init__(self) -> None:
        if not union.is_instance(self.union_base):
            raise ValueError(
                f"union_base must be a type annotated with @union: was {self.union_base} "
                f"(type {type(self.union_base).__name__})"
            )
