from __future__ import annotations

import collections.abc
from typing import Mapping, Sequence, Union, \
    overload


class ReadOnlyMapping(collections.abc.Mapping):
    """Read-only view of an underlying mapping.

    Contains an explicit membership check
    to avoid side effects that modify the underlying mapping,
    such as of `collections.defaultdict`.
    """

    def __init__(self, *args, data: Mapping, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__data = data

    def __getitem__(self, __key):
        if __key not in self.__data:
            raise KeyError(__key)
        return self.__data[__key]

    def __len__(self):
        return len(self.__data)

    def __iter__(self):
        return iter(self.__data)


class ReadOnlySequence(collections.abc.Sequence):
    """Read-only view of an underlying sequence.

    Contains an explicit index check
    to avoid side effects that modify the underlying sequence.
    """

    def __init__(self, *args, data: Sequence, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__data = data

    def __getitem__(self, index: Union[int, slice]):
        data_len = len(self.__data)
        if isinstance(index, int) and not (-data_len <= index < data_len):
            raise IndexError(index)
        return self.__data[index]

    def __len__(self):
        return len(self.__data)


@overload
def read_only_collection(c: collections.abc.Sequence) -> ReadOnlySequence:
    pass


@overload
def read_only_collection(c: collections.abc.Mapping) -> ReadOnlyMapping:
    pass


def read_only_collection(c):
    """Read-only view of an underlying collection (sequence or map)."""
    if isinstance(c, collections.abc.Sequence):
        return ReadOnlySequence(data=c)
    elif isinstance(c, collections.abc.Mapping):
        return ReadOnlyMapping(data=c)
    else:
        msg = f"{c=} is not a sequence or a mapping"
        raise TypeError(msg)


class IdentityMapping:
    """An identity mapping."""

    def __getitem__(self, __key):
        return __key


identity_mapping = IdentityMapping()