from __future__ import annotations

import logging
import os.path
import random
import string
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from tempfile import TemporaryFile
from typing import (Any, BinaryIO, ClassVar, Hashable, Literal, Optional,
                    Type, TypedDict)
from urllib.parse import urlsplit

import boto3
import dataclasses_json
import numpy as np
import pandas as pd
from dataclasses_json import LetterCase, config, dataclass_json
from dataclasses_json.core import Json

from .utils import ReadOnlyMapping, ReadOnlySequence, \
    identity_mapping, read_only_collection

_logger = logging.getLogger(__name__)

PeerId = Hashable
"""Trust peer identifier.

Must be usable as a `dict` key.
"""


class PeerId2Index(Mapping[PeerId, int]):
    """Identifier-to-index dict, with automatic allocation.

    Linked to an `Index2Id` counterpart; see `make_map`.
    """

    def __init__(self, *poargs, idx2id: list[PeerId], **kwargs):
        super().__init__(*poargs, **kwargs)
        self.__idx2id = idx2id
        self.__id2idx: dict[PeerId, int] = {}

    def __getitem__(self, id_):
        try:
            idx = self.__id2idx[id_]
        except KeyError:
            idx = len(self.__idx2id)
            self.__idx2id.append(id_)
            self.__id2idx[id_] = idx
        return idx

    def __contains__(self, id_):
        return id_ in self.__id2idx

    def __len__(self):
        return len(self.__id2idx)

    def __iter__(self):
        return iter(self.__id2idx)


class PeerIndex2Id(Sequence[PeerId]):
    """Index-to-identifier sequence.

    Linked to an `Id2Index` counterpart; see `make_map`.
    """

    def __init__(self, *poargs, data: list[PeerId], **kwargs):
        super().__init__(*poargs, **kwargs)
        self.__data = data

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, index):
        return self.__data[index]


PeerMap = tuple[PeerId2Index, PeerIndex2Id]


def make_peer_map(ids: Optional[Iterable[PeerId]] = None) -> PeerMap:
    """Make a linked pair of id-to-index and index-to-id mappings.

    The id-to-index dict allocates a new index to the lookup key (id)
    if the key is not found.

    :param ids: identifiers with which to initialize the mappings.
    :return: an (identifier-to-index dict, index-to-identifier sequence) pair.
    """
    data = []
    id2idx, idx2id = PeerId2Index(idx2id=data), PeerIndex2Id(data=data)
    for id_ in ids or []:
        _ = id2idx[id_]
    return id2idx, idx2id


IJV = TypedDict('IJV', dict(i=PeerId, j=PeerId, v=float))
IJV_CSV_HEADERS = ['i', 'j', 'v']

IV = TypedDict('IV', dict(i=PeerId, v=float))
IV_CSV_HEADERS = ['i', 'v']
Score = IV
SCORE_CSV_HEADERS = IV_CSV_HEADERS


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Ref:
    """Trust collection (local trust, pre-trust, global trust) reference."""
    scheme: str = field(init=False, default=None)

    @classmethod
    def make_scheme_root(cls):
        assert cls._scheme_id2cls is None and cls._scheme_cls2id is None
        cls._scheme_id2cls = {}
        cls._scheme_cls2id = {}
        Ref._roots.add(cls)
        dataclasses_json.cfg.global_config.decoders[cls] = cls.decode

    @classmethod
    def register_scheme(cls, scheme, concrete):
        cls._scheme_id2cls[scheme] = concrete
        cls._scheme_cls2id[concrete] = scheme

    def __post_init__(self):
        if self.scheme is None:
            self.scheme = self._scheme_cls2id[type(self)]
        else:
            assert self.scheme == self._scheme_cls2id[type(self)]

    def encode(self) -> Json:
        return self.to_dict()

    @classmethod
    def decode(cls, d: dict[str, Any]) -> Ref:
        if cls in Ref._roots:
            concrete = cls._scheme_id2cls[d['scheme']]
            return concrete.decode(d)
        else:
            return cls.from_dict(d)

    # noinspection PyShadowingBuiltins
    def _load(self, _type: Optional[str] = None):
        raise NotImplementedError

    _scheme_id2cls: ClassVar[Optional[dict[str, Type[Ref]]]] = None
    _scheme_cls2id: ClassVar[Optional[dict[Type[Ref], str]]] = None
    _scheme: ClassVar[str]
    _roots: ClassVar[set[Type[Ref]]] = set()


class Matrix(Ref):
    """Trust matrix reference."""

    # noinspection PyShadowingBuiltins
    def load(self, type: Optional[str] = None,
             value: str = 'v', coords: Optional[Iterable[str]] = None,
             coord_map: Optional[PeerId2Index] = None, *,
             on_missing: OnMissingPeer = 'raise') -> InlineMatrix:
        """Load the contents into an InlineMatrix.

        Support ``file:`` and ``s3:`` URLs.

        :param type: file type, ``csv`` or ``parquet``.
            Default (`None`): autodetect from filename extension.
        :param value: value column name.
        :param coords: coordinate column names (row/col).
            Default (`None`): use all non-value columns.
        :param coord_map: mapping from peer identifier to integer indices.
            Default (`None`): coordinate columns already have integer indices,
            do not map.
        :param on_missing: what to do upon peer mapping failure.
        """
        df = self._load(type)
        return InlineMatrix.from_df(df, value, coords, coord_map,
                                    on_missing=on_missing)


class Vector(Ref):
    """Trust vector reference."""

    # noinspection PyShadowingBuiltins
    def load(self, type: Optional[str] = None,
             value: str = 'v', coords: Optional[Iterable[str]] = None,
             coord_map: Optional[PeerId2Index] = None, *,
             on_missing: OnMissingPeer = 'raise') -> InlineVector:
        """Load the contents into an InlineVector.

        Support ``file:`` and ``s3:`` URLs.

        :param type: file type, ``csv`` or ``parquet``.
            Default (`None`): autodetect from filename extension.
        :param value: value column name.
        :param coords: coordinate column names (row/col).
            Default (`None`): use all non-value columns.
        :param coord_map: mapping from peer identifier to integer indices.
            Default (`None`): coordinate columns already have integer indices,
            do not map.
        :param on_missing: what to do upon peer mapping failure.
        """
        df = self._load(type)
        return InlineVector.from_df(df, value, coords, coord_map,
                                    on_missing=on_missing)


Matrix.make_scheme_root()
Vector.make_scheme_root()

OnMissingPeer = Literal['allocate', 'drop', 'raise']
"""What to do with unknown peers while translating their id/index in-place.

* `allocate` a new index,
* `drop` the entry, or
* `raise` a `KeyError`.
"""


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Inline(Ref):
    """"Inline" reference to a trust collection, i.e. its contents."""

    entries: pd.DataFrame = field(metadata=config(
        decoder=pd.DataFrame.from_records,
        encoder=lambda df: df.to_dict(orient='records'),
    ))
    size: Optional[int] = None

    coords: ClassVar[tuple[str, ...]]

    def __post_init__(self):
        super().__post_init__()
        expected_columns = self.coords + ('v',)
        actual_columns = tuple(self.entries.columns)
        assert actual_columns == expected_columns, \
            f"{type(self)} expects columns {expected_columns}; " \
            f"entries has {actual_columns}"
        for n in self.coords:
            dt = self.entries.dtypes[n]
            assert issubclass(dt.type, np.integer), \
                f"coordinate column {n} has non-integer dtype {dt}"
        dt = self.entries.dtypes['v']
        assert issubclass(dt.type, np.number), \
            f"value column v has non-number dtype {dt}"
        if self.size is None:
            self.size = max(chain(*(self.entries[n] for n in self.coords)),
                            default=-1) + 1

    def __bool__(self):
        return len(self.entries) > 0

    @classmethod
    def from_dicts(cls, dicts: Iterable[dict]) -> Inline:
        """Create a new instance using entry dicts.

        :param dicts: entries; coordinates are integer indices.
        """
        return cls(entries=pd.DataFrame(dicts))

    @classmethod
    def from_entries(cls, entries: Iterable[Mapping],
                     coord_map: PeerId2Index, *,
                     on_missing: OnMissingPeer = 'raise') -> Inline:
        """Create a new instance using entries.

        :param entries: entries; coordinates are PeerIds, not integer indices.
        :param coord_map: mapping from peer identifier to index.
        :param on_missing: what to do upon mapping failure.
        """
        if on_missing != 'allocate':  # drop or raise
            coord_map = ReadOnlyMapping(data=coord_map)

        def translate() -> Iterator[dict]:
            for entry in entries:
                try:
                    entry = dict(**{n: coord_map[entry[n]]
                                    for n in cls.coords}, v=entry['v'])
                except KeyError:
                    if on_missing == 'raise':
                        raise
                    continue
                yield entry

        return cls(entries=pd.DataFrame(translate()))

    @staticmethod
    def _gen_dicts(source_df,
                   source_coords, source_value,
                   target_coords, target_value,
                   coord_map, on_missing) -> Iterable[dict]:
        assert len(source_coords) == len(target_coords), \
            (f"expects {len(target_coords)} coordinates, "
             f"got {len(source_coords)}")
        target_column_names = target_coords + (target_value,)
        source_coord_columns = tuple(source_df[n] for n in source_coords)
        source_value_column = source_df[source_value]
        if coord_map is None:
            coord_map = identity_mapping
        elif on_missing != 'allocate':
            coord_map = read_only_collection(coord_map)
        for pos, v in zip(zip(*source_coord_columns), source_value_column):
            try:
                pos = tuple(coord_map[c] for c in pos)
            except KeyError:
                if on_missing == 'drop':
                    continue
                raise
            yield dict(zip(target_column_names, pos + (v,)))

    @classmethod
    def from_df(cls, df: pd.DataFrame,
                value: str = 'v', coords: Optional[Iterable[str]] = None,
                coord_map: Optional[PeerId2Index] = None, *,
                on_missing: OnMissingPeer = 'raise') -> Inline:
        """Create a new instance using a dataframe.

        :param df: the source dataframe.
        :param value: the name of the value column.
        :param coords: the names of coordinate columns.
            Defaults to all the other columns than the value column.
        :param coord_map: mapping from peer identifier to integer indices.
            If given, translate coordinate values to indices using this.
        :param on_missing: what to do upon mapping failure.
        """
        columns = tuple(df.dtypes.keys())
        assert value in columns, \
            f"value column {value!r} not in dataframe {columns=}"
        if coords is None:
            coords = tuple(col
                           for col in df.dtypes.keys()
                           if col != value)
            _logger.debug(f"auto-detected {coords=}")
        else:
            coords = tuple(coords)

        return cls(entries=pd.DataFrame(cls._gen_dicts(
            df, coords, value, cls.coords, 'v', coord_map, on_missing)))

    def to_dicts(self) -> Iterable[dict]:
        """Generate records as dicts, leaving the coordinates as integers.

        :return: (an iterable that yields) record dicts.
        """
        field_names = self.coords + ('v',)
        return (dict(zip(field_names, field_values))
                for field_values in zip(*(self.entries[field_name]
                                          for field_name in field_names)))

    def to_entries(self, coord_map: PeerIndex2Id, *,
                   on_missing: OnMissingPeer = 'raise') -> Iterable[dict]:
        """Generate entries as dicts, translating the coordinates into ids.

        :param coord_map: mapping from peer index to identifier.
        :param on_missing: what to do upon mapping failure.
        """
        if on_missing != 'allocate':  # drop or raise
            coord_map = ReadOnlySequence(data=coord_map)
        for coord_values, value in zip(zip(*(self.entries[n]
                                             for n in self.coords)),
                                       self.entries['v']):
            try:
                coord_values = tuple(coord_map[v] for v in coord_values)
            except IndexError:
                if on_missing == 'raise':
                    raise
                continue
            yield dict(zip(self.coords + ('v',), coord_values + (value,)))

    def to_df(self, value: str = 'v', coords: Optional[Iterable[str]] = None,
              coord_map: Optional[PeerIndex2Id] = None,
              on_missing: OnMissingPeer = 'raise') -> pd.DataFrame:
        """Convert to a dataframe.

        :param value: the name of the value column.
        :param coords: the names of coordinate columns.
            Defaults to the native column names, e.g. ``i`` and ``j``.
        :param coord_map: mapping from peer identifier to integer indices.
            If given, translate coordinate values to indices using this.
        :param on_missing: what to do upon mapping failure.
        """
        if coords is None:
            coords = self.coords
        else:
            coords = tuple(coords)

        return pd.DataFrame(self._gen_dicts(
            self.entries, self.coords, 'v', coords, value,
            coord_map, on_missing))

    # noinspection PyShadowingBuiltins,PyUnusedLocal
    def load(self, type: Optional[str] = None,
             value: str = 'v', coords: Optional[Iterable[str]] = None,
             coord_map: Optional[PeerId2Index] = None, *,
             on_missing: OnMissingPeer = 'raise') -> Inline:
        return self


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class InlineMatrix(Inline, Matrix):
    """Inline trust matrix."""
    coords = ('i', 'j')


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class InlineVector(Inline, Vector):
    """Inline trust vector."""
    coords = ('i',)


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Stored(Ref):
    """Reference to a trust collection stored in the go-eigentrust server."""
    id: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class StoredMatrix(Stored, Matrix):
    """Reference to a trust matrix stored in the go-eigentrust server."""


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class StoredVector(Stored, Vector):
    """Reference to a trust vector stored in the go-eigentrust server."""


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ObjectStorage(Ref):
    """Reference to a trust collection stored in an object storage, e.g. S3."""
    url: str

    @classmethod
    def from_path(cls, path: str) -> ObjectStorage:
        return cls(url=f"file://{os.path.realpath(path)}")

    def _open(self) -> BinaryIO:
        c = urlsplit(self.url)
        scheme = c.scheme
        if scheme == 's3':
            s3 = boto3.resource('s3')
            o = s3.Object(bucket_name=c.hostname, key=c.path.lstrip('/'))
            resp = o.get()
            return resp['Body']
        elif scheme == 'file':
            return open(c.path, 'rb')
        else:
            msg = f"unsupported URL {scheme=}"
            raise RuntimeError(msg)

    def _load(self, type_: Optional[str] = None) -> pd.DataFrame:
        c = urlsplit(self.url)
        if type_ is None:
            ext = os.path.splitext(c.path)[1].lower()
            if ext == '.csv':
                type_ = 'csv'
            elif ext == '.pqt':
                type_ = 'parquet'
            else:
                msg = f"cannot autodetect file type of {self.url}"
                raise RuntimeError(msg)
        with self._open() as r:
            if type_ == 'csv':
                return pd.read_csv(r)
            elif type_ == 'parquet':
                with TemporaryFile() as f:
                    while True:
                        d = r.read(1048576)
                        if not d:
                            break
                        f.write(d)
                    f.seek(0)
                    return pd.read_parquet(f)
            else:
                msg = f"unsupported file type {type_!r}"
                raise RuntimeError(msg)

    def upload_to_s3(self, bucket_name: str,
                     key: Optional[str] = None) -> ObjectStorage:
        ext = os.path.splitext(self.url)[1].lower()
        if key is None:
            rand = ''.join(
                random.choice(string.ascii_letters + string.digits)
                for _ in range(8))
            now = datetime.now().strftime("%Y%m%d_%H%M%S") + ext
            key = f"{rand}_{now}{ext}"
        with self._open() as r:
            s3 = boto3.resource('s3')
            o = s3.Object(bucket_name=bucket_name, key=key)
            with TemporaryFile() as f:
                while True:
                    d = r.read(1048576)
                    if not d:
                        break
                    f.write(d)
                f.seek(0)
                o.put(Body=f)
        cls = type(self)
        # noinspection PyArgumentList
        return cls(f"s3://{bucket_name}/{key}")


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ObjectStorageMatrix(ObjectStorage, Matrix):
    """Reference to a trust matrix stored in an object storage, e.g. S3."""


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ObjectStorageVector(ObjectStorage, Vector):
    """Reference to a trust vector stored in an object storage, e.g. S3."""


Matrix.register_scheme('inline', InlineMatrix)
Matrix.register_scheme('stored', StoredMatrix)
Matrix.register_scheme('objectstorage', ObjectStorageMatrix)
Vector.register_scheme('inline', InlineVector)
Vector.register_scheme('stored', StoredVector)
Vector.register_scheme('objectstorage', ObjectStorageVector)
