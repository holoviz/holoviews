from __future__ import annotations

import datetime as dt
import inspect
import sys
import typing as t
from types import GeneratorType

import narwhals.stable.v2 as nw
import numpy as np

from .dependencies import cftime, pd

if t.TYPE_CHECKING:
    from pandas.core.arrays.masked import BaseMaskedArray
    from pandas.core.dtypes.dtypes import DatetimeTZDtype
    from pandas.core.dtypes.generic import ABCExtensionArray, ABCIndex, ABCSeries

    _PdDatetimeT: t.TypeAlias = pd.Timestamp | pd.Period | DatetimeTZDtype
    _PdTimedeltaT: t.TypeAlias = pd.Timedelta
    _CftimeT: t.TypeAlias = cftime.datetime
    _DatetimeT: t.TypeAlias = (
        dt.datetime | dt.date | dt.time | np.datetime64 | _PdDatetimeT | _CftimeT
    )
    _TimedeltaT: t.TypeAlias = dt.timedelta | np.timedelta64 | _PdTimedeltaT
    _ArraylikeT: t.TypeAlias = np.ndarray | nw.Series | ABCIndex | ABCSeries | ABCExtensionArray
    _MaskedT: t.TypeAlias = np.ma.core.MaskedArray | BaseMaskedArray

    _YieldT = t.TypeVar("_YieldT")

    class _GenFunc(t.Protocol[_YieldT]):
        __name__: str

        def __call__(self) -> t.Iterator[_YieldT]: ...


_module_count = 0


class _GeneratorIsMeta(type):
    types: t.Callable[[], t.Iterable[type]]

    def _get_types(cls) -> tuple[type, ...]:
        # Cache types and invalidate on new imports
        global _module_count  # noqa: PLW0603
        n = len(sys.modules)
        if n != _module_count:
            _module_count = n
            for sub in _GeneratorIs.__subclasses__():
                sub._cached_types = None
        if cls._cached_types is None:
            cls._cached_types = tuple(cls.types())
        return cls._cached_types

    def __instancecheck__(cls, inst):
        return isinstance(inst, cls._get_types())

    def __subclasscheck__(cls, sub):
        return issubclass(sub, cls._get_types())

    def __iter__(cls):
        yield from cls._get_types()


class _GeneratorIs(metaclass=_GeneratorIsMeta):
    _cached_types: tuple[type, ...] | None = None

    @classmethod
    def __iter__(cls):
        yield from cls._get_types()


def gen_types(gen_func: _GenFunc[_YieldT]) -> tuple[_YieldT, ...]:
    """
    Decorator which takes a generator function which yields difference types
    make it so it can be called with isinstance and issubclass.
    """
    if t.TYPE_CHECKING:
        return tuple(gen_func())
    if not inspect.isgeneratorfunction(gen_func):
        msg = "gen_types decorator can only be applied to generator"
        raise TypeError(msg)
    return type(gen_func.__name__, (_GeneratorIs,), {"types": staticmethod(gen_func)})


# Types
generator_types = (zip, range, GeneratorType)


@gen_types
def pandas_datetime_types() -> t.Iterator[type[_PdDatetimeT]]:
    if pd:
        from pandas.core.dtypes.dtypes import DatetimeTZDtype

        yield from (pd.Timestamp, pd.Period, DatetimeTZDtype)


@gen_types
def pandas_timedelta_types() -> t.Iterator[type[_PdTimedeltaT]]:
    if pd:
        yield pd.Timedelta


@gen_types
def cftime_types() -> t.Iterator[type[_CftimeT]]:
    if cftime:
        yield cftime.datetime


@gen_types
def datetime_types() -> t.Iterator[type[_DatetimeT]]:
    yield from (dt.datetime, dt.date, dt.time, np.datetime64)
    yield from pandas_datetime_types
    yield from cftime_types


@gen_types
def timedelta_types() -> t.Iterator[type[_TimedeltaT]]:
    yield from (dt.timedelta, np.timedelta64)
    yield from pandas_timedelta_types


@gen_types
def arraylike_types() -> t.Iterator[type[_ArraylikeT]]:
    yield from (np.ndarray, nw.Series)
    if pd:
        from pandas.core.dtypes.generic import ABCExtensionArray, ABCIndex, ABCSeries

        yield from (ABCIndex, ABCSeries, ABCExtensionArray)


@gen_types
def masked_types():
    yield np.ma.core.MaskedArray

    if pd:
        from pandas.core.arrays.masked import BaseMaskedArray

        yield BaseMaskedArray


__all__ = [
    "arraylike_types",
    "cftime_types",
    "datetime_types",
    "generator_types",
    "masked_types",
    "pandas_datetime_types",
    "pandas_timedelta_types",
    "timedelta_types",
]
