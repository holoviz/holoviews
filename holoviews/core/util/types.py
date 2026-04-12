from __future__ import annotations

import datetime as dt
import inspect
import sys
import typing as t
from types import GeneratorType

import narwhals.stable.v2 as nw
import numpy as np

from .dependencies import cftime, pd

# gen_types is copied from param, can be removed when
# we support 2.2 or greater

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


def gen_types(gen_func):
    """
    Decorator which takes a generator function which yields difference types
    make it so it can be called with isinstance and issubclass.
    """
    if not inspect.isgeneratorfunction(gen_func):
        msg = "gen_types decorator can only be applied to generator"
        raise TypeError(msg)
    return type(gen_func.__name__, (_GeneratorIs,), {"types": staticmethod(gen_func)})


# Types
generator_types = (zip, range, GeneratorType)


@gen_types
def pandas_datetime_types():
    if pd:
        from pandas.core.dtypes.dtypes import DatetimeTZDtype

        yield from (pd.Timestamp, pd.Period, DatetimeTZDtype)


@gen_types
def pandas_timedelta_types():
    if pd:
        yield pd.Timedelta


@gen_types
def cftime_types():
    if cftime:
        yield cftime.datetime


@gen_types
def datetime_types():
    yield from (dt.datetime, dt.date, dt.time, np.datetime64)
    yield from pandas_datetime_types
    yield from cftime_types


@gen_types
def timedelta_types():
    yield from (dt.timedelta, np.timedelta64)
    yield from pandas_timedelta_types


@gen_types
def arraylike_types():
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
