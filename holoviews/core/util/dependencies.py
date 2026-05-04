from __future__ import annotations

import re
import sys
import typing as t
from functools import cache
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec

if t.TYPE_CHECKING:
    from types import ModuleType

_re_no = re.compile(r"\d+")


class VersionError(Exception):
    """Raised when there is a library version mismatch."""

    def __init__(self, msg, version=None, min_version=None, **kwargs):
        self.version = version
        self.min_version = min_version
        super().__init__(msg, **kwargs)


@cache
def _is_installed(module_name: str) -> bool:
    # So we don't accidentally import it
    module_name, *_ = module_name.split(".")
    return find_spec(module_name) is not None


@cache
def _get_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "0.0.0"


def _convert_int(version_str: str) -> tuple[int, ...]:
    """Convert a version string to a tuple of integers."""
    return tuple(map(int, _re_no.findall(version_str)[:3]))


@cache
def _no_import_version(package_name) -> tuple[int, ...]:
    """Get version number without importing the library"""
    return _convert_int(_get_version(package_name))


_MIN_SUPPORTED_VERSION = {
    "pandas": (1, 3, 0),
}


class _LazyModule:
    def __init__(self, module_name: str, package_name: str | None = None):
        """
        Lazy import module

        This will wait and import the module when an attribute is accessed.

        Parameters
        ----------
        module_name: str
            The import name of the module, e.g. `import PIL`
        package_name: str, optional
            Name of the package, this is the named used for installing the package, e.g. `pip install pillow`.
            Used for the __version__ if the module is not imported.
            If not set uses the module_name.
        """
        self.__module = None
        self.__module_name = module_name
        self.__package_name = package_name or module_name

    @property
    def _module(self) -> ModuleType:
        if self.__module is None:
            self.__module = import_module(self.__module_name)
            if self.__package_name in _MIN_SUPPORTED_VERSION:
                min_version = _MIN_SUPPORTED_VERSION[self.__package_name]
                mod_version = _no_import_version(self.__package_name)
                if mod_version < min_version:
                    min_version_str = ".".join(map(str, min_version))
                    mod_version_str = ".".join(map(str, mod_version))
                    msg = f"{self.__package_name} requires {min_version_str} or higher (found {mod_version_str})"
                    raise VersionError(msg, mod_version_str, min_version_str)

        return self.__module

    def __getattr__(self, attr):
        return getattr(self._module, attr)

    def __dir__(self) -> list[str]:
        return dir(self._module)

    def __bool__(self) -> bool:
        return bool(
            self.__module
            or (_is_installed(self.__module_name) and self.__module_name in sys.modules)
        )

    def __repr__(self) -> str:
        if self.__module:
            return repr(self.__module).replace("<module", "<lazy-module")
        else:
            return f"<lazy-module {self.__module_name!r}>"

    @property
    def __version__(self):
        return self.__module and self.__module.__version__ or _get_version(self.__package_name)


if t.TYPE_CHECKING:
    import cftime
    import cudf
    import cupy as cp
    import dask.array as da
    import dask.dataframe as dd
    import ibis
    import pandas as pd
    import polars as pl
else:
    cftime = _LazyModule("cftime")
    cudf = _LazyModule("cudf")
    cp = _LazyModule("cupy")
    da = _LazyModule("dask.array")
    dd = _LazyModule("dask.dataframe")
    ibis = _LazyModule("ibis", "ibis-framework")
    pd = _LazyModule("pandas")
    pl = _LazyModule("polars")

# Versions
NUMPY_VERSION = _no_import_version("numpy")
PARAM_VERSION = _no_import_version("param")
PANDAS_VERSION = _no_import_version("pandas")

NUMPY_GE_2_0_0 = NUMPY_VERSION >= (2, 0, 0)
PANDAS_GE_2_1_0 = PANDAS_VERSION >= (2, 1, 0)
PANDAS_GE_2_2_0 = PANDAS_VERSION >= (2, 2, 0)
PANDAS_GE_3_0_0 = PANDAS_VERSION >= (3, 0, 0)

__all__ = [
    "NUMPY_GE_2_0_0",
    "NUMPY_VERSION",
    "PANDAS_GE_2_1_0",
    "PANDAS_GE_2_2_0",
    "PANDAS_GE_3_0_0",
    "PANDAS_VERSION",
    "PARAM_VERSION",
]
