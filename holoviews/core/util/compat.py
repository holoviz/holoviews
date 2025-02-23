from functools import lru_cache
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec

from packaging.version import Version


class VersionError(Exception):
    """Raised when there is a library version mismatch."""

    def __init__(self, msg, version=None, min_version=None, **kwargs):
        self.version = version
        self.min_version = min_version
        super().__init__(msg, **kwargs)


@lru_cache
def _get_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "0.0.0"


def _no_import_version(name) -> tuple[int, int, int]:
    """Get version number without importing the library"""
    return Version(_get_version(name)).release


class _lazy_module:
    __module = None
    __module_name = None

    def __init__(self, module_name):
        self.__module_name = module_name

    @property
    def _module(self):
        if self.__module is None:
            try:
                self.__module = import_module(self.__module_name)
            except PackageNotFoundError:
                raise ModuleNotFoundError(f"No module named {self.__module_name!r}") from None
        return self.__module

    def __getattr__(self, attr):
        return getattr(self._module, attr)

    def __dir__(self):
        return list(self._module.__all__)

    def __bool__(self):
        return bool(self.__module or find_spec(self.__module_name))

    @property
    def __version__(self):
        return _get_version(self.__module_name)


# Versions
NUMPY_VERSION = _no_import_version("numpy")
PARAM_VERSION = _no_import_version("param")
PANDAS_VERSION = _no_import_version("pandas")

NUMPY_GE_2_0_0 = NUMPY_VERSION >= (2, 0, 0)
PANDAS_GE_2_1_0 = PANDAS_VERSION >= (2, 1, 0)
PANDAS_GE_2_2_0 = PANDAS_VERSION >= (2, 2, 0)

__all__ = ["NUMPY_GE_2_0_0", "NUMPY_VERSION", "PANDAS_GE_2_1_0", "PANDAS_GE_2_2_0", "PANDAS_VERSION", "PARAM_VERSION"]
