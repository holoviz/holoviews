import sys
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
def _is_installed(module_name):
    return find_spec(module_name) is not None


@lru_cache
def _get_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "0.0.0"


def _no_import_version(package_name) -> tuple[int, int, int]:
    """Get version number without importing the library"""
    return Version(_get_version(package_name)).release


class _lazy_module:
    __slots__ = ("__module", "__module_bool_check", "__module_name", "__package_name")

    def __init__(self, module_name, package_name=None, *, module_bool_check=False):
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
        module_bool_check: bool, optional, default False
            Use sys.modules for __bool__ check if True, else uses importlib.util.find_spec.
        """
        self.__module = None
        self.__module_name = module_name
        self.__package_name = package_name or module_name
        self.__module_bool_check = module_bool_check

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
        return dir(self._module)

    def __bool__(self):
        if self.__module_bool_check:
            return bool(self.__module or self.__module_name in sys.modules)
        else:
            return bool(self.__module or _is_installed(self.__module_name))

    def __repr__(self):
        if self.__module:
            return repr(self.__module).replace("<module", "<lazy-module")
        else:
            return f"<lazy-module {self.__module_name!r}>"

    @property
    def __version__(self):
        return self.__module and self.__module.__version__ or _get_version(self.__package_name)


# Versions
NUMPY_VERSION = _no_import_version("numpy")
PARAM_VERSION = _no_import_version("param")
PANDAS_VERSION = _no_import_version("pandas")

NUMPY_GE_2_0_0 = NUMPY_VERSION >= (2, 0, 0)
PANDAS_GE_2_1_0 = PANDAS_VERSION >= (2, 1, 0)
PANDAS_GE_2_2_0 = PANDAS_VERSION >= (2, 2, 0)

__all__ = ["NUMPY_GE_2_0_0", "NUMPY_VERSION", "PANDAS_GE_2_1_0", "PANDAS_GE_2_2_0", "PANDAS_VERSION", "PARAM_VERSION"]
