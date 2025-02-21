from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version


class VersionError(Exception):
    """Raised when there is a library version mismatch."""

    def __init__(self, msg, version=None, min_version=None, **kwargs):
        self.version = version
        self.min_version = min_version
        super().__init__(msg, **kwargs)


def _no_import_version(name) -> tuple[int, int, int]:
    """Get version number without importing the library"""
    try:
        return Version(version(name)).release
    except PackageNotFoundError:
        return (0, 0, 0)

# Versions
NUMPY_VERSION = _no_import_version("numpy")
PARAM_VERSION = _no_import_version("param")
PANDAS_VERSION = _no_import_version("pandas")

NUMPY_GE_2_0_0 = NUMPY_VERSION >= (2, 0, 0)
PANDAS_GE_2_1_0 = PANDAS_VERSION >= (2, 1, 0)
PANDAS_GE_2_2_0 = PANDAS_VERSION >= (2, 2, 0)

__all__ = ["NUMPY_GE_2_0_0", "NUMPY_VERSION", "PANDAS_GE_2_1_0", "PANDAS_GE_2_2_0", "PANDAS_VERSION", "PARAM_VERSION"]
