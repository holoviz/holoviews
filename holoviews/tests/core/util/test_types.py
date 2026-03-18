import sys

import numpy as np
import pytest

from holoviews.core.util.types import gen_types


@pytest.fixture(autouse=True)
def np_absent(monkeypatch):
    monkeypatch.delitem(sys.modules, "numpy", raising=False)


@gen_types
def my_types():
    yield int
    if "numpy" in sys.modules:
        yield np.ndarray


class TestGenTypesCacheInvalidation:
    def test_before_import_excludes_optional_type(self):
        assert tuple(my_types) == (int,)
        assert isinstance(42, my_types)
        assert not isinstance(np.array([1]), my_types)

    def test_late_import_invalidates_cache(self, monkeypatch):
        assert tuple(my_types) == (int,)

        monkeypatch.setitem(sys.modules, "numpy", np)

        result = tuple(my_types)
        assert result == (int, np.ndarray)
        assert isinstance(np.array([1, 2, 3]), my_types)

    def test_cache_reused_when_sys_modules_unchanged(self):
        tuple(my_types)
        cached = my_types._cached_types

        tuple(my_types)
        assert my_types._cached_types is cached
