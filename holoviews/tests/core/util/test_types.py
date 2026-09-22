from __future__ import annotations

import sys

import numpy as np
import pytest

from holoviews.core.util.types import gen_types


@pytest.fixture(autouse=True)
def np_absent(monkeypatch):
    monkeypatch.delitem(sys.modules, "numpy", raising=False)


@gen_types
def int_types():
    yield int
    if "numpy" in sys.modules:
        yield np.integer


class TestGenTypesCacheInvalidation:
    def test_before_import_excludes_optional_type(self):
        assert tuple(int_types) == (int,)
        assert isinstance(42, int_types)
        assert not isinstance(np.int64(1), int_types)

    def test_late_import_invalidates_cache(self, monkeypatch):
        assert tuple(int_types) == (int,)

        monkeypatch.setitem(sys.modules, "numpy", np)

        result = tuple(int_types)
        assert result == (int, np.integer)
        assert isinstance(np.int64(1), int_types)

    def test_cache_reused_when_sys_modules_unchanged(self):
        tuple(int_types)
        cached = int_types._cached_types

        tuple(int_types)
        assert int_types._cached_types is cached
