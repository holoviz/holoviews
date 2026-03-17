import sys
import types

import pytest

from holoviews.core.util.dependencies import _LazyModule
from holoviews.core.util.types import gen_types


class _FakeIsInstalled:
    """Stand-in for the lru_cached _is_installed that treats a fake module as installed."""

    def __init__(self, original, fake_name: str):
        self._original = original
        self._fake_name = fake_name

    def __call__(self, module_name: str) -> bool:
        if module_name == self._fake_name:
            return True
        return self._original(module_name)

    def cache_clear(self):
        pass


FAKE_MODULE_NAME = "_hv_test_fake_lib"


@pytest.fixture
def _fake_optional_library():
    """Patch _is_installed so _LazyModule considers our fake module installable,
    then clean up sys.modules and restore the original after the test."""
    import holoviews.core.util.dependencies as deps

    original = deps._is_installed
    deps._is_installed = _FakeIsInstalled(original, FAKE_MODULE_NAME)
    try:
        yield
    finally:
        deps._is_installed = original
        sys.modules.pop(FAKE_MODULE_NAME, None)


@pytest.mark.usefixtures("_fake_optional_library")
class TestGenTypesCacheInvalidation:
    """Cache invalidation must pick up types from late-imported optional libraries."""

    def setup_method(self):
        self.fake_lib = _LazyModule(FAKE_MODULE_NAME, bool_use_sys_modules=True)

        class MyType:
            pass

        self.MyType = MyType

        fake_lib = self.fake_lib

        @gen_types
        def my_types():
            yield int
            if fake_lib:
                yield fake_lib.MyType

        self.my_types = my_types

    def test_before_import_excludes_optional_type(self):
        assert FAKE_MODULE_NAME not in sys.modules
        assert tuple(self.my_types) == (int,)
        assert isinstance(42, self.my_types)
        assert not isinstance(self.MyType(), self.my_types)

    def test_late_import_invalidates_cache(self):
        # Populate cache without the optional library
        assert tuple(self.my_types) == (int,)

        # Simulate late import
        fake_module = types.ModuleType(FAKE_MODULE_NAME)
        fake_module.MyType = self.MyType
        sys.modules[FAKE_MODULE_NAME] = fake_module

        # Cache should invalidate and pick up the new type
        result = tuple(self.my_types)
        assert result == (int, self.MyType)
        assert isinstance(self.MyType(), self.my_types)

    def test_cache_reused_when_sys_modules_unchanged(self):
        # Populate cache
        tuple(self.my_types)
        cached = self.my_types._cached_types

        # Access again without changing sys.modules
        tuple(self.my_types)
        assert self.my_types._cached_types is cached
