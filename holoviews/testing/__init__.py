import sys

from ..core.util.dependencies import _is_installed

if _is_installed("pytest") and "holoviews.testing._testing" not in sys.modules:
    import pytest
    pytest.register_assert_rewrite("holoviews.testing._testing")

from ._testing import (
    add_comparison,
    assert_data_equal,
    assert_dict_equal,
    assert_element_equal,
)

__all__ = ["add_comparison", "assert_data_equal", "assert_dict_equal", "assert_element_equal"]
