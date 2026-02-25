import importlib
import types

import pytest

MODULES = [
    "holoviews",
    "holoviews.core",
    "holoviews.element",
    "holoviews.operation",
    "holoviews.plotting",
]


@pytest.fixture(params=MODULES)
def module(request):
    return importlib.import_module(request.param)


def test_all_no_underscores(module):
    underscored = [name for name in module.__all__ if name.startswith("_") and name != "__version__"]
    assert not underscored


def test_all_exists(module):
    assert hasattr(module, "__all__")
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert not missing


def _is_defined_in(obj, module_name):
    obj_module = getattr(obj, "__module__", None)
    if obj_module is None:
        return True
    return obj_module.startswith(module_name)


def test_all_complete(module):
    mod_attrs = {
        name
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and name != "TYPE_CHECKING"
        and not isinstance(obj, types.ModuleType)
        and _is_defined_in(obj, module.__name__)
    }
    exported = set(module.__all__)
    not_exported = sorted(mod_attrs - exported)
    assert not not_exported
