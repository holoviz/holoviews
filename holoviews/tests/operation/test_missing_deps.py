from __future__ import annotations

import subprocess
import sys
from textwrap import dedent

import numpy as np
import pytest

from holoviews.element.selection import spatial_select_gridded
from holoviews.plotting.util import categorical_legend, flatten_stack

from .._deps import ds_skip, xr_skip


@ds_skip
@xr_skip
def test_datashader_operation_imports_when_dependencies_present():
    from holoviews.operation.datashader import datashade, rasterize

    assert callable(rasterize)
    assert callable(datashade)


def test_missing_datashader_requires_install():
    script = """\
        import sys

        sys.modules["datashader"] = None
        try:
            import holoviews.operation.datashader
        except ImportError as exc:
            if type(exc) is not ImportError or "datashader must be installed" not in str(exc):
                raise SystemExit(f"unexpected import error: {exc!r}")
            if not isinstance(exc.__cause__, ModuleNotFoundError):
                raise SystemExit(f"unexpected cause: {exc.__cause__!r}")
        else:
            raise SystemExit("import succeeded")
        """
    subprocess.run([sys.executable, "-c", dedent(script)], check=True)


def test_categorical_legend_names_missing_datashader(monkeypatch):
    class _Element:
        def __getattribute__(self, name):
            raise AssertionError(name)

    legend = categorical_legend.instance()
    monkeypatch.setitem(sys.modules, "datashader", None)
    with pytest.raises(ImportError, match="datashader must be installed"):
        legend._process(_Element())


def test_missing_xarray_requires_install():
    # datashader is imported before xarray and is absent in test-core and test-315.
    script = """\
        import sys
        import types

        datashader = types.ModuleType("datashader")
        datashader.__path__ = []
        sys.modules["datashader"] = datashader
        for name in ("reductions", "transfer_functions", "colors"):
            module = types.ModuleType(f"datashader.{name}")
            sys.modules[f"datashader.{name}"] = module
        sys.modules["datashader.colors"].color_lookup = None
        sys.modules["xarray"] = None
        try:
            import holoviews.operation.datashader
        except ImportError as exc:
            if type(exc) is not ImportError or "xarray must be installed" not in str(exc):
                raise SystemExit(f"unexpected import error: {exc!r}")
            if not isinstance(exc.__cause__, ModuleNotFoundError):
                raise SystemExit(f"unexpected cause: {exc.__cause__!r}")
        else:
            raise SystemExit("import succeeded")
        """
    subprocess.run([sys.executable, "-c", dedent(script)], check=True)


def test_broken_datashader_import_propagates():
    script = """\
        import sys

        class _Broken:
            def find_spec(self, name, path, target=None):
                if name == "datashader" or name.startswith("datashader."):
                    raise RuntimeError("datashader broke")
                return None

        sys.meta_path.insert(0, _Broken())
        try:
            import holoviews.operation.datashader
        except RuntimeError as exc:
            if str(exc) != "datashader broke" or exc.__cause__ is not None:
                raise SystemExit(f"unexpected runtime error: {exc!r}")
        else:
            raise SystemExit("import succeeded")
        """
    subprocess.run([sys.executable, "-c", dedent(script)], check=True)


def test_flatten_stack_keeps_message_when_datashader_operation_missing(monkeypatch):
    stack = flatten_stack.instance()
    monkeypatch.setitem(sys.modules, "holoviews.operation.datashader", None)
    with pytest.raises(ImportError, match="Flattening ImageStacks requires datashader"):
        stack._process(object())


def test_gridded_lasso_keeps_message_when_datashader_operation_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "holoviews.operation.datashader", None)
    xvals = np.array([[0.0, 1.0], [0.0, 1.0]])
    yvals = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(
        ImportError, match="Lasso selection on gridded data requires datashader to be available"
    ):
        spatial_select_gridded(xvals, yvals, None)
