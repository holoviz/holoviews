import contextlib
import sys
from collections.abc import Callable

import numpy as np
import panel as pn
import pytest
from panel.tests.conftest import port, server_cleanup  # noqa: F401
from panel.tests.util import serve_and_wait

import holoviews as hv

CUSTOM_MARKS = ("ui", "gpu")


def pytest_addoption(parser):
    for marker in CUSTOM_MARKS:
        parser.addoption(
            f"--{marker}",
            action="store_true",
            default=False,
            help=f"Run {marker} related tests",
        )


def pytest_configure(config):
    for marker in CUSTOM_MARKS:
        config.addinivalue_line("markers", f"{marker}: {marker} test marker")


def pytest_collection_modifyitems(config, items):
    skipped, selected = [], []
    markers = [m for m in CUSTOM_MARKS if config.getoption(f"--{m}")]
    empty = not markers
    for item in items:
        if empty and any(m in item.keywords for m in CUSTOM_MARKS):
            skipped.append(item)
        elif empty:
            selected.append(item)
        elif not empty and any(m in item.keywords for m in markers):
            selected.append(item)
        else:
            skipped.append(item)

    config.hook.pytest_deselected(items=skipped)
    items[:] = selected


with contextlib.suppress(ImportError):
    import matplotlib as mpl

    mpl.use("agg")


with contextlib.suppress(Exception):
    # From Dask 2023.7.1 they now automatically convert strings
    # https://docs.dask.org/en/stable/changelog.html#v2023-7-1
    import dask

    dask.config.set({"dataframe.convert-string": False})


@pytest.fixture
def ibis_sqlite_backend():
    try:
        import ibis
    except ImportError:
        yield None
    else:
        ibis.set_backend("sqlite")
        yield
        ibis.set_backend(None)


def _plotting_backend(backend):
    pytest.importorskip(backend)
    if not hv.extension._loaded:
        hv.extension(backend)
    hv.renderer(backend)
    curent_backend = hv.Store.current_backend
    hv.Store.set_current_backend(backend)
    yield
    hv.Store.set_current_backend(curent_backend)


@pytest.fixture
def bokeh_backend():
    yield from _plotting_backend("bokeh")


@pytest.fixture
def mpl_backend():
    yield from _plotting_backend("matplotlib")


@pytest.fixture
def plotly_backend():
    yield from _plotting_backend("plotly")


@pytest.fixture
def unimport(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    """
    Return a function for unimporting modules and preventing reimport.

    This will block any new modules from being imported.
    """

    def unimport_module(modname: str) -> None:
        # Remove if already imported
        monkeypatch.delitem(sys.modules, modname, raising=False)
        # Prevent import:
        monkeypatch.setattr(sys, "path", [])

    return unimport_module


@pytest.fixture
def serve_hv(page, port):  # noqa: F811
    def serve_and_return_page(hv_obj):
        serve_and_wait(pn.pane.HoloViews(hv_obj), port=port)
        page.goto(f"http://localhost:{port}")
        return page

    return serve_and_return_page

@pytest.fixture
def serve_panel(page, port):  # noqa: F811
    def serve_and_return_page(pn_obj):
        serve_and_wait(pn.panel(pn_obj), port=port)
        page.goto(f"http://localhost:{port}")
        return page

    return serve_and_return_page

@pytest.fixture(autouse=True)
def reset_store():
    _custom_options = {k: {} for k in hv.Store._custom_options}
    _options = hv.Store._options.copy()
    current_backend = hv.Store.current_backend
    renderers = hv.Store.renderers.copy()
    yield
    hv.Store._custom_options = _custom_options
    hv.Store._options = _options
    hv.Store._weakrefs = {}
    hv.Store.renderers = renderers
    hv.Store.set_current_backend(current_backend)


@pytest.fixture
def rng():
    return np.random.default_rng(1)
