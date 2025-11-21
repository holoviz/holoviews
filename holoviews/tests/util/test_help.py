from contextlib import contextmanager

import pytest

import holoviews as hv
from holoviews.core.options import Store


@contextmanager
def _set_store(store):
    info_store = hv.core.pprint.InfoPrinter.store
    try:
        hv.core.pprint.InfoPrinter.store = store
        yield
    finally:
        hv.core.pprint.InfoPrinter.store = info_store


@pytest.mark.usefixtures("bokeh_backend")
def test_help_pattern(capsys):
    pytest.importorskip("IPython")
    with _set_store(Store):
        hv.help(hv.Curve, pattern='border')
        captured = capsys.readouterr()
        assert '\x1b[43;1;30mborder\x1b[0m' in captured.out


@pytest.mark.usefixtures("bokeh_backend")
def test_help_pattern_no_ipython(capsys):
    with _set_store(None):
        hv.help(hv.Curve)
        captured = capsys.readouterr()
        assert captured.out.startswith('Help on class Curve')
