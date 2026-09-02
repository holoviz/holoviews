from __future__ import annotations

from contextlib import contextmanager

import pytest

import holoviews as hv

from .._deps import ipython_skip


@contextmanager
def _set_store(store):
    info_store = hv.core.pprint.InfoPrinter.store
    try:
        hv.core.pprint.InfoPrinter.store = store
        yield
    finally:
        hv.core.pprint.InfoPrinter.store = info_store


@ipython_skip
@pytest.mark.usefixtures("bokeh_backend")
def test_help_pattern(capsys):
    with _set_store(hv.Store):
        hv.help(hv.Curve, pattern="border")
        captured = capsys.readouterr()
        assert "\x1b[43;1;30mborder\x1b[0m" in captured.out


@pytest.mark.usefixtures("bokeh_backend")
def test_help_pattern_no_ipython(capsys):
    with _set_store(None):
        hv.help(hv.Curve)
        captured = capsys.readouterr()
        assert captured.out.startswith("Help on class Curve")
