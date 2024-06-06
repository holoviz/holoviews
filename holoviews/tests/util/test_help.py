import pytest

import holoviews as hv


@pytest.mark.usefixtures("bokeh_backend")
def test_help_pattern(capsys):
    pytest.importorskip("IPython")
    hv.help(hv.Curve, pattern='border')
    captured = capsys.readouterr()
    assert '\x1b[43;1;30mborder\x1b[0m' in captured.out


@pytest.mark.usefixtures("bokeh_backend")
def test_help_pattern_no_ipython(capsys):
    info_store = hv.core.pprint.InfoPrinter.store
    try:
        hv.core.pprint.InfoPrinter.store = None
        hv.help(hv.Curve)
        captured = capsys.readouterr()
        assert captured.out.startswith('Help on class Curve')
    finally:
        hv.core.pprint.InfoPrinter.store = info_store
