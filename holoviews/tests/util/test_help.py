import pytest

import holoviews as hv


@pytest.mark.usefixtures("bokeh_backend")
def test_help_pattern(capsys):
    pytest.importorskip("IPython")
    hv.help(hv.Curve, pattern='border')
    captured = capsys.readouterr()
    assert '\x1b[43;1;30mborder\x1b[0m' in captured.out
