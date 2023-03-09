import holoviews as hv

def test_help_pattern(capsys):
    import holoviews.plotting.bokeh  # noqa
    hv.help(hv.Curve, pattern='border')
    captured = capsys.readouterr()
    assert '\x1b[43;1;30mborder\x1b[0m' in captured.out
