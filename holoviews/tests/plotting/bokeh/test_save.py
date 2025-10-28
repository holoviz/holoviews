import logging

import pytest

import holoviews as hv


@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize(
    "opts",
    [
        {"frame_width": 200},
        {"frame_height": 200},
        {"frame_height": 200, "frame_width": 200},
        {},
    ],
)
def test_save_suppresses_bokeh_fixed_sizing_mode(tmp_path, caplog, opts):
    curve = hv.Curve([1, 2, 3]).opts(**opts, backend="bokeh")

    out = tmp_path / "curve.html"
    logging.getLogger("bokeh").propagate = True
    with caplog.at_level(logging.WARNING):
        hv.save(curve, out)

    assert out.exists()
    assert all(["FIXED_SIZING_MODE" not in e.msg for e in caplog.records])
