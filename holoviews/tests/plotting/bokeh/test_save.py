import pytest

import holoviews as hv

hv.extension("bokeh")


@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize(
    "opts",
    [
        {"frame_width": 200},
        {"frame_height": 200},
    ],
)
def test_save_suppresses_bokeh_fixed_sizing_mode(tmp_path, caplog, opts):
    curve = hv.Curve([1, 2, 3]).opts(**opts, backend="bokeh")

    out = tmp_path / "curve.html"
    with caplog.at_level("WARNING", logger="bokeh.core.validation.check"):
        hv.save(curve, str(out), backend="bokeh")

    assert out.exists()
    assert not any(
        "FIXED_SIZING_MODE"
        in (r.getMessage() if hasattr(r, "getMessage") else str(r.msg))
        for r in caplog.records
    )
