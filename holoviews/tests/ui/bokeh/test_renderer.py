from __future__ import annotations

import pytest

import holoviews as hv
from holoviews.plotting.bokeh.renderer import BokehRenderer
from holoviews.plotting.bokeh.util import BOKEH_GE_3_10_0

pytestmark = [
    pytest.mark.ui,
    pytest.mark.skipif(not BOKEH_GE_3_10_0, reason="Playwright export needs Bokeh 3.10"),
]


@pytest.mark.parametrize(
    ("renderer_kwargs", "obj", "ext", "magic"),
    [
        ({"fig": "png"}, hv.Curve([1, 2, 3]), "png", b"\x89PNG"),
        (
            {"holomap": "gif"},
            hv.HoloMap({i: hv.Curve([1, 2, i]) for i in range(3)}),
            "gif",
            b"GIF8",
        ),
    ],
    ids=["png", "gif"],
)
def test_render_static_export(renderer_kwargs, obj, ext, magic):
    renderer = BokehRenderer.instance(**renderer_kwargs)
    data, info = renderer(obj)
    assert data.startswith(magic)
    assert info["file-ext"] == ext
