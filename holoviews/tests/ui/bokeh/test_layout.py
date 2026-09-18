from __future__ import annotations

import numpy as np
import pytest

import holoviews as hv

from .. import expect

pytestmark = pytest.mark.ui


@pytest.mark.usefixtures("bokeh_backend")
def test_gridspace_toolbar(serve_hv):
    def sine_curve(phase, freq):
        xvals = [0.1 * i for i in range(100)]
        return hv.Curve((xvals, [np.sin(phase + freq * x) for x in xvals]))

    phases = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    frequencies = [0.5, 0.75, 1.0, 1.25]
    curve_dict_2D = {(p, f): sine_curve(p, f) for p in phases for f in frequencies}
    gridspace = hv.GridSpace(curve_dict_2D, kdims=["phase", "frequency"])

    page = serve_hv(gridspace)
    bokeh_logo = page.locator(".bk-logo")
    expect(bokeh_logo).to_have_count(1)


@pytest.mark.usefixtures("bokeh_backend")
def test_adjoint_layout_responsive(page, serve_hv):
    page.set_viewport_size({"width": 1200, "height": 800})

    main = hv.Curve([]).opts(responsive=True)
    adj1 = hv.Curve([]).opts(width=80)
    adj2 = hv.Curve([]).opts(height=80)
    layout = (main << adj1 << adj2).opts(toolbar=None)

    page = serve_hv(layout)
    canvases = page.locator(".bk-Canvas")
    expect(canvases).to_have_count(3)

    boxes = [canvases.nth(i).bounding_box() for i in range(3)]
    main_box = max(boxes, key=lambda b: b["width"] * b["height"])

    assert main_box["width"] == 1200 - 80
    assert main_box["height"] == 800 - 80
