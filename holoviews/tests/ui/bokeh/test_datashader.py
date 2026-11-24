from __future__ import annotations

import numpy as np
import pytest

import holoviews as hv
from holoviews.streams import RangeXY

from ..._deps import ds_skip
from .. import expect, wait_until

pytestmark = [pytest.mark.ui, ds_skip]


@pytest.mark.usefixtures("bokeh_backend")
def test_rasterize_dynspread_per_element_overlay_initial_range(serve_hv):
    from holoviews.operation.datashader import dynspread, rasterize

    x = np.arange(10_000)
    s1 = hv.Scatter((x, np.full(x.size, 5.0)), "x", "y", label="s1")
    s2 = hv.Scatter((x, np.full(x.size, 8.0)), "x", "y", label="s2")

    d1 = dynspread(rasterize(s1), max_px=2, threshold=1)
    d2 = dynspread(rasterize(s2), max_px=2, threshold=1)

    # Attach streams so the seeded range can be inspected once the plot
    # is served, without needing any pan/zoom interaction to trigger it.
    stream1 = RangeXY(source=d1)
    stream2 = RangeXY(source=d2)

    overlay = (d1 * d2).opts(padding=0.1)

    page = serve_hv(overlay)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)

    expected_xrange = (-999.9, 10998.9)
    expected_yrange = (4.7, 8.3)

    def check():
        np.testing.assert_allclose(stream1.x_range, expected_xrange)
        np.testing.assert_allclose(stream1.y_range, expected_yrange)
        np.testing.assert_allclose(stream2.x_range, expected_xrange)
        np.testing.assert_allclose(stream2.y_range, expected_yrange)

    wait_until(check, page)


@pytest.mark.usefixtures("bokeh_backend")
def test_datashade_dynspread_constant_values_overlay_not_empty(serve_hv):
    from holoviews.operation.datashader import datashade, dynspread

    x = np.arange(10_000)
    y1 = np.full(x.size, 5.0)
    y2 = np.full(x.size, 8.0)

    curve1 = hv.Scatter((x, y1), label="variable1")
    curve2 = hv.Scatter((x, y2), label="variable2")

    curve_datashade1 = dynspread(datashade(curve1))
    curve_datashade2 = dynspread(datashade(curve2))

    stream1 = RangeXY(source=curve_datashade1)
    stream2 = RangeXY(source=curve_datashade2)

    overlay = (curve_datashade1 * curve_datashade2).opts(ylim=(4, 9))

    plot = hv.renderer("bokeh").get_plot(overlay)
    page = serve_hv(plot)
    hv_plot = page.locator(".bk-events")
    expect(hv_plot).to_have_count(1)

    def check():
        assert stream1.y_range == (4, 9)
        assert stream2.y_range == (4, 9)
        for subplot in plot.subplots.values():
            source = subplot.handles["source"]
            image = np.asarray(source.data["image"][0])
            assert image.size > 1, "image was never resized past its 1x1 placeholder"
            alpha = image.view(np.uint8).reshape(*image.shape, 4)[..., 3]
            assert np.count_nonzero(alpha) > 0, "image is fully transparent"

    wait_until(check, page)
