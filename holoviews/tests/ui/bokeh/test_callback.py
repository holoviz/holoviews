import time

import numpy as np
import pytest

try:
    from playwright.sync_api import expect
except ImportError:
    pytestmark = pytest.mark.skip('playwright not available')

pytestmark = pytest.mark.ui

from panel.io.server import serve
from panel.pane.holoviews import HoloViews
from panel.tests.util import wait_until

from holoviews import Curve, Scatter
from holoviews.plotting.bokeh import BokehRenderer
from holoviews.streams import BoundsXY, Lasso, RangeXY


def test_box_select(page, port):
    hv_scatter = Scatter([1, 2, 3]).opts(
        backend='bokeh', tools=['box_select'], active_tools=['box_select']
    )

    bounds = BoundsXY(source=hv_scatter)

    pn_scatter = HoloViews(hv_scatter, renderer=BokehRenderer)

    serve(pn_scatter, port=port, threaded=True, show=False)

    time.sleep(0.5)

    page.goto(f"http://localhost:{port}")

    hv_plot = page.locator('.bk-events')

    expect(hv_plot).to_have_count(1)

    bbox = hv_plot.bounding_box()
    hv_plot.click()

    page.mouse.move(bbox['x']+100, bbox['y']+100)
    page.mouse.down()
    page.mouse.move(bbox['x']+150, bbox['y']+150, steps=5)
    page.mouse.up()

    expected_bounds = (0.32844036697247725, 1.8285714285714285, 0.8788990825688077, 2.3183673469387758)
    wait_until(lambda: bounds.bounds == expected_bounds, page)



def test_lasso_select(page, port):
    hv_scatter = Scatter([1, 2, 3]).opts(
        backend='bokeh', tools=['lasso_select'], active_tools=['lasso_select']
    )

    lasso = Lasso(source=hv_scatter)

    pn_scatter = HoloViews(hv_scatter, renderer=BokehRenderer)

    serve(pn_scatter, port=port, threaded=True, show=False)

    time.sleep(0.5)

    page.goto(f"http://localhost:{port}")

    hv_plot = page.locator('.bk-events')

    expect(hv_plot).to_have_count(1)

    bbox = hv_plot.bounding_box()
    hv_plot.click()

    page.mouse.move(bbox['x']+100, bbox['y']+100)
    page.mouse.down()
    page.mouse.move(bbox['x']+150, bbox['y']+150, steps=5)
    page.mouse.move(bbox['x']+50, bbox['y']+150, steps=5)
    page.mouse.up()

    time.sleep(1)

    expected_array = np.array([
        [ 3.28440367e-01,  2.31836735e+00],
        [ 5.48623853e-01,  2.12244898e+00],
        [ 6.58715596e-01,  2.02448980e+00],
        [ 7.68807339e-01,  1.92653061e+00],
        [ 8.78899083e-01,  1.82857143e+00],
        [ 6.58715596e-01,  1.82857143e+00],
        [ 4.38532110e-01,  1.82857143e+00],
        [ 2.18348624e-01,  1.82857143e+00],
        [-1.83486239e-03,  1.82857143e+00],
        [-2.00000000e-01,  1.82857143e+00],
        [-2.00000000e-01,  1.82857143e+00]
    ])
    np.testing.assert_almost_equal(lasso.geometry, expected_array)


def test_rangexy(page, port):
    hv_scatter = Scatter([1, 2, 3]).opts(backend='bokeh', active_tools=['box_zoom'])

    rangexy = RangeXY(source=hv_scatter)

    pn_scatter = HoloViews(hv_scatter, renderer=BokehRenderer)

    serve(pn_scatter, port=port, threaded=True, show=False)

    time.sleep(0.5)

    page.goto(f"http://localhost:{port}")

    hv_plot = page.locator('.bk-events')

    expect(hv_plot).to_have_count(1)

    bbox = hv_plot.bounding_box()
    hv_plot.click()

    page.mouse.move(bbox['x']+100, bbox['y']+100)
    page.mouse.down()
    page.mouse.move(bbox['x']+150, bbox['y']+150, steps=5)
    page.mouse.up()

    expected_xrange = (0.32844036697247725, 0.8788990825688077)
    expected_yrange = (1.8285714285714285, 2.3183673469387758)
    wait_until(lambda: rangexy.x_range == expected_xrange and rangexy.y_range == expected_yrange, page)

def test_multi_axis_rangexy(page, port):
    c1 = Curve(np.arange(100).cumsum(), vdims='y')
    c2 = Curve(-np.arange(100).cumsum(), vdims='y2')
    s1 = RangeXY(source=c1)
    s2 = RangeXY(source=c2)

    overlay = (c1 * c2).opts(backend='bokeh', multi_y=True)

    pn_scatter = HoloViews(overlay, renderer=BokehRenderer)

    serve(pn_scatter, port=port, threaded=True, show=False)

    time.sleep(0.5)

    page.goto(f"http://localhost:{port}")

    hv_plot = page.locator('.bk-events')

    expect(hv_plot).to_have_count(1)

    bbox = hv_plot.bounding_box()
    hv_plot.click()

    page.mouse.move(bbox['x']+100, bbox['y']+100)
    page.mouse.down()
    page.mouse.move(bbox['x']+150, bbox['y']+150, steps=5)
    page.mouse.up()

    expected_xrange = (-35.1063829787234, 63.89361702127659)
    expected_yrange1 = (717.2448979591848, 6657.244897959185)
    expected_yrange2 = (-4232.7551020408155, 1707.2448979591848)
    wait_until(lambda: (
        s1.x_range == expected_xrange and
        s1.y_range == expected_yrange1 and
        s2.y_range == expected_yrange2
    ), page)
