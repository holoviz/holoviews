import time

import numpy as np
import pytest

try:
    from playwright.sync_api import expect
except ImportError:
    pytestmark = pytest.mark.skip('playwright not available')

pytestmark = pytest.mark.ui

import panel as pn
from panel.pane.holoviews import HoloViews
from panel.tests.util import serve_and_wait, wait_until

import holoviews as hv
from holoviews import Curve, DynamicMap, Scatter
from holoviews.streams import BoundsXY, Lasso, RangeXY


@pytest.mark.usefixtures("bokeh_backend")
def test_box_select(page, port):
    hv_scatter = Scatter([1, 2, 3]).opts(
        tools=['box_select'], active_tools=['box_select']
    )

    bounds = BoundsXY(source=hv_scatter)

    pn_scatter = HoloViews(hv_scatter)

    serve_and_wait(pn_scatter, port=port)
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


@pytest.mark.usefixtures("bokeh_backend")
def test_lasso_select(page, port):
    hv_scatter = Scatter([1, 2, 3]).opts(
        tools=['lasso_select'], active_tools=['lasso_select']
    )

    lasso = Lasso(source=hv_scatter)

    pn_scatter = HoloViews(hv_scatter)

    serve_and_wait(pn_scatter, port=port)
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

@pytest.mark.usefixtures("bokeh_backend")
def test_rangexy(page, port):
    hv_scatter = Scatter([1, 2, 3]).opts(active_tools=['box_zoom'])

    rangexy = RangeXY(source=hv_scatter)

    pn_scatter = HoloViews(hv_scatter)

    serve_and_wait(pn_scatter, port=port)
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

@pytest.mark.usefixtures("bokeh_backend")
def test_multi_axis_rangexy(page, port):
    c1 = Curve(np.arange(100).cumsum(), vdims='y')
    c2 = Curve(-np.arange(100).cumsum(), vdims='y2')
    s1 = RangeXY(source=c1)
    s2 = RangeXY(source=c2)

    overlay = (c1 * c2).opts(multi_y=True)

    pn_scatter = HoloViews(overlay)

    serve_and_wait(pn_scatter, port=port)
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


@pytest.mark.usefixtures("bokeh_backend")
def test_bind_trigger(page, port):
    # Regression test for https://github.com/holoviz/holoviews/issues/6013

    BOUND_COUNT, RANGE_COUNT = [0], [0]

    def bound_function():
        BOUND_COUNT[0] += 1
        return Curve([])


    def range_function(x_range, y_range):
        RANGE_COUNT[0] += 1
        return Curve([])

    range_dmap = DynamicMap(range_function, streams=[hv.streams.RangeXY()])
    bind_dmap = DynamicMap(pn.bind(bound_function))
    widget = pn.pane.HoloViews(bind_dmap * range_dmap)

    serve_and_wait(widget, port=port)
    page.goto(f"http://localhost:{port}")

    hv_plot = page.locator('.bk-events')
    expect(hv_plot).to_have_count(1)

    bbox = hv_plot.bounding_box()
    hv_plot.click()

    page.mouse.move(bbox['x']+100, bbox['y']+100)
    page.mouse.down()
    page.mouse.move(bbox['x']+150, bbox['y']+150, steps=5)
    page.mouse.up()

    wait_until(lambda: RANGE_COUNT[0] > 2, page)

    assert BOUND_COUNT[0] == 1
