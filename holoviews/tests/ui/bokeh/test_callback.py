import numpy as np
import panel as pn
import pytest

import holoviews as hv
from holoviews import Curve, DynamicMap, Scatter
from holoviews.plotting.bokeh.util import bokeh34
from holoviews.streams import BoundsX, BoundsXY, BoundsY, Lasso, RangeXY

from .. import expect, wait_until

pytestmark = pytest.mark.ui

skip_popup = pytest.mark.skipif(not bokeh34, reason="Pop ups needs Bokeh 3.4")

@pytest.fixture
def points():
    rng = np.random.default_rng(10)
    return hv.Points(rng.normal(size=(1000, 2)))

@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize(
    ["BoundsTool", "bound_slice", "bound_attr"],
    [
        (BoundsXY, slice(None), "bounds"),
        (BoundsX, slice(0, None, 2), "boundsx"),
        (BoundsY, slice(1, None, 2), "boundsy"),
    ],
)
def test_box_select(serve_hv, BoundsTool, bound_slice, bound_attr):
    hv_scatter = Scatter([1, 2, 3]).opts(
        tools=['box_select'], active_tools=['box_select']
    )

    bounds = BoundsTool(source=hv_scatter)

    page = serve_hv(hv_scatter)
    hv_plot = page.locator('.bk-events')

    expect(hv_plot).to_have_count(1)

    bbox = hv_plot.bounding_box()
    hv_plot.click()

    page.mouse.move(bbox['x']+100, bbox['y']+100)
    page.mouse.down()
    page.mouse.move(bbox['x']+150, bbox['y']+150, steps=5)
    page.mouse.up()

    expected_bounds = (0.32844036697247725, 1.8285714285714285, 0.8788990825688077, 2.3183673469387758)
    wait_until(lambda: getattr(bounds, bound_attr) == expected_bounds[bound_slice], page)


@pytest.mark.usefixtures("bokeh_backend")
def test_lasso_select(serve_hv):
    hv_scatter = Scatter([1, 2, 3]).opts(
        tools=['lasso_select'], active_tools=['lasso_select']
    )

    lasso = Lasso(source=hv_scatter)

    page = serve_hv(hv_scatter)
    hv_plot = page.locator('.bk-events')

    expect(hv_plot).to_have_count(1)

    bbox = hv_plot.bounding_box()
    hv_plot.click()

    page.mouse.move(bbox['x']+100, bbox['y']+100)
    page.mouse.down()
    page.mouse.move(bbox['x']+150, bbox['y']+150, steps=5)
    page.mouse.move(bbox['x']+50, bbox['y']+150, steps=5)
    page.mouse.up()

    if bokeh34:
        expected_array = np.array([
            [3.28440367e-01, 2.31836735e00],
            [4.38532110e-01, 2.22040816e00],
            [5.48623853e-01, 2.12244898e00],
            [6.58715596e-01, 2.02448980e00],
            [7.68807339e-01, 1.92653061e00],
            [8.78899083e-01, 1.82857143e00],
            [6.58715596e-01, 1.82857143e00],
            [4.38532110e-01, 1.82857143e00],
            [2.18348624e-01, 1.82857143e00],
            [-1.83486239e-03, 1.82857143e00],
            [-2.00000000e-01, 1.82857143e00],
        ])
    else:
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

    def compare_array():
        if lasso.geometry is None:
            return False
        np.testing.assert_almost_equal(lasso.geometry, expected_array)

    wait_until(compare_array, page)

@pytest.mark.usefixtures("bokeh_backend")
def test_rangexy(serve_hv):
    hv_scatter = Scatter([1, 2, 3]).opts(active_tools=['box_zoom'])

    rangexy = RangeXY(source=hv_scatter)

    page = serve_hv(hv_scatter)
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
def test_multi_axis_rangexy(serve_hv):
    c1 = Curve(np.arange(100).cumsum(), vdims='y')
    c2 = Curve(-np.arange(100).cumsum(), vdims='y2')
    s1 = RangeXY(source=c1)
    s2 = RangeXY(source=c2)

    overlay = (c1 * c2).opts(multi_y=True)

    page = serve_hv(overlay)
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
        np.testing.assert_almost_equal(s1.x_range, expected_xrange) and
        np.testing.assert_almost_equal(s1.y_range, expected_yrange1) and
        np.testing.assert_almost_equal(s2.y_range, expected_yrange2)
    ), page)


@pytest.mark.usefixtures("bokeh_backend")
def test_bind_trigger(serve_hv):
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

    page = serve_hv(bind_dmap * range_dmap)
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


@skip_popup
@pytest.mark.usefixtures("bokeh_backend")
def test_stream_popup(serve_hv):
    def popup_form(name):
        return f"# {name}"

    points = hv.Points(np.random.randn(10, 2)).opts(tools=["tap"])
    hv.streams.Tap(source=points, popup=popup_form("Tap"))

    page = serve_hv(points)
    hv_plot = page.locator('.bk-events')
    hv_plot.click()
    expect(hv_plot).to_have_count(1)

    locator = page.locator("#tap")
    expect(locator).to_have_count(1)


@skip_popup
@pytest.mark.usefixtures("bokeh_backend")
def test_stream_popup_none(serve_hv, points):
    def popup_form(name):
        return

    hv.streams.Tap(source=points, popup=popup_form("Tap"))

    page = serve_hv(points)
    hv_plot = page.locator('.bk-events')
    expect(hv_plot).to_have_count(1)

    bbox = hv_plot.bounding_box()
    hv_plot.click()

    page.mouse.move(bbox['x']+100, bbox['y']+100)
    page.mouse.down()
    page.mouse.move(bbox['x']+150, bbox['y']+150, steps=5)
    page.mouse.up()

    locator = page.locator("#tap")
    expect(locator).to_have_count(0)


@skip_popup
@pytest.mark.usefixtures("bokeh_backend")
def test_stream_popup_callbacks(serve_hv):
    def popup_form(x, y):
        return pn.widgets.Button(name=f"{x},{y}")

    points = hv.Points(np.random.randn(10, 2)).opts(tools=["tap"])
    hv.streams.Tap(source=points, popup=popup_form)

    page = serve_hv(points)
    hv_plot = page.locator('.bk-events')
    hv_plot.click()
    expect(hv_plot).to_have_count(1)

    locator = page.locator(".bk-btn")
    expect(locator).to_have_count(2)


@skip_popup
@pytest.mark.usefixtures("bokeh_backend")
def test_stream_popup_visible(serve_hv, points):
    def popup_form(x, y):
        def hide(_):
            col.visible = False
        button = pn.widgets.Button(
            name=f"{x},{y}",
            on_click=hide,
            css_classes=["custom-button"]
        )
        col = pn.Column(button)
        return col

    points = points.opts(tools=["tap"])
    hv.streams.Tap(source=points, popup=popup_form)

    page = serve_hv(points)
    hv_plot = page.locator('.bk-events')
    hv_plot.click()
    expect(hv_plot).to_have_count(1)

    # initial appearance
    locator = page.locator(".bk-btn")
    expect(locator).to_have_count(2)
    expect(locator.first).to_be_visible()

    # click custom button to hide
    locator = page.locator(".custom-button")
    locator.click()
    locator = page.locator(".bk-btn")
    expect(locator.first).not_to_be_visible()



@skip_popup
@pytest.mark.usefixtures("bokeh_backend")
def test_stream_popup_close_button(serve_hv, points):
    def popup_form(x, y):
        return "Hello"

    points = points.opts(tools=["tap", "box_select"])
    hv.streams.Tap(source=points, popup=popup_form)
    hv.streams.BoundsXY(source=points, popup=popup_form)

    page = serve_hv(points)
    hv_plot = page.locator('.bk-events')
    expect(hv_plot).to_have_count(1)
    hv_plot.click()

    locator = page.locator(".bk-btn.bk-btn-default")
    expect(locator).to_have_count(1)
    expect(locator).to_be_visible()
    page.click(".bk-btn.bk-btn-default")
    expect(locator).not_to_be_visible()


@skip_popup
@pytest.mark.usefixtures("bokeh_backend")
def test_stream_popup_selection1d_undefined(serve_hv, points):
    hv.streams.Selection1D(source=points)

    page = serve_hv(points)
    hv_plot = page.locator('.bk-events')
    expect(hv_plot).to_have_count(1)
    hv_plot.click()  # should not raise any error; properly guarded


@skip_popup
@pytest.mark.usefixtures("bokeh_backend")
def test_stream_popup_selection1d_tap(serve_hv, points):
    def popup_form(index):
        return "# Tap"

    points = points.opts(hit_dilation=5)
    hv.streams.Selection1D(source=points, popup=popup_form)
    points.opts(tools=["tap"], active_tools=["tap"])

    page = serve_hv(points)
    hv_plot = page.locator('.bk-events')
    expect(hv_plot).to_have_count(1)
    hv_plot.click()

    locator = page.locator("#tap")
    expect(locator).to_have_count(1)


@skip_popup
@pytest.mark.usefixtures("bokeh_backend")
def test_stream_popup_selection1d_lasso_select(serve_hv, points):
    def popup_form(index):
        if index:
            return f"# lasso\n{len(index)}"

    hv.streams.Selection1D(source=points, popup=popup_form)
    points.opts(tools=["tap", "lasso_select"], active_tools=["lasso_select"])

    page = serve_hv(points)
    hv_plot = page.locator('.bk-events')
    expect(hv_plot).to_have_count(1)

    box = hv_plot.bounding_box()
    start_x, start_y = box['x'] + 10, box['y'] + box['height'] - 10
    mid_x, mid_y = box['x'] + 10, box['y'] + 10
    end_x, end_y = box['x'] + box['width'] - 10, box['y'] + 10

    page.mouse.move(start_x, start_y)
    hv_plot.click()
    page.mouse.down()
    page.mouse.move(mid_x, mid_y)
    page.mouse.move(end_x, end_y)
    page.mouse.up()

    wait_until(lambda: expect(page.locator("#lasso")).to_have_count(1), page)
    locator = page.locator("#lasso")
    expect(locator).to_have_count(1)
    expect(locator).not_to_have_text("lasso\n0")


@pytest.mark.usefixtures("bokeh_backend")
def test_stream_subcoordinate_y_range(serve_hv, points):
    def cb(x_range, y_range):
        return (
            Curve(np.arange(100).cumsum(), vdims='y', label='A').opts(subcoordinate_y=True) *
            Curve(-np.arange(100).cumsum(), vdims='y2', label='B').opts(subcoordinate_y=True)
        )

    stream = RangeXY()
    dmap = DynamicMap(cb, streams=[stream]).opts(active_tools=['box_zoom'])

    page = serve_hv(dmap)

    hv_plot = page.locator('.bk-events')

    expect(hv_plot).to_have_count(1)

    bbox = hv_plot.bounding_box()
    hv_plot.click()

    page.mouse.move(bbox['x']+60, bbox['y']+60)
    page.mouse.down()
    page.mouse.move(bbox['x']+190, bbox['y']+190, steps=5)
    page.mouse.up()

    expected_xrange = (7.008849557522124, 63.95575221238938)
    expected_yrange = (0.030612244897959183, 1.0918367346938775)
    wait_until(lambda: stream.x_range == expected_xrange and stream.y_range == expected_yrange, page)
