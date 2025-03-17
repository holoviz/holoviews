import numpy as np
import pandas as pd
import panel as pn
import pytest

import holoviews as hv
from holoviews import Curve, DynamicMap, Scatter
from holoviews.plotting.bokeh.util import BOKEH_GE_3_4_0, BOKEH_GE_3_7_0
from holoviews.streams import (
    BoundsX,
    BoundsXY,
    BoundsY,
    Lasso,
    MultiAxisTap,
    RangeXY,
    Tap,
)

from .. import expect, wait_until

pytestmark = pytest.mark.ui

skip_popup = pytest.mark.skipif(not BOKEH_GE_3_4_0, reason="Pop ups needs Bokeh 3.4")

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

    if BOKEH_GE_3_4_0:
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
def test_multi_axis_tap(serve_hv):
    c1 = Curve(np.arange(10).cumsum(), vdims='y1')
    c2 = Curve(np.arange(20).cumsum(), vdims='y2')

    overlay = (c1 * c2).opts(multi_y=True)

    s = MultiAxisTap(source=overlay)

    page = serve_hv(overlay)

    hv_plot = page.locator('.bk-events')

    expect(hv_plot).to_have_count(1)

    hv_plot.click()

    def test():
        assert s.xs == {'x': 11.560240963855422}
        assert len(s.ys) == 2
        assert np.isclose(s.ys["y1"], 18.642857142857146)
        assert np.isclose(s.ys["y2"], 78.71428571428572)

    wait_until(test, page)


@pytest.mark.usefixtures("bokeh_backend")
def test_multi_axis_tap_datetime(serve_hv):
    c1 = Curve((pd.date_range('2024-01-01', '2024-01-10'), np.arange(10).cumsum()), vdims='y1')
    c2 = Curve((pd.date_range('2024-01-01', '2024-01-20'), np.arange(20).cumsum()), vdims='y2')

    overlay = (c1 * c2).opts(multi_y=True)

    s = MultiAxisTap(source=overlay)

    page = serve_hv(overlay)

    hv_plot = page.locator('.bk-events')

    expect(hv_plot).to_have_count(1)

    hv_plot.click()

    def test():
        assert s.xs == {'x': np.datetime64('2024-01-12T13:26:44.819277')}
        assert s.xs == {'x': np.datetime64('2024-01-12T13:26:44.819277')}
        assert len(s.ys) == 2
        if BOKEH_GE_3_7_0:
            assert np.isclose(s.ys["y1"], 16.19603524229075)
            assert np.isclose(s.ys["y2"], 68.38325991189429)
        else:
            assert np.isclose(s.ys["y1"], 18.130705394191)
            assert np.isclose(s.ys["y2"], 76.551867219917)

    wait_until(test, page)


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


@pytest.mark.usefixtures("bokeh_backend")
@skip_popup
class TestPopup:
    def _select_points_based_on_tool(self, tool, page, plot):
        """Helper method to perform point selection based on tool type."""
        box = plot.bounding_box()

        if tool == "box_select":
            start_x, start_y = box['x'] + 90, box['y'] + 90
            end_x, end_y = box['x'] + 170, box['y'] + 125
            page.mouse.move(start_x, start_y)
            plot.click()
            page.mouse.down()
            page.mouse.move(end_x, end_y)
            page.mouse.up()
        elif tool == "lasso_select":
            start_x, start_y = box['x'] + 1, box['y'] + box['height'] - 1
            mid_x, mid_y = box['x'] + 1, box['y'] + 1
            end_x, end_y = box['x'] + box['width'] - 1, box['y'] + 1
            page.mouse.move(start_x, start_y)
            plot.click()
            page.mouse.down()
            page.mouse.move(mid_x, mid_y)
            page.mouse.move(end_x, end_y)
            page.mouse.up()
        elif tool == "tap":
            plot.click()

    def _get_popup_distances_relative_to_bbox(self, popup_box, plot_box):
        return {
            'left': abs(popup_box['x'] - plot_box['x']),
            'right': abs((popup_box['x'] + popup_box['width']) - (plot_box['x'] + plot_box['width'])),
            'top': abs(popup_box['y'] - plot_box['y']),
            'bottom': abs((popup_box['y'] + popup_box['height']) - (plot_box['y'] + plot_box['height']))
        }

    def _verify_popup_position(self, distances, popup_position):
        if "right" in popup_position:
            assert distances['right'] <= distances['left']
        elif "left" in popup_position:
            assert distances['left'] <= distances['right']

        if "top" in popup_position:
            assert distances['top'] <= distances['bottom']
        elif "bottom" in popup_position:
            assert distances['bottom'] <= distances['top']

    def _serve_plot(self, serve_hv, plot):
        page = serve_hv(plot)
        hv_plot = page.locator('.bk-events')
        expect(hv_plot).to_have_count(1)
        return page, hv_plot

    def _locate_popup(self, page, count=1):
        locator = page.locator(".markdown")
        expect(locator).to_have_count(count)
        return locator

    def test_basic(self, serve_hv, points):
        def popup_form(name):
            return f"# {name}"

        points.opts(tools=["tap"])
        hv.streams.Tap(source=points, popup=popup_form("Tap"))

        page, hv_plot = self._serve_plot(serve_hv, points)
        self._select_points_based_on_tool("tap", page, hv_plot)
        self._locate_popup(page)

    @pytest.mark.parametrize("popup_position", [
        "top_right", "top_left", "bottom_left", "bottom_right",
        "right", "left", "top", "bottom"
    ])
    def test_polygons_tap(self, serve_hv, popup_position):
        def popup_form(name):
            return "# selection"

        points = hv.Polygons([(0, 0), (0, 1), (1, 1), (1, 0)]).opts(tools=["tap"])
        hv.streams.Tap(source=points, popup=popup_form("Tap"), popup_position=popup_position)

        page, hv_plot = self._serve_plot(serve_hv, points)
        self._select_points_based_on_tool("tap", page, hv_plot)
        locator = self._locate_popup(page)

        box = hv_plot.bounding_box()
        popup_box = locator.bounding_box()
        distances = self._get_popup_distances_relative_to_bbox(popup_box, box)
        self._verify_popup_position(distances, popup_position)

    def test_return_none(self, serve_hv, points):
        def popup_form(name):
            return None

        hv.streams.Tap(source=points, popup=popup_form("Tap"))

        page, hv_plot = self._serve_plot(serve_hv, points)
        self._select_points_based_on_tool("tap", page, hv_plot)
        self._locate_popup(page, count=0)

    def test_callbacks(self, serve_hv, points):
        def popup_form(x, y):
            return pn.widgets.Button(name=f"{x},{y}")

        points.opts(tools=["tap"])
        hv.streams.Tap(source=points, popup=popup_form)

        page, hv_plot = self._serve_plot(serve_hv, points)
        self._select_points_based_on_tool("tap", page, hv_plot)
        locator = page.locator(".bk-btn")
        expect(locator).to_have_count(2)

    def test_async_callbacks(self, serve_hv, points):
        async def popup_form(x, y):
            return pn.widgets.Button(name=f"{x},{y}")

        points.opts(tools=["tap"])
        hv.streams.Tap(source=points, popup=popup_form)

        page, hv_plot = self._serve_plot(serve_hv, points)
        self._select_points_based_on_tool("tap", page, hv_plot)
        locator = page.locator(".bk-btn")
        expect(locator).to_have_count(2)

    @pytest.mark.filterwarnings("ignore:reference already known")
    def test_callback_visible(self, serve_hv, points):
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

        points.opts(tools=["tap"])
        hv.streams.Tap(source=points, popup=popup_form)

        page, hv_plot = self._serve_plot(serve_hv, points)
        self._select_points_based_on_tool("tap", page, hv_plot)

        locator = page.locator(".bk-btn")
        expect(locator).to_have_count(2)
        expect(locator.first).to_be_visible()

        locator = page.locator(".custom-button")
        locator.click()
        locator = page.locator(".bk-btn")
        expect(locator.first).not_to_be_visible()

    @pytest.mark.parametrize("tool", ["box_select", "lasso_select", "tap"])
    @pytest.mark.parametrize("popup_position", [
        "top_right", "top_left", "bottom_left", "bottom_right",
        "right", "left", "top", "bottom"
    ])
    def test_position_selection1d(self, serve_hv, points, tool, popup_position):
        def popup_form(index):
            if index:
                return f"# selection\n{len(index)} {popup_position}"

        hv.streams.Selection1D(source=points, popup=popup_form, popup_position=popup_position)
        points.opts(tools=[tool], active_tools=[tool])

        page, hv_plot = self._serve_plot(serve_hv, points)
        self._select_points_based_on_tool(tool, page, hv_plot)

        locator = self._locate_popup(page)
        expect(locator).not_to_have_text("selection\n0")

        box = hv_plot.bounding_box()
        popup_box = locator.bounding_box()
        distances = self._get_popup_distances_relative_to_bbox(popup_box, box)
        self._verify_popup_position(distances, popup_position)

    def test_anchor_selection1d(self, serve_hv, points):
        def popup_form(index):
            if index:
                return f"# selection\n{len(index)}"

        hv.streams.Selection1D(source=points, popup=popup_form, popup_position="top", popup_anchor="top_right")
        points.opts(tools=["tap"], active_tools=["tap"])

        page, hv_plot = self._serve_plot(serve_hv, points)
        self._select_points_based_on_tool("tap", page, hv_plot)
        locator = self._locate_popup(page)
        expect(locator).not_to_have_text("selection\n0")

        box = hv_plot.bounding_box()
        popup_box = locator.bounding_box()
        distances = self._get_popup_distances_relative_to_bbox(popup_box, box)

        assert distances['right'] >= distances['left']


    @pytest.mark.parametrize("tool, tool_type", [
        ("box_select", BoundsXY),
        ("lasso_select", Lasso),
        ("tap", Tap)
    ])
    def test_anchor_tools(self, serve_hv, points, tool, tool_type):
        def popup_form(*args, **kwargs):
            return "# selection"

        points = points.opts(tools=[tool], active_tools=[tool])
        tool_type(source=points, popup=popup_form, popup_position="bottom", popup_anchor="bottom_right")
        page, hv_plot = self._serve_plot(serve_hv, points)

        self._select_points_based_on_tool(tool, page, hv_plot)
        locator = self._locate_popup(page)

        box = hv_plot.bounding_box()
        popup_box = locator.bounding_box()
        distances = self._get_popup_distances_relative_to_bbox(popup_box, box)

        assert distances['right'] >= distances['left']
        assert distances['bottom'] >= distances['top']


    @pytest.mark.parametrize("tool, tool_type", [("box_select", BoundsXY), ("lasso_select", Lasso)])
    @pytest.mark.parametrize("popup_position", [
        "top_right", "top_left", "bottom_left", "bottom_right",
        "right", "left", "top", "bottom"
    ])
    def test_position_tools(self, serve_hv, points, tool, tool_type, popup_position):
        def popup_form(*args, **kwargs):
            return "# selection"

        points = points.opts(tools=[tool], active_tools=[tool])
        tool_type(source=points, popup=popup_form, popup_position=popup_position)

        page, hv_plot = self._serve_plot(serve_hv, points)
        self._select_points_based_on_tool(tool, page, hv_plot)
        locator = self._locate_popup(page)

        box = hv_plot.bounding_box()
        popup_box = locator.bounding_box()
        distances = self._get_popup_distances_relative_to_bbox(popup_box, box)

        self._verify_popup_position(distances, popup_position)
