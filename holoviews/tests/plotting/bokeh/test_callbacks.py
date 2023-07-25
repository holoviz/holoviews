import datetime as dt

from collections import deque, namedtuple
from unittest import SkipTest

import numpy as np
import pandas as pd
import pytest
import pyviz_comms as comms

from holoviews.core import DynamicMap
from holoviews.core.options import Store
from holoviews.element import Points, Polygons, Box, Curve, Table, Rectangles
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import (
    PointDraw, PolyDraw, PolyEdit, BoxEdit, PointerXY, PointerX,
    PlotReset, Selection1D, RangeXY, PlotSize, CDSStream, SingleTap
)

from bokeh.events import Tap
from bokeh.io.doc import set_curdoc
from bokeh.models import Range1d, Plot, ColumnDataSource, Selection, PolyEditTool
from holoviews.plotting.bokeh.callbacks import (
    Callback, CDSCallback, PointDrawCallback, PolyDrawCallback, PolyEditCallback,
    BoxEditCallback, PointerXCallback, TapCallback
)
from holoviews.plotting.bokeh.renderer import BokehRenderer

bokeh_server_renderer = BokehRenderer.instance(mode='server')
bokeh_renderer = BokehRenderer.instance()


class CallbackTestCase(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        Store.current_backend = 'bokeh'
        self.comm_manager = bokeh_renderer.comm_manager
        bokeh_renderer.comm_manager = comms.CommManager

    def tearDown(self):
        Store.current_backend = self.previous_backend
        bokeh_server_renderer.last_plot = None
        bokeh_renderer.last_plot = None
        Callback._callbacks = {}
        bokeh_renderer.comm_manager = self.comm_manager


class TestCallbacks(CallbackTestCase):

    def test_stream_callback(self):
        dmap = DynamicMap(lambda x, y: Points([(x, y)]), kdims=[], streams=[PointerXY()])
        plot = bokeh_server_renderer.get_plot(dmap)
        bokeh_server_renderer(plot)
        set_curdoc(plot.document)
        plot.callbacks[0].on_msg({"x": 0.3, "y": 0.2})
        data = plot.handles['source'].data
        self.assertEqual(data['x'], np.array([0.3]))
        self.assertEqual(data['y'], np.array([0.2]))

    def test_point_stream_callback_clip(self):
        dmap = DynamicMap(lambda x, y: Points([(x, y)]), kdims=[], streams=[PointerXY()])
        plot = bokeh_server_renderer.get_plot(dmap)
        bokeh_server_renderer(plot)
        set_curdoc(plot.document)
        plot.callbacks[0].on_msg({"x": -0.3, "y": 1.2})
        data = plot.handles['source'].data
        self.assertEqual(data['x'], np.array([0]))
        self.assertEqual(data['y'], np.array([1]))

    def test_stream_callback_on_clone(self):
        points = Points([])
        stream = PointerXY(source=points)
        plot = bokeh_server_renderer.get_plot(points.clone())
        bokeh_server_renderer(plot)
        set_curdoc(plot.document)
        plot.callbacks[0].on_msg({"x": 0.8, "y": 0.3})
        self.assertEqual(stream.x, 0.8)
        self.assertEqual(stream.y, 0.3)

    def test_stream_callback_on_unlinked_clone(self):
        points = Points([])
        PointerXY(source=points)
        plot = bokeh_server_renderer.get_plot(points.clone(link=False))
        bokeh_server_renderer(plot)
        self.assertTrue(len(plot.callbacks) == 0)

    def test_stream_callback_with_ids(self):
        dmap = DynamicMap(lambda x, y: Points([(x, y)]), kdims=[], streams=[PointerXY()])
        plot = bokeh_server_renderer.get_plot(dmap)
        bokeh_server_renderer(plot)
        set_curdoc(plot.document)
        model = plot.state
        plot.callbacks[0].on_msg({"x": {'id': model.ref['id'], 'value': 0.5},
                                  "y": {'id': model.ref['id'], 'value': 0.4}})
        data = plot.handles['source'].data
        self.assertEqual(data['x'], np.array([0.5]))
        self.assertEqual(data['y'], np.array([0.4]))

    def test_stream_callback_single_call(self):
        def history_callback(x, history=deque(maxlen=10)):
            history.append(x)
            return Curve(list(history))
        stream = PointerX(x=0)
        dmap = DynamicMap(history_callback, kdims=[], streams=[stream])
        plot = bokeh_server_renderer.get_plot(dmap)
        bokeh_server_renderer(plot)
        set_curdoc(plot.document)
        for i in range(20):
            stream.event(x=i)
        data = plot.handles['source'].data
        self.assertEqual(data['x'], np.arange(10))
        self.assertEqual(data['y'], np.arange(10, 20))

    def test_callback_cleanup(self):
        stream = PointerX(x=0)
        dmap = DynamicMap(lambda x: Curve([x]), streams=[stream])
        plot = bokeh_server_renderer.get_plot(dmap)
        self.assertTrue(bool(stream._subscribers))
        self.assertTrue(bool(Callback._callbacks))
        plot.cleanup()
        self.assertFalse(bool(stream._subscribers))
        self.assertFalse(bool(Callback._callbacks))

    def test_selection1d_syncs_to_selected(self):
        points = Points([(0, 0), (1, 1), (2, 2)]).opts(selected=[0, 2])
        stream = Selection1D(source=points)
        bokeh_renderer.get_plot(points)
        self.assertEqual(stream.index, [0, 2])


class TestResetCallback(CallbackTestCase):

    def test_reset_callback(self):
        resets = []
        def record(resetting):
            resets.append(resetting)
        curve = Curve([])
        stream = PlotReset(source=curve)
        stream.add_subscriber(record)
        plot = bokeh_server_renderer.get_plot(curve)
        plot.callbacks[0].on_msg({'reset': True})
        self.assertEqual(resets, [True])
        self.assertIs(stream.source, curve)



class TestPointerCallbacks(CallbackTestCase):

    def test_pointer_x_datetime_out_of_bounds(self):
        points = Points([(dt.datetime(2017, 1, 1), 1), (dt.datetime(2017, 1, 3), 3)]).opts(padding=0)
        PointerX(source=points)
        plot = bokeh_server_renderer.get_plot(points)
        set_curdoc(plot.document)
        callback = plot.callbacks[0]
        self.assertIsInstance(callback, PointerXCallback)
        msg = callback._process_msg({'x': 1000})
        self.assertEqual(msg['x'], np.datetime64(dt.datetime(2017, 1, 1)))
        msg = callback._process_msg({'x': 10000000000000})
        self.assertEqual(msg['x'], np.datetime64(dt.datetime(2017, 1, 3)))

    def test_tap_datetime_out_of_bounds(self):
        points = Points([(dt.datetime(2017, 1, 1), 1), (dt.datetime(2017, 1, 3), 3)])
        SingleTap(source=points)
        plot = bokeh_server_renderer.get_plot(points)
        set_curdoc(plot.document)
        callback = plot.callbacks[0]
        self.assertIsInstance(callback, TapCallback)
        msg = callback._process_msg({'x': 1000, 'y': 2})
        self.assertEqual(msg, {})
        msg = callback._process_msg({'x': 10000000000000, 'y': 1})
        self.assertEqual(msg, {})



class TestEditToolCallbacks(CallbackTestCase):

    def test_point_draw_callback(self):
        points = Points([(0, 1)])
        point_draw = PointDraw(source=points)
        plot = bokeh_server_renderer.get_plot(points)
        self.assertIsInstance(plot.callbacks[0], PointDrawCallback)
        callback = plot.callbacks[0]
        data = {'x': [1, 2, 3], 'y': [1, 2, 3]}
        callback.on_msg({'data': data})
        self.assertEqual(point_draw.element, Points(data))

    def test_point_draw_callback_initialized_server(self):
        points = Points([(0, 1)])
        PointDraw(source=points)
        plot = bokeh_server_renderer.get_plot(points)
        assert 'data' in plot.handles['source']._callbacks

    def test_point_draw_callback_with_vdims_initialization(self):
        points = Points([(0, 1, 'A')], vdims=['A'])
        stream = PointDraw(source=points)
        bokeh_server_renderer.get_plot(points)
        self.assertEqual(stream.element.dimension_values('A'), np.array(['A']))

    def test_point_draw_callback_with_vdims(self):
        points = Points([(0, 1, 'A')], vdims=['A'])
        point_draw = PointDraw(source=points)
        plot = bokeh_server_renderer.get_plot(points)
        self.assertIsInstance(plot.callbacks[0], PointDrawCallback)
        callback = plot.callbacks[0]
        data = {'x': [1, 2, 3], 'y': [1, 2, 3], 'A': [None, None, 1]}
        callback.on_msg({'data': data})
        processed = dict(data, A=[np.nan, np.nan, 1])
        self.assertEqual(point_draw.element, Points(processed, vdims=['A']))

    def test_poly_draw_callback(self):
        polys = Polygons([[(0, 0), (2, 2), (4, 0)]])
        poly_draw = PolyDraw(source=polys)
        plot = bokeh_server_renderer.get_plot(polys)
        self.assertIsInstance(plot.callbacks[0], PolyDrawCallback)
        callback = plot.callbacks[0]
        data = {'x': [[1, 2, 3], [3, 4, 5]], 'y': [[1, 2, 3], [3, 4, 5]]}
        callback.on_msg({'data': data})
        element = Polygons([[(1, 1), (2, 2), (3, 3)], [(3, 3), (4, 4), (5, 5)]])
        self.assertEqual(poly_draw.element, element)

    def test_poly_draw_callback_initialized_server(self):
        polys = Polygons([[(0, 0), (2, 2), (4, 0)]])
        PolyDraw(source=polys)
        plot = bokeh_server_renderer.get_plot(polys)
        assert 'data' in plot.handles['source']._callbacks

    def test_poly_draw_callback_with_vdims(self):
        polys = Polygons([{'x': [0, 2, 4], 'y': [0, 2, 0], 'A': 1}], vdims=['A'])
        poly_draw = PolyDraw(source=polys)
        plot = bokeh_server_renderer.get_plot(polys)
        self.assertIsInstance(plot.callbacks[0], PolyDrawCallback)
        callback = plot.callbacks[0]
        data = {'x': [[1, 2, 3], [3, 4, 5]], 'y': [[1, 2, 3], [3, 4, 5]], 'A': [1, 2]}
        callback.on_msg({'data': data})
        element = Polygons([{'x': [1, 2, 3], 'y': [1, 2, 3], 'A': 1},
                            {'x': [3, 4, 5], 'y': [3, 4, 5], 'A': 2}], vdims=['A'])
        self.assertEqual(poly_draw.element, element)

    def test_poly_draw_callback_with_vdims_no_color_index(self):
        polys = Polygons([{'x': [0, 2, 4], 'y': [0, 2, 0], 'A': 1}], vdims=['A']).options(color_index=None)
        poly_draw = PolyDraw(source=polys)
        plot = bokeh_server_renderer.get_plot(polys)
        self.assertIsInstance(plot.callbacks[0], PolyDrawCallback)
        callback = plot.callbacks[0]
        data = {'x': [[1, 2, 3], [3, 4, 5]], 'y': [[1, 2, 3], [3, 4, 5]], 'A': [1, 2]}
        callback.on_msg({'data': data})
        element = Polygons([{'x': [1, 2, 3], 'y': [1, 2, 3], 'A': 1},
                            {'x': [3, 4, 5], 'y': [3, 4, 5], 'A': 2}], vdims=['A'])
        self.assertEqual(poly_draw.element, element)

    def test_box_edit_callback(self):
        boxes = Rectangles([(-0.5, -0.5, 0.5, 0.5)])
        box_edit = BoxEdit(source=boxes)
        plot = bokeh_server_renderer.get_plot(boxes)
        self.assertIsInstance(plot.callbacks[0], BoxEditCallback)
        callback = plot.callbacks[0]
        source = plot.handles['cds']
        self.assertEqual(source.data['left'], [-0.5])
        self.assertEqual(source.data['bottom'], [-0.5])
        self.assertEqual(source.data['right'], [0.5])
        self.assertEqual(source.data['top'], [0.5])
        data = {'left': [-0.25, 0], 'bottom': [-1, 0.75], 'right': [0.25, 2], 'top': [1, 1.25]}
        callback.on_msg({'data': data})
        element = Rectangles([(-0.25, -1, 0.25, 1), (0, 0.75, 2, 1.25)])
        self.assertEqual(box_edit.element, element)

    def test_box_edit_callback_legacy(self):
        boxes = Polygons([Box(0, 0, 1)])
        box_edit = BoxEdit(source=boxes)
        plot = bokeh_server_renderer.get_plot(boxes)
        self.assertIsInstance(plot.callbacks[0], BoxEditCallback)
        callback = plot.callbacks[0]
        source = plot.handles['cds']
        self.assertEqual(source.data['left'], [-0.5])
        self.assertEqual(source.data['bottom'], [-0.5])
        self.assertEqual(source.data['right'], [0.5])
        self.assertEqual(source.data['top'], [0.5])
        data = {'left': [-0.25, 0], 'bottom': [-1, 0.75], 'right': [0.25, 2], 'top': [1, 1.25]}
        callback.on_msg({'data': data})
        element = Polygons([Box(0, 0, (0.5, 2)), Box(1, 1, (2, 0.5))])
        self.assertEqual(box_edit.element, element)

    def test_box_edit_callback_initialized_server(self):
        boxes = Polygons([Box(0, 0, 1)])
        BoxEdit(source=boxes)
        plot = bokeh_server_renderer.get_plot(boxes)
        assert 'data' in plot.handles['cds']._callbacks

    @pytest.mark.flaky(max_runs=3)
    def test_poly_edit_callback(self):
        polys = Polygons([[(0, 0), (2, 2), (4, 0)]])
        poly_edit = PolyEdit(source=polys)
        plot = bokeh_server_renderer.get_plot(polys)
        self.assertIsInstance(plot.callbacks[0], PolyEditCallback)
        callback = plot.callbacks[0]
        data = {'x': [[1, 2, 3], [3, 4, 5]], 'y': [[1, 2, 3], [3, 4, 5]]}
        callback.on_msg({'data': data})
        element = Polygons([[(1, 1), (2, 2), (3, 3)], [(3, 3), (4, 4), (5, 5)]])
        self.assertEqual(poly_edit.element, element)

    def test_poly_edit_callback_initialized_server(self):
        polys = Polygons([[(0, 0), (2, 2), (4, 0)]])
        PolyEdit(source=polys)
        plot = bokeh_server_renderer.get_plot(polys)
        assert 'data' in plot.handles['source']._callbacks

    def test_poly_edit_shared_callback(self):
        polys = Polygons([[(0, 0), (2, 2), (4, 0)]])
        polys2 = Polygons([[(0, 0), (2, 2), (4, 0)]])
        poly_edit = PolyEdit(source=polys, shared=True)
        poly_edit2 = PolyEdit(source=polys2, shared=True)
        plot = bokeh_server_renderer.get_plot(polys*polys2)
        edit_tools = [t for t in plot.state.tools if isinstance(t, PolyEditTool)]
        self.assertEqual(len(edit_tools), 1)
        plot1, plot2 = plot.subplots.values()
        self.assertIsInstance(plot1.callbacks[0], PolyEditCallback)
        callback = plot1.callbacks[0]
        data = {'x': [[1, 2, 3], [3, 4, 5]], 'y': [[1, 2, 3], [3, 4, 5]]}
        callback.on_msg({'data': data})
        self.assertIsInstance(plot2.callbacks[0], PolyEditCallback)
        callback = plot2.callbacks[0]
        data = {'x': [[1, 2, 3], [3, 4, 5]], 'y': [[1, 2, 3], [3, 4, 5]]}
        callback.on_msg({'data': data})
        element = Polygons([[(1, 1), (2, 2), (3, 3)], [(3, 3), (4, 4), (5, 5)]])
        self.assertEqual(poly_edit.element, element)
        self.assertEqual(poly_edit2.element, element)

    def test_point_draw_shared_datasource_callback(self):
        points = Points([1, 2, 3])
        table = Table(points.data, ['x', 'y'])
        layout = (points + table).opts(shared_datasource=True, clone=False)
        PointDraw(source=points)
        self.assertIs(points.data, table.data)
        plot = bokeh_renderer.get_plot(layout)
        point_plot = plot.subplots[(0, 0)].subplots['main']
        table_plot = plot.subplots[(0, 1)].subplots['main']
        self.assertIs(point_plot.handles['source'], table_plot.handles['source'])


class TestServerCallbacks(CallbackTestCase):

    def test_server_callback_resolve_attr_spec_range1d_start(self):
        range1d = Range1d(start=0, end=10)
        msg = Callback.resolve_attr_spec('x_range.attributes.start', range1d)
        self.assertEqual(msg, {'id': range1d.ref['id'], 'value': 0})

    def test_server_callback_resolve_attr_spec_range1d_end(self):
        range1d = Range1d(start=0, end=10)
        msg = Callback.resolve_attr_spec('x_range.attributes.end', range1d)
        self.assertEqual(msg, {'id': range1d.ref['id'], 'value': 10})

    def test_server_callback_resolve_attr_spec_source_selected(self):
        source = ColumnDataSource()
        source.selected.indices = [1, 2, 3]
        msg = Callback.resolve_attr_spec('cb_obj.selected.indices', source)
        self.assertEqual(msg, {'id': source.ref['id'], 'value': [1, 2, 3]})

    def test_server_callback_resolve_attr_spec_tap_event(self):
        plot = Plot()
        event = Tap(plot, x=42)
        msg = Callback.resolve_attr_spec('cb_obj.x', event, plot)
        self.assertEqual(msg, {'id': plot.ref['id'], 'value': 42})

    def test_selection1d_resolves(self):
        points = Points([1, 2, 3])
        Selection1D(source=points)
        plot = bokeh_server_renderer.get_plot(points)
        selected = Selection(indices=[0, 2])
        callback = plot.callbacks[0]
        spec = callback.attributes['index']
        resolved = callback.resolve_attr_spec(spec, selected, model=selected)
        self.assertEqual(resolved, {'id': selected.ref['id'], 'value': [0, 2]})

    def test_selection1d_resolves_table(self):
        table = Table([1, 2, 3], 'x')
        Selection1D(source=table)
        plot = bokeh_server_renderer.get_plot(table)
        selected = Selection(indices=[0, 2])
        callback = plot.callbacks[0]
        spec = callback.attributes['index']
        resolved = callback.resolve_attr_spec(spec, selected, model=selected)
        self.assertEqual(resolved, {'id': selected.ref['id'], 'value': [0, 2]})

    def test_plotsize_resolves(self):
        points = Points([1, 2, 3])
        PlotSize(source=points)
        plot = bokeh_server_renderer.get_plot(points)
        callback = plot.callbacks[0]
        model = namedtuple('Plot', 'inner_width inner_height ref')(400, 300, {'id': 'Test'})
        width_spec = callback.attributes['width']
        height_spec = callback.attributes['height']
        resolved = callback.resolve_attr_spec(width_spec, model, model=model)
        self.assertEqual(resolved, {'id': 'Test', 'value': 400})
        resolved = callback.resolve_attr_spec(height_spec, model, model=model)
        self.assertEqual(resolved, {'id': 'Test', 'value': 300})

    def test_cds_resolves(self):
        points = Points([1, 2, 3])
        CDSStream(source=points)
        plot = bokeh_server_renderer.get_plot(points)
        cds = plot.handles['cds']
        callback = plot.callbacks[0]
        data_spec = callback.attributes['data']
        resolved = callback.resolve_attr_spec(data_spec, cds, model=cds)
        self.assertEqual(resolved, {'id': cds.ref['id'],
                                    'value': points.columns()})

    def test_rangexy_datetime(self):
        curve = Curve(pd._testing.makeTimeDataFrame(), 'index', 'C')
        stream = RangeXY(source=curve)
        plot = bokeh_server_renderer.get_plot(curve)
        callback = plot.callbacks[0]
        callback.on_msg({"x0": curve.iloc[0, 0], 'x1': curve.iloc[3, 0],
                         "y0": 0.2, 'y1': 0.8})
        self.assertEqual(stream.x_range[0], curve.iloc[0, 0])
        self.assertEqual(stream.x_range[1], curve.iloc[3, 0])
        self.assertEqual(stream.y_range, (0.2, 0.8))

    def test_rangexy_framewise_reset(self):
        raise SkipTest('The fix for this was reverted, see #4396')
        stream = RangeXY(x_range=(0, 2), y_range=(0, 1))
        curve = DynamicMap(lambda z, x_range, y_range: Curve([1, 2, z]),
                           kdims=['z'], streams=[stream]).redim.range(z=(0, 3))
        plot = bokeh_server_renderer.get_plot(curve.opts(framewise=True))
        plot.update((1,))
        self.assertEqual(stream.y_range, None)

    def test_rangexy_framewise_not_reset_if_triggering(self):
        stream = RangeXY(x_range=(0, 2), y_range=(0, 1))
        curve = DynamicMap(lambda z, x_range, y_range: Curve([1, 2, z]),
                           kdims=['z'], streams=[stream]).redim.range(z=(0, 3))
        bokeh_server_renderer.get_plot(curve.opts(framewise=True
        ))
        stream.event(x_range=(0, 3))
        self.assertEqual(stream.x_range, (0, 3))


def test_msg_with_base64_array():
    # Account for issue seen in https://github.com/holoviz/geoviews/issues/584
    data_before = ["AAAAAAAAJEAAAAAAAAA0QAAAAAAAAD5AAAAAAAAAREA=", "float64", "little", [4]]
    msg_before = {"data": {"x": data_before}}
    msg_after = CDSCallback(None, None, None)._process_msg(msg_before)
    data_after = msg_after["data"]["x"]

    data_expected = np.array([10.0, 20.0, 30.0, 40.0])
    assert np.equal(data_expected, data_after).all()


def test_rangexy_multi_yaxes():
    c1 = Curve(np.arange(100).cumsum(), vdims='y')
    c2 = Curve(-np.arange(100).cumsum(), vdims='y2')
    RangeXY(source=c1)
    RangeXY(source=c2)

    overlay = (c1 * c2).opts(backend='bokeh', multi_y=True)
    plot = bokeh_server_renderer.get_plot(overlay)

    p1, p2 = plot.subplots.values()

    assert plot.state.y_range is p1.handles['y_range']
    assert 'y2' in plot.state.extra_y_ranges
    assert plot.state.extra_y_ranges['y2'] is p2.handles['y_range']

    # Ensure both callbacks are attached
    assert p1.callbacks[0].plot is p1
    assert p2.callbacks[0].plot is p2
