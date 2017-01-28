"""
Tests of plot instantiation (not display tests, just instantiation)
"""
from __future__ import unicode_literals

import logging
import datetime as dt
from collections import deque
from unittest import SkipTest
from io import BytesIO, StringIO

import param
import numpy as np
from holoviews import (Dimension, Overlay, DynamicMap, Store,
                       NdOverlay, GridSpace, HoloMap, Layout)
from holoviews.element import (Curve, Scatter, Image, VLine, Points,
                               HeatMap, QuadMesh, Spikes, ErrorBars,
                               Scatter3D, Path, Polygons, Bars, BoxWhisker)
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PositionXY, PositionX
from holoviews.plotting import comms

# Standardize backend due to random inconsistencies
try:
    from matplotlib import pyplot
    pyplot.switch_backend('agg')
    from holoviews.plotting.mpl import OverlayPlot
    mpl_renderer = Store.renderers['matplotlib']
except:
    mpl_renderer = None

try:
    import holoviews.plotting.bokeh
    bokeh_renderer = Store.renderers['bokeh']
    from holoviews.plotting.bokeh.callbacks import Callback
    from bokeh.models import Div, ColumnDataSource
    from bokeh.models.mappers import LinearColorMapper, LogColorMapper
    from bokeh.models.tools import HoverTool
except:
    bokeh_renderer = None

try:
    import holoviews.plotting.plotly
    plotly_renderer = Store.renderers['plotly']
except:
    plotly_renderer = None


class ParamLogStream(object):
    """
    Context manager that replaces the param logger and captures
    log messages in a StringIO stream.
    """

    def __enter__(self):
        self.stream = StringIO()
        self._handler = logging.StreamHandler(self.stream)
        self._logger = logging.getLogger('testlogger')
        for handler in self._logger.handlers:
            self._logger.removeHandler(handler)
        self._logger.addHandler(self._handler)
        self._param_logger = param.parameterized.logger
        param.parameterized.logger = self._logger
        return self

    def __exit__(self, *args):
        param.parameterized.logger = self._param_logger
        self._handler.close()
        self.stream.seek(0)


class TestMPLPlotInstantiation(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        Store.current_backend = 'matplotlib'
        if mpl_renderer is None:
            raise SkipTest("Matplotlib required to test plot instantiation")
        self.default_comm = mpl_renderer.comms['default']
        mpl_renderer.comms['default'] = (comms.Comm, '')

    def teardown(self):
        mpl_renderer.comms['default'] = self.default_comm
        Store.current_backend = self.previous_backend

    def test_interleaved_overlay(self):
        """
        Test to avoid regression after fix of https://github.com/ioam/holoviews/issues/41
        """
        o = Overlay([Curve(np.array([[0, 1]])) , Scatter([[1,1]]) , Curve(np.array([[0, 1]]))])
        OverlayPlot(o)

    def test_dynamic_nonoverlap(self):
        kdims = [Dimension('File', range=(0.01, 1)),
                 Dimension('SliceDimension', range=(0.01, 1)),
                 Dimension('Coordinates', range=(0.01, 1))]
        dmap1 = DynamicMap(lambda x, y, z: Image(np.random.rand(10,10)), kdims=kdims)
        dmap2 = DynamicMap(lambda x: Curve(np.random.rand(10,2))*VLine(x),
                           kdims=kdims[:1])
        mpl_renderer.get_widget(dmap1 + dmap2, 'selection')


    def test_dynamic_streams_refresh(self):
        stream = PositionXY()
        dmap = DynamicMap(lambda x, y: Points([(x, y)]),
                             kdims=[], streams=[stream])
        plot = mpl_renderer.get_plot(dmap)
        plot.initialize_plot()
        pre = mpl_renderer(plot, fmt='png')
        stream.update(x=1, y=1)
        plot.refresh()
        post = mpl_renderer(plot, fmt='png')
        self.assertNotEqual(pre, post)

    def test_errorbar_test(self):
        errorbars = ErrorBars(([0,1],[1,2],[0.1,0.2]))
        plot = mpl_renderer.get_plot(errorbars)
        plot.initialize_plot()

    def test_stream_callback_single_call(self):
        def history_callback(x, history=deque(maxlen=10)):
            history.append(x)
            return Curve(list(history))
        stream = PositionX()
        dmap = DynamicMap(history_callback, kdims=[], streams=[stream])
        plot = mpl_renderer.get_plot(dmap)
        mpl_renderer(plot)
        for i in range(20):
            stream.update(x=i)
        x, y = plot.handles['artist'].get_data()
        self.assertEqual(x, np.arange(10))
        self.assertEqual(y, np.arange(10, 20))

    def test_points_non_numeric_size_warning(self):
        data = (np.arange(10), np.arange(10), list(map(chr, range(94,104))))
        points = Points(data, vdims=['z'])(plot=dict(size_index=2))
        with ParamLogStream() as log:
            plot = mpl_renderer.get_plot(points)
        log_msg = log.stream.read()
        warning = ('%s: z dimension is not numeric, '
                   'cannot use to scale Points size.\n' % plot.name)
        self.assertEqual(log_msg, warning)



class TestBokehPlotInstantiation(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        if not bokeh_renderer:
            raise SkipTest("Bokeh required to test plot instantiation")
        Store.current_backend = 'bokeh'
        Callback._comm_type = comms.Comm
        self.default_comm = bokeh_renderer.comms['default']
        bokeh_renderer.comms['default'] = (comms.Comm, '')

    def teardown(self):
        Store.current_backend = self.previous_backend
        Callback._comm_type = comms.JupyterCommJS
        mpl_renderer.comms['default'] = self.default_comm

    def test_batched_plot(self):
        overlay = NdOverlay({i: Points(np.arange(i)) for i in range(1, 100)})
        plot = bokeh_renderer.get_plot(overlay)
        extents = plot.get_extents(overlay, {})
        self.assertEqual(extents, (0, 0, 98, 98))

    def _test_hover_info(self, element, tooltips):
        plot = bokeh_renderer.get_plot(element)
        plot.initialize_plot()
        fig = plot.state
        hover = fig.select(dict(type=HoverTool))
        self.assertTrue(len(hover))
        self.assertEqual(hover[0].tooltips, tooltips)

    def test_curve_overlay_hover(self):
        obj = NdOverlay({i: Curve(np.random.rand(10,2)) for i in range(5)},
                        kdims=['Test'])
        opts = {'Curve': {'tools': ['hover']},
                'NdOverlay': {'legend_limit': 0}}
        obj = obj(plot=opts)
        self._test_hover_info(obj, [('Test', '@Test')])


    def test_points_overlay_hover(self):
        obj = NdOverlay({i: Points(np.random.rand(10,2)) for i in range(5)},
                        kdims=['Test'])
        opts = {'Points': {'tools': ['hover']},
                'NdOverlay': {'legend_limit': 0}}
        obj = obj(plot=opts)
        self._test_hover_info(obj, [('Test', '@Test'), ('x', '@x'),
                                    ('y', '@y')])

    def test_path_overlay_hover(self):
        obj = NdOverlay({i: Path([np.random.rand(10,2)]) for i in range(5)},
                        kdims=['Test'])
        opts = {'Path': {'tools': ['hover']},
                'NdOverlay': {'legend_limit': 0}}
        obj = obj(plot=opts)
        self._test_hover_info(obj, [('Test', '@Test')])

    def test_polygons_overlay_hover(self):
        obj = NdOverlay({i: Polygons([np.random.rand(10,2)], vdims=['z'], level=0)
                         for i in range(5)}, kdims=['Test'])
        opts = {'Polygons': {'tools': ['hover']},
                'NdOverlay': {'legend_limit': 0}}
        obj = obj(plot=opts)
        self._test_hover_info(obj, [('Test', '@Test'), ('z', '@z')])


    def _test_colormapping(self, element, dim, log=False):
        plot = bokeh_renderer.get_plot(element)
        plot.initialize_plot()
        fig = plot.state
        cmapper = plot.handles['color_mapper']
        low, high = element.range(dim)
        self.assertEqual(cmapper.low, low)
        self.assertEqual(cmapper.high, high)
        mapper_type = LogColorMapper if log else LinearColorMapper
        self.assertTrue(isinstance(cmapper, mapper_type))

    def test_points_colormapping(self):
        points = Points(np.random.rand(10, 4), vdims=['a', 'b'])
        self._test_colormapping(points, 3)

    def test_image_colormapping(self):
        img = Image(np.random.rand(10, 10))(plot=dict(logz=True))
        self._test_colormapping(img, 2, True)

    def test_heatmap_colormapping(self):
        hm = HeatMap([(1,1,1), (2,2,0)])
        self._test_colormapping(hm, 2)

    def test_quadmesh_colormapping(self):
        n = 21
        xs = np.logspace(1, 3, n)
        ys = np.linspace(1, 10, n)
        qmesh = QuadMesh((xs, ys, np.random.rand(n-1, n-1)))
        self._test_colormapping(qmesh, 2)

    def test_spikes_colormapping(self):
        spikes = Spikes(np.random.rand(20, 2), vdims=['Intensity'])
        self._test_colormapping(spikes, 1)


    def test_stream_callback(self):
        dmap = DynamicMap(lambda x, y: Points([(x, y)]), kdims=[], streams=[PositionXY()])
        plot = bokeh_renderer.get_plot(dmap)
        bokeh_renderer(plot)
        plot.callbacks[0].on_msg({"x": 10, "y": -10})
        data = plot.handles['source'].data
        self.assertEqual(data['x'], np.array([10]))
        self.assertEqual(data['y'], np.array([-10]))

    def test_stream_callback_with_ids(self):
        dmap = DynamicMap(lambda x, y: Points([(x, y)]), kdims=[], streams=[PositionXY()])
        plot = bokeh_renderer.get_plot(dmap)
        bokeh_renderer(plot)
        hover = plot.state.select(type=HoverTool)[0]
        plot.callbacks[0].on_msg({"x": {'id': hover.ref['id'], 'value': 10},
                                  "y": {'id': hover.ref['id'], 'value': -10}})
        data = plot.handles['source'].data
        self.assertEqual(data['x'], np.array([10]))
        self.assertEqual(data['y'], np.array([-10]))

    def test_stream_callback_single_call(self):
        def history_callback(x, history=deque(maxlen=10)):
            history.append(x)
            return Curve(list(history))
        stream = PositionX()
        dmap = DynamicMap(history_callback, kdims=[], streams=[stream])
        plot = bokeh_renderer.get_plot(dmap)
        bokeh_renderer(plot)
        for i in range(20):
            stream.update(x=i)
        data = plot.handles['source'].data
        self.assertEqual(data['x'], np.arange(10))
        self.assertEqual(data['y'], np.arange(10, 20))

    def test_bars_suppress_legend(self):
        bars = Bars([('A', 1), ('B', 2)])(plot=dict(show_legend=False))
        plot = bokeh_renderer.get_plot(bars)
        plot.initialize_plot()
        fig = plot.state
        assert len(fig.legend[0].items) == 0

    def test_image_boolean_array(self):
        img = Image(np.array([[True, False], [False, True]]))
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        source = plot.handles['source']
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 1)
        self.assertEqual(source.data['image'][0],
                         np.array([[0, 1], [1, 0]]))

    def test_layout_title(self):
        hmap1 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        hmap2 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        plot = bokeh_renderer.get_plot(hmap1+hmap2)
        title = plot.handles['title']
        self.assertIsInstance(title, Div)
        text = "<span style='font-size: 16pt'><b>Default: 0</b></font>"
        self.assertEqual(title.text, text)

    def test_layout_title_fontsize(self):
        hmap1 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        hmap2 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        layout = Layout([hmap1, hmap2])(plot=dict(fontsize={'title': '12pt'}))
        plot = bokeh_renderer.get_plot(layout)
        title = plot.handles['title']
        self.assertIsInstance(title, Div)
        text = "<span style='font-size: 12pt'><b>Default: 0</b></font>"
        self.assertEqual(title.text, text)

    def test_layout_title_show_title_false(self):
        hmap1 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        hmap2 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        layout = Layout([hmap1, hmap2])(plot=dict(show_title=False))
        plot = bokeh_renderer.get_plot(layout)
        self.assertTrue('title' not in plot.handles)

    def test_layout_title_update(self):
        hmap1 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        hmap2 = HoloMap({a: Image(np.random.rand(10,10)) for a in range(3)})
        plot = bokeh_renderer.get_plot(hmap1+hmap2)
        plot.update(1)
        title = plot.handles['title']
        self.assertIsInstance(title, Div)
        text = "<span style='font-size: 16pt'><b>Default: 1</b></font>"
        self.assertEqual(title.text, text)

    def test_grid_title(self):
        grid = GridSpace({(i, j): HoloMap({a: Image(np.random.rand(10,10))
                                           for a in range(3)}, kdims=['X'])
                          for i in range(2) for j in range(3)})
        plot = bokeh_renderer.get_plot(grid)
        title = plot.handles['title']
        self.assertIsInstance(title, Div)
        text = "<span style='font-size: 16pt'><b>X: 0</b></font>"
        self.assertEqual(title.text, text)

    def test_grid_title_update(self):
        grid = GridSpace({(i, j): HoloMap({a: Image(np.random.rand(10,10))
                                           for a in range(3)}, kdims=['X'])
                          for i in range(2) for j in range(3)})
        plot = bokeh_renderer.get_plot(grid)
        plot.update(1)
        title = plot.handles['title']
        self.assertIsInstance(title, Div)
        text = "<span style='font-size: 16pt'><b>X: 1</b></font>"
        self.assertEqual(title.text, text)

    def test_points_non_numeric_size_warning(self):
        data = (np.arange(10), np.arange(10), list(map(chr, range(94,104))))
        points = Points(data, vdims=['z'])(plot=dict(size_index=2))
        with ParamLogStream() as log:
            plot = bokeh_renderer.get_plot(points)
        log_msg = log.stream.read()
        warning = ('%s: z dimension is not numeric, '
                   'cannot use to scale Points size.\n' % plot.name)
        self.assertEqual(log_msg, warning)

    def test_box_whisker_datetime(self):
        times = np.arange(dt.datetime(2017,1,1), dt.datetime(2017,2,1),
                          dt.timedelta(days=1))
        box = BoxWhisker((times, np.random.rand(len(times))), kdims=['Date'])
        plot = bokeh_renderer.get_plot(box)
        formatted = [box.kdims[0].pprint_value(t).replace(':', ';') for t in times]
        self.assertTrue(all('Date' in cds.data for cds in
                            plot.state.select(ColumnDataSource)))
        self.assertTrue(cds.data['Date'][0] in formatted for cds in
                        plot.state.select(ColumnDataSource))


class TestPlotlyPlotInstantiation(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        Store.current_backend = 'plotly'
        self.default_comm = bokeh_renderer.comms['default']
        if not plotly_renderer:
            raise SkipTest("Plotly required to test plot instantiation")
        plotly_renderer.comms['default'] = (comms.Comm, '')


    def teardown(self):
        Store.current_backend = self.previous_backend
        plotly_renderer.comms['default'] = self.default_comm

    def _get_plot_state(self, element):
        plot = plotly_renderer.get_plot(element)
        plot.initialize_plot()
        return plot.state

    def test_curve_state(self):
        curve = Curve([1, 2, 3])
        state = self._get_plot_state(curve)
        self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3]))
        self.assertEqual(state['layout']['yaxis']['range'], [1, 3])

    def test_scatter3d_state(self):
        scatter = Scatter3D(([0,1], [2,3], [4,5]))
        state = self._get_plot_state(scatter)
        self.assertEqual(state['data'][0]['x'], np.array([0, 1]))
        self.assertEqual(state['data'][0]['y'], np.array([2, 3]))
        self.assertEqual(state['data'][0]['z'], np.array([4, 5]))
        self.assertEqual(state['layout']['scene']['xaxis']['range'], [0, 1])
        self.assertEqual(state['layout']['scene']['yaxis']['range'], [2, 3])
        self.assertEqual(state['layout']['scene']['zaxis']['range'], [4, 5])

    def test_overlay_state(self):
        layout = Curve([1, 2, 3]) * Curve([2, 4, 6])
        state = self._get_plot_state(layout)
        self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3]))
        self.assertEqual(state['data'][1]['y'], np.array([2, 4, 6]))
        self.assertEqual(state['layout']['yaxis']['range'], [1, 6])

    def test_layout_state(self):
        layout = Curve([1, 2, 3]) + Curve([2, 4, 6])
        state = self._get_plot_state(layout)
        self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3]))
        self.assertEqual(state['data'][0]['yaxis'], 'y1')
        self.assertEqual(state['data'][1]['y'], np.array([2, 4, 6]))
        self.assertEqual(state['data'][1]['yaxis'], 'y2')

    def test_grid_state(self):
        grid = GridSpace({(i, j): Curve([i, j]) for i in [0, 1]
                          for j in [0, 1]})
        state = self._get_plot_state(grid)
        self.assertEqual(state['data'][0]['y'], np.array([0, 0]))
        self.assertEqual(state['data'][0]['xaxis'], 'x1')
        self.assertEqual(state['data'][0]['yaxis'], 'y1')
        self.assertEqual(state['data'][1]['y'], np.array([1, 0]))
        self.assertEqual(state['data'][1]['xaxis'], 'x2')
        self.assertEqual(state['data'][1]['yaxis'], 'y1')
        self.assertEqual(state['data'][2]['y'], np.array([0, 1]))
        self.assertEqual(state['data'][2]['xaxis'], 'x1')
        self.assertEqual(state['data'][2]['yaxis'], 'y2')
        self.assertEqual(state['data'][3]['y'], np.array([1, 1]))
        self.assertEqual(state['data'][3]['xaxis'], 'x2')
        self.assertEqual(state['data'][3]['yaxis'], 'y2')

    def test_stream_callback_single_call(self):
        def history_callback(x, history=deque(maxlen=10)):
            history.append(x)
            return Curve(list(history))
        stream = PositionX()
        dmap = DynamicMap(history_callback, kdims=[], streams=[stream])
        plot = plotly_renderer.get_plot(dmap)
        plotly_renderer(plot)
        for i in range(20):
            stream.update(x=i)
        state = plot.state
        self.assertEqual(state['data'][0]['x'], np.arange(10))
        self.assertEqual(state['data'][0]['y'], np.arange(10, 20))
