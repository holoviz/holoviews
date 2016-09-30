"""
Tests of plot instantiation (not display tests, just instantiation)
"""

from unittest import SkipTest
from io import BytesIO

import numpy as np
from holoviews import (Dimension, Overlay, DynamicMap, Store,
                       NdOverlay, GridSpace)
from holoviews.element import (Curve, Scatter, Image, VLine, Points,
                               HeatMap, QuadMesh, Spikes, ErrorBars,
                               Scatter3D)
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PositionXY
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
    from bokeh.models.mappers import LinearColorMapper, LogColorMapper
except:
    bokeh_renderer = None

try:
    import holoviews.plotting.plotly
    plotly_renderer = Store.renderers['plotly']
except:
    plotly_renderer = None


class TestMPLPlotInstantiation(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        Store.current_backend = 'matplotlib'
        if mpl_renderer is None:
            raise SkipTest("Matplotlib required to test plot instantiation")
        self.default_comm, _ = mpl_renderer.comms['default']
        mpl_renderer.comms['default'] = (comms.Comm, '')

    def teardown(self):
        mpl_renderer.comms['default'] = (self.default_comm, '')
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



class TestBokehPlotInstantiation(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        if not bokeh_renderer:
            raise SkipTest("Bokeh required to test plot instantiation")
        Store.current_backend = 'bokeh'
        Callback._comm_type = comms.Comm
        self.default_comm, _ = bokeh_renderer.comms['default']
        bokeh_renderer.comms['default'] = (comms.Comm, '')

    def teardown(self):
        Store.current_backend = self.previous_backend
        Callback._comm_type = comms.JupyterCommJS
        mpl_renderer.comms['default'] = (self.default_comm, '')

    def test_batched_plot(self):
        overlay = NdOverlay({i: Points(np.arange(i)) for i in range(1, 100)})
        plot = bokeh_renderer.get_plot(overlay)
        extents = plot.get_extents(overlay, {})
        self.assertEqual(extents, (0, 0, 98, 98))

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
        print plot.document
        plot.callbacks[0].on_msg('{"x": 10, "y": -10}')
        data = plot.handles['source'].data
        self.assertEqual(data['x'], np.array([10]))
        self.assertEqual(data['y'], np.array([-10]))


class TestPlotlyPlotInstantiation(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        Store.current_backend = 'plotly'
        if not plotly_renderer:
            raise SkipTest("Plotly required to test plot instantiation")

    def teardown(self):
        Store.current_backend = self.previous_backend

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
