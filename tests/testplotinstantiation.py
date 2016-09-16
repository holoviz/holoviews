"""
Tests of plot instantiation (not display tests, just instantiation)
"""

from unittest import SkipTest
from io import BytesIO

import numpy as np
from holoviews import (Dimension, Overlay, DynamicMap, Store, NdOverlay)
from holoviews.element import (Curve, Scatter, Image, VLine, Points,
                               HeatMap, QuadMesh, Spikes)
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PositionXY

# Standardize backend due to random inconsistencies
try:
    from matplotlib import pyplot
    pyplot.switch_backend('agg')
    from holoviews.plotting.mpl import OverlayPlot
    from holoviews.plotting.comms import Comm
    mpl_renderer = Store.renderers['matplotlib']
except:
    mpl_renderer = None

try:
    import holoviews.plotting.bokeh
    bokeh_renderer = Store.renderers['bokeh']
    from bokeh.models.mappers import LinearColorMapper, LogColorMapper
except:
    bokeh_renderer = None


class TestMPLPlotInstantiation(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        Store.current_backend = 'matplotlib'
        if mpl_renderer is None:
            raise SkipTest("Matplotlib required to test plot instantiation")
        self.default_comm, _ = mpl_renderer.comms['default']
        mpl_renderer.comms['default'] = (Comm, '')

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


class TestBokehPlotInstantiation(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        Store.current_backend = 'bokeh'

        if not bokeh_renderer:
            raise SkipTest("Bokeh required to test plot instantiation")

    def teardown(self):
        Store.current_backend = self.previous_backend

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
