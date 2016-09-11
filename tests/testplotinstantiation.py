"""
Tests of plot instantiation (not display tests, just instantiation)
"""

from unittest import SkipTest
import numpy as np
from holoviews import (Dimension, Curve, Scatter, Overlay, DynamicMap,
                       Store, Image, VLine, NdOverlay, Points)
from holoviews.element.comparison import ComparisonTestCase

# Standardize backend due to random inconsistencies
try:
    from matplotlib import pyplot
    pyplot.switch_backend('agg')
    from holoviews.plotting.mpl import OverlayPlot
    from holoviews.plotting.comms import JupyterPushComm
    mpl_renderer = Store.renderers['matplotlib']
except:
    mpl_renderer = None

try:
    import holoviews.plotting.bokeh
    bokeh_renderer = Store.renderers['bokeh']
except:
    bokeh_renderer = None


class TestMPLPlotInstantiation(ComparisonTestCase):

    def setUp(self):
        if mpl_renderer is None:
            raise SkipTest("Matplotlib required to test plot instantiation")
        self.default_comm, _ = mpl_renderer.comms['default']
        mpl_renderer.comms['default'] = (JupyterPushComm, '')

    def teardown(self):
        mpl_renderer.comms['default'] = (self.default_comm, '')

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



class TestBokehPlotInstantiation(ComparisonTestCase):

    def setUp(self):
        if not bokeh_renderer:
            raise SkipTest("Bokeh required to test plot instantiation")

    def test_batched_plot(self):
        overlay = NdOverlay({i: Points(np.arange(i)) for i in range(1, 100)})
        plot = bokeh_renderer.get_plot(overlay)
        extents = plot.get_extents(overlay, {})
        self.assertEqual(extents, (0, 0, 98, 98))
