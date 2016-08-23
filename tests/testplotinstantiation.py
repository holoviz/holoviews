"""
Tests of plot instantiation (not display tests, just instantiation)
"""

from unittest import SkipTest
import numpy as np
from holoviews import (Dimension, Curve, Scatter, Overlay, DynamicMap,
                       Store, Image, VLine)
from holoviews.element.comparison import ComparisonTestCase

try:
    # Standardize backend due to random inconsistencies
    from matplotlib import pyplot
    pyplot.switch_backend('agg')
    from holoviews.plotting.mpl import OverlayPlot
    renderer = Store.renderers['matplotlib']
except:
    pyplot = None


class TestPlotInstantiation(ComparisonTestCase):

    def setUp(self):
        if pyplot is None:
            raise SkipTest("Matplotlib required to test plot instantiation")

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
        renderer.get_widget(dmap1 + dmap2, 'selection')
