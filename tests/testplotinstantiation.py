"""
Tests of plot instantiation (not display tests, just instantiation)
"""

from unittest import SkipTest
import numpy as np
from holoviews import Curve, Scatter, Overlay
from holoviews.element.comparison import ComparisonTestCase

try:
    # Standardize backend due to random inconsistencies
    from matplotlib import pyplot
    pyplot.switch_backend('agg')
    from holoviews.plotting import OverlayPlot
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

