import numpy as np

from holoviews.core import Overlay, NdOverlay
from holoviews.element import Curve, Scatter

from .testplot import TestMPLPlot, mpl_renderer

try:
    from holoviews.plotting.mpl import OverlayPlot
except:
    pass


class TestOverlayPlot(TestMPLPlot):

    def test_interleaved_overlay(self):
        """
        Test to avoid regression after fix of https://github.com/ioam/holoviews/issues/41
        """
        o = Overlay([Curve(np.array([[0, 1]])) , Scatter([[1,1]]) , Curve(np.array([[0, 1]]))])
        OverlayPlot(o)

    def test_overlay_empty_layers(self):
        overlay = Curve(range(10)) * NdOverlay()
        plot = mpl_renderer.get_plot(overlay)
        self.assertEqual(len(plot.subplots), 1)
