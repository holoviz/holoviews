import numpy as np

from holoviews.core import HoloMap, NdOverlay
from holoviews.element import Image, Curve

from .testplot import TestMPLPlot, mpl_renderer


class TestLayoutPlot(TestMPLPlot):

    def test_layout_instantiate_subplots(self):
        layout = (Curve(range(10)) + Curve(range(10)) + Image(np.random.rand(10,10)) +
                  Curve(range(10)) + Curve(range(10)))
        plot = mpl_renderer.get_plot(layout)
        positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]
        self.assertEqual(sorted(plot.subplots.keys()), positions)
        for i, pos in enumerate(positions):
            adjoint = plot.subplots[pos]
            if 'main' in adjoint.subplots:
                self.assertEqual(adjoint.subplots['main'].layout_num, i+1)

    def test_layout_empty_subplots(self):
        layout = Curve(range(10)) + NdOverlay() + HoloMap() + HoloMap({1: Image(np.random.rand(10,10))})
        plot = mpl_renderer.get_plot(layout)
        self.assertEqual(len(plot.subplots.values()), 2)

    def test_layout_instantiate_subplots_transposed(self):
        layout = (Curve(range(10)) + Curve(range(10)) + Image(np.random.rand(10,10)) +
                  Curve(range(10)) + Curve(range(10)))
        plot = mpl_renderer.get_plot(layout(plot=dict(transpose=True)))
        positions = [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0)]
        self.assertEqual(sorted(plot.subplots.keys()), positions)
        nums = [1, 5, 2, 3, 4]
        for pos, num in zip(positions, nums):
            adjoint = plot.subplots[pos]
            if 'main' in adjoint.subplots:
                self.assertEqual(adjoint.subplots['main'].layout_num, num)
