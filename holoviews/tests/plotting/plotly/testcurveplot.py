import numpy as np

from holoviews.element import Curve

from .testplot import TestPlotlyPlot


class TestCurvePlot(TestPlotlyPlot):

    def test_curve_state(self):
        curve = Curve([1, 2, 3])
        state = self._get_plot_state(curve)
        self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3]))
        self.assertEqual(state['data'][0]['mode'], 'lines')
        self.assertEqual(state['layout']['yaxis']['range'], [1, 3])
