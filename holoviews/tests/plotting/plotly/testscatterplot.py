import numpy as np

from holoviews.element import Scatter

from .testplot import TestPlotlyPlot


class TestScatterPlot(TestPlotlyPlot):

    def test_scatter_state(self):
        scatter = Scatter([3, 2, 1])
        state = self._get_plot_state(scatter)
        self.assertEqual(state['data'][0]['y'], np.array([3, 2, 1]))
        self.assertEqual(state['data'][0]['mode'], 'markers')
        self.assertEqual(state['layout']['yaxis']['range'], [1, 3])
