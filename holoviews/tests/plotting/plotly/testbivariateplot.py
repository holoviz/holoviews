import numpy as np

from holoviews.element import Bivariate

from .testplot import TestPlotlyPlot


class TestBivariatePlot(TestPlotlyPlot):

    def test_bivariate_state(self):
        bivariate = Bivariate(([3, 2, 1], [0, 1, 2]))
        state = self._get_plot_state(bivariate)
        self.assertEqual(state['data'][0]['type'], 'histogram2dcontour')
        self.assertEqual(state['data'][0]['x'], np.array([3, 2, 1]))
        self.assertEqual(state['data'][0]['y'], np.array([0, 1, 2]))
        self.assertEqual(state['layout']['xaxis']['range'], [1, 3])
        self.assertEqual(state['layout']['yaxis']['range'], [0, 2])
        self.assertEqual(state['data'][0]['contours']['coloring'], 'lines')

    def test_bivariate_filled(self):
        bivariate = Bivariate(([3, 2, 1], [0, 1, 2])).options(
            filled=True)
        state = self._get_plot_state(bivariate)
        self.assertEqual(state['data'][0]['contours']['coloring'], 'fill')
    
    def test_bivariate_ncontours(self):
        bivariate = Bivariate(([3, 2, 1], [0, 1, 2])).options(ncontours=5)
        state = self._get_plot_state(bivariate)
        self.assertEqual(state['data'][0]['ncontours'], 5)
        self.assertEqual(state['data'][0]['autocontour'], False)
