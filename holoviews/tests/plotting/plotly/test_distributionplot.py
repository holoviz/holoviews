from holoviews.element import Distribution

from .test_plot import TestPlotlyPlot


class TestDistributionPlot(TestPlotlyPlot):

    def test_distribution_filled(self):
        dist = Distribution([1, 1.1,  2.1, 3, 2, 1, 2.2])
        state = self._get_plot_state(dist)
        self.assertEqual(state['data'][0]['type'], 'scatter')
        self.assertEqual(state['data'][0]['mode'], 'lines')
        self.assertEqual(state['data'][0]['fill'], 'tozeroy')

    def test_distribution_not_filled(self):
        dist = Distribution([1, 1.1,  2.1, 3, 2, 1, 2.2]).opts(filled=False)
        state = self._get_plot_state(dist)
        self.assertEqual(state['data'][0]['type'], 'scatter')
        self.assertEqual(state['data'][0]['mode'], 'lines')
        self.assertEqual(state['data'][0].get('fill'), None)

    def test_visible(self):
        element = Distribution([1, 1.1,  2.1, 3, 2, 1, 2.2]).opts(visible=False)
        state = self._get_plot_state(element)
        self.assertEqual(state['data'][0]['visible'], False)
