import numpy as np

from holoviews.element import Spread

from .testplot import TestPlotlyPlot


class TestSpreadPlot(TestPlotlyPlot):

    def test_spread_fill_between_ys(self):
        spread = Spread([(0, 1, 0.5), (1, 2, 1), (2, 3, 2.25)], vdims=['y', 'y2'])
        state = self._get_plot_state(spread)
        self.assertEqual(state['data'][0]['y'], np.array([0.5, 1, 0.75]))
        self.assertEqual(state['data'][0]['mode'], 'lines')
        self.assertEqual(state['data'][0]['fill'], None)
        self.assertEqual(state['data'][1]['y'], np.array([1.5, 3, 5.25]))
        self.assertEqual(state['data'][1]['mode'], 'lines')
        self.assertEqual(state['data'][1]['fill'], 'tonexty')
        self.assertEqual(state['layout']['yaxis']['range'], [0.5, 5.25])

    def test_spread_fill_between_xs(self):
        spread = Spread([(0, 1, 0.5), (1, 2, 1), (2, 3, 2.25)], vdims=['y', 'y2']).options(invert_axes=True)
        state = self._get_plot_state(spread)
        self.assertEqual(state['data'][0]['x'], np.array([0.5, 1, 0.75]))
        self.assertEqual(state['data'][0]['mode'], 'lines')
        self.assertEqual(state['data'][0]['fill'], None)
        self.assertEqual(state['data'][1]['x'], np.array([1.5, 3, 5.25]))
        self.assertEqual(state['data'][1]['mode'], 'lines')
        self.assertEqual(state['data'][1]['fill'], 'tonextx')
        self.assertEqual(state['layout']['xaxis']['range'], [0.5, 5.25])
        self.assertEqual(state['layout']['yaxis']['range'], [0, 2])
