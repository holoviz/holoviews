import numpy as np

from holoviews.element import Spread
from holoviews.testing import assert_data_equal

from .test_plot import TestPlotlyPlot


class TestSpreadPlot(TestPlotlyPlot):

    def test_spread_fill_between_ys(self):
        spread = Spread([(0, 1, 0.5), (1, 2, 1), (2, 3, 2.25)], vdims=['y', 'y2'])
        state = self._get_plot_state(spread)
        assert_data_equal(state['data'][0]['y'], np.array([0.5, 1, 0.75]))
        assert state['data'][0]['mode'] == 'lines'
        assert state['data'][0].get('fill', None) is None
        assert_data_equal(state['data'][1]['y'], np.array([1.5, 3, 5.25]))
        assert state['data'][1]['mode'] == 'lines'
        assert state['data'][1]['fill'] == 'tonexty'
        assert state['layout']['yaxis']['range'] == [0.5, 5.25]

    def test_spread_fill_between_xs(self):
        spread = Spread([(0, 1, 0.5), (1, 2, 1), (2, 3, 2.25)], vdims=['y', 'y2']).options(invert_axes=True)
        state = self._get_plot_state(spread)
        assert_data_equal(state['data'][0]['x'], np.array([0.5, 1, 0.75]))
        assert state['data'][0]['mode'] == 'lines'
        assert state['data'][0].get('fill', None) is None
        assert_data_equal(state['data'][1]['x'], np.array([1.5, 3, 5.25]))
        assert state['data'][1]['mode'] == 'lines'
        assert state['data'][1]['fill'] == 'tonextx'
        assert state['layout']['xaxis']['range'] == [0.5, 5.25]
        assert state['layout']['yaxis']['range'] == [0, 2]

    def test_visible(self):
        element = Spread(
            [(0, 1, 0.5), (1, 2, 1), (2, 3, 2.25)],
            vdims=['y', 'y2']
        ).opts(visible=False)
        state = self._get_plot_state(element)
        assert state['data'][0]['visible'] is False
