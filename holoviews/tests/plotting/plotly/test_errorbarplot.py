import numpy as np

from holoviews.element import ErrorBars
from holoviews.testing import assert_data_equal

from .test_plot import TestPlotlyPlot


class TestErrorBarsPlot(TestPlotlyPlot):

    def test_errorbars_plot(self):
        errorbars = ErrorBars([(0, 1, 0.5), (1, 2, 1), (2, 3, 2.25)], vdims=['y', 'y2'])
        state = self._get_plot_state(errorbars)
        assert_data_equal(state['data'][0]['x'], np.array([0, 1, 2]))
        assert_data_equal(state['data'][0]['y'], np.array([1, 2, 3]))
        assert_data_equal(state['data'][0]['error_y']['array'], np.array([0.5, 1, 2.25]))
        assert_data_equal(state['data'][0]['error_y']['arrayminus'], np.array([0.5, 1, 2.25]))
        assert state['data'][0]['mode'] == 'lines'
        assert state['layout']['yaxis']['range'] == [0.5, 5.25]

    def test_errorbars_plot_inverted(self):
        errorbars = ErrorBars([(0, 1, 0.5), (1, 2, 1), (2, 3, 2.25)], vdims=['y', 'y2']).opts(invert_axes=True)
        state = self._get_plot_state(errorbars)
        assert_data_equal(state['data'][0]['x'], np.array([1, 2, 3]))
        assert_data_equal(state['data'][0]['y'], np.array([0, 1, 2]))
        assert_data_equal(state['data'][0]['error_x']['array'], np.array([0.5, 1, 2.25]))
        assert_data_equal(state['data'][0]['error_x']['arrayminus'], np.array([0.5, 1, 2.25]))
        assert state['data'][0]['mode'] == 'lines'
        assert state['layout']['xaxis']['range'] == [0.5, 5.25]

    def test_visible(self):
        element = ErrorBars(
            [(0, 1, 0.5), (1, 2, 1), (2, 3, 2.25)],
            vdims=['y', 'y2']
        ).opts(visible=False)
        state = self._get_plot_state(element)
        assert state['data'][0]['visible'] is False
