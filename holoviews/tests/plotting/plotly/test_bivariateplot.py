import numpy as np

from holoviews.element import Bivariate
from holoviews.testing import assert_data_equal

from .test_plot import TestPlotlyPlot


class TestBivariatePlot(TestPlotlyPlot):

    def test_bivariate_state(self):
        bivariate = Bivariate(([3, 2, 1], [0, 1, 2]))
        state = self._get_plot_state(bivariate)
        assert state['data'][0]['type'] == 'histogram2dcontour'
        assert_data_equal(state['data'][0]['x'], np.array([3, 2, 1]))
        assert_data_equal(state['data'][0]['y'], np.array([0, 1, 2]))
        assert state['layout']['xaxis']['range'] == [1, 3]
        assert state['layout']['yaxis']['range'] == [0, 2]
        assert state['data'][0]['contours']['coloring'] == 'lines'

    def test_bivariate_filled(self):
        bivariate = Bivariate(([3, 2, 1], [0, 1, 2])).opts(
            filled=True)
        state = self._get_plot_state(bivariate)
        assert state['data'][0]['contours']['coloring'] == 'fill'

    def test_bivariate_ncontours(self):
        bivariate = Bivariate(([3, 2, 1], [0, 1, 2])).opts(ncontours=5)
        state = self._get_plot_state(bivariate)
        assert state['data'][0]['ncontours'] == 5
        assert state['data'][0]['autocontour'] is False

    def test_bivariate_colorbar(self):
        bivariate = Bivariate(([3, 2, 1], [0, 1, 2]))\

        bivariate.opts(colorbar=True)
        state = self._get_plot_state(bivariate)
        trace = state['data'][0]
        assert trace['showscale']

        bivariate.opts(colorbar=False)
        state = self._get_plot_state(bivariate)
        trace = state['data'][0]
        assert not trace['showscale']

    def test_visible(self):
        element = Bivariate(([3, 2, 1], [0, 1, 2])).opts(visible=False)
        state = self._get_plot_state(element)
        assert state['data'][0]['visible'] is False
