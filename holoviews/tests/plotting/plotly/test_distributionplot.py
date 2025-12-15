
from holoviews.element import Distribution

from ...utils import optional_dependencies
from .test_plot import TestPlotlyPlot

_, scipy_skip = optional_dependencies("scipy")


@scipy_skip
class TestDistributionPlot(TestPlotlyPlot):

    def test_distribution_filled(self):
        dist = Distribution([1, 1.1,  2.1, 3, 2, 1, 2.2])
        state = self._get_plot_state(dist)
        assert state['data'][0]['type'] == 'scatter'
        assert state['data'][0]['mode'] == 'lines'
        assert state['data'][0]['fill'] == 'tozeroy'

    def test_distribution_not_filled(self):
        dist = Distribution([1, 1.1,  2.1, 3, 2, 1, 2.2]).opts(filled=False)
        state = self._get_plot_state(dist)
        assert state['data'][0]['type'] == 'scatter'
        assert state['data'][0]['mode'] == 'lines'
        assert state['data'][0].get('fill') is None

    def test_visible(self):
        element = Distribution([1, 1.1,  2.1, 3, 2, 1, 2.2]).opts(visible=False)
        state = self._get_plot_state(element)
        assert state['data'][0]['visible'] is False
