from holoviews.element import Table

from .test_plot import TestPlotlyPlot


class TestTablePlot(TestPlotlyPlot):

    def test_table_state(self):
        table = Table([(0, 1), (1, 2), (2, 3)], 'x', 'y')
        state = self._get_plot_state(table)
        assert state['data'][0]['type'] == 'table'
        assert state['data'][0]['header']['values'] == ['x', 'y']
        assert state['data'][0]['cells']['values'] == [['0', '1', '2'], ['1', '2', '3']]

    def test_visible(self):
        element = Table([(0, 1), (1, 2), (2, 3)], 'x', 'y').options(visible=False)
        state = self._get_plot_state(element)
        assert state['data'][0]['visible'] is False
