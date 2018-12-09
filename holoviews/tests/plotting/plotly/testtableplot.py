from holoviews.element import Table

from .testplot import TestPlotlyPlot


class TestTablePlot(TestPlotlyPlot):

    def test_table_state(self):
        table = Table([(0, 1), (1, 2), (2, 3)], 'x', 'y')
        state = self._get_plot_state(table)
        self.assertEqual(state['data'][0]['type'], 'table')
        self.assertEqual(state['data'][0]['header']['values'], ['x', 'y'])
        self.assertEqual(state['data'][0]['cells']['values'],
                         [['0', '1', '2'], ['1', '2', '3']])
