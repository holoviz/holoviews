import numpy as np

from holoviews.element import Bars

from .testplot import TestPlotlyPlot


class TestBarsPlot(TestPlotlyPlot):

    def test_bars_plot(self):
        bars = Bars([3, 2, 1])
        state = self._get_plot_state(bars)
        self.assertEqual(state['data'][0]['x'], ['0', '1', '2'])
        self.assertEqual(state['data'][0]['y'], np.array([3, 2, 1]))
        self.assertEqual(state['data'][0]['type'], 'bar')
        self.assertEqual(state['layout']['xaxis']['range'], [-0.5, 2.5])
        self.assertEqual(state['layout']['xaxis']['title'], 'x')
        self.assertEqual(state['layout']['yaxis']['range'], [0, 3])
        self.assertEqual(state['layout']['yaxis']['title'], 'y')

    def test_bars_plot_inverted(self):
        bars = Bars([3, 2, 1]).options(invert_axes=True)
        state = self._get_plot_state(bars)
        self.assertEqual(state['data'][0]['y'], ['0', '1', '2'])
        self.assertEqual(state['data'][0]['x'], np.array([3, 2, 1]))
        self.assertEqual(state['data'][0]['type'], 'bar')
        self.assertEqual(state['layout']['xaxis']['range'], [0, 3])
        self.assertEqual(state['layout']['xaxis']['title'], 'y')
        self.assertEqual(state['layout']['yaxis']['range'], [-0.5, 2.5])
        self.assertEqual(state['layout']['yaxis']['title'], 'x')

    def test_bars_grouped(self):
        bars = Bars([('A', 1, 1), ('B', 2, 2), ('C', 2, 3), ('C', 1, 4)],
                    kdims=['A', 'B'])
        state = self._get_plot_state(bars)
        self.assertEqual(state['data'][0]['x'], ['A', 'C'])
        self.assertEqual(state['data'][0]['y'], np.array([1, 4]))
        self.assertEqual(state['data'][0]['type'], 'bar')
        self.assertEqual(state['data'][1]['x'], ['B', 'C'])
        self.assertEqual(state['data'][1]['y'], np.array([2, 3]))
        self.assertEqual(state['data'][1]['type'], 'bar')
        self.assertEqual(state['layout']['barmode'], 'group')
        self.assertEqual(state['layout']['xaxis']['range'], [-0.5, 2.5])
        self.assertEqual(state['layout']['xaxis']['title'], 'A, B')
        self.assertEqual(state['layout']['yaxis']['range'], [0, 4])
        self.assertEqual(state['layout']['yaxis']['title'], 'y')

    def test_bars_grouped_inverted(self):
        bars = Bars([('A', 1, 1), ('B', 2, 2), ('C', 2, 3), ('C', 1, 4)],
                    kdims=['A', 'B']).options(invert_axes=True)
        state = self._get_plot_state(bars)
        self.assertEqual(state['data'][0]['y'], ['A', 'C'])
        self.assertEqual(state['data'][0]['x'], np.array([1, 4]))
        self.assertEqual(state['data'][0]['type'], 'bar')
        self.assertEqual(state['data'][1]['y'], ['B', 'C'])
        self.assertEqual(state['data'][1]['x'], np.array([2, 3]))
        self.assertEqual(state['data'][1]['type'], 'bar')
        self.assertEqual(state['layout']['barmode'], 'group')
        self.assertEqual(state['layout']['yaxis']['range'], [-0.5, 2.5])
        self.assertEqual(state['layout']['yaxis']['title'], 'A, B')
        self.assertEqual(state['layout']['xaxis']['range'], [0, 4])
        self.assertEqual(state['layout']['xaxis']['title'], 'y')
        
    def test_bars_stacked(self):
        bars = Bars([('A', 1, 1), ('B', 2, 2), ('C', 2, 3), ('C', 1, 4)],
                    kdims=['A', 'B']).options(stacked=True)
        state = self._get_plot_state(bars)
        self.assertEqual(state['data'][0]['x'], ['A', 'C'])
        self.assertEqual(state['data'][0]['y'], np.array([1, 4]))
        self.assertEqual(state['data'][0]['type'], 'bar')
        self.assertEqual(state['data'][1]['x'], ['B', 'C'])
        self.assertEqual(state['data'][1]['y'], np.array([2, 3]))
        self.assertEqual(state['data'][1]['type'], 'bar')
        self.assertEqual(state['layout']['barmode'], 'stack')
        self.assertEqual(state['layout']['xaxis']['range'], [-0.5, 2.5])
        self.assertEqual(state['layout']['xaxis']['title'], 'A')
        self.assertEqual(state['layout']['yaxis']['range'], [0, 7])
        self.assertEqual(state['layout']['yaxis']['title'], 'y')

    def test_bars_stacked_inverted(self):
        bars = Bars([('A', 1, 1), ('B', 2, 2), ('C', 2, 3), ('C', 1, 4)],
                    kdims=['A', 'B']).options(stacked=True, invert_axes=True)
        state = self._get_plot_state(bars)
        self.assertEqual(state['data'][0]['y'], ['A', 'C'])
        self.assertEqual(state['data'][0]['x'], np.array([1, 4]))
        self.assertEqual(state['data'][0]['type'], 'bar')
        self.assertEqual(state['data'][1]['y'], ['B', 'C'])
        self.assertEqual(state['data'][1]['x'], np.array([2, 3]))
        self.assertEqual(state['data'][1]['type'], 'bar')
        self.assertEqual(state['layout']['barmode'], 'stack')
        self.assertEqual(state['layout']['yaxis']['range'], [-0.5, 2.5])
        self.assertEqual(state['layout']['yaxis']['title'], 'A')
        self.assertEqual(state['layout']['xaxis']['range'], [0, 7])
        self.assertEqual(state['layout']['xaxis']['title'], 'y')
