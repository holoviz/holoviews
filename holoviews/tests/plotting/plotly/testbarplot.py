import numpy as np

from holoviews.element import Bars

from .testplot import TestPlotlyPlot


class TestBarsPlot(TestPlotlyPlot):

    def test_bars_plot(self):
        bars = Bars([3, 2, 1])
        state = self._get_plot_state(bars)
        self.assertEqual(state['data'][0]['x'], ['0', '1', '2'])
        self.assertEqual(state['data'][0]['y'], [3, 2, 1])
        self.assertEqual(state['data'][0]['type'], 'bar')
        self.assertEqual(state['layout']['xaxis']['range'], [None, None])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'x')
        self.assertEqual(state['layout']['yaxis']['range'], [0, 3.2])
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'y')

    def test_bars_plot_inverted(self):
        bars = Bars([3, 2, 1]).options(invert_axes=True)
        state = self._get_plot_state(bars)
        self.assertEqual(state['data'][0]['y'], ['0', '1', '2'])
        self.assertEqual(state['data'][0]['x'], [3, 2, 1])
        self.assertEqual(state['data'][0]['type'], 'bar')
        self.assertEqual(state['layout']['xaxis']['range'], [0, 3.2])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'y')
        self.assertEqual(state['layout']['yaxis']['range'], [None, None])
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'x')

    def test_bars_grouped(self):
        bars = Bars([('A', 1, 1), ('B', 2, 2), ('C', 2, 3), ('C', 1, 4)],
                    kdims=['A', 'B'])
        state = self._get_plot_state(bars)
        self.assertEqual(state['data'][0]['x'], [['A', 'B', 'C', 'C'], ['1', '2', '2', '1']])
        self.assertEqual(state['data'][0]['y'], np.array([1, 2, 3, 4]))
        self.assertEqual(state['data'][0]['type'], 'bar')
        self.assertEqual(state['layout']['barmode'], 'group')
        self.assertEqual(state['layout']['xaxis']['range'], [None, None])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'A, B')
        self.assertEqual(state['layout']['yaxis']['range'], [0, 4.3])
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'y')

    def test_bars_grouped_inverted(self):
        bars = Bars([('A', 1, 1), ('B', 2, 2), ('C', 2, 3), ('C', 1, 4)],
                    kdims=['A', 'B']).options(invert_axes=True)
        state = self._get_plot_state(bars)
        self.assertEqual(state['data'][0]['y'], [['A', 'B', 'C', 'C'], ['1', '2', '2', '1']])
        self.assertEqual(state['data'][0]['x'], np.array([1, 2, 3, 4]))
        self.assertEqual(state['data'][0]['type'], 'bar')
        self.assertEqual(state['layout']['barmode'], 'group')
        self.assertEqual(state['layout']['yaxis']['range'], [None, None])
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'A, B')
        self.assertEqual(state['layout']['xaxis']['range'], [0, 4.3])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'y')

    def test_bars_stacked(self):
        bars = Bars([('A', 1, 1), ('B', 2, 2), ('C', 2, 3), ('C', 1, 4)],
                    kdims=['A', 'B']).options(stacked=True)
        state = self._get_plot_state(bars)
        self.assertEqual(state['data'][0]['x'], ['A', 'B', 'C'])
        self.assertEqual(state['data'][0]['y'], [0, 2, 3])
        self.assertEqual(state['data'][0]['type'], 'bar')
        self.assertEqual(state['data'][1]['x'], ['A', 'B', 'C'])
        self.assertEqual(state['data'][1]['y'], [1, 0, 4])
        self.assertEqual(state['data'][1]['type'], 'bar')
        self.assertEqual(state['layout']['barmode'], 'stack')
        self.assertEqual(state['layout']['xaxis']['range'], [None, None])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'A')
        self.assertEqual(state['layout']['yaxis']['range'], [0, 7.6])
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'y')

    def test_bars_stacked_inverted(self):
        bars = Bars([('A', 1, 1), ('B', 2, 2), ('C', 2, 3), ('C', 1, 4)],
                    kdims=['A', 'B']).options(stacked=True, invert_axes=True)
        state = self._get_plot_state(bars)
        self.assertEqual(state['data'][0]['y'], ['A', 'B', 'C'])
        self.assertEqual(state['data'][0]['x'], [0, 2, 3])
        self.assertEqual(state['data'][0]['type'], 'bar')
        self.assertEqual(state['data'][1]['y'], ['A', 'B', 'C'])
        self.assertEqual(state['data'][1]['x'], [1, 0, 4])
        self.assertEqual(state['data'][1]['type'], 'bar')
        self.assertEqual(state['layout']['barmode'], 'stack')
        self.assertEqual(state['layout']['yaxis']['range'], [None, None])
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'A')
        self.assertEqual(state['layout']['xaxis']['range'], [0, 7.6])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'y')

    def test_visible(self):
        element = Bars([3, 2, 1]).options(visible=False)
        state = self._get_plot_state(element)
        self.assertEqual(state['data'][0]['visible'], False)
