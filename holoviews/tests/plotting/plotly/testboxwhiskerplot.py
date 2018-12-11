import numpy as np

from holoviews.element import BoxWhisker

from .testplot import TestPlotlyPlot


class TestBoxWhiskerPlot(TestPlotlyPlot):

    def test_boxwhisker_single(self):
        box = BoxWhisker([1, 1, 2, 3, 3, 4, 5, 5])
        state = self._get_plot_state(box)
        self.assertEqual(len(state['data']), 1)
        self.assertEqual(state['data'][0]['type'], 'box')
        self.assertEqual(state['data'][0]['name'], '')
        self.assertEqual(state['data'][0]['y'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
        self.assertEqual(state['layout']['xaxis'], {})
        self.assertEqual(state['layout']['yaxis']['range'], [1, 5])
        self.assertEqual(state['layout']['yaxis']['title'], 'y')

    def test_boxwhisker_single_invert_axes(self):
        box = BoxWhisker([1, 1, 2, 3, 3, 4, 5, 5]).options(invert_axes=True)
        state = self._get_plot_state(box)
        self.assertEqual(len(state['data']), 1)
        self.assertEqual(state['data'][0]['type'], 'box')
        self.assertEqual(state['data'][0]['name'], '')
        self.assertEqual(state['data'][0]['x'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
        self.assertEqual(state['layout']['yaxis'], {})
        self.assertEqual(state['layout']['xaxis']['range'], [1, 5])
        self.assertEqual(state['layout']['xaxis']['title'], 'y')

    def test_boxwhisker_multi(self):
        box = BoxWhisker((['A']*8+['B']*8, [1, 1, 2, 3, 3, 4, 5, 5]*2), 'x', 'y')
        state = self._get_plot_state(box)
        self.assertEqual(len(state['data']), 2)
        self.assertEqual(state['data'][0]['type'], 'box')
        self.assertEqual(state['data'][0]['name'], 'A')
        self.assertEqual(state['data'][0]['y'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
        self.assertEqual(state['data'][1]['type'], 'box')
        self.assertEqual(state['data'][1]['name'], 'B')
        self.assertEqual(state['data'][1]['y'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
        self.assertEqual(state['layout']['xaxis']['title'], 'x')
        self.assertEqual(state['layout']['yaxis']['range'], [1, 5])
        self.assertEqual(state['layout']['yaxis']['title'], 'y')

    def test_boxwhisker_multi_invert_axes(self):
        box = BoxWhisker((['A']*8+['B']*8, [1, 1, 2, 3, 3, 4, 5, 5]*2), 'x', 'y').options(
            invert_axes=True)
        state = self._get_plot_state(box)
        self.assertEqual(len(state['data']), 2)
        self.assertEqual(state['data'][0]['type'], 'box')
        self.assertEqual(state['data'][0]['name'], 'A')
        self.assertEqual(state['data'][0]['x'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
        self.assertEqual(state['data'][1]['type'], 'box')
        self.assertEqual(state['data'][1]['name'], 'B')
        self.assertEqual(state['data'][1]['x'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
        self.assertEqual(state['layout']['yaxis']['title'], 'x')
        self.assertEqual(state['layout']['xaxis']['range'], [1, 5])
        self.assertEqual(state['layout']['xaxis']['title'], 'y')
