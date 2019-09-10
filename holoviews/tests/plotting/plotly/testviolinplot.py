import numpy as np

from holoviews.element import Violin

from .testplot import TestPlotlyPlot


class TestViolinPlot(TestPlotlyPlot):

    def test_violin_single(self):
        violin = Violin([1, 1, 2, 3, 3, 4, 5, 5])
        state = self._get_plot_state(violin)
        self.assertEqual(len(state['data']), 1)
        self.assertEqual(state['data'][0]['type'], 'violin')
        self.assertEqual(state['data'][0]['name'], '')
        self.assertEqual(state['data'][0]['y'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
        self.assertEqual(state['layout'].get('xaxis', {}), {})
        self.assertEqual(state['layout']['yaxis']['range'], [1, 5])
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'y')

    def test_violin_single_invert_axes(self):
        violin = Violin([1, 1, 2, 3, 3, 4, 5, 5]).options(invert_axes=True)
        state = self._get_plot_state(violin)
        self.assertEqual(len(state['data']), 1)
        self.assertEqual(state['data'][0]['type'], 'violin')
        self.assertEqual(state['data'][0]['name'], '')
        self.assertEqual(state['data'][0]['x'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
        self.assertEqual(state['layout'].get('yaxis', {}), {})
        self.assertEqual(state['layout']['xaxis']['range'], [1, 5])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'y')

    def test_violin_multi(self):
        violin = Violin((['A']*8+['B']*8, [1, 1, 2, 3, 3, 4, 5, 5]*2), 'x', 'y')
        state = self._get_plot_state(violin)
        self.assertEqual(len(state['data']), 2)
        self.assertEqual(state['data'][0]['type'], 'violin')
        self.assertEqual(state['data'][0]['name'], 'A')
        self.assertEqual(state['data'][0]['y'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
        self.assertEqual(state['data'][1]['type'], 'violin')
        self.assertEqual(state['data'][1]['name'], 'B')
        self.assertEqual(state['data'][1]['y'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'x')
        self.assertEqual(state['layout']['yaxis']['range'], [1, 5])
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'y')

    def test_violin_multi_invert_axes(self):
        violin = Violin((['A']*8+['B']*8, [1, 1, 2, 3, 3, 4, 5, 5]*2), 'x', 'y').options(
            invert_axes=True)
        state = self._get_plot_state(violin)
        self.assertEqual(len(state['data']), 2)
        self.assertEqual(state['data'][0]['type'], 'violin')
        self.assertEqual(state['data'][0]['name'], 'A')
        self.assertEqual(state['data'][0]['x'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
        self.assertEqual(state['data'][1]['type'], 'violin')
        self.assertEqual(state['data'][1]['name'], 'B')
        self.assertEqual(state['data'][1]['x'], np.array([1, 1, 2, 3, 3, 4, 5, 5]))
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'x')
        self.assertEqual(state['layout']['xaxis']['range'], [1, 5])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'y')

    def test_visible(self):
        element = Violin([1, 1, 2, 3, 3, 4, 5, 5]).options(visible=False)
        state = self._get_plot_state(element)
        self.assertEqual(state['data'][0]['visible'], False)
