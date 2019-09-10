import numpy as np

from holoviews.element import Labels

from .testplot import TestPlotlyPlot


class TestLabelsPlot(TestPlotlyPlot):

    def test_labels_state(self):
        labels = Labels([(0, 3, 0), (1, 2, 1), (2, 1, 1)])
        state = self._get_plot_state(labels)
        self.assertEqual(state['data'][0]['x'], np.array([0, 1, 2]))
        self.assertEqual(state['data'][0]['y'], np.array([3, 2, 1]))
        self.assertEqual(state['data'][0]['text'], ['0', '1', '1'])
        self.assertEqual(state['data'][0]['mode'], 'text')
        self.assertEqual(state['layout']['xaxis']['range'], [0, 2])
        self.assertEqual(state['layout']['yaxis']['range'], [1, 3])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'x')
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'y')

    def test_labels_inverted(self):
        labels = Labels([(0, 3, 0), (1, 2, 1), (2, 1, 1)]).options(invert_axes=True)
        state = self._get_plot_state(labels)
        self.assertEqual(state['data'][0]['x'], np.array([3, 2, 1]))
        self.assertEqual(state['data'][0]['y'], np.array([0, 1, 2]))
        self.assertEqual(state['data'][0]['text'], ['0', '1', '1'])
        self.assertEqual(state['data'][0]['mode'], 'text')
        self.assertEqual(state['layout']['xaxis']['range'], [1, 3])
        self.assertEqual(state['layout']['yaxis']['range'], [0, 2])
        self.assertEqual(state['layout']['xaxis']['title']['text'], 'y')
        self.assertEqual(state['layout']['yaxis']['title']['text'], 'x')

    def test_labels_size(self):
        labels = Labels([(0, 3, 0), (0, 2, 1), (0, 1, 1)]).options(size='y')
        state = self._get_plot_state(labels)
        self.assertEqual(state['data'][0]['textfont']['size'], np.array([3, 2, 1]))

    def test_labels_xoffset(self):
        labels = Labels([(0, 3, 0), (1, 2, 1), (2, 1, 1)]).options(xoffset=0.5)
        state = self._get_plot_state(labels)
        self.assertEqual(state['data'][0]['x'], np.array([0.5, 1.5, 2.5]))

    def test_labels_yoffset(self):
        labels = Labels([(0, 3, 0), (1, 2, 1), (2, 1, 1)]).options(yoffset=0.5)
        state = self._get_plot_state(labels)
        self.assertEqual(state['data'][0]['y'], np.array([3.5, 2.5, 1.5]))

    def test_visible(self):
        element = Labels([(0, 3, 0), (1, 2, 1), (2, 1, 1)]).options(visible=False)
        state = self._get_plot_state(element)
        self.assertEqual(state['data'][0]['visible'], False)
