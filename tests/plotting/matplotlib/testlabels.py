import numpy as np

from holoviews.core.dimension import Dimension
from holoviews.element import Labels

from .testplot import TestMPLPlot, mpl_renderer


class TestLabelsPlot(TestMPLPlot):

    def test_labels_simple(self):
        labels = Labels([(0, 1, 'A'), (1, 0, 'B')])
        plot = mpl_renderer.get_plot(labels)
        artist = plot.handles['artist']
        expected = {'x': np.array([0, 1]), 'y': np.array([1, 0]),
                    'Label': ['A', 'B']}
        for i, text in enumerate(artist):
            self.assertEqual(text._x, expected['x'][i])
            self.assertEqual(text._y, expected['y'][i])
            self.assertEqual(text.get_text(), expected['Label'][i])

    def test_labels_empty(self):
        labels = Labels([])
        plot = mpl_renderer.get_plot(labels)
        artist = plot.handles['artist']
        self.assertEqual(artist, [])

    def test_labels_formatter(self):
        vdim = Dimension('text', value_format=lambda x: '%.1f' % x)
        labels = Labels([(0, 1, 0.33333), (1, 0, 0.66666)], vdims=vdim)
        plot = mpl_renderer.get_plot(labels)
        artist = plot.handles['artist']
        expected = {'x': np.array([0, 1]), 'y': np.array([1, 0]),
                    'text': ['0.3', '0.7']}
        for i, text in enumerate(artist):
            self.assertEqual(text._x, expected['x'][i])
            self.assertEqual(text._y, expected['y'][i])
            self.assertEqual(text.get_text(), expected['text'][i])

    def test_labels_inverted(self):
        labels = Labels([(0, 1, 'A'), (1, 0, 'B')]).options(invert_axes=True)
        plot = mpl_renderer.get_plot(labels)
        artist = plot.handles['artist']
        expected = {'x': np.array([0, 1]), 'y': np.array([1, 0]), 'Label': ['A', 'B']}
        for i, text in enumerate(artist):
            self.assertEqual(text._x, expected['y'][i])
            self.assertEqual(text._y, expected['x'][i])
            self.assertEqual(text.get_text(), expected['Label'][i])

    def test_labels_color_mapped(self):
        labels = Labels([(0, 1, 0.33333), (1, 0, 0.66666)]).options(color_index=2)
        plot = mpl_renderer.get_plot(labels)
        artist = plot.handles['artist']
        expected = {'x': np.array([0, 1]), 'y': np.array([1, 0]),
                    'Label': ['0.33333', '0.66666']}
        colors = [(0.26666666666666666, 0.0039215686274509803, 0.32941176470588235, 1.0),
                  (0.99215686274509807, 0.90588235294117647, 0.14117647058823529, 1.0)]
        for i, text in enumerate(artist):
            self.assertEqual(text._x, expected['x'][i])
            self.assertEqual(text._y, expected['y'][i])
            self.assertEqual(text.get_text(), expected['Label'][i])
            self.assertEqual(text.get_color(), colors[i])
