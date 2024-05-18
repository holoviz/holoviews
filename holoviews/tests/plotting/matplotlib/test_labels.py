import numpy as np

from holoviews.core.dimension import Dimension
from holoviews.core.options import Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Labels
from holoviews.plotting.util import rgb2hex

from .test_plot import TestMPLPlot, mpl_renderer


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
        vdim = Dimension('text', value_format=lambda x: f'{x:.1f}')
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
        labels = Labels([(0, 1, 'A'), (1, 0, 'B')]).opts(invert_axes=True)
        plot = mpl_renderer.get_plot(labels)
        artist = plot.handles['artist']
        expected = {'x': np.array([0, 1]), 'y': np.array([1, 0]), 'Label': ['A', 'B']}
        for i, text in enumerate(artist):
            self.assertEqual(text._x, expected['y'][i])
            self.assertEqual(text._y, expected['x'][i])
            self.assertEqual(text.get_text(), expected['Label'][i])

    def test_labels_color_mapped(self):
        labels = Labels([(0, 1, 0.33333), (1, 0, 0.66666)]).opts(color_index=2)
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

    ###########################
    #    Styling mapping      #
    ###########################

    def test_label_color_op(self):
        labels = Labels([(0, 0, '#000000'), (0, 1, '#FF0000'), (0, 2, '#00FF00')],
                        vdims='color').opts(color='color')
        plot = mpl_renderer.get_plot(labels)
        artist = plot.handles['artist']
        self.assertEqual([a.get_color() for a in artist],
                         ['#000000', '#FF0000', '#00FF00'])

    def test_label_color_op_update(self):
        labels = HoloMap({
            0: Labels([(0, 0, '#000000'), (0, 1, '#FF0000'), (0, 2, '#00FF00')],
                      vdims='color'),
            1: Labels([(0, 0, '#FF0000'), (0, 1, '#00FF00'), (0, 2, '#0000FF')],
                      vdims='color')}).opts(color='color')
        plot = mpl_renderer.get_plot(labels)
        artist = plot.handles['artist']
        self.assertEqual([a.get_color() for a in artist],
                         ['#000000', '#FF0000', '#00FF00'])
        plot.update((1,))
        artist = plot.handles['artist']
        self.assertEqual([a.get_color() for a in artist],
                         ['#FF0000', '#00FF00', '#0000FF'])

    def test_label_linear_color_op(self):
        labels = Labels([(0, 0, 0), (0, 1, 1), (0, 2, 2)],
                        vdims='color').opts(color='color')
        plot = mpl_renderer.get_plot(labels)
        artist = plot.handles['artist']
        self.assertEqual([rgb2hex(a.get_color()) for a in artist],
                         ['#440154', '#20908c', '#fde724'])

    def test_label_categorical_color_op(self):
        labels = Labels([(0, 0, 'A'), (0, 1, 'B'), (0, 2, 'A')],
                        vdims='color').opts(color='color', cmap='tab10')
        plot = mpl_renderer.get_plot(labels)
        artist = plot.handles['artist']
        self.assertEqual([rgb2hex(a.get_color()) for a in artist],
                         ['#1f77b4', '#ff7f0e', '#1f77b4'])

    def test_label_size_op(self):
        labels = Labels([(0, 0, 8), (0, 1, 12), (0, 2, 6)],
                        vdims='size').opts(size='size')
        plot = mpl_renderer.get_plot(labels)
        artist = plot.handles['artist']
        self.assertEqual([a.get_fontsize() for a in artist], [8, 12, 6])

    def test_label_size_op_update(self):
        labels = HoloMap({
            0: Labels([(0, 0, 8), (0, 1, 6), (0, 2, 12)],
                      vdims='size'),
            1: Labels([(0, 0, 9), (0, 1, 4), (0, 2, 3)],
                      vdims='size')}).opts(size='size')
        plot = mpl_renderer.get_plot(labels)
        artist = plot.handles['artist']
        self.assertEqual([a.get_fontsize() for a in artist], [8, 6, 12])
        plot.update((1,))
        artist = plot.handles['artist']
        self.assertEqual([a.get_fontsize() for a in artist], [9, 4, 3])

    def test_label_alpha_op(self):
        labels = Labels([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)],
                        vdims='alpha').opts(alpha='alpha')
        plot = mpl_renderer.get_plot(labels)
        artist = plot.handles['artist']
        self.assertEqual([a.get_alpha() for a in artist],
                         [0, 0.2, 0.7])

    def test_label_alpha_op_update(self):
        labels = HoloMap({
            0: Labels([(0, 0, 0.3), (0, 1, 1), (0, 2, 0.6)],
                      vdims='alpha'),
            1: Labels([(0, 0, 0.6), (0, 1, 0.1), (0, 2, 1)],
                      vdims='alpha')}).opts(alpha='alpha')
        plot = mpl_renderer.get_plot(labels)
        artist = plot.handles['artist']
        self.assertEqual([a.get_alpha() for a in artist],
                         [0.3, 1, 0.6])
        plot.update((1,))
        artist = plot.handles['artist']
        self.assertEqual([a.get_alpha() for a in artist],
                         [0.6, 0.1, 1])

    def test_label_rotation_op(self):
        labels = Labels([(0, 0, 90), (0, 1, 180), (0, 2, 270)],
                        vdims='rotation').opts(rotation='rotation')
        plot = mpl_renderer.get_plot(labels)
        artist = plot.handles['artist']
        self.assertEqual([a.get_rotation() for a in artist],
                         [90, 180, 270])

    def test_label_rotation_op_update(self):
        labels = HoloMap({
            0: Labels([(0, 0, 45), (0, 1, 180), (0, 2, 90)],
                      vdims='rotation'),
            1: Labels([(0, 0, 30), (0, 1, 120), (0, 2, 60)],
                      vdims='rotation')}).opts(rotation='rotation')
        plot = mpl_renderer.get_plot(labels)
        artist = plot.handles['artist']
        self.assertEqual([a.get_rotation() for a in artist],
                         [45, 180, 90])
        plot.update((1,))
        artist = plot.handles['artist']
        self.assertEqual([a.get_rotation() for a in artist],
                         [30, 120, 60])

    def test_labels_text_color_cycle(self):
        hm = HoloMap(
            {i: Labels([
                (0, 0 + i, "Label 1"),
                (1, 1 + i, "Label 2")
            ]) for i in range(3)}
        ).overlay()
        assert isinstance(hm[0].opts["color"], Cycle)
