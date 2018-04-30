import numpy as np

from holoviews.core.dimension import Dimension
from holoviews.element import Labels

from .testplot import TestBokehPlot, bokeh_renderer


class TestLabelsPlot(TestBokehPlot):

    def test_labels_simple(self):
        labels = Labels([(0, 1, 'A'), (1, 0, 'B')])
        plot = bokeh_renderer.get_plot(labels)
        source = plot.handles['source']
        glyph = plot.handles['glyph']
        expected = {'x': np.array([0, 1]), 'y': np.array([1, 0]),
                    'Label': ['A', 'B']}
        for k, vals in expected.items():
            self.assertEqual(source.data[k], vals)
        self.assertEqual(glyph.x, 'x')
        self.assertEqual(glyph.y, 'y')
        self.assertEqual(glyph.text, 'Label')

    def test_labels_empty(self):
        labels = Labels([])
        plot = bokeh_renderer.get_plot(labels)
        source = plot.handles['source']
        glyph = plot.handles['glyph']
        expected = {'x': np.array([]), 'y': np.array([]), 'Label': []}
        for k, vals in expected.items():
            self.assertEqual(source.data[k], vals)
        self.assertEqual(glyph.x, 'x')
        self.assertEqual(glyph.y, 'y')
        self.assertEqual(glyph.text, 'Label')
        
    def test_labels_formatter(self):
        vdim = Dimension('text', value_format=lambda x: '%.1f' % x)
        labels = Labels([(0, 1, 0.33333), (1, 0, 0.66666)], vdims=vdim)
        plot = bokeh_renderer.get_plot(labels)
        source = plot.handles['source']
        glyph = plot.handles['glyph']
        expected = {'x': np.array([0, 1]), 'y': np.array([1, 0]),
                    'text': ['0.3', '0.7']}
        for k, vals in expected.items():
            self.assertEqual(source.data[k], vals)
        self.assertEqual(glyph.x, 'x')
        self.assertEqual(glyph.y, 'y')
        self.assertEqual(glyph.text, 'text')

    def test_labels_inverted(self):
        labels = Labels([(0, 1, 'A'), (1, 0, 'B')]).options(invert_axes=True)
        plot = bokeh_renderer.get_plot(labels)
        source = plot.handles['source']
        glyph = plot.handles['glyph']
        expected = {'x': np.array([0, 1]), 'y': np.array([1, 0]), 'Label': ['A', 'B']}
        for k, vals in expected.items():
            self.assertEqual(source.data[k], vals)
        self.assertEqual(glyph.x, 'y')
        self.assertEqual(glyph.y, 'x')
        self.assertEqual(glyph.text, 'Label')

    def test_labels_color_mapped(self):
        labels = Labels([(0, 1, 0.33333), (1, 0, 0.66666)]).options(color_index=2)
        plot = bokeh_renderer.get_plot(labels)
        source = plot.handles['source']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_mapper']
        expected = {'x': np.array([0, 1]), 'y': np.array([1, 0]),
                    'Label': ['0.33333', '0.66666'],
                    'text_color': np.array([0.33333, 0.66666])}
        for k, vals in expected.items():
            self.assertEqual(source.data[k], vals)
        self.assertEqual(glyph.x, 'x')
        self.assertEqual(glyph.y, 'y')
        self.assertEqual(glyph.text, 'Label')
        self.assertEqual(cmapper.low, 0.33333)
        self.assertEqual(cmapper.high, 0.66666)
