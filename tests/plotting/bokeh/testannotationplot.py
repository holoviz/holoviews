import numpy as np

from holoviews.element import HLine, VLine, Text, Labels

from .testplot import TestBokehPlot, bokeh_renderer


class TestHVLinePlot(TestBokehPlot):

    def test_hline_invert_axes(self):
        hline = HLine(1.1).opts(plot=dict(invert_axes=True))
        plot = bokeh_renderer.get_plot(hline)
        span = plot.handles['glyph']
        self.assertEqual(span.dimension, 'height')
        self.assertEqual(span.location, 1.1)

    def test_hline_plot(self):
        hline = HLine(1.1)
        plot = bokeh_renderer.get_plot(hline)
        span = plot.handles['glyph']
        self.assertEqual(span.dimension, 'width')
        self.assertEqual(span.location, 1.1)

    def test_vline_invert_axes(self):
        vline = VLine(1.1).opts(plot=dict(invert_axes=True))
        plot = bokeh_renderer.get_plot(vline)
        span = plot.handles['glyph']
        self.assertEqual(span.dimension, 'width')
        self.assertEqual(span.location, 1.1)

    def test_vline_plot(self):
        vline = VLine(1.1)
        plot = bokeh_renderer.get_plot(vline)
        span = plot.handles['glyph']
        self.assertEqual(span.dimension, 'height')
        self.assertEqual(span.location, 1.1)


class TestTextPlot(TestBokehPlot):

    def test_text_plot(self):
        text = Text(0, 0, 'Test')
        plot = bokeh_renderer.get_plot(text)
        source = plot.handles['source']
        self.assertEqual(source.data, {'x': [0], 'y': [0], 'text': ['Test']})

    def test_text_plot_fontsize(self):
        text = Text(0, 0, 'Test', fontsize=18)
        plot = bokeh_renderer.get_plot(text)
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.text_font_size, '18Pt')

    def test_text_plot_rotation(self):
        text = Text(0, 0, 'Test', rotation=90)
        plot = bokeh_renderer.get_plot(text)
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.angle, np.pi/2.)
    
    def test_text_plot_rotation_style(self):
        text = Text(0, 0, 'Test').options(angle=90)
        plot = bokeh_renderer.get_plot(text)
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.angle, np.pi/2.)



class TestLabelsPlot(TestBokehPlot):

    def test_labels_plot(self):
        text = Labels([(0, 0, 'Test')])
        plot = bokeh_renderer.get_plot(text)
        source = plot.handles['source']
        data = {'x': np.array([0]), 'y': np.array([0]), 'Label': ['Test']}
        for c, col in source.data.items():
            self.assertEqual(col, data[c])

    def test_labels_plot_rotation_style(self):
        text = Labels([(0, 0, 'Test')]).options(angle=90)
        plot = bokeh_renderer.get_plot(text)
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.angle, np.pi/2.)
