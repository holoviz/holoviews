from holoviews.element import HLine, VLine, Text

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
