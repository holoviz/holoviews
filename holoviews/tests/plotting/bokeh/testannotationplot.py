import numpy as np

from holoviews.element import HLine, VLine, Text, Labels, Arrow

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


class TestArrowPlot(TestBokehPlot):

    def _compare_arrow_plot(self, plot, start, end):
        arrow_glyph = plot.handles['arrow_glyph']
        label_glyph = plot.handles['label_glyph']
        label_cds = plot.handles['label_source']
        x0, y0 = start
        x1, y1 = end
        self.assertEqual(label_glyph.x, 'x')
        self.assertEqual(label_glyph.y, 'y')
        self.assertEqual(label_cds.data, {'x': [x0], 'y': [y0], 'text': ['Test']})
        self.assertEqual(arrow_glyph.x_start, x0)
        self.assertEqual(arrow_glyph.x_end, x1)
        self.assertEqual(arrow_glyph.y_start, y0)
        self.assertEqual(arrow_glyph.y_end, y1)

    def test_arrow_plot_left(self):
        arrow = Arrow(0, 0, 'Test')
        plot = bokeh_renderer.get_plot(arrow)
        self._compare_arrow_plot(plot, (1/6., 0), (0, 0))

    def test_arrow_plot_up(self):
        arrow = Arrow(0, 0, 'Test', '^')
        plot = bokeh_renderer.get_plot(arrow)
        self._compare_arrow_plot(plot, (0, -1/6.), (0, 0))

    def test_arrow_plot_right(self):
        arrow = Arrow(0, 0, 'Test', '>')
        plot = bokeh_renderer.get_plot(arrow)
        self._compare_arrow_plot(plot, (-1/6., 0), (0, 0))

    def test_arrow_plot_down(self):
        arrow = Arrow(0, 0, 'Test', 'v')
        plot = bokeh_renderer.get_plot(arrow)
        self._compare_arrow_plot(plot, (0, 1/6.), (0, 0))


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
