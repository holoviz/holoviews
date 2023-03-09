import numpy as np

from holoviews.element import (
    HLine, VLine, Text, Labels, Arrow, HSpan, VSpan, Slope
)

from .test_plot import TestBokehPlot, bokeh_renderer


class TestHVLinePlot(TestBokehPlot):

    def test_hline_invert_axes(self):
        hline = HLine(1.1).opts(invert_axes=True)
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
        vline = VLine(1.1).opts(invert_axes=True)
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


class TestHVSpanPlot(TestBokehPlot):

    def test_hspan_invert_axes(self):
        hspan = HSpan(1.1, 1.5).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(hspan)
        span = plot.handles['glyph']

        self.assertEqual(span.left, 1.1)
        self.assertEqual(span.right, 1.5)
        self.assertEqual(span.bottom, None)
        self.assertEqual(span.top, None)

    def test_hspan_plot(self):
        hspan = HSpan(1.1, 1.5)
        plot = bokeh_renderer.get_plot(hspan)
        span = plot.handles['glyph']
        self.assertEqual(span.left, None)
        self.assertEqual(span.right, None)
        self.assertEqual(span.bottom, 1.1)
        self.assertEqual(span.top, 1.5)

    def test_vspan_invert_axes(self):
        vspan = VSpan(1.1, 1.5).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(vspan)
        span = plot.handles['glyph']
        self.assertEqual(span.left, None)
        self.assertEqual(span.right, None)
        self.assertEqual(span.bottom, 1.1)
        self.assertEqual(span.top, 1.5)

    def test_vspan_plot(self):
        vspan = VSpan(1.1, 1.5)
        plot = bokeh_renderer.get_plot(vspan)
        span = plot.handles['glyph']
        self.assertEqual(span.left, 1.1)
        self.assertEqual(span.right, 1.5)
        self.assertEqual(span.bottom, None)
        self.assertEqual(span.top, None)



class TestSlopePlot(TestBokehPlot):

    def test_slope(self):
        hspan = Slope(2, 10)
        plot = bokeh_renderer.get_plot(hspan)
        slope = plot.handles['glyph']
        self.assertEqual(slope.gradient, 2)
        self.assertEqual(slope.y_intercept, 10)

    def test_slope_invert_axes(self):
        hspan = Slope(2, 10).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(hspan)
        slope = plot.handles['glyph']
        self.assertEqual(slope.gradient, 0.5)
        self.assertEqual(slope.y_intercept, -5)



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
        text = Text(0, 0, 'Test').opts(angle=90)
        plot = bokeh_renderer.get_plot(text)
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.angle, np.pi/2.)


class TestArrowPlot(TestBokehPlot):

    def _compare_arrow_plot(self, plot, start, end):
        print(plot.handles)
        arrow_glyph = plot.handles['arrow_1_glyph']
        arrow_cds = plot.handles['arrow_1_source']
        label_glyph = plot.handles['text_1_glyph']

        label_cds = plot.handles['text_1_source']
        x0, y0 = start
        x1, y1 = end
        self.assertEqual(label_glyph.x, 'x')
        self.assertEqual(label_glyph.y, 'y')
        self.assertEqual(label_cds.data, {'x': [x0], 'y': [y0], 'text': ['Test']})
        self.assertEqual(arrow_glyph.x_start, 'x_start')
        self.assertEqual(arrow_glyph.y_start, 'y_start')
        self.assertEqual(arrow_glyph.x_end, 'x_end')
        self.assertEqual(arrow_glyph.y_end, 'y_end')
        self.assertEqual(arrow_cds.data, {'x_start': [x0], 'x_end': [x1],
                                          'y_start': [y0], 'y_end': [y1]})

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
        text = Labels([(0, 0, 'Test')]).opts(angle=90)
        plot = bokeh_renderer.get_plot(text)
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.angle, np.pi/2.)
