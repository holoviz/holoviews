import pandas as pd
from bokeh.models import FactorRange

from holoviews.core import NdOverlay
from holoviews.element import Segments

from .test_plot import TestBokehPlot, bokeh_renderer


class TestSegmentPlot(TestBokehPlot):

    def test_segments_color_selection_nonselection(self):
        opts = dict(color='green', selection_color='red', nonselection_color='blue')
        segments = Segments([(i, i*2, i*3, i*4, i*5, chr(65+i)) for i in range(10)],
                            vdims=['a', 'b']).opts(**opts)
        plot = bokeh_renderer.get_plot(segments)
        glyph_renderer = plot.handles['glyph_renderer']
        self.assertEqual(glyph_renderer.glyph.line_color, 'green')
        self.assertEqual(glyph_renderer.selection_glyph.line_color, 'red')
        self.assertEqual(glyph_renderer.nonselection_glyph.line_color, 'blue')

    def test_segments_alpha_selection_nonselection(self):
        opts = dict(alpha=0.8, selection_alpha=1.0, nonselection_alpha=0.2)
        segments = Segments([(i, i*2, i*3, i*4, i*5, chr(65+i)) for i in range(10)],
                            vdims=['a', 'b']).opts(**opts)
        plot = bokeh_renderer.get_plot(segments)
        glyph_renderer = plot.handles['glyph_renderer']
        self.assertEqual(glyph_renderer.glyph.line_alpha, 0.8)
        self.assertEqual(glyph_renderer.selection_glyph.line_alpha, 1)
        self.assertEqual(glyph_renderer.nonselection_glyph.line_alpha, 0.2)

    def test_segments_overlay_hover(self):
        obj = NdOverlay({
            i: Segments((range(31), range(31),range(1, 32), range(31)))
            for i in range(5)
        }, kdims=['Test']).opts({'Segments': {'tools': ['hover']}})
        tooltips = [
            ('Test', '@{Test}'),
            ('x0', '@{x0}'),
            ('y0', '@{y0}'),
            ('x1', '@{x1}'),
            ('y1', '@{y1}')
        ]
        self._test_hover_info(obj, tooltips)

    def test_segments_overlay_datetime_hover(self):
        obj = NdOverlay({
            i: Segments((
                list(pd.date_range('2016-01-01', '2016-01-31')),
                range(31),
                pd.date_range('2016-01-02', '2016-02-01'),
                range(31)
            ))
            for i in range(5)
        }, kdims=['Test']).opts({'Segments': {'tools': ['hover']}})
        tooltips = [
            ('Test', '@{Test}'),
            ('x0', '@{x0}{%F %T}'),
            ('y0', '@{y0}'),
            ('x1', '@{x1}{%F %T}'),
            ('y1', '@{y1}')
        ]
        formatters = {'@{x0}': "datetime", '@{x1}': "datetime"}
        self._test_hover_info(obj, tooltips, formatters=formatters)

    def test_segments_categorical_xaxis(self):
        segments = Segments((['A', 'B', 'C'], [1, 2, 3], ['A', 'B', 'C'], [4, 5, 6]))
        plot = bokeh_renderer.get_plot(segments)
        x_range = plot.handles['x_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, ['A', 'B', 'C'])

    def test_segments_categorical_yaxis(self):
        segments = Segments(([1, 2, 3], ['A', 'B', 'C'], [4, 5, 6], ['A', 'B', 'C']))
        plot = bokeh_renderer.get_plot(segments)
        y_range = plot.handles['y_range']
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['A', 'B', 'C'])

    def test_segments_categorical_yaxis_invert_axes(self):
        segments = Segments(([1, 2, 3], ['A', 'B', 'C'], [4, 5, 6], ['A', 'B', 'C']))
        plot = bokeh_renderer.get_plot(segments)
        y_range = plot.handles['y_range']
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['A', 'B', 'C'])

    def test_segments_overlay_categorical_yaxis(self):
        segments = Segments(([1, 2, 3], ['A', 'B', 'C'], [4, 5, 6], ['A', 'B', 'C']))
        segments2 = Segments(([1, 2, 3], ['B', 'C', 'D'], [4, 5, 6], ['B', 'C', 'D']))
        plot = bokeh_renderer.get_plot(segments*segments2)
        y_range = plot.handles['y_range']
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['A', 'B', 'C', 'D'])

    def test_segments_overlay_categorical_yaxis_invert_yaxis(self):
        segments = Segments(([1, 2, 3], ['A', 'B', 'C'], [4, 5, 6], ['A', 'B', 'C'])).opts(invert_yaxis=True)
        segments2 = Segments(([1, 2, 3], ['B', 'C', 'D'], [4, 5, 6], ['B', 'C', 'D']))
        plot = bokeh_renderer.get_plot(segments*segments2)
        y_range = plot.handles['y_range']
        self.assertIsInstance(y_range, FactorRange)
        self.assertEqual(y_range.factors, ['A', 'B', 'C', 'D'][::-1])

    def test_segments_overlay_categorical_xaxis_invert_axes(self):
        segments = Segments(([1, 2, 3], ['A', 'B', 'C'], [4, 5, 6], ['A', 'B', 'C'])).opts(invert_axes=True)
        segments2 = Segments(([1, 2, 3], ['B', 'C', 'D'], [4, 5, 6], ['B', 'C', 'D']))
        plot = bokeh_renderer.get_plot(segments*segments2)
        x_range = plot.handles['x_range']
        self.assertIsInstance(x_range, FactorRange)
        self.assertEqual(x_range.factors, ['A', 'B', 'C', 'D'])
