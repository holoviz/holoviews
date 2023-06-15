import numpy as np

from bokeh.models import CategoricalColorMapper, LinearColorMapper

from holoviews.element import ErrorBars
from holoviews.plotting.bokeh.util import property_to_dict

from .test_plot import TestBokehPlot, bokeh_renderer


class TestErrorBarsPlot(TestBokehPlot):

    def test_errorbars_padding_square(self):
        errorbars = ErrorBars([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(errorbars)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0.19999999999999996)
        self.assertEqual(y_range.end, 3.8)

    def test_errorbars_padding_hard_range(self):
        errorbars = ErrorBars([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).redim.range(y=(0, 4)).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(errorbars)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 4)

    def test_errorbars_padding_soft_range(self):
        errorbars = ErrorBars([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).redim.soft_range(y=(0, 3.5)).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(errorbars)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.5)

    def test_errorbars_padding_nonsquare(self):
        errorbars = ErrorBars([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).opts(padding=0.1, width=600)
        plot = bokeh_renderer.get_plot(errorbars)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.9)
        self.assertEqual(x_range.end, 3.1)
        self.assertEqual(y_range.start, 0.19999999999999996)
        self.assertEqual(y_range.end, 3.8)

    def test_errorbars_padding_logx(self):
        errorbars = ErrorBars([(1, 1, 0.5), (2, 2, 0.5), (3,3, 0.5)]).opts(padding=0.1, logx=True)
        plot = bokeh_renderer.get_plot(errorbars)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.89595845984076228)
        self.assertEqual(x_range.end, 3.3483695221017129)
        self.assertEqual(y_range.start, 0.19999999999999996)
        self.assertEqual(y_range.end, 3.8)

    def test_errorbars_padding_logy(self):
        errorbars = ErrorBars([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).opts(padding=0.1, logy=True)
        plot = bokeh_renderer.get_plot(errorbars)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0.41158562699652224)
        self.assertEqual(y_range.end, 4.2518491541367327)

    ###########################
    #    Styling mapping      #
    ###########################

    def test_errorbars_color_op(self):
        errorbars = ErrorBars([(0, 0, 0.1, 0.2, '#000'), (0, 1, 0.2, 0.4, '#F00'), (0, 2, 0.6, 1.2, '#0F0')],
                              vdims=['y', 'perr', 'nerr', 'color']).opts(color='color')
        plot = bokeh_renderer.get_plot(errorbars)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['color'], np.array(['#000', '#F00', '#0F0']))
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color'})

    def test_errorbars_linear_color_op(self):
        errorbars = ErrorBars([(0, 0, 0.1, 0.2, 0), (0, 1, 0.2, 0.4, 1), (0, 2, 0.6, 1.2, 2)],
                              vdims=['y', 'perr', 'nerr', 'color']).opts(color='color')
        plot = bokeh_renderer.get_plot(errorbars)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertTrue(cmapper, LinearColorMapper)
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 2)
        self.assertEqual(cds.data['color'], np.array([0, 1, 2]))
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color', 'transform': cmapper})

    def test_errorbars_categorical_color_op(self):
        errorbars = ErrorBars([(0, 0, 0.1, 0.2, 'A'), (0, 1, 0.2, 0.4, 'B'), (0, 2, 0.6, 1.2, 'C')],
                              vdims=['y', 'perr', 'nerr', 'color']).opts(color='color')
        plot = bokeh_renderer.get_plot(errorbars)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertTrue(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, ['A', 'B', 'C'])
        self.assertEqual(cds.data['color'], np.array(['A', 'B', 'C']))
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color', 'transform': cmapper})

    def test_errorbars_line_color_op(self):
        errorbars = ErrorBars([(0, 0, 0.1, 0.2, '#000'), (0, 1, 0.2, 0.4, '#F00'), (0, 2, 0.6, 1.2, '#0F0')],
                              vdims=['y', 'perr', 'nerr', 'color']).opts(line_color='color')
        plot = bokeh_renderer.get_plot(errorbars)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['line_color'], np.array(['#000', '#F00', '#0F0']))
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'line_color'})

    def test_errorbars_alpha_op(self):
        errorbars = ErrorBars([(0, 0, 0.1, 0.2, 0), (0, 1, 0.2, 0.4, 0.2), (0, 2, 0.6, 1.2, 0.7)],
                              vdims=['y', 'perr', 'nerr', 'alpha']).opts(alpha='alpha')
        plot = bokeh_renderer.get_plot(errorbars)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['alpha'], np.array([0, 0.2, 0.7]))
        self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'alpha'})

    def test_errorbars_line_alpha_op(self):
        errorbars = ErrorBars([(0, 0, 0.1, 0.2, 0), (0, 1, 0.2, 0.4, 0.2), (0, 2, 0.6, 1.2, 0.7)],
                              vdims=['y', 'perr', 'nerr', 'alpha']).opts(line_alpha='alpha')
        plot = bokeh_renderer.get_plot(errorbars)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['line_alpha'], np.array([0, 0.2, 0.7]))
        self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'line_alpha'})

    def test_errorbars_line_width_op(self):
        errorbars = ErrorBars([(0, 0, 0.1, 0.2, 1), (0, 1, 0.2, 0.4, 4), (0, 2, 0.6, 1.2, 8)],
                              vdims=['y', 'perr', 'nerr', 'line_width']).opts(line_width='line_width')
        plot = bokeh_renderer.get_plot(errorbars)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['line_width'], np.array([1, 4, 8]))
        self.assertEqual(property_to_dict(glyph.line_width), {'field': 'line_width'})
