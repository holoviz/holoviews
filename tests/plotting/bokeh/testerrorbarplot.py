from holoviews.element import ErrorBars

from .testplot import TestBokehPlot, bokeh_renderer


class TestErrorBarsPlot(TestBokehPlot):

    def test_errorbars_padding_square(self):
        errorbars = ErrorBars([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).options(padding=0.2)
        plot = bokeh_renderer.get_plot(errorbars)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0.19999999999999996)
        self.assertEqual(y_range.end, 3.8)

    def test_errorbars_padding_hard_range(self):
        errorbars = ErrorBars([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).redim.range(y=(0, 4)).options(padding=0.2)
        plot = bokeh_renderer.get_plot(errorbars)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 4)

    def test_errorbars_padding_soft_range(self):
        errorbars = ErrorBars([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).redim.soft_range(y=(0, 3.5)).options(padding=0.2)
        plot = bokeh_renderer.get_plot(errorbars)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.8)

    def test_errorbars_padding_nonsquare(self):
        errorbars = ErrorBars([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).options(padding=0.2, width=600)
        plot = bokeh_renderer.get_plot(errorbars)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.9)
        self.assertEqual(x_range.end, 3.1)
        self.assertEqual(y_range.start, 0.19999999999999996)
        self.assertEqual(y_range.end, 3.8)

    def test_errorbars_padding_logx(self):
        errorbars = ErrorBars([(1, 1, 0.5), (2, 2, 0.5), (3,3, 0.5)]).options(padding=0.2, logx=True)
        plot = bokeh_renderer.get_plot(errorbars)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.89595845984076228)
        self.assertEqual(x_range.end, 3.3483695221017129)
        self.assertEqual(y_range.start, 0.19999999999999996)
        self.assertEqual(y_range.end, 3.8)
    
    def test_errorbars_padding_logy(self):
        errorbars = ErrorBars([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).options(padding=0.2, logy=True)
        plot = bokeh_renderer.get_plot(errorbars)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0.41158562699652224)
        self.assertEqual(y_range.end, 4.2518491541367327)
