import numpy as np

from holoviews.element import Area

from .testplot import TestBokehPlot, bokeh_renderer


class TestAreaPlot(TestBokehPlot):

    def test_area_with_nans(self):
        area = Area([1, 2, 3, np.nan, 5, 6, 7])
        plot = bokeh_renderer.get_plot(area)
        cds = plot.handles['cds']
        self.assertEqual(cds.data['x'], np.array([0., 1., 2., 2., 1., 0., np.nan,
                                                  4., 5., 6., 6., 5., 4.]))
        self.assertEqual(cds.data['y'], np.array([0., 0., 0., 3., 2., 1., np.nan,
                                                  0., 0., 0., 7., 6., 5.]))

    def test_area_empty(self):
        area = Area([])
        plot = bokeh_renderer.get_plot(area)
        cds = plot.handles['cds']
        self.assertEqual(cds.data['x'], [])
        self.assertEqual(cds.data['y'], [])

    def test_area_padding_square(self):
        area = Area([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).options(padding=0.2)
        plot = bokeh_renderer.get_plot(area)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0.19999999999999996)
        self.assertEqual(y_range.end, 3.8)

    def test_area_padding_hard_range(self):
        area = Area([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).redim.range(y=(0, 4)).options(padding=0.2)
        plot = bokeh_renderer.get_plot(area)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 4)

    def test_area_padding_soft_range(self):
        area = Area([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).redim.soft_range(y=(0, 3.5)).options(padding=0.2)
        plot = bokeh_renderer.get_plot(area)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.8)

    def test_area_padding_nonsquare(self):
        area = Area([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).options(padding=0.2, width=600)
        plot = bokeh_renderer.get_plot(area)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.9)
        self.assertEqual(x_range.end, 3.1)
        self.assertEqual(y_range.start, 0.19999999999999996)
        self.assertEqual(y_range.end, 3.8)

    def test_area_padding_logx(self):
        area = Area([(1, 1, 0.5), (2, 2, 0.5), (3,3, 0.5)]).options(padding=0.2, logx=True)
        plot = bokeh_renderer.get_plot(area)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.89595845984076228)
        self.assertEqual(x_range.end, 3.3483695221017129)
        self.assertEqual(y_range.start, -0.30000000000000004)
        self.assertEqual(y_range.end, 3.8)
    
    def test_area_padding_logy(self):
        area = Area([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).options(padding=0.2, logy=True)
        plot = bokeh_renderer.get_plot(area)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0.41158562699652224)
        self.assertEqual(y_range.end, 4.2518491541367327)
