import datetime as dt

import numpy as np

from holoviews.element import Dataset, Histogram
from holoviews.operation import histogram

from .testplot import TestMPLPlot, mpl_renderer


class TestHistogramPlot(TestMPLPlot):

    def test_histogram_datetime64_plot(self):
        dates = np.array([dt.datetime(2017, 1, i) for i in range(1, 5)])
        hist = histogram(Dataset(dates, 'Date'), num_bins=4)
        plot = mpl_renderer.get_plot(hist)
        artist = plot.handles['artist']
        ax = plot.handles['axis']
        self.assertEqual(ax.get_xlim(), (736330.0, 736333.0))
        bounds = [736330.0, 736330.75, 736331.5, 736332.25]
        self.assertEqual([p.get_x() for p in artist.patches], bounds)

    def test_histogram_padding_square(self):
        points = Histogram([(1, 2), (2, -1), (3, 3)]).options(padding=0.2)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.19999999999999996)
        self.assertEqual(x_range[1], 3.8)
        self.assertEqual(y_range[0], -1.4)
        self.assertEqual(y_range[1], 3.4)

    def test_histogram_padding_square_positive(self):
        points = Histogram([(1, 2), (2, 1), (3, 3)]).options(padding=0.2)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.19999999999999996)
        self.assertEqual(x_range[1], 3.8)
        self.assertEqual(y_range[0], 0)
        self.assertEqual(y_range[1], 3.2)

    def test_histogram_padding_square_negative(self):
        points = Histogram([(1, -2), (2, -1), (3, -3)]).options(padding=0.2)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.19999999999999996)
        self.assertEqual(x_range[1], 3.8)
        self.assertEqual(y_range[0], -3.2)
        self.assertEqual(y_range[1], 0)

    def test_histogram_padding_nonsquare(self):
        histogram = Histogram([(1, 2), (2, 1), (3, 3)]).options(padding=0.2, aspect=2)
        plot = mpl_renderer.get_plot(histogram)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.35)
        self.assertEqual(x_range[1], 3.65)
        self.assertEqual(y_range[0], 0)
        self.assertEqual(y_range[1], 3.2)

    def test_histogram_padding_logx(self):
        histogram = Histogram([(1, 1), (2, 2), (3,3)]).options(padding=0.2, logx=True)
        plot = mpl_renderer.get_plot(histogram)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.41158562699652224)
        self.assertEqual(x_range[1], 4.2518491541367327)
        self.assertEqual(y_range[0], 0)
        self.assertEqual(y_range[1], 3.2)

    def test_histogram_padding_logy(self):
        histogram = Histogram([(1, 2), (2, 1), (3, 3)]).options(padding=0.2, logy=True)
        plot = mpl_renderer.get_plot(histogram)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.19999999999999996)
        self.assertEqual(x_range[1], 3.8)
        self.assertEqual(y_range[0], 1)
        self.assertEqual(y_range[1], 3.3483695221017129)

    def test_histogram_padding_datetime_square(self):
        histogram = Histogram([(np.datetime64('2016-04-0%d' % i, 'ns'), i) for i in range(1, 4)]).options(
            padding=0.2
        )
        plot = mpl_renderer.get_plot(histogram)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 736054.19999999995)
        self.assertEqual(x_range[1], 736057.80000000005)
        self.assertEqual(y_range[0], 0)
        self.assertEqual(y_range[1], 3.2)

    def test_histogram_padding_datetime_nonsquare(self):
        histogram = Histogram([(np.datetime64('2016-04-0%d' % i, 'ns'), i) for i in range(1, 4)]).options(
            padding=0.2, aspect=2
        )
        plot = mpl_renderer.get_plot(histogram)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 736054.34999999998)
        self.assertEqual(x_range[1], 736057.65000000002)
        self.assertEqual(y_range[0], 0)
        self.assertEqual(y_range[1], 3.2)
