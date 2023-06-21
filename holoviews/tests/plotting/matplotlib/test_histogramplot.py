import datetime as dt

import numpy as np
import pytest

from holoviews.core.overlay import NdOverlay
from holoviews.element import Dataset, Histogram
from holoviews.operation import histogram
from holoviews.plotting.util import hex2rgb
from holoviews.core.options import AbbreviatedException

from ...utils import LoggingComparisonTestCase
from .test_plot import TestMPLPlot, mpl_renderer


class TestHistogramPlot(LoggingComparisonTestCase, TestMPLPlot):

    def test_histogram_datetime64_plot(self):
        dates = np.array([dt.datetime(2017, 1, i) for i in range(1, 5)])
        hist = histogram(Dataset(dates, 'Date'), num_bins=4)
        plot = mpl_renderer.get_plot(hist)
        artist = plot.handles['artist']
        ax = plot.handles['axis']
        self.assertEqual(ax.get_xlim(), (17167.0, 17170.0))
        bounds = [17167.0, 17167.75, 17168.5, 17169.25]
        self.assertEqual([p.get_x() for p in artist.patches], bounds)

    def test_histogram_padding_square(self):
        points = Histogram([(1, 2), (2, -1), (3, 3)]).opts(padding=0.1)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.19999999999999996)
        self.assertEqual(x_range[1], 3.8)
        self.assertEqual(y_range[0], -1.4)
        self.assertEqual(y_range[1], 3.4)

    def test_histogram_padding_square_positive(self):
        points = Histogram([(1, 2), (2, 1), (3, 3)]).opts(padding=0.1)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.19999999999999996)
        self.assertEqual(x_range[1], 3.8)
        self.assertEqual(y_range[0], 0)
        self.assertEqual(y_range[1], 3.2)

    def test_histogram_padding_square_negative(self):
        points = Histogram([(1, -2), (2, -1), (3, -3)]).opts(padding=0.1)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.19999999999999996)
        self.assertEqual(x_range[1], 3.8)
        self.assertEqual(y_range[0], -3.2)
        self.assertEqual(y_range[1], 0)

    def test_histogram_padding_nonsquare(self):
        histogram = Histogram([(1, 2), (2, 1), (3, 3)]).opts(padding=0.1, aspect=2)
        plot = mpl_renderer.get_plot(histogram)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.35)
        self.assertEqual(x_range[1], 3.65)
        self.assertEqual(y_range[0], 0)
        self.assertEqual(y_range[1], 3.2)

    def test_histogram_padding_logx(self):
        histogram = Histogram([(1, 1), (2, 2), (3,3)]).opts(padding=0.1, logx=True)
        plot = mpl_renderer.get_plot(histogram)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.41158562699652224)
        self.assertEqual(x_range[1], 4.2518491541367327)
        self.assertEqual(y_range[0], 0)
        self.assertEqual(y_range[1], 3.2)

    def test_histogram_padding_logy(self):
        histogram = Histogram([(1, 2), (2, 1), (3, 3)]).opts(padding=0.1, logy=True)
        plot = mpl_renderer.get_plot(histogram)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.19999999999999996)
        self.assertEqual(x_range[1], 3.8)
        self.assertEqual(y_range[0], 0.03348369522101712)
        self.assertEqual(y_range[1], 3.3483695221017129)
        self.log_handler.assertContains('WARNING', 'Logarithmic axis range encountered value less than')

    def test_histogram_padding_datetime_square(self):
        histogram = Histogram([(np.datetime64('2016-04-0%d' % i, 'ns'), i) for i in range(1, 4)]).opts(
            padding=0.1
        )
        plot = mpl_renderer.get_plot(histogram)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 16891.2)
        self.assertEqual(x_range[1], 16894.8)
        self.assertEqual(y_range[0], 0)
        self.assertEqual(y_range[1], 3.2)

    def test_histogram_padding_datetime_nonsquare(self):
        histogram = Histogram([(np.datetime64('2016-04-0%d' % i, 'ns'), i) for i in range(1, 4)]).opts(
            padding=0.1, aspect=2
        )
        plot = mpl_renderer.get_plot(histogram)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 16891.35)
        self.assertEqual(x_range[1], 16894.65)
        self.assertEqual(y_range[0], 0)
        self.assertEqual(y_range[1], 3.2)

    ###########################
    #    Styling mapping      #
    ###########################

    def test_histogram_color_op(self):
        histogram = Histogram([(0, 0, '#000000'), (0, 1, '#FF0000'), (0, 2, '#00FF00')],
                              vdims=['y', 'color']).opts(color='color')
        plot = mpl_renderer.get_plot(histogram)
        artist = plot.handles['artist']
        children = artist.get_children()
        for c, w in zip(children, ['#000000', '#FF0000', '#00FF00']):
            self.assertEqual(c.get_facecolor(), tuple(c/255. for c in hex2rgb(w))+(1,))

    def test_histogram_linear_color_op(self):
        histogram = Histogram([(0, 0, 0), (0, 1, 1), (0, 2, 2)],
                              vdims=['y', 'color']).opts(color='color')
        msg = 'ValueError: Mapping a continuous dimension to a color'
        with pytest.raises(AbbreviatedException, match=msg):
            mpl_renderer.get_plot(histogram)

    def test_histogram_categorical_color_op(self):
        histogram = Histogram([(0, 0, 'A'), (0, 1, 'B'), (0, 2, 'C')],
                              vdims=['y', 'color']).opts(color='color')
        msg = 'ValueError: Mapping a continuous dimension to a color'
        with pytest.raises(AbbreviatedException, match=msg):
            mpl_renderer.get_plot(histogram)

    def test_histogram_line_color_op(self):
        histogram = Histogram([(0, 0, '#000'), (0, 1, '#F00'), (0, 2, '#0F0')],
                              vdims=['y', 'color']).opts(edgecolor='color')
        plot = mpl_renderer.get_plot(histogram)
        artist = plot.handles['artist']
        children = artist.get_children()
        self.assertEqual(children[0].get_edgecolor(), (0, 0, 0, 1))
        self.assertEqual(children[1].get_edgecolor(), (1, 0, 0, 1))
        self.assertEqual(children[2].get_edgecolor(), (0, 1, 0, 1))

    def test_histogram_alpha_op(self):
        histogram = Histogram([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)],
                              vdims=['y', 'alpha']).opts(alpha='alpha')
        msg = 'ValueError: Mapping a dimension to the "alpha" style'
        with pytest.raises(AbbreviatedException, match=msg):
            mpl_renderer.get_plot(histogram)

    def test_histogram_line_width_op(self):
        histogram = Histogram([(0, 0, 1), (0, 1, 4), (0, 2, 8)],
                              vdims=['y', 'line_width']).opts(linewidth='line_width')
        plot = mpl_renderer.get_plot(histogram)
        artist = plot.handles['artist']
        children = artist.get_children()
        for c, w in zip(children, np.array([1, 4, 8])):
            self.assertEqual(c.get_linewidth(), w)

    def test_op_ndoverlay_value(self):
        colors = ['blue', 'red']
        overlay = NdOverlay({color: Histogram(np.arange(i+2))
                             for i, color in enumerate(colors)}, 'Color').opts(
                                     'Histogram', facecolor='Color'
                             )
        plot = mpl_renderer.get_plot(overlay)
        colors = [(0, 0, 1, 1), (1, 0, 0, 1)]
        for subplot, color in zip(plot.subplots.values(),  colors):
            children = subplot.handles['artist'].get_children()
            for c in children:
                self.assertEqual(c.get_facecolor(), color)
