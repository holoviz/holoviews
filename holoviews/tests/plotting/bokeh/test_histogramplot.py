import datetime as dt

import numpy as np

from holoviews.core.overlay import NdOverlay
from holoviews.element import Image, Points, Dataset, Histogram
from holoviews.operation import histogram
from holoviews.plotting.bokeh.util import property_to_dict

from bokeh.models import DatetimeAxis, CategoricalColorMapper, LinearColorMapper

from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer


class TestSideHistogramPlot(LoggingComparisonTestCase, TestBokehPlot):

    def test_side_histogram_no_cmapper(self):
        points = Points(np.random.rand(100, 2))
        plot = bokeh_renderer.get_plot(points.hist())
        plot.initialize_plot()
        adjoint_plot = next(iter(plot.subplots.values()))
        main_plot = adjoint_plot.subplots['main']
        right_plot = adjoint_plot.subplots['right']
        self.assertTrue('color_mapper' not in main_plot.handles)
        self.assertTrue('color_mapper' not in right_plot.handles)

    def test_side_histogram_cmapper(self):
        """Assert histogram shares colormapper"""
        x,y = np.mgrid[-50:51, -50:51] * 0.1
        img = Image(np.sin(x**2+y**2), bounds=(-1,-1,1,1))
        plot = bokeh_renderer.get_plot(img.hist())
        plot.initialize_plot()
        adjoint_plot = next(iter(plot.subplots.values()))
        main_plot = adjoint_plot.subplots['main']
        right_plot = adjoint_plot.subplots['right']
        self.assertIs(main_plot.handles['color_mapper'],
                      right_plot.handles['color_mapper'])
        self.assertEqual(main_plot.handles['color_dim'], img.vdims[0])

    def test_side_histogram_cmapper_weighted(self):
        """Assert weighted histograms share colormapper"""
        x,y = np.mgrid[-50:51, -50:51] * 0.1
        img = Image(np.sin(x**2+y**2), bounds=(-1,-1,1,1))
        adjoint = img.hist(dimension=['x', 'y'], weight_dimension='z',
                           mean_weighted=True)
        plot = bokeh_renderer.get_plot(adjoint)
        plot.initialize_plot()
        adjoint_plot = next(iter(plot.subplots.values()))
        main_plot = adjoint_plot.subplots['main']
        right_plot = adjoint_plot.subplots['right']
        top_plot = adjoint_plot.subplots['top']
        self.assertIs(main_plot.handles['color_mapper'],
                      right_plot.handles['color_mapper'])
        self.assertIs(main_plot.handles['color_mapper'],
                      top_plot.handles['color_mapper'])
        self.assertEqual(main_plot.handles['color_dim'], img.vdims[0])

    def test_histogram_datetime64_plot(self):
        dates = np.array([dt.datetime(2017, 1, i) for i in range(1, 5)])
        hist = histogram(Dataset(dates, 'Date'), num_bins=4)
        plot = bokeh_renderer.get_plot(hist)
        source = plot.handles['source']
        data = {
            'top': np.array([1, 1, 1, 1]),
            'left': np.array([
                '2017-01-01T00:00:00.000000', '2017-01-01T18:00:00.000000',
                '2017-01-02T12:00:00.000000', '2017-01-03T06:00:00.000000'],
                dtype='datetime64[us]'),
            'right': np.array([
                '2017-01-01T18:00:00.000000', '2017-01-02T12:00:00.000000',
                '2017-01-03T06:00:00.000000', '2017-01-04T00:00:00.000000'],
                dtype='datetime64[us]')
        }
        for k, v in data.items():
            self.assertEqual(source.data[k], v)
        xaxis = plot.handles['xaxis']
        range_x = plot.handles['x_range']
        self.assertIsInstance(xaxis, DatetimeAxis)
        self.assertEqual(range_x.start, np.datetime64('2017-01-01T00:00:00.000000', 'us'))
        self.assertEqual(range_x.end, np.datetime64('2017-01-04T00:00:00.000000', 'us'))

    def test_histogram_padding_square(self):
        points = Histogram([(1, 2), (2, -1), (3, 3)]).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.19999999999999996)
        self.assertEqual(x_range.end, 3.8)
        self.assertEqual(y_range.start, -1.4)
        self.assertEqual(y_range.end, 3.4)

    def test_histogram_padding_square_positive(self):
        points = Histogram([(1, 2), (2, 1), (3, 3)]).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.19999999999999996)
        self.assertEqual(x_range.end, 3.8)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.2)

    def test_histogram_padding_square_negative(self):
        points = Histogram([(1, -2), (2, -1), (3, -3)]).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.19999999999999996)
        self.assertEqual(x_range.end, 3.8)
        self.assertEqual(y_range.start, -3.2)
        self.assertEqual(y_range.end, 0)

    def test_histogram_padding_nonsquare(self):
        histogram = Histogram([(1, 2), (2, 1), (3, 3)]).opts(padding=0.1, width=600)
        plot = bokeh_renderer.get_plot(histogram)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.35)
        self.assertEqual(x_range.end, 3.65)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.2)

    def test_histogram_padding_logx(self):
        histogram = Histogram([(1, 1), (2, 2), (3,3)]).opts(padding=0.1, logx=True)
        plot = bokeh_renderer.get_plot(histogram)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.41158562699652224)
        self.assertEqual(x_range.end, 4.2518491541367327)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.2)

    def test_histogram_padding_logy(self):
        histogram = Histogram([(1, 2), (2, 1), (3, 3)]).opts(padding=0.1, logy=True)
        plot = bokeh_renderer.get_plot(histogram)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.19999999999999996)
        self.assertEqual(x_range.end, 3.8)
        self.assertEqual(y_range.start, 0.01)
        self.assertEqual(y_range.end, 3.3483695221017129)
        self.log_handler.assertContains('WARNING', 'Logarithmic axis range encountered value less than')

    def test_histogram_padding_datetime_square(self):
        histogram = Histogram([(np.datetime64('2016-04-0%d' % i, 'ns'), i) for i in range(1, 4)]).opts(
            padding=0.1
        )
        plot = bokeh_renderer.get_plot(histogram)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, np.datetime64('2016-03-31T04:48:00.000000000'))
        self.assertEqual(x_range.end, np.datetime64('2016-04-03T19:12:00.000000000'))
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.2)

    def test_histogram_padding_datetime_nonsquare(self):
        histogram = Histogram([(np.datetime64('2016-04-0%d' % i, 'ns'), i) for i in range(1, 4)]).opts(
            padding=0.1, width=600
        )
        plot = bokeh_renderer.get_plot(histogram)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, np.datetime64('2016-03-31T08:24:00.000000000'))
        self.assertEqual(x_range.end, np.datetime64('2016-04-03T15:36:00.000000000'))
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.2)

    ###########################
    #    Styling mapping      #
    ###########################

    def test_histogram_color_op(self):
        histogram = Histogram([(0, 0, '#000'), (0, 1, '#F00'), (0, 2, '#0F0')],
                              vdims=['y', 'color']).opts(color='color')
        plot = bokeh_renderer.get_plot(histogram)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['color'], np.array(['#000', '#F00', '#0F0']))
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'color'})
        self.assertEqual(glyph.line_color, 'black')

    def test_histogram_linear_color_op(self):
        histogram = Histogram([(0, 0, 0), (0, 1, 1), (0, 2, 2)],
                              vdims=['y', 'color']).opts(color='color')
        plot = bokeh_renderer.get_plot(histogram)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertTrue(cmapper, LinearColorMapper)
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 2)
        self.assertEqual(cds.data['color'], np.array([0, 1, 2]))
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'color', 'transform': cmapper})
        self.assertEqual(glyph.line_color, 'black')

    def test_histogram_categorical_color_op(self):
        histogram = Histogram([(0, 0, 'A'), (0, 1, 'B'), (0, 2, 'C')],
                              vdims=['y', 'color']).opts(color='color')
        plot = bokeh_renderer.get_plot(histogram)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertTrue(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, ['A', 'B', 'C'])
        self.assertEqual(cds.data['color'], np.array(['A', 'B', 'C']))
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'color', 'transform': cmapper})
        self.assertEqual(glyph.line_color, 'black')

    def test_histogram_line_color_op(self):
        histogram = Histogram([(0, 0, '#000'), (0, 1, '#F00'), (0, 2, '#0F0')],
                              vdims=['y', 'color']).opts(line_color='color')
        plot = bokeh_renderer.get_plot(histogram)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['line_color'], np.array(['#000', '#F00', '#0F0']))
        self.assertNotEqual(property_to_dict(glyph.fill_color), {'field': 'line_color'})
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'line_color'})

    def test_histogram_fill_color_op(self):
        histogram = Histogram([(0, 0, '#000'), (0, 1, '#F00'), (0, 2, '#0F0')],
                              vdims=['y', 'color']).opts(fill_color='color')
        plot = bokeh_renderer.get_plot(histogram)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['fill_color'], np.array(['#000', '#F00', '#0F0']))
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'fill_color'})
        self.assertNotEqual(property_to_dict(glyph.line_color), {'field': 'fill_color'})

    def test_histogram_alpha_op(self):
        histogram = Histogram([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)],
                              vdims=['y', 'alpha']).opts(alpha='alpha')
        plot = bokeh_renderer.get_plot(histogram)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['alpha'], np.array([0, 0.2, 0.7]))
        self.assertEqual(property_to_dict(glyph.fill_alpha), {'field': 'alpha'})

    def test_histogram_line_alpha_op(self):
        histogram = Histogram([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)],
                              vdims=['y', 'alpha']).opts(line_alpha='alpha')
        plot = bokeh_renderer.get_plot(histogram)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['line_alpha'], np.array([0, 0.2, 0.7]))
        self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'line_alpha'})
        self.assertNotEqual(property_to_dict(glyph.fill_alpha), {'field': 'line_alpha'})

    def test_histogram_fill_alpha_op(self):
        histogram = Histogram([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)],
                              vdims=['y', 'alpha']).opts(fill_alpha='alpha')
        plot = bokeh_renderer.get_plot(histogram)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['fill_alpha'], np.array([0, 0.2, 0.7]))
        self.assertNotEqual(property_to_dict(glyph.line_alpha), {'field': 'fill_alpha'})
        self.assertEqual(property_to_dict(glyph.fill_alpha), {'field': 'fill_alpha'})

    def test_histogram_line_width_op(self):
        histogram = Histogram([(0, 0, 1), (0, 1, 4), (0, 2, 8)],
                              vdims=['y', 'line_width']).opts(line_width='line_width')
        plot = bokeh_renderer.get_plot(histogram)
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        self.assertEqual(cds.data['line_width'], np.array([1, 4, 8]))
        self.assertEqual(property_to_dict(glyph.line_width), {'field': 'line_width'})

    def test_op_ndoverlay_value(self):
        colors = ['blue', 'red']
        overlay = NdOverlay({color: Histogram(np.arange(i+2)) for i, color in enumerate(colors)}, 'Color').opts('Histogram', fill_color='Color')
        plot = bokeh_renderer.get_plot(overlay)
        for subplot, color in zip(plot.subplots.values(),  colors):
            self.assertEqual(subplot.handles['glyph'].fill_color, color)
