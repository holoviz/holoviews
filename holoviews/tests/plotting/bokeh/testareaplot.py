import datetime as dt

import numpy as np

from holoviews.element import Area

from ...utils import LoggingComparisonTestCase
from .testplot import TestBokehPlot, bokeh_renderer


class TestAreaPlot(LoggingComparisonTestCase, TestBokehPlot):

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

    def test_area_datetime_nat(self):
        values = [(np.datetime64(dt.datetime(2017, 1, i)), i) for i in range(1, 4)]
        values.append((np.datetime64('nat'), np.nan))
        values += [(np.datetime64(dt.datetime(2017, 1, i)), i) for i in range(4, 6)]
        area = Area(values)
        plot = bokeh_renderer.get_plot(area)
        cds = plot.handles['cds']
        xs = np.array([
            '2017-01-01T00:00:00.000000000', '2017-01-02T00:00:00.000000000',
            '2017-01-03T00:00:00.000000000', '2017-01-03T00:00:00.000000000',
            '2017-01-02T00:00:00.000000000', '2017-01-01T00:00:00.000000000',
            'NaT', '2017-01-04T00:00:00.000000000',
            '2017-01-05T00:00:00.000000000', '2017-01-05T00:00:00.000000000',
            '2017-01-04T00:00:00.000000000'
        ], dtype='datetime64[ns]')

        ys = np.array([
            0.,  0.,  0.,  3.,  2.,  1., np.nan,  0.,  0.,  5.,  4.
        ])
        self.assertEqual(cds.data['x'], xs)
        self.assertEqual(cds.data['y'], ys)

    def test_area_padding_square(self):
        area = Area([(1, 1), (2, 2), (3, 3)]).options(padding=0.1)
        plot = bokeh_renderer.get_plot(area)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.2)

    def test_area_padding_square_per_axis(self):
        area = Area([(1, 1), (2, 2), (3, 3)]).options(padding=((0, 0.1), (0.1, 0.2)))
        plot = bokeh_renderer.get_plot(area)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 1.0)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.4)

    def test_area_with_lower_vdim(self):
        area = Area([(1, 0.5, 1), (2, 1.5, 2), (3, 2.5, 3)], vdims=['y', 'y2']).options(padding=0.1)
        plot = bokeh_renderer.get_plot(area)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0.25)
        self.assertEqual(y_range.end, 3.25)

    def test_area_padding_negative(self):
        area = Area([(1, -1), (2, -2), (3, -3)]).options(padding=0.1)
        plot = bokeh_renderer.get_plot(area)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, -3.2)
        self.assertEqual(y_range.end, 0)

    def test_area_padding_mixed(self):
        area = Area([(1, 1), (2, -2), (3, 3)]).options(padding=0.1)
        plot = bokeh_renderer.get_plot(area)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, -2.5)
        self.assertEqual(y_range.end, 3.5)
        
    def test_area_padding_hard_range(self):
        area = Area([(1, 1), (2, 2), (3, 3)]).redim.range(y=(0, 4)).options(padding=0.1)
        plot = bokeh_renderer.get_plot(area)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 4)

    def test_area_padding_soft_range(self):
        area = Area([(1, 1), (2, 2), (3, 3)]).redim.soft_range(y=(0, 3.5)).options(padding=0.1)
        plot = bokeh_renderer.get_plot(area)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.5)

    def test_area_padding_nonsquare(self):
        area = Area([(1, 1), (2, 2), (3, 3)]).options(padding=0.1, width=600)
        plot = bokeh_renderer.get_plot(area)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.9)
        self.assertEqual(x_range.end, 3.1)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.2)

    def test_area_padding_logx(self):
        area = Area([(1, 1), (2, 2), (3,3)]).options(padding=0.1, logx=True)
        plot = bokeh_renderer.get_plot(area)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.89595845984076228)
        self.assertEqual(x_range.end, 3.3483695221017129)
        self.assertEqual(y_range.start, 0)
        self.assertEqual(y_range.end, 3.2)
    
    def test_area_padding_logy(self):
        area = Area([(1, 1), (2, 2), (3, 3)]).options(padding=0.1, logy=True)
        plot = bokeh_renderer.get_plot(area)
        x_range, y_range = plot.handles['x_range'], plot.handles['y_range']
        self.assertEqual(x_range.start, 0.8)
        self.assertEqual(x_range.end, 3.2)
        self.assertEqual(y_range.start, 0.033483695221017122)
        self.assertEqual(y_range.end, 3.3483695221017129)
        self.log_handler.assertContains('WARNING', 'Logarithmic axis range encountered value less than')

    def test_area_legend(self):
        python = np.array([2, 3, 7, 5, 26, 221, 44, 233, 254, 265, 266, 267, 120, 111])
        pypy = np.array([12, 33, 47, 15, 126, 121, 144, 233, 254, 225, 226, 267, 110, 130])
        jython = np.array([22, 43, 10, 25, 26, 101, 114, 203, 194, 215, 201, 227, 139, 160])

        dims = dict(kdims="time", vdims="memory")
        python = Area(python, label="python", **dims)
        pypy = Area(pypy, label="pypy", **dims)
        jython = Area(jython, label="jython", **dims)

        overlay = Area.stack(python * pypy * jython)
        labels = [n[1] for n in overlay.data]
        self.assertEqual(labels, ['Python', 'Pypy', 'Jython'])
