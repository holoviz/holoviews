import datetime as dt

import numpy as np

from holoviews.element import Image, Points, Dataset
from holoviews.operation import histogram

from bokeh.models import DatetimeAxis

from .testplot import TestBokehPlot, bokeh_renderer


class TestSideHistogramPlot(TestBokehPlot):

    def test_side_histogram_no_cmapper(self):
        points = Points(np.random.rand(100, 2))
        plot = bokeh_renderer.get_plot(points.hist())
        plot.initialize_plot()
        adjoint_plot = list(plot.subplots.values())[0]
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
        adjoint_plot = list(plot.subplots.values())[0]
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
        adjoint_plot = list(plot.subplots.values())[0]
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
        data = {'top': np.array([  3.85802469e-18,   3.85802469e-18,   3.85802469e-18, 3.85802469e-18]),
                'left': np.array(['2017-01-01T00:00:00.000000', '2017-01-01T17:59:59.999999',
                                  '2017-01-02T12:00:00.000000', '2017-01-03T06:00:00.000000'],
                                 dtype='datetime64[us]'),
                'right': np.array(['2017-01-01T17:59:59.999999', '2017-01-02T12:00:00.000000',
                                   '2017-01-03T06:00:00.000000', '2017-01-04T00:00:00.000000'],
                                  dtype='datetime64[us]')}
        for k, v in data.items():
            self.assertEqual(source.data[k], v)
        xaxis = plot.handles['xaxis']
        range_x = plot.handles['x_range']
        self.assertIsInstance(xaxis, DatetimeAxis)
        self.assertEqual(range_x.start, np.datetime64('2017-01-01T00:00:00.000000', 'us'))
        self.assertEqual(range_x.end, np.datetime64('2017-01-04T00:00:00.000000', 'us'))
