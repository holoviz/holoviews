import numpy as np

from holoviews.element import Image, Points

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
