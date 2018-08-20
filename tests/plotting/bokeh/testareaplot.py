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
