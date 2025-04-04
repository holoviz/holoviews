from itertools import product

import numpy as np

from holoviews.core.spaces import HoloMap
from holoviews.element.raster import HeatMap

try:
    from holoviews.plotting.mpl import RadialHeatMapPlot
except ImportError:
    pass

from .test_plot import TestMPLPlot, mpl_renderer


class RadialHeatMapPlotTests(TestMPLPlot):

    def setUp(self):
        super().setUp()

        # set up dummy data for convenient tests
        x = [f"Seg {idx}" for idx in range(2)]
        y = [f"Ann {idx}" for idx in range(2)]
        self.z = list(range(4))
        self.x, self.y = zip(*product(x, y), strict=None)

        self.wedge_data = [((0.5, 0.5), 0.125, 0.375, 180.0, 360.0),
                           ((0.5, 0.5), 0.125, 0.375,   0.0, 180.0),
                           ((0.5, 0.5), 0.125, 0.5,   180.0, 360.0),
                           ((0.5, 0.5), 0.125, 0.5,     0.0, 180.0)]

        self.xticks = [(0.0, 'Seg 0'), (3.1415926535897931, 'Seg 1')]
        self.yticks = [(0.25, 'Ann 0'), (0.375, 'Ann 1')]

        # set up plot options for convenient tests
        plot_opts = dict(start_angle=0,
                         max_radius=1,
                         radius_inner=0.5,
                         radius_outer=0.2,
                         radial=True)

        opts = dict(HeatMap=plot_opts)

        # provide element and plot instances for tests
        self.element = HeatMap((self.x, self.y, self.z)).opts(opts)

    def test_get_data(self):
        plot = mpl_renderer.get_plot(self.element)
        data, style, ticks = plot.get_data(self.element, {'z': {'combined': (0, 3)}}, {})
        wedges = data['annular']
        for wedge, wdata in zip(wedges, self.wedge_data, strict=None):
            self.assertEqual((wedge.center, wedge.width, wedge.r,
                              wedge.theta1, wedge.theta2), wdata)
        self.assertEqual(ticks['xticks'], self.xticks)
        self.assertEqual(ticks['yticks'], self.yticks)

    def test_get_data_xseparators(self):
        plot = mpl_renderer.get_plot(self.element.opts(xmarks=4))
        data, style, ticks = plot.get_data(self.element, {'z': {'combined': (0, 3)}}, {})
        xseparators = data['xseparator']
        arrays = [np.array([[0., 0.25],
                            [0., 0.5 ]]),
                  np.array([[3.14159265, 0.25],
                            [3.14159265, 0.5]])]
        self.assertEqual(xseparators, arrays)

    def test_get_data_yseparators(self):
        plot = mpl_renderer.get_plot(self.element.opts(ymarks=4))
        data, style, ticks = plot.get_data(self.element, {'z': {'combined': (0, 3)}}, {})
        yseparators = data['yseparator']
        for circle, r in zip(yseparators, [0.25, 0.375], strict=None):
            self.assertEqual(circle.radius, r)

    def test_heatmap_holomap(self):
        hm = HoloMap({'A': HeatMap(np.random.randint(0, 10, (100, 3))),
                      'B': HeatMap(np.random.randint(0, 10, (100, 3)))})
        plot = mpl_renderer.get_plot(hm.opts(radial=True))
        self.assertIsInstance(plot, RadialHeatMapPlot)
