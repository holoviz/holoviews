from itertools import product

from holoviews.element.raster import HeatMap

from .testplot import TestMPLPlot, mpl_renderer


class RadialHeatMapPlotTests(TestMPLPlot):

    def setUp(self):
        super(RadialHeatMapPlotTests, self).setUp()

        # set up dummy data for convenient tests
        x = ["Seg {}".format(idx) for idx in range(2)]
        y = ["Ann {}".format(idx) for idx in range(2)]
        self.z = list(range(4))
        self.x, self.y = zip(*product(x, y))

        self.wedge_data = [((0.5, 0.5), 0.125, 0.375, 180.0, 360.0),
                           ((0.5, 0.5), 0.125, 0.375,   0.0, 180.0),
                           ((0.5, 0.5), 0.125, 0.5,   180.0, 360.0),
                           ((0.5, 0.5), 0.125, 0.5,     0.0, 180.0)]

        self.xticks = [(0.0, 'Seg 0'), (3.1415926535897931, 'Seg 1')]
        self.yticks = [(0.25, 'Ann 0'), (0.375, 'Ann 1')]

        # set up plot options for convenient tests
        plot_opts = dict(start_angle=0,
                         max_radius=1,
                         padding_inner=0.5,
                         padding_outer=0.2,
                         radial=True)

        opts = dict(HeatMap=dict(plot=plot_opts))

        # provide element and plot instances for tests
        self.element = HeatMap((self.x, self.y, self.z)).opts(opts)
        self.plot = mpl_renderer.get_plot(self.element)

    def test_get_data(self):
        data, style, ticks = self.plot.get_data(self.element, {'z': (0, 3)}, {})
        wedges = data['annular']
        for wedge, wdata in zip(wedges, self.wedge_data):
            self.assertEqual((wedge.center, wedge.width, wedge.r,
                              wedge.theta1, wedge.theta2), wdata)
        self.assertEqual(ticks['xticks'], self.xticks)
        self.assertEqual(ticks['yticks'], self.yticks)


