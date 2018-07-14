import numpy as np

from holoviews.element import Points

from .testplot import TestMPLPlot, mpl_renderer
from ..utils import ParamLogStream

try:
    from matplotlib import pyplot
except:
    pass


class TestPointPlot(TestMPLPlot):

    def test_points_non_numeric_size_warning(self):
        data = (np.arange(10), np.arange(10), list(map(chr, range(94,104))))
        points = Points(data, vdims=['z']).opts(plot=dict(size_index=2))
        with ParamLogStream() as log:
            plot = mpl_renderer.get_plot(points)
        log_msg = log.stream.read()
        warning = ('%s: z dimension is not numeric, '
                   'cannot use to scale Points size.\n' % plot.name)
        self.assertEqual(log_msg, warning)

    def test_points_cbar_extend_both(self):
        img = Points(([0, 1], [0, 3])).redim(y=dict(range=(1,2)))
        plot = mpl_renderer.get_plot(img(plot=dict(colorbar=True, color_index=1)))
        self.assertEqual(plot.handles['cbar'].extend, 'both')

    def test_points_cbar_extend_min(self):
        img = Points(([0, 1], [0, 3])).redim(y=dict(range=(1, None)))
        plot = mpl_renderer.get_plot(img(plot=dict(colorbar=True, color_index=1)))
        self.assertEqual(plot.handles['cbar'].extend, 'min')

    def test_points_cbar_extend_max(self):
        img = Points(([0, 1], [0, 3])).redim(y=dict(range=(None, 2)))
        plot = mpl_renderer.get_plot(img(plot=dict(colorbar=True, color_index=1)))
        self.assertEqual(plot.handles['cbar'].extend, 'max')

    def test_points_cbar_extend_clime(self):
        img = Points(([0, 1], [0, 3])).opts(style=dict(clim=(None, None)))
        plot = mpl_renderer.get_plot(img(plot=dict(colorbar=True, color_index=1)))
        self.assertEqual(plot.handles['cbar'].extend, 'neither')

    def test_points_rcparams_do_not_persist(self):
        opts = dict(fig_rcparams={'text.usetex': True})
        points = Points(([0, 1], [0, 3])).opts(plot=opts)
        mpl_renderer.get_plot(points)
        self.assertFalse(pyplot.rcParams['text.usetex'])

    def test_points_rcparams_used(self):
        opts = dict(fig_rcparams={'grid.color': 'red'})
        points = Points(([0, 1], [0, 3])).opts(plot=opts)
        plot = mpl_renderer.get_plot(points)
        ax = plot.state.axes[0]
        lines = ax.get_xgridlines()
        self.assertEqual(lines[0].get_color(), 'red')
    
    def test_points_padding_square(self):
        points = Points([1, 2, 3]).options(padding=0.2)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], -0.2)
        self.assertEqual(x_range[1], 2.2)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_points_padding_hard_xrange(self):
        points = Points([1, 2, 3]).redim.range(x=(0, 3)).options(padding=0.2)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0)
        self.assertEqual(x_range[1], 3)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_points_padding_soft_xrange(self):
        points = Points([1, 2, 3]).redim.soft_range(x=(0, 3)).options(padding=0.2)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], -0.2)
        self.assertEqual(x_range[1], 3)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_points_padding_unequal(self):
        points = Points([1, 2, 3]).options(padding=(0.1, 0.2))
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], -0.1)
        self.assertEqual(x_range[1], 2.1)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_points_padding_nonsquare(self):
        points = Points([1, 2, 3]).options(padding=0.2, aspect=2)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], -0.1)
        self.assertEqual(x_range[1], 2.1)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_points_padding_logx(self):
        points = Points([(1, 1), (2, 2), (3,3)]).options(padding=0.2, logx=True)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.89595845984076228)
        self.assertEqual(x_range[1], 3.3483695221017129)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_points_padding_logy(self):
        points = Points([1, 2, 3]).options(padding=0.2, logy=True)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], -0.2)
        self.assertEqual(x_range[1], 2.2)
        self.assertEqual(y_range[0], 0.89595845984076228)
        self.assertEqual(y_range[1], 3.3483695221017129)

    def test_points_padding_datetime_square(self):
        points = Points([(np.datetime64('2016-04-0%d' % i), i) for i in range(1, 4)]).options(
            padding=0.2
        )
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 736054.80000000005)
        self.assertEqual(x_range[1], 736057.19999999995)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_points_padding_datetime_nonsquare(self):
        points = Points([(np.datetime64('2016-04-0%d' % i), i) for i in range(1, 4)]).options(
            padding=0.2, aspect=2
        )
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 736054.90000000002)
        self.assertEqual(x_range[1], 736057.09999999998)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)
