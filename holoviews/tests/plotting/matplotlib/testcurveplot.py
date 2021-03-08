import datetime as dt
from unittest import skipIf

import numpy as np

from holoviews.core.overlay import NdOverlay
from holoviews.core.util import pd
from holoviews.element import Curve
from holoviews.util.transform import dim

from .testplot import TestMPLPlot, mpl_renderer

pd_skip = skipIf(pd is None, 'Pandas is not available')


class TestCurvePlot(TestMPLPlot):

    def test_curve_datetime64(self):
        dates = [np.datetime64(dt.datetime(2016,1,i)) for i in range(1, 11)]
        curve = Curve((dates, np.random.rand(10)))
        plot = mpl_renderer.get_plot(curve)
        self.assertEqual(plot.handles['axis'].get_xlim(), (16801.0, 16810.0))

    @pd_skip
    def test_curve_pandas_timestamps(self):
        dates = pd.date_range('2016-01-01', '2016-01-10', freq='D')
        curve = Curve((dates, np.random.rand(10)))
        plot = mpl_renderer.get_plot(curve)
        self.assertEqual(plot.handles['axis'].get_xlim(), (16801.0, 16810.0))

    def test_curve_dt_datetime(self):
        dates = [dt.datetime(2016,1,i) for i in range(1, 11)]
        curve = Curve((dates, np.random.rand(10)))
        plot = mpl_renderer.get_plot(curve)
        self.assertEqual(tuple(map(round, plot.handles['axis'].get_xlim())), (16801.0, 16810.0))

    def test_curve_heterogeneous_datetime_types_overlay(self):
        dates64 = [np.datetime64(dt.datetime(2016,1,i)) for i in range(1, 11)]
        dates = [dt.datetime(2016,1,i) for i in range(2, 12)]
        curve_dt64 = Curve((dates64, np.random.rand(10)))
        curve_dt = Curve((dates, np.random.rand(10)))
        plot = mpl_renderer.get_plot(curve_dt*curve_dt64)
        self.assertEqual(tuple(map(round, plot.handles['axis'].get_xlim())), (16801.0, 16811.0))

    @pd_skip
    def test_curve_heterogeneous_datetime_types_with_pd_overlay(self):
        dates_pd = pd.date_range('2016-01-04', '2016-01-13', freq='D')
        dates64 = [np.datetime64(dt.datetime(2016,1,i)) for i in range(1, 11)]
        dates = [dt.datetime(2016,1,i) for i in range(2, 12)]
        curve_dt64 = Curve((dates64, np.random.rand(10)))
        curve_dt = Curve((dates, np.random.rand(10)))
        curve_pd = Curve((dates_pd, np.random.rand(10)))
        plot = mpl_renderer.get_plot(curve_dt*curve_dt64*curve_pd)
        self.assertEqual(plot.handles['axis'].get_xlim(), (16801.0, 16813.0))

    def test_curve_padding_square(self):
        curve = Curve([1, 2, 3]).options(padding=0.1)
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], -0.2)
        self.assertEqual(x_range[1], 2.2)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_curve_padding_square_per_axis(self):
        curve = Curve([1, 2, 3]).options(padding=((0, 0.1), (0.1, 0.2)))
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0)
        self.assertEqual(x_range[1], 2.2)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.4)

    def test_curve_padding_hard_xrange(self):
        curve = Curve([1, 2, 3]).redim.range(x=(0, 3)).options(padding=0.1)
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0)
        self.assertEqual(x_range[1], 3)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_curve_padding_soft_xrange(self):
        curve = Curve([1, 2, 3]).redim.soft_range(x=(0, 3)).options(padding=0.1)
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0)
        self.assertEqual(x_range[1], 3)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_curve_padding_unequal(self):
        curve = Curve([1, 2, 3]).options(padding=(0.05, 0.1))
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], -0.1)
        self.assertEqual(x_range[1], 2.1)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_curve_padding_nonsquare(self):
        curve = Curve([1, 2, 3]).options(padding=0.1, aspect=2)
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], -0.1)
        self.assertEqual(x_range[1], 2.1)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_curve_padding_logx(self):
        curve = Curve([(1, 1), (2, 2), (3,3)]).options(padding=0.1, logx=True)
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.89595845984076228)
        self.assertEqual(x_range[1], 3.3483695221017129)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_curve_padding_logy(self):
        curve = Curve([1, 2, 3]).options(padding=0.1, logy=True)
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], -0.2)
        self.assertEqual(x_range[1], 2.2)
        self.assertEqual(y_range[0], 0.89595845984076228)
        self.assertEqual(y_range[1], 3.3483695221017129)

    def test_curve_padding_datetime_square(self):
        curve = Curve([(np.datetime64('2016-04-0%d' % i), i) for i in range(1, 4)]).options(
            padding=0.1
        )
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 16891.8)
        self.assertEqual(x_range[1], 16894.2)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_curve_padding_datetime_nonsquare(self):
        curve = Curve([(np.datetime64('2016-04-0%d' % i), i) for i in range(1, 4)]).options(
            padding=0.1, aspect=2
        )
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 16891.9)
        self.assertEqual(x_range[1], 16894.1)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    ###########################
    #    Styling mapping      #
    ###########################

    def test_curve_scalar_color_op(self):
        curve = Curve([(0, 0, 'red'), (0, 1, 'red'), (0, 2, 'red')],
                       vdims=['y', 'color']).options(color='color')
        plot = mpl_renderer.get_plot(curve)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_color(), 'red')

    def test_op_ndoverlay_color_value(self):
        colors = ['blue', 'red']
        overlay = NdOverlay({color: Curve(np.arange(i))
                             for i, color in enumerate(colors)},
                            'color').options('Curve', color='color')
        plot = mpl_renderer.get_plot(overlay)
        for subplot, color in zip(plot.subplots.values(), colors):
            style = dict(subplot.style[subplot.cyclic_index])
            style = subplot._apply_transforms(subplot.current_frame, {}, style)
            self.assertEqual(style['color'], color)

    def test_curve_color_op(self):
        curve = Curve([(0, 0, 'red'), (0, 1, 'blue'), (0, 2, 'red')],
                       vdims=['y', 'color']).options(color='color')
        with self.assertRaises(Exception):
            mpl_renderer.get_plot(curve)

    def test_curve_alpha_op(self):
        curve = Curve([(0, 0, 0.1), (0, 1, 0.3), (0, 2, 1)],
                       vdims=['y', 'alpha']).options(alpha='alpha')
        with self.assertRaises(Exception):
            mpl_renderer.get_plot(curve)

    def test_curve_linewidth_op(self):
        curve = Curve([(0, 0, 0.1), (0, 1, 0.3), (0, 2, 1)],
                       vdims=['y', 'linewidth']).options(linewidth='linewidth')
        with self.assertRaises(Exception):
            mpl_renderer.get_plot(curve)

    def test_curve_style_mapping_ndoverlay_dimensions(self):
        ndoverlay = NdOverlay({
            (0, 'A'): Curve([1, 2, 0]), (0, 'B'): Curve([1, 2, 1]),
            (1, 'A'): Curve([1, 2, 2]), (1, 'B'): Curve([1, 2, 3])},
                              ['num', 'cat']
        ).opts({
            'Curve': dict(
                color=dim('num').categorize({0: 'red', 1: 'blue'}),
                linestyle=dim('cat').categorize({'A': '-.', 'B': '-'})
            )
        })
        plot = mpl_renderer.get_plot(ndoverlay)
        for (num, cat), sp in plot.subplots.items():
            artist = sp.handles['artist']
            color = artist.get_color()
            if num == 0:
                self.assertEqual(color, 'red')
            else:
                self.assertEqual(color, 'blue')
            linestyle = artist.get_linestyle()
            if cat == 'A':
                self.assertEqual(linestyle, '-.')
            else:
                self.assertEqual(linestyle, '-')

    def test_curve_style_mapping_constant_value_dimensions(self):
        vdims = ['y', 'num', 'cat']
        ndoverlay = NdOverlay({
            0: Curve([(0, 1, 0, 'A'), (1, 0, 0, 'A')], vdims=vdims),
            1: Curve([(0, 1, 0, 'B'), (1, 1, 0, 'B')], vdims=vdims),
            2: Curve([(0, 1, 1, 'A'), (1, 2, 1, 'A')], vdims=vdims),
            3: Curve([(0, 1, 1, 'B'), (1, 3, 1, 'B')], vdims=vdims)}
        ).opts({
            'Curve': dict(
                color=dim('num').categorize({0: 'red', 1: 'blue'}),
                linestyle=dim('cat').categorize({'A': '-.', 'B': '-'})
            )
        })
        plot = mpl_renderer.get_plot(ndoverlay)
        for k, sp in plot.subplots.items():
            artist = sp.handles['artist']
            color = artist.get_color()
            if ndoverlay[k].iloc[0, 2] == 0:
                self.assertEqual(color, 'red')
            else:
                self.assertEqual(color, 'blue')
            linestyle = artist.get_linestyle()
            if ndoverlay[k].iloc[0, 3] == 'A':
                self.assertEqual(linestyle, '-.')
            else:
                self.assertEqual(linestyle, '-')
