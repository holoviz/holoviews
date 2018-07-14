import datetime as dt
from unittest import SkipTest

import numpy as np

from holoviews.core.util import pd
from holoviews.element import Curve

from .testplot import TestMPLPlot, mpl_renderer


class TestCurvePlot(TestMPLPlot):

    def test_curve_datetime64(self):
        dates = [np.datetime64(dt.datetime(2016,1,i)) for i in range(1, 11)]
        curve = Curve((dates, np.random.rand(10)))
        plot = mpl_renderer.get_plot(curve)
        self.assertEqual(plot.handles['axis'].get_xlim(), (735964.0, 735973.0))

    def test_curve_pandas_timestamps(self):
        if not pd:
            raise SkipTest("Pandas not available")
        dates = pd.date_range('2016-01-01', '2016-01-10', freq='D')
        curve = Curve((dates, np.random.rand(10)))
        plot = mpl_renderer.get_plot(curve)
        self.assertEqual(plot.handles['axis'].get_xlim(), (735964.0, 735973.0))

    def test_curve_dt_datetime(self):
        dates = [dt.datetime(2016,1,i) for i in range(1, 11)]
        curve = Curve((dates, np.random.rand(10)))
        plot = mpl_renderer.get_plot(curve)
        self.assertEqual(plot.handles['axis'].get_xlim(), (735964.0, 735973.0))

    def test_curve_heterogeneous_datetime_types_overlay(self):
        dates64 = [np.datetime64(dt.datetime(2016,1,i)) for i in range(1, 11)]
        dates = [dt.datetime(2016,1,i) for i in range(2, 12)]
        curve_dt64 = Curve((dates64, np.random.rand(10)))
        curve_dt = Curve((dates, np.random.rand(10)))
        plot = mpl_renderer.get_plot(curve_dt*curve_dt64)
        self.assertEqual(plot.handles['axis'].get_xlim(), (735964.0, 735974.0))

    def test_curve_heterogeneous_datetime_types_with_pd_overlay(self):
        if not pd:
            raise SkipTest("Pandas not available")
        dates_pd = pd.date_range('2016-01-04', '2016-01-13', freq='D')
        dates64 = [np.datetime64(dt.datetime(2016,1,i)) for i in range(1, 11)]
        dates = [dt.datetime(2016,1,i) for i in range(2, 12)]
        curve_dt64 = Curve((dates64, np.random.rand(10)))
        curve_dt = Curve((dates, np.random.rand(10)))
        curve_pd = Curve((dates_pd, np.random.rand(10)))
        plot = mpl_renderer.get_plot(curve_dt*curve_dt64*curve_pd)
        self.assertEqual(plot.handles['axis'].get_xlim(), (735964.0, 735976.0))

    def test_curve_padding_square(self):
        curve = Curve([1, 2, 3]).options(padding=0.2)
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], -0.2)
        self.assertEqual(x_range[1], 2.2)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_curve_padding_hard_xrange(self):
        curve = Curve([1, 2, 3]).redim.range(x=(0, 3)).options(padding=0.2)
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0)
        self.assertEqual(x_range[1], 3)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_curve_padding_soft_xrange(self):
        curve = Curve([1, 2, 3]).redim.soft_range(x=(0, 3)).options(padding=0.2)
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], -0.2)
        self.assertEqual(x_range[1], 3)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_curve_padding_unequal(self):
        curve = Curve([1, 2, 3]).options(padding=(0.1, 0.2))
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], -0.1)
        self.assertEqual(x_range[1], 2.1)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_curve_padding_nonsquare(self):
        curve = Curve([1, 2, 3]).options(padding=0.2, aspect=2)
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], -0.1)
        self.assertEqual(x_range[1], 2.1)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_curve_padding_logx(self):
        curve = Curve([(1, 1), (2, 2), (3,3)]).options(padding=0.2, logx=True)
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 0.89595845984076228)
        self.assertEqual(x_range[1], 3.3483695221017129)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_curve_padding_logy(self):
        curve = Curve([1, 2, 3]).options(padding=0.2, logy=True)
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], -0.2)
        self.assertEqual(x_range[1], 2.2)
        self.assertEqual(y_range[0], 0.89595845984076228)
        self.assertEqual(y_range[1], 3.3483695221017129)

    def test_curve_padding_datetime_square(self):
        curve = Curve([(np.datetime64('2016-04-0%d' % i), i) for i in range(1, 4)]).options(
            padding=0.2
        )
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 736054.80000000005)
        self.assertEqual(x_range[1], 736057.19999999995)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)

    def test_curve_padding_datetime_nonsquare(self):
        curve = Curve([(np.datetime64('2016-04-0%d' % i), i) for i in range(1, 4)]).options(
            padding=0.2, aspect=2
        )
        plot = mpl_renderer.get_plot(curve)
        x_range, y_range = plot.handles['axis'].get_xlim(), plot.handles['axis'].get_ylim()
        self.assertEqual(x_range[0], 736054.90000000002)
        self.assertEqual(x_range[1], 736057.09999999998)
        self.assertEqual(y_range[0], 0.8)
        self.assertEqual(y_range[1], 3.2)
