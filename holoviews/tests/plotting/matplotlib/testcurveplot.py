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
