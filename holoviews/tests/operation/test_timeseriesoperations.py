import numpy as np
import pandas as pd

from holoviews import Curve, Scatter
from holoviews.operation.timeseries import resample, rolling, rolling_outlier_std
from holoviews.testing import assert_element_equal

from ..utils import optional_dependencies

scipy, scipy_skip = optional_dependencies('scipy')

class TimeseriesOperationTests:
    """
    Tests for the various timeseries operations including rolling,
    resample and rolling_outliers_std.
    """

    def setup_method(self):
        self.dates = pd.date_range("2016-01-01", "2016-01-07", freq='D')
        self.values = [1, 2, 3, 4, 5, 6, 7]
        self.outliers = [1, 2, 1, 2, 10., 2, 1]
        self.date_curve = Curve((self.dates, self.values))
        self.int_curve = Curve(self.values)
        self.date_outliers = Curve((self.dates, self.outliers))
        self.int_outliers = Curve(self.outliers)

    def test_roll_dates(self):
        rolled = rolling(self.date_curve, rolling_window=2)
        rolled_vals = [np.nan, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        assert_element_equal(rolled, Curve((self.dates, rolled_vals)))

    def test_roll_ints(self):
        rolled = rolling(self.int_curve, rolling_window=2)
        rolled_vals = [np.nan, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        assert_element_equal(rolled, Curve(rolled_vals))

    @scipy_skip
    def test_roll_date_with_window_type(self):
        rolled = rolling(self.date_curve, rolling_window=3, window_type='triang')
        rolled_vals = [np.nan, 2, 3, 4, 5, 6, np.nan]
        assert_element_equal(rolled, Curve((self.dates, rolled_vals)))

    @scipy_skip
    def test_roll_ints_with_window_type(self):
        rolled = rolling(self.int_curve, rolling_window=3, window_type='triang')
        rolled_vals = [np.nan, 2, 3, 4, 5, 6, np.nan]
        assert_element_equal(rolled, Curve(rolled_vals))

    def test_resample_weekly(self):
        resampled = resample(self.date_curve, rule='W')
        dates = list(map(pd.Timestamp, ["2016-01-03", "2016-01-10"]))
        vals = [2, 5.5]
        assert_element_equal(resampled, Curve((dates, vals)))

    def test_resample_weekly_closed_left(self):
        resampled = resample(self.date_curve, rule='W', closed='left')
        dates = list(map(pd.Timestamp, ["2016-01-03", "2016-01-10"]))
        vals = [1.5, 5]
        assert_element_equal(resampled, Curve((dates, vals)))

    def test_resample_weekly_label_left(self):
        resampled = resample(self.date_curve, rule='W', label='left')
        dates = list(map(pd.Timestamp, ["2015-12-27", "2016-01-03"]))
        vals = [2, 5.5]
        assert_element_equal(resampled, Curve((dates, vals)))

    def test_rolling_outliers_std_ints(self):
        outliers = rolling_outlier_std(self.int_outliers, rolling_window=2, sigma=1)
        assert_element_equal(outliers, Scatter([(4, 10)]))

    def test_rolling_outliers_std_dates(self):
        outliers = rolling_outlier_std(self.date_outliers, rolling_window=2, sigma=1)
        assert_element_equal(outliers, Scatter([(pd.Timestamp("2016-01-05"), 10)]))
