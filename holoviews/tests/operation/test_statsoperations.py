import numpy as np

import holoviews as hv
from holoviews.operation.stats import bivariate_kde, univariate_kde
from holoviews.testing import assert_element_equal

from ..utils import optional_dependencies

scipy, scipy_skip = optional_dependencies("scipy")


@scipy_skip
class KDEOperationTests:
    """
    Tests for the various timeseries operations including rolling,
    resample and rolling_outliers_std.
    """

    def setup_method(self):
        self.values = np.arange(4)
        self.dist = hv.Distribution(self.values)
        self.nans = np.full(5, np.nan)
        self.values2d = [(i, j) for i in np.linspace(0, 4, 10) for j in np.linspace(0, 4, 10)]
        self.bivariate = hv.Bivariate(self.values2d)
        self.dist_nans = hv.Distribution(self.nans)
        self.bivariate_nans = hv.Bivariate(np.column_stack([self.nans, self.nans]))

    def test_univariate_kde(self):
        kde = univariate_kde(self.dist, n_samples=5, bin_range=(0, 4))
        xs = np.arange(5)
        ys = [0.17594505, 0.23548218, 0.23548218, 0.17594505, 0.0740306]
        area = hv.Area((xs, ys), "Value", ("Value_density", "Density"))
        assert_element_equal(kde, area)

    def test_univariate_kde_flat_distribution(self):
        dist = hv.Distribution([1, 1, 1])
        kde = univariate_kde(dist, n_samples=5, bin_range=(0, 4))
        area = hv.Area([], "Value", ("Value_density", "Density"))
        assert_element_equal(kde, area)

    def test_univariate_kde_nans(self):
        kde = univariate_kde(self.dist_nans, n_samples=5, bin_range=(0, 4))
        xs = np.arange(5)
        ys = [0, 0, 0, 0, 0]
        area = hv.Area((xs, ys), "Value", ("Value_density", "Density"))
        assert_element_equal(kde, area)

    def test_bivariate_kde(self):
        kde = bivariate_kde(
            self.bivariate, n_samples=2, x_range=(0, 4), y_range=(0, 4), contours=False
        )
        img = hv.Image(
            np.array([[0.021315, 0.021315], [0.021315, 0.021315]]),
            bounds=(-2, -2, 6, 6),
            vdims=["Density"],
        )
        assert_element_equal(kde, img)

    def test_bivariate_kde_contours(self):
        np.random.seed(1)
        bivariate = hv.Bivariate(np.random.rand(100, 2))
        kde = bivariate_kde(
            bivariate, n_samples=100, x_range=(0, 1), y_range=(0, 1), contours=True, levels=10
        )
        assert isinstance(kde, hv.Contours)
        assert len(kde.data) == 9

    def test_bivariate_kde_contours_filled(self):
        np.random.seed(1)
        bivariate = hv.Bivariate(np.random.rand(100, 2))
        kde = bivariate_kde(
            bivariate,
            n_samples=100,
            x_range=(0, 1),
            y_range=(0, 1),
            contours=True,
            filled=True,
            levels=10,
        )
        assert isinstance(kde, hv.Polygons)
        assert len(kde.data) == 10

    def test_bivariate_kde_nans(self):
        kde = bivariate_kde(
            self.bivariate_nans, n_samples=2, x_range=(0, 4), y_range=(0, 4), contours=False
        )
        img = hv.Image(np.zeros((2, 2)), bounds=(-2, -2, 6, 6), vdims=["Density"])
        assert_element_equal(kde, img)
