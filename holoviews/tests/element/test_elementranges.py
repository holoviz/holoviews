"""
Test cases for computing ranges on elements which are not simply
the (min, max) of the dimension values array.
"""
import holoviews as hv


class HistogramRangeTests:

    def test_histogram_range_x(self):
        r = hv.Histogram(([0, 1, 2, 3], [1, 2, 3])).range(0)
        assert r == (0., 3.0)

    def test_histogram_range_x_explicit(self):
        r = hv.Histogram(([0, 1, 2, 3], [1, 2, 3]),
                      kdims=[hv.Dimension('x', range=(-1, 4.))]).range(0)
        assert r == (-1., 4.)

    def test_histogram_range_x_explicit_upper(self):
        r = hv.Histogram(([0, 1, 2, 3], [1, 2, 3]),
                      kdims=[hv.Dimension('x', range=(None, 4.))]).range(0)
        assert r == (0, 4.)

    def test_histogram_range_x_explicit_lower(self):
        r = hv.Histogram(([0, 1, 2, 3], [1, 2, 3]),
                      kdims=[hv.Dimension('x', range=(-1, None))]).range(0)
        assert r == (-1., 3.)

    def test_histogram_range_y(self):
        r = hv.Histogram(([0, 1, 2, 3], [1, 2, 3])).range(1)
        assert r == (1., 3.0)

    def test_histogram_range_y_explicit(self):
        r = hv.Histogram(([0, 1, 2, 3], [1, 2, 3]),
                      vdims=[hv.Dimension('y', range=(0, 4.))]).range(1)
        assert r == (0., 4.)

    def test_histogram_range_y_explicit_upper(self):
        r = hv.Histogram(([0, 1, 2, 3], [1, 2, 3]),
                      vdims=[hv.Dimension('y', range=(None, 4.))]).range(1)
        assert r == (1., 4.)

    def test_histogram_range_y_explicit_lower(self):
        r = hv.Histogram(([0, 1, 2, 3], [1, 2, 3]),
                      vdims=[hv.Dimension('y', range=(0., None))]).range(1)
        assert r == (0., 3.)



class ErrorBarsRangeTests:

    def test_errorbars_range_x(self):
        r = hv.ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5])).range(0)
        assert r == (1., 3.0)

    def test_errorbars_range_x_explicit(self):
        r = hv.ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5]),
                      kdims=[hv.Dimension('x', range=(-1, 4.))]).range(0)
        assert r == (-1., 4.)

    def test_errorbars_range_x_explicit_upper(self):
        r = hv.ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5]),
                      kdims=[hv.Dimension('x', range=(None, 4.))]).range(0)
        assert r == (1, 4.)

    def test_errorbars_range_x_explicit_lower(self):
        r = hv.ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5]),
                      kdims=[hv.Dimension('x', range=(-1, None))]).range(0)
        assert r == (-1., 3.)

    def test_errorbars_range_y(self):
        r = hv.ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5])).range(1)
        assert r == (1.5, 4.5)

    def test_errorbars_range_y_explicit(self):
        r = hv.ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5]),
                      vdims=[hv.Dimension('y', range=(0, 4.)), 'yerr']).range(1)
        assert r == (0., 4.)

    def test_errorbars_range_y_explicit_upper(self):
        r = hv.ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5]),
                      vdims=[hv.Dimension('y', range=(None, 4.)), 'yerr']).range(1)
        assert r == (1.5, 4.)

    def test_errorbars_range_y_explicit_lower(self):
        r = hv.ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5]),
                      vdims=[hv.Dimension('y', range=(0., None)), 'yerr']).range(1)
        assert r == (0., 4.5)

    def test_errorbars_range_horizontal(self):
        r = hv.ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5]),
                      horizontal=True).range(0)
        assert r == (0.5, 3.5)

    def test_errorbars_range_explicit_horizontal(self):
        r = hv.ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5]),
                      kdims=[hv.Dimension('x', range=(-1, 4.))],
                      vdims=['y', 'xerr'],
                      horizontal=True).range(0)
        assert r == (-1., 4.)
