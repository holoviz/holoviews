"""
Test cases for computing ranges on elements which are not simply
the (min, max) of the dimension values array.
"""
from holoviews import Dimension, ErrorBars, Histogram
from holoviews.element.comparison import ComparisonTestCase


class HistogramRangeTests(ComparisonTestCase):

    def test_histogram_range_x(self):
        r = Histogram(([0, 1, 2, 3], [1, 2, 3])).range(0)
        self.assertEqual(r, (0., 3.0))

    def test_histogram_range_x_explicit(self):
        r = Histogram(([0, 1, 2, 3], [1, 2, 3]),
                      kdims=[Dimension('x', range=(-1, 4.))]).range(0)
        self.assertEqual(r, (-1., 4.))

    def test_histogram_range_x_explicit_upper(self):
        r = Histogram(([0, 1, 2, 3], [1, 2, 3]),
                      kdims=[Dimension('x', range=(None, 4.))]).range(0)
        self.assertEqual(r, (0, 4.))

    def test_histogram_range_x_explicit_lower(self):
        r = Histogram(([0, 1, 2, 3], [1, 2, 3]),
                      kdims=[Dimension('x', range=(-1, None))]).range(0)
        self.assertEqual(r, (-1., 3.))

    def test_histogram_range_y(self):
        r = Histogram(([0, 1, 2, 3], [1, 2, 3])).range(1)
        self.assertEqual(r, (1., 3.0))

    def test_histogram_range_y_explicit(self):
        r = Histogram(([0, 1, 2, 3], [1, 2, 3]),
                      vdims=[Dimension('y', range=(0, 4.))]).range(1)
        self.assertEqual(r, (0., 4.))

    def test_histogram_range_y_explicit_upper(self):
        r = Histogram(([0, 1, 2, 3], [1, 2, 3]),
                      vdims=[Dimension('y', range=(None, 4.))]).range(1)
        self.assertEqual(r, (1., 4.))

    def test_histogram_range_y_explicit_lower(self):
        r = Histogram(([0, 1, 2, 3], [1, 2, 3]),
                      vdims=[Dimension('y', range=(0., None))]).range(1)
        self.assertEqual(r, (0., 3.))



class ErrorBarsRangeTests(ComparisonTestCase):

    def test_errorbars_range_x(self):
        r = ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5])).range(0)
        self.assertEqual(r, (1., 3.0))

    def test_errorbars_range_x_explicit(self):
        r = ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5]),
                      kdims=[Dimension('x', range=(-1, 4.))]).range(0)
        self.assertEqual(r, (-1., 4.))

    def test_errorbars_range_x_explicit_upper(self):
        r = ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5]),
                      kdims=[Dimension('x', range=(None, 4.))]).range(0)
        self.assertEqual(r, (1, 4.))

    def test_errorbars_range_x_explicit_lower(self):
        r = ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5]),
                      kdims=[Dimension('x', range=(-1, None))]).range(0)
        self.assertEqual(r, (-1., 3.))

    def test_errorbars_range_y(self):
        r = ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5])).range(1)
        self.assertEqual(r, (1.5, 4.5))

    def test_errorbars_range_y_explicit(self):
        r = ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5]),
                      vdims=[Dimension('y', range=(0, 4.)), 'yerr']).range(1)
        self.assertEqual(r, (0., 4.))

    def test_errorbars_range_y_explicit_upper(self):
        r = ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5]),
                      vdims=[Dimension('y', range=(None, 4.)), 'yerr']).range(1)
        self.assertEqual(r, (1.5, 4.))

    def test_errorbars_range_y_explicit_lower(self):
        r = ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5]),
                      vdims=[Dimension('y', range=(0., None)), 'yerr']).range(1)
        self.assertEqual(r, (0., 4.5))

    def test_errorbars_range_horizontal(self):
        r = ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5]),
                      horizontal=True).range(0)
        self.assertEqual(r, (0.5, 3.5))

    def test_errorbars_range_explicit_horizontal(self):
        r = ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5]),
                      kdims=[Dimension('x', range=(-1, 4.))],
                      vdims=['y', 'xerr'],
                      horizontal=True).range(0)
        self.assertEqual(r, (-1., 4.))
