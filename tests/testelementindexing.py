"""
Test cases for both indexing and slicing of elements
"""
import numpy as np
from holoviews import Histogram
from holoviews.element.comparison import ComparisonTestCase


class HistogramIndexingTest(ComparisonTestCase):

    def setUp(self):
        self.values = [i for i in range(10)]
        self.edges =  [i for i in range(11)]
        self.hist=Histogram(self.values, self.edges)

    def test_slice_all(self):
        sliced = self.hist[:]
        self.assertEqual(np.all(sliced.values == self.values), True)
        self.assertEqual(np.all(sliced.edges == self.edges), True)

    def test_slice_exclusive_upper(self):
        "Exclusive upper boundary semantics for bin centers"
        sliced = self.hist[:6.5]
        self.assertEqual(np.all(sliced.values == [0, 1, 2, 3, 4, 5]), True)
        self.assertEqual(np.all(sliced.edges == [0, 1, 2, 3, 4, 5, 6]), True)

    def test_slice_exclusive_upper_exceeded(self):
        "Slightly above the boundary in the previous test"
        sliced = self.hist[:6.55]
        self.assertEqual(np.all(sliced.values == [0, 1, 2, 3, 4, 5, 6]), True)
        self.assertEqual(np.all(sliced.edges == [0, 1, 2, 3, 4, 5, 6, 7]), True)

    def test_slice_inclusive_lower(self):
        "Inclusive lower boundary semantics for bin centers"
        sliced = self.hist[3.5:]
        self.assertEqual(np.all(sliced.values == [3, 4, 5, 6, 7, 8, 9]), True)
        self.assertEqual(np.all(sliced.edges == [3, 4, 5, 6, 7, 8, 9, 10]), True)

    def test_slice_inclusive_lower_undershot(self):
        "Inclusive lower boundary semantics for bin centers"
        sliced = self.hist[3.45:]
        self.assertEqual(np.all(sliced.values == [3, 4, 5, 6, 7, 8, 9]), True)
        self.assertEqual(np.all(sliced.edges == [3, 4, 5, 6, 7, 8, 9, 10]), True)

    def test_slice_bounded(self):
        sliced = self.hist[3.5:6.5]
        self.assertEqual(np.all(sliced.values == [3, 4, 5]), True)
        self.assertEqual(np.all(sliced.edges == [3, 4, 5, 6]), True)

    def test_slice_lower_out_of_bounds(self):
        sliced = self.hist[-3:]
        self.assertEqual(np.all(sliced.values == self.values), True)
        self.assertEqual(np.all(sliced.edges == self.edges), True)

    def test_slice_upper_out_of_bounds(self):
        sliced = self.hist[:12]
        self.assertEqual(np.all(sliced.values == self.values), True)
        self.assertEqual(np.all(sliced.edges == self.edges), True)

    def test_slice_both_out_of_bounds(self):
        sliced = self.hist[-3:13]
        self.assertEqual(np.all(sliced.values == self.values), True)
        self.assertEqual(np.all(sliced.edges == self.edges), True)

    def test_scalar_index(self):
        self.assertEqual(self.hist[4.5], 4)
        self.assertEqual(self.hist[3.7], 3)
        self.assertEqual(self.hist[9.9], 9)

    def test_scalar_index_boundary(self):
        """
        Scalar at boundary indexes next bin.
        (exclusive upper boundary for current bin)
        """
        self.assertEqual(self.hist[4], 4)
        self.assertEqual(self.hist[5], 5)

    def test_scalar_lowest_index(self):
        self.assertEqual(self.hist[0], 0)

    def test_scalar_lowest_index_out_of_bounds(self):
        try:
            self.hist[-0.1]
        except Exception as e:
            if not str(e).startswith("'Key value -0.1 is out of the histogram bounds"):
                raise AssertionError("Out of bound exception not generated")

    def test_scalar_highest_index_out_of_bounds(self):
        try:
            self.hist[10]
        except Exception as e:
            if not str(e).startswith("'Key value 10 is out of the histogram bounds"):
                raise AssertionError("Out of bound exception not generated")
