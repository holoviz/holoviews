"""
Test cases for both indexing and slicing of elements
"""
import numpy as np
from holoviews import Histogram, QuadMesh
from holoviews.element.comparison import ComparisonTestCase


class HistogramIndexingTest(ComparisonTestCase):

    def setUp(self):
        self.values = np.arange(10)
        self.edges =  np.arange(11)
        self.hist=Histogram((self.edges, self.values))

    def test_slice_all(self):
        sliced = self.hist[:]
        values = sliced.dimension_values(1)
        edges = sliced.interface.coords(sliced, sliced.kdims[0], edges=True)
        self.assertEqual(values, self.values)
        self.assertEqual(edges, self.edges)

    def test_slice_exclusive_upper(self):
        "Exclusive upper boundary semantics for bin centers"
        sliced = self.hist[:6.5]
        values = sliced.dimension_values(1)
        edges = sliced.interface.coords(sliced, sliced.kdims[0], edges=True)
        self.assertEqual(values, np.arange(6))
        self.assertEqual(edges, np.arange(7))

    def test_slice_exclusive_upper_exceeded(self):
        "Slightly above the boundary in the previous test"
        sliced = self.hist[:6.55]
        values = sliced.dimension_values(1)
        edges = sliced.interface.coords(sliced, sliced.kdims[0], edges=True)
        self.assertEqual(values, np.arange(7))
        self.assertEqual(edges, np.arange(8))

    def test_slice_inclusive_lower(self):
        "Inclusive lower boundary semantics for bin centers"
        sliced = self.hist[3.5:]
        values = sliced.dimension_values(1)
        edges = sliced.interface.coords(sliced, sliced.kdims[0], edges=True)
        self.assertEqual(values, np.arange(3, 10))
        self.assertEqual(edges, np.arange(3, 11))

    def test_slice_inclusive_lower_undershot(self):
        "Inclusive lower boundary semantics for bin centers"
        sliced = self.hist[3.45:]
        values = sliced.dimension_values(1)
        edges = sliced.interface.coords(sliced, sliced.kdims[0], edges=True)
        self.assertEqual(values, np.arange(3, 10))
        self.assertEqual(edges, np.arange(3, 11))

    def test_slice_bounded(self):
        sliced = self.hist[3.5:6.5]
        values = sliced.dimension_values(1)
        edges = sliced.interface.coords(sliced, sliced.kdims[0], edges=True)
        self.assertEqual(values, np.arange(3, 6))
        self.assertEqual(edges, np.arange(3, 7))

    def test_slice_lower_out_of_bounds(self):
        sliced = self.hist[-3:]
        values = sliced.dimension_values(1)
        edges = sliced.interface.coords(sliced, sliced.kdims[0], edges=True)
        self.assertEqual(values, self.values)
        self.assertEqual(edges, self.edges)

    def test_slice_upper_out_of_bounds(self):
        sliced = self.hist[:12]
        values = sliced.dimension_values(1)
        edges = sliced.interface.coords(sliced, sliced.kdims[0], edges=True)
        self.assertEqual(values, self.values)
        self.assertEqual(edges, self.edges)

    def test_slice_both_out_of_bounds(self):
        sliced = self.hist[-3:13]
        values = sliced.dimension_values(1)
        edges = sliced.interface.coords(sliced, sliced.kdims[0], edges=True)
        self.assertEqual(values, self.values)
        self.assertEqual(edges, self.edges)

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
        with self.assertRaises(IndexError):
            self.hist[-1]

    def test_scalar_highest_index_out_of_bounds(self):
        with self.assertRaises(IndexError):
            self.hist[10]

class QuadMeshIndexingTest(ComparisonTestCase):


    def setUp(self):
        n = 4
        self.xs = np.logspace(1, 3, n)
        self.ys = np.linspace(1, 10, n)
        self.zs = np.arange((n-1)**2).reshape(n-1, n-1)
        self.qmesh = QuadMesh((self.xs, self.ys, self.zs))

    def test_qmesh_index_lower_left(self):
        self.assertEqual(self.qmesh[10, 1], 0)

    def test_qmesh_index_lower_right(self):
        self.assertEqual(self.qmesh[800, 3.9], 2)

    def test_qmesh_index_top_left(self):
        self.assertEqual(self.qmesh[10, 9.9], 6)

    def test_qmesh_index_top_right(self):
        self.assertEqual(self.qmesh[216, 7], 8)

    def test_qmesh_index_xcoords(self):
        sliced = QuadMesh((self.xs[2:4], self.ys, self.zs[:, 2:3]))
        self.assertEqual(self.qmesh[300, :], sliced)

    def test_qmesh_index_ycoords(self):
        sliced = QuadMesh((self.xs, self.ys[-2:], self.zs[-1:, :]))
        self.assertEqual(self.qmesh[:, 7], sliced)

    def test_qmesh_slice_xcoords(self):
        sliced = QuadMesh((self.xs[1:], self.ys, self.zs[:, 1:]))
        self.assertEqual(self.qmesh[100:1000, :], sliced)

    def test_qmesh_slice_ycoords(self):
        sliced = QuadMesh((self.xs, self.ys[:-1], self.zs[:-1, :]))
        self.assertEqual(self.qmesh[:, 2:7], sliced)

    def test_qmesh_slice_xcoords_ycoords(self):
        sliced = QuadMesh((self.xs[1:], self.ys[:-1], self.zs[:-1, 1:]))
        self.assertEqual(self.qmesh[100:1000, 2:7], sliced)
