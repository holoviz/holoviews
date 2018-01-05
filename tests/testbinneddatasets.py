"""
Tests for binned interfaces including GridInterface and XArrayInterface
"""

from unittest import SkipTest

import numpy as np

from holoviews.core.spaces import HoloMap
from holoviews.core.data import Dataset
from holoviews.element import Histogram, QuadMesh
from holoviews.element.comparison import ComparisonTestCase


class Binned1DTest(ComparisonTestCase):

    def setUp(self):
        self.values = np.arange(10)
        self.edges =  np.arange(11)
        self.dataset1d = Histogram((self.edges, self.values))

    def test_slice_all(self):
        sliced = self.dataset1d[:]
        self.assertEqual(sliced.values, self.values)
        self.assertEqual(sliced.edges, self.edges)

    def test_slice_exclusive_upper(self):
        "Exclusive upper boundary semantics for bin centers"
        sliced = self.dataset1d[:6.5]
        self.assertEqual(sliced.values, np.arange(6))
        self.assertEqual(sliced.edges, np.arange(7))

    def test_slice_exclusive_upper_exceeded(self):
        "Slightly above the boundary in the previous test"
        sliced = self.dataset1d[:6.55]
        self.assertEqual(sliced.values, np.arange(7))
        self.assertEqual(sliced.edges, np.arange(8))

    def test_slice_inclusive_lower(self):
        "Inclusive lower boundary semantics for bin centers"
        sliced = self.dataset1d[3.5:]
        self.assertEqual(sliced.values, np.arange(3, 10))
        self.assertEqual(sliced.edges, np.arange(3, 11))

    def test_slice_inclusive_lower_undershot(self):
        "Inclusive lower boundary semantics for bin centers"
        sliced = self.dataset1d[3.45:]
        self.assertEqual(sliced.values, np.arange(3, 10))
        self.assertEqual(sliced.edges, np.arange(3, 11))

    def test_slice_bounded(self):
        sliced = self.dataset1d[3.5:6.5]
        self.assertEqual(sliced.values, np.arange(3, 6))
        self.assertEqual(sliced.edges, np.arange(3, 7))

    def test_slice_lower_out_of_bounds(self):
        sliced = self.dataset1d[-3:]
        self.assertEqual(sliced.values, self.values)
        self.assertEqual(sliced.edges, self.edges)

    def test_slice_upper_out_of_bounds(self):
        sliced = self.dataset1d[:12]
        self.assertEqual(sliced.values, self.values)
        self.assertEqual(sliced.edges, self.edges)

    def test_slice_both_out_of_bounds(self):
        sliced = self.dataset1d[-3:13]
        self.assertEqual(sliced.values, self.values)
        self.assertEqual(sliced.edges, self.edges)

    def test_scalar_index(self):
        self.assertEqual(self.dataset1d[4.5], 4)
        self.assertEqual(self.dataset1d[3.7], 3)
        self.assertEqual(self.dataset1d[9.9], 9)

    def test_scalar_index_boundary(self):
        """
        Scalar at boundary indexes next bin.
        (exclusive upper boundary for current bin)
        """
        self.assertEqual(self.dataset1d[4], 4)
        self.assertEqual(self.dataset1d[5], 5)

    def test_scalar_lowest_index(self):
        self.assertEqual(self.dataset1d[0], 0)

    def test_scalar_lowest_index_out_of_bounds(self):
        with self.assertRaises(IndexError):
            self.dataset1d[-1]

    def test_scalar_highest_index_out_of_bounds(self):
        with self.assertRaises(IndexError):
            self.dataset1d[10]

    def test_groupby_kdim(self):
        grouped = self.dataset1d.groupby('x', group_type=Dataset)
        holomap = HoloMap({self.edges[i:i+2].mean(): Dataset([(i,)], vdims=['Frequency'])
                           for i in range(10)}, kdims=['x'])
        self.assertEqual(grouped, holomap)


class Binned2DTest(ComparisonTestCase):

    def setUp(self):
        n = 4
        self.xs = np.logspace(1, 3, n)
        self.ys = np.linspace(1, 10, n)
        self.zs = np.arange((n-1)**2).reshape(n-1, n-1)
        self.dataset2d = QuadMesh((self.xs, self.ys, self.zs))

    def test_qmesh_index_lower_left(self):
        self.assertEqual(self.dataset2d[10, 1], 0)

    def test_qmesh_index_lower_right(self):
        self.assertEqual(self.dataset2d[800, 3.9], 2)

    def test_qmesh_index_top_left(self):
        self.assertEqual(self.dataset2d[10, 9.9], 6)

    def test_qmesh_index_top_right(self):
        self.assertEqual(self.dataset2d[216, 7], 8)

    def test_qmesh_index_xcoords(self):
        sliced = QuadMesh((self.xs[2:4], self.ys, self.zs[:, 2:3]))
        self.assertEqual(self.dataset2d[300, :], sliced)

    def test_qmesh_index_ycoords(self):
        sliced = QuadMesh((self.xs, self.ys[-2:], self.zs[-1:, :]))
        self.assertEqual(self.dataset2d[:, 7], sliced)

    def test_qmesh_slice_xcoords(self):
        sliced = QuadMesh((self.xs[1:], self.ys, self.zs[:, 1:]))
        self.assertEqual(self.dataset2d[100:1000, :], sliced)

    def test_qmesh_slice_ycoords(self):
        sliced = QuadMesh((self.xs, self.ys[:-1], self.zs[:-1, :]))
        self.assertEqual(self.dataset2d[:, 2:7], sliced)

    def test_qmesh_slice_xcoords_ycoords(self):
        sliced = QuadMesh((self.xs[1:], self.ys[:-1], self.zs[:-1, 1:]))
        self.assertEqual(self.dataset2d[100:1000, 2:7], sliced)

    def test_groupby_xdim(self):
        grouped = self.dataset2d.groupby('x', group_type=Dataset)
        holomap = HoloMap({self.xs[i:i+2].mean(): Dataset((self.ys, self.zs[:, i]), 'y', 'z')
                           for i in range(3)}, kdims=['x'])
        self.assertEqual(grouped, holomap)

    def test_groupby_ydim(self):
        grouped = self.dataset2d.groupby('y', group_type=Dataset)
        holomap = HoloMap({self.ys[i:i+2].mean(): Dataset((self.xs, self.zs[i]), 'x', 'z')
                           for i in range(3)}, kdims=['y'])
        self.assertEqual(grouped, holomap)



class Irregular2DBinsTest(ComparisonTestCase):

    def setUp(self):
        lon, lat = np.meshgrid(np.linspace(-20, 20, 6), np.linspace(0, 30, 4))
        lon += lat/10
        lat += lon/10
        self.xs = lon
        self.ys = lat
        self.zs = np.arange(24).reshape(4, 6)

    def test_construct_from_dict(self):
        dataset = Dataset((self.xs, self.ys, self.zs), ['x', 'y'], 'z')
        self.assertEqual(dataset.dimension_values('x'), self.xs.T.flatten())
        self.assertEqual(dataset.dimension_values('y'), self.ys.T.flatten())
        self.assertEqual(dataset.dimension_values('z'), self.zs.T.flatten())

    def test_construct_from_xarray(self):
        try:
            import xarray as xr
        except:
            raise SkipError("Test requires xarray")
        da = xr.DataArray(self.zs, dims=['y', 'x'],
                          coords = {'lat': (('y', 'x'), self.ys),
                                    'lon': (('y', 'x'), self.xs)}, name='z')
        dataset = Dataset(da, ['lon', 'lat'], 'z')
        self.assertEqual(dataset.dimension_values('lon'), self.xs.T.flatten())
        self.assertEqual(dataset.dimension_values('lat'), self.ys.T.flatten())
        self.assertEqual(dataset.dimension_values('z'), self.zs.T.flatten())

    def test_construct_3d_from_xarray(self):
        try:
            import xarray as xr
        except:
            raise SkipError("Test requires xarray")
        zs = np.arange(48).reshape(2, 4, 6)
        da = xr.DataArray(zs, dims=['z', 'y', 'x'],
                          coords = {'lat': (('y', 'x'), self.ys),
                                    'lon': (('y', 'x'), self.xs),
                                    'z': [0, 1]}, name='A')
        dataset = Dataset(da, ['lon', 'lat', 'z'], 'A')
        self.assertEqual(dataset.dimension_values('lon'), self.xs.T.flatten())
        self.assertEqual(dataset.dimension_values('lat'), self.ys.T.flatten())
        self.assertEqual(dataset.dimension_values('z', expanded=False), np.array([0, 1]))
        self.assertEqual(dataset.dimension_values('A'), zs.T.flatten())

    def test_groupby_3d_from_xarray(self):
        try:
            import xarray as xr
        except:
            raise SkipError("Test requires xarray")
        zs = np.arange(48).reshape(2, 4, 6)
        da = xr.DataArray(zs, dims=['z', 'y', 'x'],
                          coords = {'lat': (('y', 'x'), self.ys),
                                    'lon': (('y', 'x'), self.xs),
                                    'z': [0, 1]}, name='A')
        grouped = Dataset(da, ['lon', 'lat', 'z'], 'A').groupby('z')
        hmap = HoloMap({0: Dataset((self.xs, self.ys, zs[0]), ['lon', 'lat'], 'A'),
                        1: Dataset((self.xs, self.ys, zs[1]), ['lon', 'lat'], 'A')}, kdims='z')
        self.assertEqual(grouped, hmap)
