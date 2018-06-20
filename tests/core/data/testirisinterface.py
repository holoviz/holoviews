from nose.plugins.attrib import attr
from unittest import SkipTest

import numpy as np

try:
    import iris
    from iris.tests.stock import lat_lon_cube
    from iris.exceptions import MergeError
except ImportError:
    raise SkipTest("Could not import iris, skipping IrisInterface tests.")

from holoviews.core.data import Dataset, concat
from holoviews.core.data.iris import coord_to_dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Image

from .testimageinterface import Image_ImageInterfaceTests
from .testgridinterface import GridInterfaceTests


@attr(optional=1)
class IrisInterfaceTests(GridInterfaceTests):
    """
    Tests for Iris interface
    """

    datatype = 'cube'
    data_type = iris.cube.Cube

    def init_data(self):
        self.cube = lat_lon_cube()
        self.epsilon = 0.01

    def test_concat_grid_3d_shape_mismatch(self):
        arr1 = np.random.rand(3, 2)
        arr2 = np.random.rand(2, 3)
        ds1 = Dataset(([0, 1], [1, 2, 3], arr1), ['x', 'y'], 'z')
        ds2 = Dataset(([0, 1, 2], [1, 2], arr2), ['x', 'y'], 'z')
        hmap = HoloMap({1: ds1, 2: ds2})
        with self.assertRaises(MergeError):
            concat(hmap)

    def test_dataset_array_init_hm(self):
        "Tests support for arrays (homogeneous)"
        raise SkipTest("Not supported")

    # Disabled tests for NotImplemented methods
    def test_dataset_add_dimensions_values_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_add_dimensions_values_hm_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_reverse_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_vdim_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_reverse_vdim_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_vdim_hm_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_1D_reduce_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_1D_reduce_hm_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_2D_reduce_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_2D_reduce_hm_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_2D_aggregate_partial_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_2D_aggregate_partial_hm_alias(self):
        raise SkipTest("Not supported")

    def test_aggregate_2d_with_spreadfn(self):
        raise SkipTest("Not supported")

    def test_dataset_sample_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sample_hm_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_groupby_drop_dims_with_vdim(self):
        raise SkipTest("Not supported")

    def test_dataset_groupby_drop_dims_dynamic_with_vdim(self):
        raise SkipTest("Not supported")

    def test_dataset_ndloc_slice_two_vdims(self):
        raise SkipTest("Not supported")

    def test_dim_to_coord(self):
        dim = coord_to_dimension(self.cube.coords()[0])
        self.assertEqual(dim.name, 'latitude')
        self.assertEqual(dim.unit, 'degrees')

    def test_initialize_cube(self):
        cube = Dataset(self.cube)
        self.assertEqual(cube.dimensions(label=True),
                         ['longitude', 'latitude', 'unknown'])

    def test_initialize_cube_with_kdims(self):
        cube = Dataset(self.cube, kdims=['longitude', 'latitude'])
        self.assertEqual(cube.dimensions('key', True),
                         ['longitude', 'latitude'])

    def test_initialize_cube_with_vdims(self):
        cube = Dataset(self.cube, vdims=['Quantity'])
        self.assertEqual(cube.dimensions('value', True),
                         ['Quantity'])

    def test_dimension_values_kdim_expanded(self):
        cube = Dataset(self.cube, kdims=['longitude', 'latitude'])
        self.assertEqual(cube.dimension_values('longitude'),
                         np.array([-1, -1, -1, 0,  0,  0,
                                   1,  1,  1, 2,  2,  2], dtype=np.int32))

    def test_dimension_values_kdim(self):
        cube = Dataset(self.cube, kdims=['longitude', 'latitude'])
        self.assertEqual(cube.dimension_values('longitude', expanded=False),
                         np.array([-1,  0,  1, 2], dtype=np.int32))

    def test_dimension_values_vdim(self):
        cube = Dataset(self.cube, kdims=['longitude', 'latitude'])
        self.assertEqual(cube.dimension_values('unknown', flat=False),
                         np.array([[ 0,  4,  8],
                                   [ 1,  5,  9],
                                   [ 2,  6, 10],
                                   [ 3,  7, 11]], dtype=np.int32).T)

    def test_range_kdim(self):
        cube = Dataset(self.cube, kdims=['longitude', 'latitude'])
        self.assertEqual(cube.range('longitude'), (-1, 2))

    def test_range_vdim(self):
        cube = Dataset(self.cube, kdims=['longitude', 'latitude'])
        self.assertEqual(cube.range('unknown'), (0, 11))

    def test_select_index(self):
        cube = Dataset(self.cube)
        self.assertEqual(cube.select(longitude=0).data.data,
                         np.array([[1, 5, 9]], dtype=np.int32))

    def test_select_slice(self):
        cube = Dataset(self.cube)
        self.assertEqual(cube.select(longitude=(0, 1.01)).data.data,
                         np.array([[1,  2], [5,  6], [9, 10]], dtype=np.int32))

    def test_select_set(self):
        cube = Dataset(self.cube)
        self.assertEqual(cube.select(longitude={0, 1}).data.data,
                         np.array([[1,  2], [5,  6], [9, 10]], dtype=np.int32))

    def test_select_multi_index(self):
        cube = Dataset(self.cube)
        self.assertEqual(cube.select(longitude=0, latitude=0), 5)

    def test_select_multi_slice1(self):
        cube = Dataset(self.cube)
        self.assertEqual(cube.select(longitude=(0, 1.01),
                                     latitude=(0, 1.01)).data.data,
                         np.array([[5,  6], [9, 10]], dtype=np.int32))

    def test_select_multi_slice2(self):
        cube = Dataset(self.cube)
        self.assertEqual(cube.select(longitude={0, 2},
                                     latitude={0, 2}).data.data,
                         np.array([[5, 7]], dtype=np.int32))

    def test_getitem_index(self):
        cube = Dataset(self.cube)
        self.assertEqual(cube[0].data.data,
                         np.array([[1, 5, 9]], dtype=np.int32))

    def test_getitem_scalar(self):
        cube = Dataset(self.cube)
        self.assertEqual(cube[0, 0], 5)


@attr(optional=1)
class Image_IrisInterfaceTests(Image_ImageInterfaceTests):

    datatype = 'cube'

    def init_data(self):
        xs = np.linspace(-9, 9, 10)
        ys = np.linspace(0.5, 9.5, 10)
        self.xs = xs
        self.ys = ys
        self.array = np.arange(10) * np.arange(10)[:, np.newaxis]
        self.image = Image((xs, ys, self.array))
        self.image_inv = Image((xs[::-1], ys[::-1], self.array[::-1, ::-1]))

    def test_init_data_datetime_xaxis(self):
        raise SkipTest("Not supported")

    def test_init_data_datetime_yaxis(self):
        raise SkipTest("Not supported")

    def test_init_bounds_datetime_xaxis(self):
        raise SkipTest("Not supported")

    def test_init_bounds_datetime_yaxis(self):
        raise SkipTest("Not supported")

    def test_init_densities_datetime_xaxis(self):
        raise SkipTest("Not supported")

    def test_init_densities_datetime_yaxis(self):
        raise SkipTest("Not supported")

    def test_range_datetime_xdim(self):
        raise SkipTest("Not supported")

    def test_range_datetime_ydim(self):
        raise SkipTest("Not supported")

    def test_dimension_values_datetime_xcoords(self):
        raise SkipTest("Not supported")

    def test_dimension_values_datetime_ycoords(self):
        raise SkipTest("Not supported")

    def test_slice_datetime_xaxis(self):
        raise SkipTest("Not supported")

    def test_slice_datetime_yaxis(self):
        raise SkipTest("Not supported")

    def test_reduce_to_scalar(self):
        raise SkipTest("Not supported")

    def test_reduce_x_dimension(self):
        raise SkipTest("Not supported")

    def test_reduce_y_dimension(self):
        raise SkipTest("Not supported")

    def test_aggregate_with_spreadfn(self):
        raise SkipTest("Not supported")

    def test_sample_datetime_xaxis(self):
        raise SkipTest("Not supported")

    def test_sample_datetime_yaxis(self):
        raise SkipTest("Not supported")
