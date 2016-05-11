import numpy as np
import unittest

try:
    from iris.tests.stock import lat_lon_cube
except ImportError:
    raise unittest.SkipTest("Could not import iris, skipping iris interface "
                            "tests.")

from holoviews.core.data import Dataset
from holoviews.core.data.iris import coord_to_dimension
from holoviews.element.comparison import ComparisonTestCase

class TestCube(ComparisonTestCase):

    def setUp(self):
        self.cube = lat_lon_cube()

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
                                   [ 3,  7, 11]], dtype=np.int32))

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
        self.assertEqual(cube.select(longitude=(0, 1)).data.data,
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
        self.assertEqual(cube.select(longitude=(0, 1),
                                     latitude=(0, 1)).data.data,
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
