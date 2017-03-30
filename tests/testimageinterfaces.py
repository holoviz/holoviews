from unittest import SkipTest

import numpy as np
from holoviews import Dimension, Image, Curve, RGB, Dataset
from holoviews.element.comparison import ComparisonTestCase

from .testdataset import DatatypeContext


class ImageInterfaceTest(ComparisonTestCase):

    datatype = 'image'

    def setUp(self):
        self.eltype = Image
        self.restore_datatype = self.eltype.datatype
        self.eltype.datatype = [self.datatype]
        self.init_data()

    def init_data(self):
        self.array = np.arange(10) * np.arange(10)[:, np.newaxis]
        self.image = Image(np.flipud(self.array), bounds=(-10, 0, 10, 10))

    def tearDown(self):
        self.eltype.datatype = self.restore_datatype

    def test_init_bounds(self):
        self.assertEqual(self.image.bounds.lbrt(), (-10, 0, 10, 10))

    def test_init_densities(self):
        self.assertEqual(self.image.xdensity, 0.5)
        self.assertEqual(self.image.ydensity, 1)

    def test_dimension_values_xs(self):
        self.assertEqual(self.image.dimension_values(0, expanded=False),
                         np.linspace(-9, 9, 10))

    def test_dimension_values_ys(self):
        self.assertEqual(self.image.dimension_values(1, expanded=False),
                         np.linspace(0.5, 9.5, 10))

    def test_dimension_values_ys(self):
        self.assertEqual(self.image.dimension_values(1, expanded=False),
                         np.linspace(0.5, 9.5, 10))

    def test_dimension_values_vdim(self):
        self.assertEqual(self.image.dimension_values(2, flat=False),
                         self.array)

    def test_index_single_coordinate(self):
        self.assertEqual(self.image[0.3, 5.1], 25)

    def test_slice_xaxis(self):
        sliced = self.image[0.3:5.2]
        self.assertEqual(sliced.bounds.lbrt(), (0, 0, 6, 10))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False),
                         self.array[:, 5:8])

    def test_slice_yaxis(self):
        sliced = self.image[:, 1.2:5.2]
        self.assertEqual(sliced.bounds.lbrt(), (-10, 1., 10, 5))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False),
                         self.array[1:5, :])

    def test_slice_both_axes(self):
        sliced = self.image[0.3:5.2, 1.2:5.2]
        self.assertEqual(sliced.bounds.lbrt(), (0, 1., 6, 5))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False),
                         self.array[1:5, 5:8])

    def test_slice_x_index_y(self):
        sliced = self.image[0.3:5.2, 5.2]
        self.assertEqual(sliced.bounds.lbrt(), (0, 5.0, 6.0, 6.0))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False),
                         self.array[5:6, 5:8])
    
    def test_index_x_slice_y(self):
        sliced = self.image[3.2, 1.2:5.2]
        self.assertEqual(sliced.bounds.lbrt(), (2.0, 1.0, 4.0, 5.0))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False),
                         self.array[1:5, 6:7])

    def test_range_xdim(self):
        self.assertEqual(self.image.range(0), (-10, 10))

    def test_range_ydim(self):
        self.assertEqual(self.image.range(1), (0, 10))

    def test_range_vdim(self):
        self.assertEqual(self.image.range(2), (0, 81))

    def test_dimension_values_xcoords(self):
        self.assertEqual(self.image.dimension_values(0, expanded=False),
                         np.linspace(-9, 9, 10))

    def test_dimension_values_ycoords(self):
        self.assertEqual(self.image.dimension_values(1, expanded=False),
                         np.linspace(0.5, 9.5, 10))

    def test_sample_xcoord(self):
        ys = np.linspace(0.5, 9.5, 10)
        zs = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63]
        with DatatypeContext([self.datatype, 'columns', 'dataframe'], self.image):
            self.assertEqual(self.image.sample(x=5),
                             Curve((ys, zs), kdims=['y'], vdims=['z']))

    def test_sample_ycoord(self):
        xs = np.linspace(-9, 9, 10)
        zs = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
        with DatatypeContext([self.datatype, 'columns', 'dataframe'], self.image):
            self.assertEqual(self.image.sample(y=5),
                             Curve((xs, zs), kdims=['x'], vdims=['z']))

    def test_reduce_to_scalar(self):
        self.assertEqual(self.image.reduce(['x', 'y'], function=np.mean),
                         20.25)

    def test_reduce_x_dimension(self):
        ys = np.linspace(0.5, 9.5, 10)
        zs = [0., 4.5, 9., 13.5, 18., 22.5, 27., 31.5, 36., 40.5]
        with DatatypeContext([self.datatype, 'columns', 'dataframe'], Image):
            self.assertEqual(self.image.reduce(x=np.mean),
                             Curve((ys, zs), kdims=['y'], vdims=['z']))

    def test_reduce_y_dimension(self):
        xs = np.linspace(-9, 9, 10)
        zs = [0., 4.5, 9., 13.5, 18., 22.5, 27., 31.5, 36., 40.5]
        with DatatypeContext([self.datatype, 'columns', 'dataframe'], Image):
            self.assertEqual(self.image.reduce(y=np.mean),
                             Curve((xs, zs), kdims=['x'], vdims=['z']))



class ImageGridInterfaceTest(ImageInterfaceTest):

    datatype = 'grid'

    def init_data(self):
        self.xs = np.linspace(-9, 9, 10)
        self.ys = np.linspace(0.5, 9.5, 10)
        self.array = np.arange(10) * np.arange(10)[:, np.newaxis]
        self.image = Image((self.xs, self.ys, self.array))



class ImageXArrayInterfaceTest(ImageGridInterfaceTest):

    datatype = 'xarray'


class ImageIrisInterfaceTest(ImageGridInterfaceTest):

    datatype = 'cube'

    def init_data(self):
        xs = np.linspace(-9, 9, 10)
        ys = np.linspace(0.5, 9.5, 10)
        self.array = np.arange(10) * np.arange(10)[:, np.newaxis]
        self.image = Image((xs, ys, self.array))

    def test_reduce_to_scalar(self):
        raise SkipTest("Not supported")

    def test_reduce_x_dimension(self):
        raise SkipTest("Not supported")

    def test_reduce_y_dimension(self):
        raise SkipTest("Not supported")


class RGBInterfaceTest(ComparisonTestCase):

    datatype = 'image'

    def setUp(self):
        self.eltype = RGB
        self.restore_datatype = self.eltype.datatype
        self.eltype.datatype = [self.datatype]
        self.init_data()

    def init_data(self):
        self.rgb_array = np.random.rand(10, 10, 3)
        self.rgb = RGB(self.rgb_array[::-1], bounds=(-10, 0, 10, 10))

    def test_init_bounds(self):
        self.assertEqual(self.rgb.bounds.lbrt(), (-10, 0, 10, 10))

    def test_init_densities(self):
        self.assertEqual(self.rgb.xdensity, 0.5)
        self.assertEqual(self.rgb.ydensity, 1)

    def test_dimension_values_xs(self):
        self.assertEqual(self.rgb.dimension_values(0, expanded=False),
                         np.linspace(-9, 9, 10))

    def test_dimension_values_ys(self):
        self.assertEqual(self.rgb.dimension_values(1, expanded=False),
                         np.linspace(0.5, 9.5, 10))

    def test_dimension_values_ys(self):
        self.assertEqual(self.rgb.dimension_values(1, expanded=False),
                         np.linspace(0.5, 9.5, 10))

    def test_dimension_values_vdims(self):
        self.assertEqual(self.rgb.dimension_values(2, flat=False),
                         self.rgb_array[:, :, 0])
        self.assertEqual(self.rgb.dimension_values(3, flat=False),
                         self.rgb_array[:, :, 1])
        self.assertEqual(self.rgb.dimension_values(4, flat=False),
                         self.rgb_array[:, :, 2])

    def test_slice_xaxis(self):
        sliced = self.rgb[0.3:5.2]
        self.assertEqual(sliced.bounds.lbrt(), (0, 0, 6, 10))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False),
                         self.rgb_array[:, 5:8, 0])

    def test_slice_yaxis(self):
        sliced = self.rgb[:, 1.2:5.2]
        self.assertEqual(sliced.bounds.lbrt(), (-10, 1., 10, 5))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False),
                         self.rgb_array[1:5, :, 0])

    def test_slice_both_axes(self):
        sliced = self.rgb[0.3:5.2, 1.2:5.2]
        self.assertEqual(sliced.bounds.lbrt(), (0, 1., 6, 5))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False),
                         self.rgb_array[1:5, 5:8, 0])

    def test_slice_x_index_y(self):
        sliced = self.rgb[0.3:5.2, 5.2]
        self.assertEqual(sliced.bounds.lbrt(), (0, 5.0, 6.0, 6.0))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False),
                         self.rgb_array[5:6, 5:8, 0])

    def test_index_x_slice_y(self):
        sliced = self.rgb[3.2, 1.2:5.2]
        self.assertEqual(sliced.bounds.lbrt(), (2.0, 1.0, 4.0, 5.0))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False),
                         self.rgb_array[1:5, 6:7, 0])

    def test_select_value_dimension_rgb(self):
        self.assertEqual(self.rgb[..., 'R'],
                         Image(np.flipud(self.rgb_array[:, :, 0]), bounds=self.rgb.bounds,
                               vdims=[Dimension('R', range=(0, 1))]))

    def test_select_single_coordinate(self):
        with DatatypeContext([self.datatype, 'columns', 'dataframe'], self.rgb):
            self.assertEqual(self.rgb[5.2, 3.1],
                             self.rgb.clone([tuple(self.rgb_array[3, 7])],
                                            kdims=[], new_type=Dataset))


    def test_reduce_to_single_values(self):
        with DatatypeContext([self.datatype, 'columns', 'dataframe'], self.rgb):
            self.assertEqual(self.rgb.reduce(['x', 'y'], function=np.mean),
                             self.rgb.clone([tuple(np.mean(self.rgb_array, axis=(0, 1)))],
                                            kdims=[], new_type=Dataset))

    def test_sample_xcoord(self):
        ys = np.linspace(0.5, 9.5, 10)
        data = (ys,) + tuple(self.rgb_array[:, 7, i] for i in range(3))
        with DatatypeContext([self.datatype, 'columns', 'dataframe'], self.rgb):
            self.assertEqual(self.rgb.sample(x=5),
                             self.rgb.clone(data, kdims=['y'],
                                            new_type=Curve))

    def test_sample_ycoord(self):
        xs = np.linspace(-9, 9, 10)
        data = (xs,) + tuple(self.rgb_array[4, :, i] for i in range(3))
        with DatatypeContext([self.datatype, 'columns', 'dataframe'], self.rgb):
            self.assertEqual(self.rgb.sample(y=5),
                             self.rgb.clone(data, kdims=['x'],
                                            new_type=Curve))



class RGBGridInterfaceTest(RGBInterfaceTest):

    datatype = 'grid'

    def init_data(self):
        self.xs = np.linspace(-9, 9, 10)
        self.ys = np.linspace(0.5, 9.5, 10)
        self.rgb_array = np.random.rand(10, 10, 3)
        self.rgb = RGB((self.xs, self.ys, self.rgb_array[:, :, 0],
                        self.rgb_array[:, :, 1], self.rgb_array[:, :, 2]))



class RGBXArrayInterfaceTest(RGBGridInterfaceTest):

    datatype = 'xarray'
