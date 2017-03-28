from unittest import SkipTest

import numpy as np
from holoviews import Dimension, Image
from holoviews.element.comparison import ComparisonTestCase


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



class ImageGridInterfaceTest(ImageInterfaceTest):

    datatype = 'grid'

    def init_data(self):
        xs = np.linspace(-9, 9, 10)
        ys = np.linspace(0.5, 9.5, 10)
        self.array = np.arange(10) * np.arange(10)[:, np.newaxis]
        self.image = Image((xs, ys, self.array))


class ImageXArrayInterfaceTest(ImageGridInterfaceTest):

    datatype = 'xarray'


class ImageIrisInterfaceTest(ImageGridInterfaceTest):

    datatype = 'cube'
