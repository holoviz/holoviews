"""
Test cases for the Comparisons class over the Raster types.
"""
import numpy as np


from holoviews.core import BoundingBox, Dimension
from holoviews.core.element import HoloMap
from holoviews.element.comparison import ComparisonTestCase
from holoviews import Image


class RasterTestCase(ComparisonTestCase):

    def setUp(self):
        self.arr1 = np.array([[1,2], [3,4]])
        self.arr2 = np.array([[10,2], [3,4]])
        self.arr3 = np.array([[10,2], [3,40]])
        # Varying arrays, default bounds
        self.mat1 = Image(self.arr1, bounds=BoundingBox())
        self.mat2 = Image(self.arr2, bounds=BoundingBox())
        self.mat3 = Image(self.arr3, bounds=BoundingBox())
        # Varying arrays, different bounds
        self.mat4 = Image(self.arr1, bounds=BoundingBox(radius=0.3))
        self.mat5 = Image(self.arr2, bounds=BoundingBox(radius=0.3))


class RasterOverlayTestCase(RasterTestCase):

    def setUp(self):
        super().setUp()
        # Two overlays of depth two with different layers
        self.overlay1_depth2 = (self.mat1 * self.mat2)
        self.overlay2_depth2 = (self.mat1 * self.mat3)
        # Overlay of depth 2 with different bounds
        self.overlay3_depth2 = (self.mat4 * self.mat5)
        # # Overlay of depth 3
        self.overlay4_depth3 = (self.mat1 * self.mat2 * self.mat3)


class RasterMapTestCase(RasterOverlayTestCase):

    def setUp(self):
        super().setUp()
        # Example 1D map
        self.map1_1D = HoloMap(kdims=['int'])
        self.map1_1D[0] = self.mat1
        self.map1_1D[1] = self.mat2
        # Changed keys...
        self.map2_1D = HoloMap(kdims=['int'])
        self.map2_1D[1] = self.mat1
        self.map2_1D[2] = self.mat2
        # Changed number of keys...
        self.map3_1D = HoloMap(kdims=['int'])
        self.map3_1D[1] = self.mat1
        self.map3_1D[2] = self.mat2
        self.map3_1D[3] = self.mat3
        # Changed values...
        self.map4_1D = HoloMap(kdims=['int'])
        self.map4_1D[0] = self.mat1
        self.map4_1D[1] = self.mat3
        # Changed bounds...
        self.map5_1D = HoloMap(kdims=['int'])
        self.map5_1D[0] = self.mat4
        self.map5_1D[1] = self.mat5
        # Example dimension label
        self.map6_1D = HoloMap(kdims=['int_v2'])
        self.map6_1D[0] = self.mat1
        self.map6_1D[1] = self.mat2
        # A HoloMap of Overlays
        self.map7_1D = HoloMap(kdims=['int'])
        self.map7_1D[0] =  self.overlay1_depth2
        self.map7_1D[1] =  self.overlay2_depth2
        # A different HoloMap of Overlays
        self.map8_1D = HoloMap(kdims=['int'])
        self.map8_1D[0] =  self.overlay2_depth2
        self.map8_1D[1] =  self.overlay1_depth2

        # Example 2D map
        self.map1_2D = HoloMap(kdims=['int', Dimension('float')])
        self.map1_2D[0, 0.5] = self.mat1
        self.map1_2D[1, 1.0] = self.mat2
        # Changed 2D keys...
        self.map2_2D = HoloMap(kdims=['int', Dimension('float')])
        self.map2_2D[0, 1.0] = self.mat1
        self.map2_2D[1, 1.5] = self.mat2



class BasicRasterComparisonTest(RasterTestCase):
    """
    This tests the ComparisonTestCase class which is an important
    component of other tests.
    """

    def test_matrices_equal(self):
        self.assertEqual(self.mat1, self.mat1)

    def test_unequal_arrays(self):
        try:
            self.assertEqual(self.mat1, self.mat2)
            raise AssertionError("Array mismatch not raised")
        except AssertionError as e:
            if not str(e).startswith('Image not almost equal to 6 decimals\n'):
                raise self.failureException("Image data mismatch error not raised.")

    def test_bounds_mismatch(self):
        try:
            self.assertEqual(self.mat1, self.mat4)
        except AssertionError as e:
            self.assertEqual(str(e), 'BoundingBoxes are mismatched: (-0.5, -0.5, 0.5, 0.5) != (-0.3, -0.3, 0.3, 0.3).')



class RasterOverlayComparisonTest(RasterOverlayTestCase):

    def test_depth_mismatch(self):
        try:
            self.assertEqual(self.overlay1_depth2, self.overlay4_depth3)
        except AssertionError as e:
            self.assertEqual(str(e), 'Overlays have mismatched path counts.')

    def test_element_mismatch(self):
        try:
            self.assertEqual(self.overlay1_depth2, self.overlay2_depth2)
        except AssertionError as e:
            if not str(e).startswith('Image not almost equal to 6 decimals\n'):
                raise self.failureException("Image mismatch error not raised.")



class RasterMapComparisonTest(RasterMapTestCase):

    def test_dimension_mismatch(self):
         try:
             self.assertEqual(self.map1_1D, self.map1_2D)
             raise AssertionError("Mismatch in dimension number not raised.")
         except AssertionError as e:
             self.assertEqual(str(e), 'Key dimension list mismatched')

    def test_dimension_label_mismatch(self):
         try:
             self.assertEqual(self.map1_1D, self.map6_1D)
             raise AssertionError("Mismatch in dimension labels not raised.")
         except AssertionError as e:
             self.assertEqual(str(e), 'Dimension names mismatched: int != int_v2')


    def test_key_len_mismatch(self):
        try:
            self.assertEqual(self.map1_1D, self.map3_1D)
            raise AssertionError("Mismatch in map key number not raised.")
        except AssertionError as e:
            self.assertEqual(str(e), 'HoloMaps have different numbers of keys.')

    def test_key_mismatch(self):
        try:
            self.assertEqual(self.map1_1D, self.map2_1D)
            raise AssertionError("Mismatch in map keys not raised.")
        except AssertionError as e:
            self.assertEqual(str(e),
                             'HoloMaps have different sets of keys.'
                             ' In first, not second [0]. In second, not first: [2].')

    def test_element_mismatch(self):
        try:
            self.assertEqual(self.map1_1D, self.map4_1D)
            raise AssertionError("Pane mismatch in array data not raised.")
        except AssertionError as e:
            if not str(e).startswith('Image not almost equal to 6 decimals\n'):
                raise self.failureException("Image mismatch error not raised.")
