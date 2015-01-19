"""
Test cases for ViewTestCase which implements view comparison.
"""
import numpy as np


from holoviews.core import BoundingBox, Dimension, ViewMap
from holoviews.testing import ViewTestCase
from holoviews.view import Matrix

class MatrixTestCase(ViewTestCase):

    def setUp(self):
        self.arr1 = np.array([[1,2], [3,4]])
        self.arr2 = np.array([[10,2], [3,4]])
        self.arr3 = np.array([[10,2], [3,40]])
        # Varying arrays, default bounds
        self.mat1 = Matrix(self.arr1, BoundingBox())
        self.mat2 = Matrix(self.arr2, BoundingBox())
        self.mat3 = Matrix(self.arr3, BoundingBox())
        # Varying arrays, different bounds
        self.mat4 = Matrix(self.arr1, BoundingBox(radius=0.3))
        self.mat5 = Matrix(self.arr2, BoundingBox(radius=0.3))


class MatrixOverlayTestCase(MatrixTestCase):

    def setUp(self):
        super(MatrixOverlayTestCase, self).setUp()
        # Two overlays of depth two with different layers
        self.overlay1_depth2 = (self.mat1 * self.mat2)
        self.overlay2_depth2 = (self.mat1 * self.mat3)
        # Layers of depth 2 with different bounds
        self.overlay3_depth2 = (self.mat4 * self.mat5)
        # # Layers of depth 3
        self.overlay4_depth3 = (self.mat1 * self.mat2 * self.mat3)


class MapTestCase(MatrixOverlayTestCase):

    def setUp(self):
        super(MapTestCase, self).setUp()
        # Example 1D map
        self.map1_1D = ViewMap(index_dimensions=['int'])
        self.map1_1D[0] = self.mat1
        self.map1_1D[1] = self.mat2
        # Changed keys...
        self.map2_1D = ViewMap(index_dimensions=['int'])
        self.map2_1D[1] = self.mat1
        self.map2_1D[2] = self.mat2
        # Changed number of keys...
        self.map3_1D = ViewMap(index_dimensions=['int'])
        self.map3_1D[1] = self.mat1
        self.map3_1D[2] = self.mat2
        self.map3_1D[3] = self.mat3
        # Changed values...
        self.map4_1D = ViewMap(index_dimensions=['int'])
        self.map4_1D[0] = self.mat1
        self.map4_1D[1] = self.mat3
        # Changed bounds...
        self.map5_1D = ViewMap(index_dimensions=['int'])
        self.map5_1D[0] = self.mat4
        self.map5_1D[1] = self.mat5
        # Example dimension label
        self.map6_1D = ViewMap(index_dimensions=['int_v2'])
        self.map6_1D[0] = self.mat1
        self.map6_1D[1] = self.mat2
        # A ViewMap of Overlays
        self.map7_1D = ViewMap(index_dimensions=['int'])
        self.map7_1D[0] =  self.overlay1_depth2
        self.map7_1D[1] =  self.overlay2_depth2
        # A different ViewMap of Overlays
        self.map8_1D = ViewMap(index_dimensions=['int'])
        self.map8_1D[0] =  self.overlay2_depth2
        self.map8_1D[1] =  self.overlay1_depth2

        # Example 2D map
        self.map1_2D = ViewMap(index_dimensions=['int', Dimension('float')])
        self.map1_2D[0, 0.5] = self.mat1
        self.map1_2D[1, 1.0] = self.mat2
        # Changed 2D keys...
        self.map2_2D = ViewMap(index_dimensions=['int', Dimension('float')])
        self.map2_2D[0, 1.0] = self.mat1
        self.map2_2D[1, 1.5] = self.mat2



class SheetComparisonTest(MatrixTestCase):
    """
    This tests the ViewTestCase class which is an important component
    of other tests.
    """

    def test_equal(self):
        self.assertEqual(self.mat1, self.mat1)

    def test_unequal_arrays(self):
        try:
            self.assertEqual(self.mat1, self.mat2)
            raise AssertionError("Array mismatch not detected")
        except AssertionError as e:
            assert str(e).startswith('Matrix: \nArrays are not almost equal to 6 decimals')

    def test_bounds_mismatch(self):
        try:
            self.assertEqual(self.mat1, self.mat4)
        except AssertionError as e:
            assert str(e).startswith('BoundingBoxes are mismatched.')



class SheetOverlayComparisonTest(MatrixOverlayTestCase):

    def test_depth_mismatch(self):
        try:
            self.assertEqual(self.overlay1_depth2, self.overlay4_depth3)
        except AssertionError as e:
            assert str(e).startswith("Overlays have different lengths.")

    def test_element_mismatch(self):
        try:
            self.assertEqual(self.overlay1_depth2, self.overlay2_depth2)
        except AssertionError as e:
            assert str(e).startswith('Matrix: \nArrays are not almost equal to 6 decimals')



class MapComparisonTest(MapTestCase):

    def test_dimension_mismatch(self):
         try:
             self.assertEqual(self.map1_1D, self.map1_2D)
             raise AssertionError("Mismatch in dimension number not detected.")
         except AssertionError as e:
             assert str(e).startswith("Maps have different numbers of dimensions.")

    def test_dimension_label_mismatch(self):
         try:
             self.assertEqual(self.map1_1D, self.map6_1D)
             raise AssertionError("Mismatch in dimension labels not detected.")
         except AssertionError as e:
             assert str(e).startswith("Maps have different dimension labels.")


    def test_key_len_mismatch(self):
        try:
            self.assertEqual(self.map1_1D, self.map3_1D)
            raise AssertionError("Mismatch in map key number not detected.")
        except AssertionError as e:
            assert str(e).startswith("Maps have different numbers of keys.")

    def test_key_mismatch(self):
        try:
            self.assertEqual(self.map1_1D, self.map2_1D)
            raise AssertionError("Mismatch in map keys not detected.")
        except AssertionError as e:
            assert str(e).startswith("Maps have different sets of keys.")

    def test_element_mismatch(self):
        try:
            self.assertEqual(self.map1_1D, self.map4_1D)
            raise AssertionError("Pane mismatch in array data not detected.")
        except AssertionError as e:
            assert str(e).startswith('Matrix: \nArrays are not almost equal to 6 decimals')


if __name__ == "__main__":
    import sys
    import nose
    nose.runmodule(argv=[sys.argv[0], "--logging-level", "ERROR"])
