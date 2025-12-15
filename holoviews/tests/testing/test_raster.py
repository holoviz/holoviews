"""
Test cases for the Comparisons class over the Raster types.
"""
import re

import numpy as np
import pytest

from holoviews import Image
from holoviews.core import BoundingBox, Dimension
from holoviews.core.element import HoloMap
from holoviews.testing import assert_element_equal


class RasterTestCase:

    def setup_method(self):
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

    def setup_method(self):
        super().setup_method()
        # Two overlays of depth two with different layers
        self.overlay1_depth2 = (self.mat1 * self.mat2)
        self.overlay2_depth2 = (self.mat1 * self.mat3)
        # Overlay of depth 2 with different bounds
        self.overlay3_depth2 = (self.mat4 * self.mat5)
        # # Overlay of depth 3
        self.overlay4_depth3 = (self.mat1 * self.mat2 * self.mat3)


class RasterMapTestCase(RasterOverlayTestCase):

    def setup_method(self):
        super().setup_method()
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
        assert_element_equal(self.mat1, self.mat1)

    def test_unequal_arrays(self):
        msg = "Arrays are not almost equal to 6 decimals"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.mat1, self.mat2)

    def test_bounds_mismatch(self):
        msg = r"(-0.5, -0.3" # output from np.isclose(..., atol=)
        with pytest.raises(AssertionError, match=re.escape(msg)):
            assert_element_equal(self.mat1, self.mat4)



class RasterOverlayComparisonTest(RasterOverlayTestCase):

    def test_depth_mismatch(self):
        msg = "Right contains one more item"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.overlay1_depth2, self.overlay4_depth3)

    def test_element_mismatch(self):
        msg = "Arrays are not almost equal to 6 decimals"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.overlay1_depth2, self.overlay2_depth2)



class RasterMapComparisonTest(RasterMapTestCase):

    def test_dimension_mismatch(self):
        msg = "1 == 2"
        with pytest.raises(AssertionError, match=msg):
             assert_element_equal(self.map1_1D, self.map1_2D)

    def test_dimension_label_mismatch(self):
        msg = "'int' == 'int_v2'"
        with pytest.raises(AssertionError, match=msg):
             assert_element_equal(self.map1_1D, self.map6_1D)

    def test_key_len_mismatch(self):
        msg = "[0, 1] == [1, 2, 3]"
        with pytest.raises(AssertionError, match=re.escape(msg)):
            assert_element_equal(self.map1_1D, self.map3_1D)

    def test_key_mismatch(self):
        msg = "[0, 1] == [1, 2]"
        with pytest.raises(AssertionError, match=re.escape(msg)):
            assert_element_equal(self.map1_1D, self.map2_1D)

    def test_element_mismatch(self):
        msg = "Arrays are not almost equal to 6 decimals"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.map1_1D, self.map4_1D)
