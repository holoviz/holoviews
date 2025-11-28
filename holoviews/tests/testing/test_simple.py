"""
Test cases for the Comparisons class over basic literal types.

Int, float, numpy array and BoundingBox comparisons are tested.
"""
import re

import numpy as np
import pytest

from holoviews.core import BoundingBox
from holoviews.testing import assert_data_equal, assert_element_equal


class SimpleComparisonTest:
    def test_arrays_equal_int(self):
        assert_data_equal(np.array([[1,2],[3,4]]),
                         np.array([[1,2],[3,4]]))

    def test_floats_unequal_int(self):
        msg = "Arrays are not almost equal to 6 decimals"
        with pytest.raises(AssertionError, match=msg):
            assert_data_equal(np.array([[1,2],[3,4]]),
                             np.array([[1,2],[3,5]]))

    def test_arrays_equal_float(self):
        assert_data_equal(np.array([[1.0,2.5],[3,4]], dtype=np.float32),
                         np.array([[1.0,2.5],[3,4]], dtype=np.float32))

    def test_floats_unequal_float(self):
        msg = "Arrays are not almost equal to 6 decimals"
        with pytest.raises(AssertionError, match=msg):
            assert_data_equal(np.array([[1,2],[3,4.5]], dtype=np.float32),
                             np.array([[1,2],[3,5]], dtype=np.float32))

    def test_bounds_equal(self):
        assert_element_equal(BoundingBox(radius=0.5), BoundingBox(radius=0.5))

    def test_bounds_unequal(self):
        msg = "BoundingBox(radius=0.5) == BoundingBox(radius=0.7)"
        with pytest.raises(AssertionError, match=re.escape(msg)):
            assert_element_equal(BoundingBox(radius=0.5), BoundingBox(radius=0.7))


    def test_bounds_equal_lbrt(self):
        assert_element_equal(BoundingBox(points=((-1,-1),(3,4.5))),
                         BoundingBox(points=((-1,-1),(3,4.5))))

    def test_bounds_unequal_lbrt(self):
        msg = 'BoundingBox(points=((-1,-1),(3,4.5))) == BoundingBox(points=((-1,-1),(3,5.0)))'
        with pytest.raises(AssertionError, match=re.escape(msg)):
            assert_element_equal(
                BoundingBox(points=((-1, -1,), (3, 4.5,),)),
                BoundingBox(points=((-1, -1,), (3, 5.0,),))
            )
