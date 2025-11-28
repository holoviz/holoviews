"""
Test cases for the Comparisons class over the Path elements
"""

import pytest

from holoviews import Bounds, Box, Contours, Ellipse, Path
from holoviews.testing import assert_element_equal


class PathComparisonTest:

    def setup_method(self):
        self.path1 = Path([(-0.3, 0.4), (-0.3, 0.3), (-0.2, 0.3),
                           (-0.2, 0.4),(-0.3, 0.4)])

        self.path2 = Path([(-0.3, 0.4), (-0.3, 0.3), (-0.2, 0.3),
                           (-0.2, 0.4),(-3, 4)])

        self.contours1 = Contours([(-0.3, 0.4, 1), (-0.3, 0.3, 1), (-0.2, 0.3, 1),
                                   (-0.2, 0.4, 1),(-0.3, 0.4, 1)], vdims='Level')

        self.contours2 = Contours([(-0.3, 0.4, 1), (-0.3, 0.3, 1), (-0.2, 0.3, 1),
                                   (-0.2, 0.4, 1), (-3, 4, 1)], vdims='Level')

        self.contours3 = Contours([(-0.3, 0.4, 2), (-0.3, 0.3, 2), (-0.2, 0.3, 2),
                                   (-0.2, 0.4, 2), (-0.3, 0.4, 2)], vdims='Level')

        self.bounds1 = Bounds(0.3)
        self.bounds2 = Bounds(0.4)

        self.box1 = Box(-0.25, 0.3, 0.3)
        self.box2 = Box(-0.25, 0.3, 0.4)

        self.ellipse1 = Ellipse(-0.25, 0.3, 0.3)
        self.ellipse2 = Ellipse(-0.25, 0.3, 0.4)

    def test_paths_equal(self):
        assert_element_equal(self.path1, self.path1)

    def test_paths_unequal(self):
        msg = "Arrays are not almost equal to 6 decimals"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.path1, self.path2)

    def test_contours_equal(self):
        assert_element_equal(self.contours1, self.contours1)

    def test_contours_unequal(self):
        msg = "Arrays are not almost equal to 6 decimals"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.contours1, self.contours2)

    def test_contour_levels_unequal(self):
        msg = "Arrays are not almost equal to 6 decimals"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.contours1, self.contours3)


    def test_bounds_equal(self):
        assert_element_equal(self.bounds1, self.bounds1)

    def test_bounds_unequal(self):
        msg = "Arrays are not almost equal to 6 decimals"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.bounds1, self.bounds2)

    def test_boxes_equal(self):
        assert_element_equal(self.box1, self.box1)

    def test_boxes_unequal(self):
        msg = "Arrays are not almost equal to 6 decimals"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.box1, self.box2)

    def test_ellipses_equal(self):
        assert_element_equal(self.ellipse1, self.ellipse1)

    def test_ellipses_unequal(self):
        msg = "Arrays are not almost equal to 6 decimals"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.ellipse1, self.ellipse2)
