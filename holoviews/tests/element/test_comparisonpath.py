"""
Test cases for the Comparisons class over the Path elements
"""

from holoviews import Bounds, Box, Contours, Ellipse, Path
from holoviews.element.comparison import ComparisonTestCase


class PathComparisonTest(ComparisonTestCase):

    def setUp(self):
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
        self.assertEqual(self.path1, self.path1)

    def test_paths_unequal(self):
        try:
            self.assertEqual(self.path1, self.path2)
        except AssertionError as e:
            if not str(e).startswith("Path not almost equal to 6 decimals"):
                raise self.failureException("Path mismatch error not raised.")

    def test_contours_equal(self):
        self.assertEqual(self.contours1, self.contours1)

    def test_contours_unequal(self):
        try:
            self.assertEqual(self.contours1, self.contours2)
        except AssertionError as e:
            if not str(e).startswith("Contours not almost equal to 6 decimals"):
                raise self.failureException("Contours mismatch error not raised.")

    def test_contour_levels_unequal(self):
        try:
            self.assertEqual(self.contours1, self.contours3)
        except AssertionError as e:
            if not str(e).startswith("Contours not almost equal to 6 decimals"):
                raise self.failureException("Contour level are mismatch error not raised.")


    def test_bounds_equal(self):
        self.assertEqual(self.bounds1, self.bounds1)

    def test_bounds_unequal(self):
        try:
            self.assertEqual(self.bounds1, self.bounds2)
        except AssertionError as e:
            if not str(e).startswith("Bounds not almost equal to 6 decimals"):
                raise self.failureException("Bounds mismatch error not raised.")

    def test_boxs_equal(self):
        self.assertEqual(self.box1, self.box1)

    def test_boxs_unequal(self):
        try:
            self.assertEqual(self.box1, self.box2)
        except AssertionError as e:
            if not str(e).startswith("Box not almost equal to 6 decimals"):
                raise self.failureException("Box mismatch error not raised.")

    def test_ellipses_equal(self):
        self.assertEqual(self.ellipse1, self.ellipse1)

    def test_ellipses_unequal(self):
        try:
            self.assertEqual(self.ellipse1, self.ellipse2)
        except AssertionError as e:
            if not str(e).startswith("Ellipse not almost equal to 6 decimals"):
                raise self.failureException("Ellipse mismatch error not raised.")
