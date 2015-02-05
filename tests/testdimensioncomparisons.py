"""
Test cases for Dimension and Dimensioned object comparison.
"""

from holoviews.core import Dimension
from . import ComparisonTestCase


class DimensionsComparisonTestCase(ComparisonTestCase):

    def setUp(self):
        super(DimensionsComparisonTestCase, self).setUp()
        self.dimension1 = Dimension('dim1', range=(0,1))
        self.dimension2 = Dimension('dim2', range=(0,1))
        self.dimension3 = Dimension('dim1', range=(0,2))
        self.dimension4 = Dimension('dim1')
        self.dimension5 = Dimension('dim1', cyclic=True)
        self.dimension6 = Dimension('dim1', cyclic=True, range=(0,1))
        self.dimension7 = Dimension('dim1', cyclic=True, range=(0,1), unit='ms')
        self.dimension8 = Dimension('dim1', values=['a', 'b'])
        self.dimension9 = Dimension('dim1', format_string='{name}')

    def test_dimension_comparison_equal1(self):
        self.assertEqual(self.dimension1, self.dimension1)

    def test_dimension_comparison_equal2(self):
        self.assertEqual(self.dimension1,
                         Dimension('dim1', range=(0,1)))

    def test_dimension_comparison_equal3(self):
        self.assertEqual(self.dimension7,
                         Dimension('dim1', cyclic=True, range=(0,1), unit='ms'))

    def test_dimension_comparison_names_unequal(self):
        try:
            self.assertEqual(self.dimension1, self.dimension2)
        except AssertionError as e:
            self.assertEqual(str(e), 'Dimension names mismatched.')

    def test_dimension_comparison_range_unequal1(self):
        try:
            self.assertEqual(self.dimension1, self.dimension3)
        except AssertionError as e:
            self.assertEqual(str(e), 'Dimension ranges mismatched.')

    def test_dimension_comparison_cyclic_unequal(self):
        try:
            self.assertEqual(self.dimension4, self.dimension5)
        except AssertionError as e:
            self.assertEqual(str(e), 'Dimension cyclic declarations mismatched.')

    def test_dimension_comparison_range_unequal2(self):
        try:
            self.assertEqual(self.dimension5, self.dimension6)
        except AssertionError as e:
            self.assertEqual(str(e), 'Dimension ranges mismatched.')

    def test_dimension_comparison_units_unequal(self):
        try:
            self.assertEqual(self.dimension6, self.dimension7)
        except AssertionError as e:
            self.assertEqual(str(e), 'Dimension unit declarations mismatched.')

    def test_dimension_comparison_values_unequal(self):
        try:
            self.assertEqual(self.dimension4, self.dimension8)
        except AssertionError as e:
            self.assertEqual(str(e), 'Dimension value declarations mismatched.')

    def test_dimension_comparison_format_unequal(self):
        try:
            self.assertEqual(self.dimension4, self.dimension9)
        except AssertionError as e:
            self.assertEqual(str(e), 'Dimension format string declarations mismatched.')



if __name__ == "__main__":
    import sys
    import nose
    nose.runmodule(argv=[sys.argv[0], "--logging-level", "ERROR"])
