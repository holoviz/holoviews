from holoviews.core import Dimension, Element
from holoviews.element.comparison import ComparisonTestCase


class ElementTests(ComparisonTestCase):

    def setUp(self):
        self.element = Element([], kdims=['A', 'B'], vdims=['C'])

    def test_key_dimension_in_element(self):
        self.assertTrue(Dimension('A') in self.element)

    def test_value_dimension_in_element(self):
        self.assertTrue(Dimension('C') in self.element)

    def test_dimension_not_in_element(self):
        self.assertFalse(Dimension('D') in self.element)

    def test_key_dimension_string_in_element(self):
        self.assertTrue('A' in self.element)

    def test_value_dimension_string_in_element(self):
        self.assertTrue('C' in self.element)

    def test_dimension_string_not_in_element(self):
        self.assertFalse('D' in self.element)
