from holoviews.core import Dimension, Element
from holoviews.element.comparison import ComparisonTestCase


class ElementTests(ComparisonTestCase):

    def setUp(self):
        self.element = Element([], kdims=['A', 'B'], vdims=['C'])

    def test_key_dimension_in_element(self):
        assert Dimension('A') in self.element

    def test_value_dimension_in_element(self):
        assert Dimension('C') in self.element

    def test_dimension_not_in_element(self):
        assert Dimension('D') not in self.element

    def test_key_dimension_string_in_element(self):
        assert 'A' in self.element

    def test_value_dimension_string_in_element(self):
        assert 'C' in self.element

    def test_dimension_string_not_in_element(self):
        assert 'D' not in self.element
