import holoviews as hv


class ElementTests:

    def setup_method(self):
        self.element = hv.Element([], kdims=['A', 'B'], vdims=['C'])

    def test_key_dimension_in_element(self):
        assert hv.Dimension('A') in self.element

    def test_value_dimension_in_element(self):
        assert hv.Dimension('C') in self.element

    def test_dimension_not_in_element(self):
        assert hv.Dimension('D') not in self.element

    def test_key_dimension_string_in_element(self):
        assert 'A' in self.element

    def test_value_dimension_string_in_element(self):
        assert 'C' in self.element

    def test_dimension_string_not_in_element(self):
        assert 'D' not in self.element
