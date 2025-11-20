"""
Test cases for Dimension and Dimensioned object comparison.
"""
import pytest

from holoviews.core import Dimension, Dimensioned
from holoviews.testing import assert_element_equal


class DimensionsComparisonTestCase:

    def setup_method(self):
        self.dimension1 = Dimension('dim1', range=(0,1))
        self.dimension2 = Dimension('dim2', range=(0,1))
        self.dimension3 = Dimension('dim1', range=(0,2))
        self.dimension4 = Dimension('dim1')
        self.dimension5 = Dimension('dim1', cyclic=True)
        self.dimension6 = Dimension('dim1', cyclic=True, range=(0,1))
        self.dimension7 = Dimension('dim1', cyclic=True, range=(0,1), unit='ms')
        self.dimension8 = Dimension('dim1', values=['a', 'b'])
        self.dimension9 = Dimension('dim1', type=int)
        self.dimension10 = Dimension('dim1', type=float)
        self.dimension11 = Dimension(('dim1','Test Dimension'), range=(0,1))
        self.dimension12 = Dimension('dim1', value_format=lambda x: x)
        self.dimension13 = Dimension('dim1', value_format=lambda x: x)

    def test_dimension_comparison_equal1(self):
        assert_element_equal(self.dimension1, self.dimension1)

    def test_dimension_comparison_equal2(self):
        assert_element_equal(self.dimension1,
                         Dimension('dim1', range=(0,1)))

    def test_dimension_comparison_equal3(self):
        assert_element_equal(self.dimension7,
                         Dimension('dim1', cyclic=True, range=(0,1), unit='ms'))

    def test_dimension_comparison_names_unequal(self):
        msg = "'dim1' == 'dim2'"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.dimension1, self.dimension2)

    def test_dimension_comparison_labels_unequal(self):
        msg = "'dim1' == 'Test Dimension'"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.dimension1, self.dimension11)

    def test_dimension_comparison_range_unequal1(self):
        msg = "1 == 2"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.dimension1, self.dimension3)

    def test_dimension_comparison_cyclic_unequal(self):
        msg = "False == True"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.dimension4, self.dimension5)

    def test_dimension_comparison_range_unequal2(self):
        msg = "None == 0"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.dimension5, self.dimension6)

    def test_dimension_comparison_units_unequal(self):
        msg = "None == 'ms'"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.dimension6, self.dimension7)

    def test_dimension_comparison_values_unequal(self):
        msg = "0 == 2"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.dimension4, self.dimension8)

    def test_dimension_comparison_types_unequal(self):
        msg = "<class 'int'> == <class 'float'>"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.dimension9, self.dimension10)

    def test_dimension_comparison_value_format_unequal(self):
        # Comparing callables is skipped
        assert_element_equal(self.dimension12, self.dimension13)
        assert str(self.dimension12.value_format) != str(self.dimension13.value_format)


class DimensionedComparisonTestCase:

    def setup_method(self):
        # Value dimension lists
        self.value_list1 = [Dimension('val1')]
        self.value_list2 = [Dimension('val2')]
        # Key dimension lists
        self.key_list1 = [Dimension('key1')]
        self.key_list2 = [Dimension('key2')]
        # Dimensioned instances
        self.dimensioned1 = Dimensioned('data1', vdims=self.value_list1,
                                        kdims=self.key_list1)
        self.dimensioned2 = Dimensioned('data2', vdims=self.value_list2,
                                        kdims=self.key_list1)

        self.dimensioned3 = Dimensioned('data3', vdims=self.value_list1,
                                        kdims=self.key_list2)

        self.dimensioned4 = Dimensioned('data4', vdims=[],
                                        kdims=self.key_list1)

        self.dimensioned5 = Dimensioned('data5', vdims=self.value_list1,
                                        kdims=[])
        # Value / Label comparison tests
        self.dimensioned6 = Dimensioned('data6', group='foo',
                                        vdims=self.value_list1,
                                        kdims=self.key_list1)

        self.dimensioned7 = Dimensioned('data7', group='foo', label='bar',
                                        vdims=self.value_list1,
                                        kdims=self.key_list1)


    def test_dimensioned_comparison_equal(self):
        "Note that the data is not compared at the Dimensioned level"
        assert_element_equal(self.dimensioned1,
                         Dimensioned('other_data',
                                     vdims=self.value_list1,
                                     kdims=self.key_list1))

    def test_dimensioned_comparison_unequal_value_dims(self):
        msg = "'val1' == 'val2'"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.dimensioned1, self.dimensioned2)

    def test_dimensioned_comparison_unequal_key_dims(self):
        msg = "'key1' == 'key2'"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.dimensioned1, self.dimensioned3)

    def test_dimensioned_comparison_unequal_value_dim_lists(self):
        msg = "1 == 0"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.dimensioned1, self.dimensioned4)

    def test_dimensioned_comparison_unequal_key_dim_lists(self):
        msg = "1 == 0"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.dimensioned1, self.dimensioned5)

    def test_dimensioned_comparison_unequal_group(self):
        msg = "'Dimensioned' == 'foo'"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.dimensioned1, self.dimensioned6)

    def test_dimensioned_comparison_unequal_label(self):
        msg = "'' == 'bar'"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(self.dimensioned6, self.dimensioned7)
