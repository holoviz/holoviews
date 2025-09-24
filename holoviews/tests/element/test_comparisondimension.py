"""
Test cases for Dimension and Dimensioned object comparison.
"""
from holoviews.core import Dimension, Dimensioned
from holoviews.core.util import NUMPY_GE_2_0_0
from holoviews.element.comparison import ComparisonTestCase


class DimensionsComparisonTestCase(ComparisonTestCase):

    def setUp(self):
        super().setUp()
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
            self.assertEqual(str(e),  'Dimension names mismatched: dim1 != dim2')

    def test_dimension_comparison_labels_unequal(self):
        try:
            self.assertEqual(self.dimension1, self.dimension11)
        except AssertionError as e:
            self.assertEqual(str(e),  'Dimension labels mismatched: dim1 != Test Dimension')

    def test_dimension_comparison_range_unequal1(self):
        try:
            self.assertEqual(self.dimension1, self.dimension3)
        except AssertionError as e:
            self.assertEqual(str(e), "Dimension parameter 'range' mismatched: (0, 1) != (0, 2)")

    def test_dimension_comparison_cyclic_unequal(self):
        try:
            self.assertEqual(self.dimension4, self.dimension5)
        except AssertionError as e:
            self.assertEqual(str(e), "Dimension parameter 'cyclic' mismatched: False != True")

    def test_dimension_comparison_range_unequal2(self):
        try:
            self.assertEqual(self.dimension5, self.dimension6)
        except AssertionError as e:
            self.assertEqual(str(e), "Dimension parameter 'range' mismatched: (None, None) != (0, 1)")

    def test_dimension_comparison_units_unequal(self):
        try:
            self.assertEqual(self.dimension6, self.dimension7)
        except AssertionError as e:
            self.assertEqual(str(e), "Dimension parameter 'unit' mismatched: None != 'ms'")

    def test_dimension_comparison_values_unequal(self):
        try:
            self.assertEqual(self.dimension4, self.dimension8)
        except AssertionError as e:
            if NUMPY_GE_2_0_0:
                msg = "Dimension parameter 'values' mismatched: [] != [np.str_('a'), np.str_('b')]"
            else:
                msg = "Dimension parameter 'values' mismatched: [] != ['a', 'b']"
            self.assertEqual(str(e), msg)

    def test_dimension_comparison_types_unequal(self):
        try:
            self.assertEqual(self.dimension9, self.dimension10)
        except AssertionError as e:
            self.assertEqual(str(e), "Dimension parameter 'type' mismatched: <class 'int'> != <class 'float'>")

    def test_dimension_comparison_value_format_unequal(self):
        # Comparing callables is skipped
        self.assertEqual(self.dimension12, self.dimension13)
        self.assertNotEqual(str(self.dimension12.value_format),
                            str(self.dimension13.value_format))


class DimensionedComparisonTestCase(ComparisonTestCase):

    def setUp(self):
        super().setUp()
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
        self.assertEqual(self.dimensioned1,
                         Dimensioned('other_data',
                                     vdims=self.value_list1,
                                     kdims=self.key_list1))

    def test_dimensioned_comparison_unequal_value_dims(self):
        try:
            self.assertEqual(self.dimensioned1, self.dimensioned2)
        except AssertionError as e:
            self.assertEqual(str(e), "Dimension names mismatched: val1 != val2")


    def test_dimensioned_comparison_unequal_key_dims(self):
        try:
            self.assertEqual(self.dimensioned1, self.dimensioned3)
        except AssertionError as e:
            self.assertEqual(str(e), 'Dimension names mismatched: key1 != key2')

    def test_dimensioned_comparison_unequal_value_dim_lists(self):
        try:
            self.assertEqual(self.dimensioned1, self.dimensioned4)
        except AssertionError as e:
            self.assertEqual(str(e), "Value dimension list mismatched")

    def test_dimensioned_comparison_unequal_key_dim_lists(self):
        try:
            self.assertEqual(self.dimensioned1, self.dimensioned5)
        except AssertionError as e:
            self.assertEqual(str(e), 'Key dimension list mismatched')

    def test_dimensioned_comparison_unequal_group(self):
        try:
            self.assertEqual(self.dimensioned1, self.dimensioned6)
        except AssertionError as e:
            self.assertEqual(str(e), 'Group labels mismatched.')

    def test_dimensioned_comparison_unequal_label(self):
        try:
            self.assertEqual(self.dimensioned6, self.dimensioned7)
        except AssertionError as e:
            self.assertEqual(str(e), 'Labels mismatched.')
