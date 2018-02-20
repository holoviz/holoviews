"""
Test cases for Dimension and Dimensioned object behaviour.
"""
from unittest import SkipTest
from holoviews.core import Dimensioned, Dimension
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase

import numpy as np
try:
    import pandas as pd
except:
    pd = None

class DimensionNameLabelTest(LoggingComparisonTestCase):

    def setUp(self):
        super(DimensionNameLabelTest, self).setUp()

    def test_dimension_name(self):
        dim = Dimension('test')
        self.assertEqual(dim.name, 'test')

    def test_dimension_name_and_label(self):
        dim = Dimension('test')
        self.assertEqual(dim.name, 'test')
        self.assertEqual(dim.label, 'test')

    def test_dimension_name_tuple(self):
        dim = Dimension(('test', 'A test'))
        self.assertEqual(dim.name, 'test')

    def test_dimension_label_tuple(self):
        dim = Dimension(('test', 'A test'))
        self.assertEqual(dim.label, 'A test')

    def test_dimension_label_kwarg(self):
        dim = Dimension('test', label='A test')
        self.assertEqual(dim.label, 'A test')

    def test_dimension_label_kwarg_and_tuple(self):
        dim = Dimension(('test', 'A test'), label='Another test')
        substr = "Using label as supplied by keyword ('Another test'), ignoring tuple value 'A test'"
        self.log_handler.assertEndsWith('WARNING', substr)
        self.assertEqual(dim.label, 'Another test')

    def test_dimension_invalid_name(self):
        regexp = 'Dimension name must only be passed as the positional argument'
        with self.assertRaisesRegexp(KeyError, regexp):
            Dimension('test', name='something else')

    def test_dimension_invalid_name_tuple(self):
        regexp = 'Dimension name must only be passed as the positional argument'
        with self.assertRaisesRegexp(KeyError, regexp):
            Dimension(('test', 'test dimension'), name='something else')


class DimensionReprTest(ComparisonTestCase):

    def test_name_dimension_repr(self):
        dim = Dimension('test')
        self.assertEqual(repr(dim), "Dimension('test')")

    def test_name_dimension_repr_eval_equality(self):
        dim = Dimension('test')
        self.assertEqual(eval(repr(dim)) == dim, True)

    def test_name_dimension_repr_tuple(self):
        dim = Dimension(('test', 'Test Dimension'))
        self.assertEqual(repr(dim), "Dimension('test', label='Test Dimension')")

    def test_name_dimension_repr_tuple_eval_equality(self):
        dim = Dimension(('test', 'Test Dimension'))
        self.assertEqual(eval(repr(dim)) == dim, True)

    def test_name_dimension_repr_params(self):
        dim = Dimension('test', label='Test Dimension', unit='m')
        self.assertEqual(repr(dim), "Dimension('test', label='Test Dimension', unit='m')")

    def test_name_dimension_repr_params_eval_equality(self):
        dim = Dimension('test', label='Test Dimension', unit='m')
        self.assertEqual(eval(repr(dim)) == dim, True)


class DimensionEqualityTest(ComparisonTestCase):

    def test_simple_dim_equality(self):
        dim1 = Dimension('test')
        dim2 = Dimension('test')
        self.assertEqual(dim1==dim2, True)

    def test_simple_str_equality(self):
        dim1 = Dimension('test')
        dim2 = Dimension('test')
        self.assertEqual(dim1==str(dim2), True)

    def test_simple_dim_inequality(self):
        dim1 = Dimension('test1')
        dim2 = Dimension('test2')
        self.assertEqual(dim1==dim2, False)

    def test_simple_str_inequality(self):
        dim1 = Dimension('test1')
        dim2 = Dimension('test2')
        self.assertEqual(dim1==str(dim2), False)

    def test_label_dim_inequality(self):
        dim1 = Dimension(('test', 'label1'))
        dim2 = Dimension(('test', 'label2'))
        self.assertEqual(dim1==dim2, False)

    def test_label_str_equality(self):
        dim1 = Dimension(('test', 'label1'))
        dim2 = Dimension(('test', 'label2'))
        self.assertEqual(dim1==str(dim2), True)

    def test_weak_dim_equality(self):
        dim1 = Dimension('test', cyclic=True, unit='m', type=float)
        dim2 = Dimension('test', cyclic=False, unit='km', type=int)
        self.assertEqual(dim1==dim2, True)

    def test_weak_str_equality(self):
        dim1 = Dimension('test', cyclic=True, unit='m', type=float)
        dim2 = Dimension('test', cyclic=False, unit='km', type=int)
        self.assertEqual(dim1==str(dim2), True)


class DimensionValuesTest(ComparisonTestCase):

    def setUp(self):
        self.values1 = [0,1,2,3,4,5,6]
        self.values2 = ['a','b','c','d']

        self.duplicates1 = [0,1,0,2,3,4,3,2,5,5,6]
        self.duplicates2 = ['a','b','b','a','c','a','c','d','d']

    def test_dimension_values_list1(self):
        dim = Dimension('test', values=self.values1)
        self.assertEqual(dim.values, self.values1)

    def test_dimension_values_list2(self):
        dim = Dimension('test', values=self.values2)
        self.assertEqual(dim.values, self.values2)

    def test_dimension_values_list_duplicates1(self):
        dim = Dimension('test', values=self.duplicates1)
        self.assertEqual(dim.values, self.values1)

    def test_dimension_values_list_duplicates2(self):
        dim = Dimension('test', values=self.duplicates2)
        self.assertEqual(dim.values, self.values2)

    def test_dimension_values_array1(self):
        dim = Dimension('test', values=np.array(self.values1))
        self.assertEqual(dim.values, self.values1)

    def test_dimension_values_array2(self):
        dim = Dimension('test', values=np.array(self.values2))
        self.assertEqual(dim.values, self.values2)

    def test_dimension_values_array_duplicates1(self):
        dim = Dimension('test', values=np.array(self.duplicates1))
        self.assertEqual(dim.values, self.values1)

    def test_dimension_values_array_duplicates2(self):
        dim = Dimension('test', values=np.array(self.duplicates2))
        self.assertEqual(dim.values, self.values2)

    def test_dimension_values_series1(self):
        if pd is None:
            raise SkipTest("Pandas not available")
        df = pd.DataFrame({'col':self.values1})
        dim = Dimension('test', values=df['col'])
        self.assertEqual(dim.values, self.values1)

    def test_dimension_values_series2(self):
        if pd is None:
            raise SkipTest("Pandas not available")
        df = pd.DataFrame({'col':self.values2})
        dim = Dimension('test', values=df['col'])
        self.assertEqual(dim.values, self.values2)


    def test_dimension_values_series_duplicates1(self):
        if pd is None:
            raise SkipTest("Pandas not available")
        df = pd.DataFrame({'col':self.duplicates1})
        dim = Dimension('test', values=df['col'])
        self.assertEqual(dim.values, self.values1)


    def test_dimension_values_series_duplicates2(self):
        if pd is None:
            raise SkipTest("Pandas not available")
        df = pd.DataFrame({'col':self.duplicates2})
        dim = Dimension('test', values=df['col'])
        self.assertEqual(dim.values, self.values2)


class DimensionCloneTest(ComparisonTestCase):

    def test_simple_clone(self):
        dim = Dimension('test')
        self.assertEqual(dim.name, 'test')
        self.assertEqual(dim.clone('bar').name, 'bar')

    def test_simple_label_clone(self):
        dim = Dimension('test')
        self.assertEqual(dim.name, 'test')
        clone = dim.clone(label='label')
        self.assertEqual(clone.name, 'test')
        self.assertEqual(clone.label, 'label')

    def test_simple_values_clone(self):
        dim = Dimension('test', values=[1,2,3])
        self.assertEqual(dim.values, [1,2,3])
        clone = dim.clone(values=[4,5,6])
        self.assertEqual(clone.name, 'test')
        self.assertEqual(clone.values, [4,5,6])

    def test_tuple_clone(self):
        dim = Dimension('test')
        self.assertEqual(dim.name, 'test')
        clone = dim.clone(('test', 'A test'))
        self.assertEqual(clone.name, 'test')
        self.assertEqual(clone.label, 'A test')


class DimensionedTest(ComparisonTestCase):

    def test_dimensioned_init(self):
        Dimensioned('An example of arbitrary data')

    def test_dimensioned_constant_label(self):
        label = 'label'
        view = Dimensioned('An example of arbitrary data', label=label)
        self.assertEqual(view.label, label)
        try:
            view.label = 'another label'
            raise AssertionError("Label should be a constant parameter.")
        except TypeError: pass

    def test_dimensioned_redim_string(self):
        dimensioned = Dimensioned('Arbitrary Data', kdims=['x'])
        redimensioned = dimensioned.clone(kdims=['Test'])
        self.assertEqual(redimensioned, dimensioned.redim(x='Test'))

    def test_dimensioned_redim_dict_label(self):
        dimensioned = Dimensioned('Arbitrary Data', kdims=['x'])
        redimensioned = dimensioned.clone(kdims=[('x', 'Test')])
        self.assertEqual(redimensioned, dimensioned.redim.label(x='Test'))

    def test_dimensioned_redim_dict_label_existing_error(self):
        dimensioned = Dimensioned('Arbitrary Data', kdims=[('x', 'Test1')])
        with self.assertRaisesRegexp(ValueError, 'Cannot override an existing Dimension label'):
            dimensioned.redim.label(x='Test2')

    def test_dimensioned_redim_dimension(self):
        dimensioned = Dimensioned('Arbitrary Data', kdims=['x'])
        redimensioned = dimensioned.clone(kdims=['Test'])
        self.assertEqual(redimensioned, dimensioned.redim(x=Dimension('Test')))

    def test_dimensioned_redim_dict(self):
        dimensioned = Dimensioned('Arbitrary Data', kdims=['x'])
        redimensioned = dimensioned.clone(kdims=['Test'])
        self.assertEqual(redimensioned, dimensioned.redim(x={'name': 'Test'}))

    def test_dimensioned_redim_dict_range(self):
        redimensioned = Dimensioned('Arbitrary Data', kdims=['x']).redim(x={'range': (0, 10)})
        self.assertEqual(redimensioned.kdims[0].range, (0, 10))

    def test_dimensioned_redim_range_aux(self):
        dimensioned = Dimensioned('Arbitrary Data', kdims=['x'])
        redimensioned = dimensioned.redim.range(x=(-10,42))
        self.assertEqual(redimensioned.kdims[0].range, (-10,42))

    def test_dimensioned_redim_cyclic_aux(self):
        dimensioned = Dimensioned('Arbitrary Data', kdims=['x'])
        redimensioned = dimensioned.redim.cyclic(x=True)
        self.assertEqual(redimensioned.kdims[0].cyclic, True)
