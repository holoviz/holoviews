"""
Tests for the Dataset Element types.
"""

from unittest import SkipTest
from itertools import product

import numpy as np
from holoviews import Dataset, NdElement, HoloMap, Dimension, Image
from holoviews.element.comparison import ComparisonTestCase

from collections import OrderedDict
from holoviews.core.dimension import OrderedDict as cyODict

try:
    import pandas as pd
except:
    pd = None

try:
    import dask.dataframe as dd
except:
    dd = None


class DatatypeContext(object):

    def __init__(self, datatypes):
        self.datatypes = datatypes
        self._old_datatypes = None

    def __enter__(self):
        self._old_datatypes = Dataset.datatype
        Dataset.datatype = self.datatypes

    def __exit__(self, *args):
        Dataset.datatype = self._old_datatypes


class HomogeneousColumnTypes(object):
    """
    Tests for data formats that require all dataset to have the same
    type (e.g numpy arrays)
    """

    def setUp(self):
        self.restore_datatype = Dataset.datatype
        self.data_instance_type = None

    def init_column_data(self):
        self.xs = range(11)
        self.xs_2 = [el**2 for el in self.xs]

        self.y_ints = [i*2 for i in range(11)]
        self.dataset_hm = Dataset((self.xs, self.y_ints),
                                  kdims=['x'], vdims=['y'])
        self.dataset_hm_alias = Dataset((self.xs, self.y_ints),
                                        kdims=[('x', 'X')], vdims=[('y', 'Y')])

    def tearDown(self):
        Dataset.datatype = self.restore_datatype

    # Test the array constructor (homogenous data) to be supported by
    # all interfaces.

    def test_dataset_array_init_hm(self):
        "Tests support for arrays (homogeneous)"
        dataset = Dataset(np.column_stack([self.xs, self.xs_2]),
                          kdims=['x'], vdims=['x2'])
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    def test_dataset_ndelement_init_hm(self):
        "Tests support for homogeneous NdElement (backwards compatibility)"
        dataset = Dataset(NdElement(zip(self.xs, self.xs_2),
                                    kdims=['x'], vdims=['x2']))
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    def test_dataset_dataframe_init_hm(self):
        "Tests support for homogeneous DataFrames"
        if pd is None:
            raise SkipTest("Pandas not available")
        dataset = Dataset(pd.DataFrame({'x':self.xs, 'x2':self.xs_2}),
                          kdims=['x'], vdims=[ 'x2'])
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    def test_dataset_dataframe_init_hm_alias(self):
        "Tests support for homogeneous DataFrames"
        if pd is None:
            raise SkipTest("Pandas not available")
        dataset = Dataset(pd.DataFrame({'x':self.xs, 'x2':self.xs_2}),
                          kdims=[('x', 'X-label')], vdims=[('x2', 'X2-label')])
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    # Properties and information

    def test_dataset_shape(self):
        self.assertEqual(self.dataset_hm.shape, (11, 2))

    def test_dataset_range(self):
        self.assertEqual(self.dataset_hm.range('y'), (0, 20))

    def test_dataset_closest(self):
        closest = self.dataset_hm.closest([0.51, 1, 9.9])
        self.assertEqual(closest, [1., 1., 10.])

    # Operations

    def test_dataset_sort_vdim_hm(self):
        xs_2 = np.array(self.xs_2)
        dataset = Dataset(np.column_stack([self.xs, -xs_2]),
                          kdims=['x'], vdims=['y'])
        dataset_sorted = Dataset(np.column_stack([self.xs[::-1], -xs_2[::-1]]),
                                 kdims=['x'], vdims=['y'])
        self.assertEqual(dataset.sort('y'), dataset_sorted)

    def test_dataset_sort_vdim_hm_alias(self):
        xs_2 = np.array(self.xs_2)
        dataset = Dataset(np.column_stack([self.xs, -xs_2]),
                          kdims=[('x', 'X-label')], vdims=[('y', 'Y-label')])
        dataset_sorted = Dataset(np.column_stack([self.xs[::-1], -xs_2[::-1]]),
                                 kdims=[('x', 'X-label')], vdims=[('y', 'Y-label')])
        self.assertEqual(dataset.sort('y'), dataset_sorted)
        self.assertEqual(dataset.sort('Y-label'), dataset_sorted)

    def test_dataset_redim_hm_kdim(self):
        redimmed = self.dataset_hm.redim(x='Time')
        self.assertEqual(redimmed.dimension_values('Time'),
                         self.dataset_hm.dimension_values('x'))

    def test_dataset_redim_hm_kdim_alias(self):
        redimmed = self.dataset_hm_alias.redim(x='Time')
        self.assertEqual(redimmed.dimension_values('Time'),
                         self.dataset_hm_alias.dimension_values('x'))

    def test_dataset_redim_hm_vdim(self):
        redimmed = self.dataset_hm.redim(y='Value')
        self.assertEqual(redimmed.dimension_values('Value'),
                         self.dataset_hm.dimension_values('y'))

    def test_dataset_redim_hm_vdim_alias(self):
        redimmed = self.dataset_hm_alias.redim(y=Dimension(('val', 'Value')))
        self.assertEqual(redimmed.dimension_values('Value'),
                         self.dataset_hm_alias.dimension_values('y'))

    def test_dataset_sample_hm(self):
        samples = self.dataset_hm.sample([0, 5, 10]).dimension_values('y')
        self.assertEqual(samples, np.array([0, 10, 20]))

    def test_dataset_sample_hm_alias(self):
        samples = self.dataset_hm_alias.sample([0, 5, 10]).dimension_values('y')
        self.assertEqual(samples, np.array([0, 10, 20]))

    def test_dataset_array_hm(self):
        self.assertEqual(self.dataset_hm.array(),
                         np.column_stack([self.xs, self.y_ints]))

    def test_dataset_array_hm_alias(self):
        self.assertEqual(self.dataset_hm_alias.array(),
                         np.column_stack([self.xs, self.y_ints]))

    def test_dataset_add_dimensions_value_hm(self):
        table = self.dataset_hm.add_dimension('z', 1, 0)
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.zeros(table.shape[0]))

    def test_dataset_add_dimensions_values_hm(self):
        table =  self.dataset_hm.add_dimension('z', 1, range(1,12))
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.array(list(range(1,12))))

    def test_dataset_slice_hm(self):
        dataset_slice = Dataset({'x':range(5, 9), 'y':[2 * i for i in range(5, 9)]},
                                kdims=['x'], vdims=['y'])
        self.assertEqual(self.dataset_hm[5:9], dataset_slice)

    def test_dataset_slice_hm_alias(self):
        dataset_slice = Dataset({'x':range(5, 9), 'y':[2 * i for i in range(5, 9)]},
                                kdims=[('x', 'X')], vdims=[('y', 'Y')])
        self.assertEqual(self.dataset_hm_alias[5:9], dataset_slice)

    def test_dataset_slice_fn_hm(self):
        dataset_slice = Dataset({'x':range(5, 9), 'y':[2 * i for i in range(5, 9)]},
                                kdims=['x'], vdims=['y'])
        self.assertEqual(self.dataset_hm[lambda x: (x >= 5) & (x < 9)], dataset_slice)

    def test_dataset_1D_reduce_hm(self):
        dataset = Dataset({'x':self.xs, 'y':self.y_ints}, kdims=['x'], vdims=['y'])
        self.assertEqual(dataset.reduce('x', np.mean), 10)

    def test_dataset_1D_reduce_hm_alias(self):
        dataset = Dataset({'x':self.xs, 'y':self.y_ints}, kdims=[('x', 'X')],
                          vdims=[('y', 'Y')])
        self.assertEqual(dataset.reduce('X', np.mean), 10)

    def test_dataset_2D_reduce_hm(self):
        dataset = Dataset({'x':self.xs, 'y':self.y_ints, 'z':[el ** 2 for el in self.y_ints]},
                          kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(np.array(dataset.reduce(['x', 'y'], np.mean)),
                         np.array(140))

    def test_dataset_2D_aggregate_partial_hm(self):
        z_ints = [el**2 for el in self.y_ints]
        dataset = Dataset({'x':self.xs, 'y':self.y_ints, 'z':z_ints},
                          kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(dataset.aggregate(['x'], np.mean),
                         Dataset({'x':self.xs, 'z':z_ints}, kdims=['x'], vdims=['z']))

    # Indexing

    def test_dataset_index_column_idx_hm(self):
        self.assertEqual(self.dataset_hm[5], self.y_ints[5])

    def test_dataset_index_column_ht(self):
        self.compare_arrays(self.dataset_hm['y'], self.y_ints)

    def test_dataset_array_ht(self):
        self.assertEqual(self.dataset_hm.array(),
                         np.column_stack([self.xs, self.y_ints]))



class HeterogeneousColumnTypes(HomogeneousColumnTypes):
    """
    Tests for data formats that all dataset to have varied types
    """

    def init_column_data(self):
        self.kdims = ['Gender', 'Age']
        self.vdims = ['Weight', 'Height']
        self.gender, self.age = ['M','M','F'], [10,16,12]
        self.weight, self.height = [15,18,10], [0.8,0.6,0.8]
        self.table = Dataset({'Gender':self.gender, 'Age':self.age,
                              'Weight':self.weight, 'Height':self.height},
                             kdims=self.kdims, vdims=self.vdims)

        self.alias_kdims = [('gender', 'Gender'), ('age', 'Age')]
        self.alias_vdims = [('weight', 'Weight'), ('height', 'Height')]
        self.alias_table = Dataset({'gender':self.gender, 'age':self.age,
                                    'weight':self.weight, 'height':self.height},
                                   kdims=self.alias_kdims, vdims=self.alias_vdims)

        super(HeterogeneousColumnTypes, self).init_column_data()
        self.ys = np.linspace(0, 1, 11)
        self.zs = np.sin(self.xs)
        self.dataset_ht = Dataset({'x':self.xs, 'y':self.ys},
                                  kdims=['x'], vdims=['y'])

    # Test the constructor to be supported by all interfaces supporting
    # heterogeneous column types.

    def test_dataset_ndelement_init_ht(self):
        "Tests support for heterogeneous NdElement (backwards compatibility)"
        dataset = Dataset(NdElement(zip(self.xs, self.ys), kdims=['x'], vdims=['y']))
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    def test_dataset_dataframe_init_ht(self):
        "Tests support for heterogeneous DataFrames"
        if pd is None:
            raise SkipTest("Pandas not available")
        dataset = Dataset(pd.DataFrame({'x':self.xs, 'y':self.ys}), kdims=['x'], vdims=['y'])
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    def test_dataset_dataframe_init_ht_alias(self):
        "Tests support for heterogeneous DataFrames"
        if pd is None:
            raise SkipTest("Pandas not available")
        dataset = Dataset(pd.DataFrame({'x':self.xs, 'y':self.ys}),
                          kdims=[('x', 'X')], vdims=[('y', 'Y')])
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    # Test literal formats

    def test_dataset_expanded_dimvals_ht(self):
        self.assertEqual(self.table.dimension_values('Gender', expanded=False),
                         np.array(['M', 'F']))

    def test_dataset_implicit_indexing_init(self):
        dataset = Dataset(self.ys, kdims=['x'], vdims=['y'])
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    def test_dataset_tuple_init(self):
        dataset = Dataset((self.xs, self.ys), kdims=['x'], vdims=['y'])
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    def test_dataset_tuple_init_alias(self):
        dataset = Dataset((self.xs, self.ys), kdims=[('x', 'X')], vdims=[('y', 'Y')])
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    def test_dataset_simple_zip_init(self):
        dataset = Dataset(zip(self.xs, self.ys), kdims=['x'], vdims=['y'])
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    def test_dataset_simple_zip_init_alias(self):
        dataset = Dataset(zip(self.xs, self.ys), kdims=[('x', 'X')], vdims=[('y', 'Y')])
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    def test_dataset_zip_init(self):
        dataset = Dataset(zip(self.gender, self.age,
                              self.weight, self.height),
                          kdims=self.kdims, vdims=self.vdims)
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    def test_dataset_zip_init_alias(self):
        dataset = self.alias_table.clone(zip(self.gender, self.age,
                                             self.weight, self.height))
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    def test_dataset_odict_init(self):
        dataset = Dataset(OrderedDict(zip(self.xs, self.ys)), kdims=['A'], vdims=['B'])
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    def test_dataset_odict_init_alias(self):
        dataset = Dataset(OrderedDict(zip(self.xs, self.ys)),
                          kdims=[('a', 'A')], vdims=[('b', 'B')])
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    def test_dataset_dict_init(self):
        dataset = Dataset(dict(zip(self.xs, self.ys)), kdims=['A'], vdims=['B'])
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    # Operations

    def test_dataset_redim_with_alias_dframe(self):
        test_df = pd.DataFrame({'x': range(10), 'y': range(0,20,2)})
        dataset = Dataset(test_df, kdims=[('x', 'X-label')], vdims=['y'])
        redim_df = pd.DataFrame({'X': range(10), 'y': range(0,20,2)})
        dataset_redim = Dataset(redim_df, kdims=['X'], vdims=['y'])
        self.assertEqual(dataset.redim(**{'X-label':'X'}), dataset_redim)
        self.assertEqual(dataset.redim(**{'x':'X'}), dataset_redim)

    def test_dataset_sort_vdim_ht(self):
        dataset = Dataset({'x':self.xs, 'y':-self.ys},
                          kdims=['x'], vdims=['y'])
        dataset_sorted = Dataset({'x': self.xs[::-1], 'y':-self.ys[::-1]},
                                 kdims=['x'], vdims=['y'])
        self.assertEqual(dataset.sort('y'), dataset_sorted)

    def test_dataset_sort_string_ht(self):
        dataset_sorted = Dataset({'Gender':['F', 'M', 'M'], 'Age':[12, 10, 16],
                                  'Weight':[10,15,18], 'Height':[0.8,0.8,0.6]},
                                 kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(self.table.sort(), dataset_sorted)

    def test_dataset_sample_ht(self):
        samples = self.dataset_ht.sample([0, 5, 10]).dimension_values('y')
        self.assertEqual(samples, np.array([0, 0.5, 1]))

    def test_dataset_reduce_ht(self):
        reduced = Dataset({'Age':self.age, 'Weight':self.weight, 'Height':self.height},
                          kdims=self.kdims[1:], vdims=self.vdims)
        self.assertEqual(self.table.reduce(['Gender'], np.mean), reduced)

    def test_dataset_1D_reduce_ht(self):
        self.assertEqual(self.dataset_ht.reduce('x', np.mean), np.float64(0.5))

    def test_dataset_2D_reduce_ht(self):
        reduced = Dataset({'Weight':[14.333333333333334], 'Height':[0.73333333333333339]},
                          kdims=[], vdims=self.vdims)
        self.assertEqual(self.table.reduce(function=np.mean), reduced)

    def test_dataset_2D_partial_reduce_ht(self):
        dataset = Dataset({'x':self.xs, 'y':self.ys, 'z':self.zs},
                          kdims=['x', 'y'], vdims=['z'])
        reduced = Dataset({'x':self.xs, 'z':self.zs},
                          kdims=['x'], vdims=['z'])
        self.assertEqual(dataset.reduce(['y'], np.mean), reduced)

    def test_dataset_aggregate_ht(self):
        aggregated = Dataset({'Gender':['M', 'F'], 'Weight':[16.5, 10], 'Height':[0.7, 0.8]},
                             kdims=self.kdims[:1], vdims=self.vdims)
        self.compare_dataset(self.table.aggregate(['Gender'], np.mean), aggregated)

    def test_dataset_aggregate_ht_alias(self):
        aggregated = Dataset({'gender':['M', 'F'], 'weight':[16.5, 10], 'height':[0.7, 0.8]},
                             kdims=self.alias_kdims[:1], vdims=self.alias_vdims)
        self.compare_dataset(self.alias_table.aggregate('Gender', np.mean), aggregated)

    def test_dataset_2D_aggregate_partial_ht(self):
        dataset = Dataset({'x':self.xs, 'y':self.ys, 'z':self.zs},
                          kdims=['x', 'y'], vdims=['z'])
        reduced = Dataset({'x':self.xs, 'z':self.zs},
                          kdims=['x'], vdims=['z'])
        self.assertEqual(dataset.aggregate(['x'], np.mean), reduced)

    def test_dataset_groupby(self):
        group1 = {'Age':[10,16], 'Weight':[15,18], 'Height':[0.8,0.6]}
        group2 = {'Age':[12], 'Weight':[10], 'Height':[0.8]}
        grouped = HoloMap([('M', Dataset(group1, kdims=['Age'], vdims=self.vdims)),
                           ('F', Dataset(group2, kdims=['Age'], vdims=self.vdims))],
                          kdims=['Gender'])
        self.assertEqual(self.table.groupby(['Gender']), grouped)

    def test_dataset_groupby_alias(self):
        group1 = {'age':[10,16], 'weight':[15,18], 'height':[0.8,0.6]}
        group2 = {'age':[12], 'weight':[10], 'height':[0.8]}
        grouped = HoloMap([('M', Dataset(group1, kdims=[('age', 'Age')],
                                         vdims=self.alias_vdims)),
                           ('F', Dataset(group2, kdims=[('age', 'Age')],
                                         vdims=self.alias_vdims))],
                          kdims=[('gender', 'Gender')])
        self.assertEqual(self.alias_table.groupby('Gender'), grouped)

    def test_dataset_groupby_dynamic(self):
        grouped_dataset = self.table.groupby('Gender', dynamic=True)
        self.assertEqual(grouped_dataset['M'],
                         self.table.select(Gender='M').reindex(['Age']))
        self.assertEqual(grouped_dataset['F'],
                         self.table.select(Gender='F').reindex(['Age']))

    def test_dataset_groupby_dynamic_alias(self):
        grouped_dataset = self.alias_table.groupby('Gender', dynamic=True)
        self.assertEqual(grouped_dataset['M'],
                         self.alias_table.select(gender='M').reindex(['Age']))
        self.assertEqual(grouped_dataset['F'],
                         self.alias_table.select(gender='F').reindex(['Age']))

    def test_dataset_add_dimensions_value_ht(self):
        table = self.dataset_ht.add_dimension('z', 1, 0)
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.zeros(table.shape[0]))

    def test_dataset_add_dimensions_value_ht_alias(self):
        table = self.dataset_ht.add_dimension(('z', 'Z'), 1, 0)
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.zeros(table.shape[0]))

    def test_dataset_add_dimensions_values_ht(self):
        table = self.dataset_ht.add_dimension('z', 1, range(1,12))
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.array(list(range(1,12))))

    # Indexing

    def test_dataset_index_row_gender_female(self):
        indexed = Dataset({'Gender':['F'], 'Age':[12],
                           'Weight':[10], 'Height':[0.8]},
                          kdims=self.kdims, vdims=self.vdims)
        row = self.table['F',:]
        self.assertEquals(row, indexed)

    def test_dataset_index_rows_gender_male(self):
        row = self.table['M',:]
        indexed = Dataset({'Gender':['M', 'M'], 'Age':[10, 16],
                           'Weight':[15,18], 'Height':[0.8,0.6]},
                          kdims=self.kdims, vdims=self.vdims)
        self.assertEquals(row, indexed)

    def test_dataset_select_rows_gender_male(self):
        row = self.table.select(Gender='M')
        indexed = Dataset({'Gender':['M', 'M'], 'Age':[10, 16],
                           'Weight':[15,18], 'Height':[0.8,0.6]},
                          kdims=self.kdims, vdims=self.vdims)
        self.assertEquals(row, indexed)

    def test_dataset_select_rows_gender_male_alias(self):
        row = self.alias_table.select(Gender='M')
        alias_row = self.alias_table.select(gender='M')
        indexed = Dataset({'gender':['M', 'M'], 'age':[10, 16],
                           'weight':[15,18], 'height':[0.8,0.6]},
                          kdims=self.alias_kdims, vdims=self.alias_vdims)
        self.assertEquals(row, indexed)
        self.assertEquals(alias_row, indexed)

    def test_dataset_index_row_age(self):
        indexed = Dataset({'Gender':['F'], 'Age':[12],
                           'Weight':[10], 'Height':[0.8]},
                          kdims=self.kdims, vdims=self.vdims)
        self.assertEquals(self.table[:, 12], indexed)

    def test_dataset_index_item_table(self):
        indexed = Dataset({'Gender':['F'], 'Age':[12],
                           'Weight':[10], 'Height':[0.8]},
                          kdims=self.kdims, vdims=self.vdims)
        self.assertEquals(self.table['F', 12], indexed)

    def test_dataset_index_value1(self):
        self.assertEquals(self.table['F', 12, 'Weight'], 10)

    def test_dataset_index_value2(self):
        self.assertEquals(self.table['F', 12, 'Height'], 0.8)

    def test_dataset_index_column_ht(self):
        self.compare_arrays(self.dataset_ht['y'], self.ys)

    def test_dataset_boolean_index(self):
        row = self.table[np.array([True, True, False])]
        indexed = Dataset({'Gender':['M', 'M'], 'Age':[10, 16],
                           'Weight':[15,18], 'Height':[0.8,0.6]},
                          kdims=self.kdims, vdims=self.vdims)
        self.assertEquals(row, indexed)

    def test_dataset_value_dim_index(self):
        row = self.table[:, :, 'Weight']
        indexed = Dataset({'Gender':['M', 'M', 'F'], 'Age':[10, 16, 12],
                           'Weight':[15,18, 10]},
                          kdims=self.kdims, vdims=self.vdims[:1])
        self.assertEquals(row, indexed)

    def test_dataset_value_dim_scalar_index(self):
        row = self.table['M', 10, 'Weight']
        self.assertEquals(row, 15)

    # Casting

    def test_dataset_array_ht(self):
        self.assertEqual(self.dataset_ht.array(),
                         np.column_stack([self.xs, self.ys]))


class ArrayDatasetTest(HomogeneousColumnTypes, ComparisonTestCase):
    """
    Test of the ArrayDataset interface.
    """
    def setUp(self):
        self.restore_datatype = Dataset.datatype
        Dataset.datatype = ['array']
        self.data_instance_type = np.ndarray
        self.init_column_data()


class DFDatasetTest(HeterogeneousColumnTypes, ComparisonTestCase):
    """
    Test of the pandas DFDataset interface.
    """

    def setUp(self):
        if pd is None:
            raise SkipTest("Pandas not available")
        self.restore_datatype = Dataset.datatype
        Dataset.datatype = ['dataframe']
        self.data_instance_type = pd.DataFrame
        self.init_column_data()


class DaskDatasetTest(HeterogeneousColumnTypes, ComparisonTestCase):
    """
    Test of the pandas DaskDataset interface.
    """

    def setUp(self):
        if dd is None:
            raise SkipTest("dask not available")
        self.restore_datatype = Dataset.datatype
        Dataset.datatype = ['dask']
        self.data_instance_type = dd.DataFrame
        self.init_column_data()

    # Disabled tests for NotImplemented methods
    def test_dataset_add_dimensions_values_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_add_dimensions_values_ht(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_vdim_ht(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_vdim_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_vdim_hm_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_string_ht(self):
        raise SkipTest("Not supported")

    def test_dataset_boolean_index(self):
        raise SkipTest("Not supported")


class DictDatasetTest(HeterogeneousColumnTypes, ComparisonTestCase):
    """
    Test of the generic dictionary interface.
    """

    def setUp(self):
        self.restore_datatype = Dataset.datatype
        Dataset.datatype = ['dictionary']
        self.data_instance_type = (dict, cyODict, OrderedDict)
        self.init_column_data()



class NdDatasetTest(HeterogeneousColumnTypes, ComparisonTestCase):
    """
    Test of the NdDataset interface (mostly for backwards compatibility)
    """

    def setUp(self):
        self.restore_datatype = Dataset.datatype
        Dataset.datatype = ['ndelement']
        self.data_instance_type = NdElement
        self.init_column_data()

    # Literal formats that have been previously been supported but
    # currently are only supported via NdElement.

    def test_dataset_double_zip_init(self):
        dataset = Dataset(zip(zip(self.gender, self.age),
                              zip(self.weight, self.height)),
                          kdims=self.kdims, vdims=self.vdims)
        self.assertTrue(isinstance(dataset.data, NdElement))


class GridTests(object):
    """
    Test of the Grid array interface
    """

    datatype = 'grid'

    def setUp(self):
        self.restore_datatype = Dataset.datatype
        Dataset.datatype = ['grid']
        self.data_instance_type = dict
        self.init_data()

    def init_data(self):
        self.xs = range(11)
        self.xs_2 = [el**2 for el in self.xs]

        self.y_ints = [i*2 for i in range(11)]
        self.dataset_hm = Dataset((self.xs, self.y_ints),
                                  kdims=['x'], vdims=['y'])
        self.dataset_hm_alias = Dataset((self.xs, self.y_ints),
                                        kdims=[('x', 'X')], vdims=[('y', 'Y')])

    def init_grid_data(self):
        self.grid_xs = [0, 1]
        self.grid_ys = [0.1, 0.2, 0.3]
        self.grid_zs = [[0, 1], [2, 3], [4, 5]]
        self.dataset_grid = self.eltype((self.grid_xs, self.grid_ys,
                                         self.grid_zs), kdims=['x', 'y'],
                                        vdims=['z'])
        self.dataset_grid_alias = self.eltype((self.grid_xs, self.grid_ys,
                                               self.grid_zs), kdims=[('x', 'X'), ('y', 'Y')],
                                              vdims=[('z', 'Z')])
        self.dataset_grid_inv = self.eltype((self.grid_xs[::-1], self.grid_ys[::-1],
                                             self.grid_zs), kdims=['x', 'y'],
                                            vdims=['z'])

    def test_canonical_vdim(self):
        x = np.array([ 0.  ,  0.75,  1.5 ])
        y = np.array([ 1.5 ,  0.75,  0.  ])
        z = np.array([[ 0.06925999,  0.05800389,  0.05620127],
                      [ 0.06240918,  0.05800931,  0.04969735],
                      [ 0.05376789,  0.04669417,  0.03880118]])
        dataset = Dataset((x, y, z), kdims=['x', 'y'], vdims=['z'])
        canonical = np.array([[ 0.05376789,  0.04669417,  0.03880118],
                              [ 0.06240918,  0.05800931,  0.04969735],
                              [ 0.06925999,  0.05800389,  0.05620127]])
        self.assertEqual(dataset.dimension_values('z', flat=False),
                         canonical)

    def test_dataset_dim_vals_grid_kdims_xs(self):
        self.assertEqual(self.dataset_grid.dimension_values(0, expanded=False),
                         np.array([0, 1]))

    def test_dataset_dim_vals_grid_kdims_xs_alias(self):
        self.assertEqual(self.dataset_grid_alias.dimension_values('x', expanded=False),
                         np.array([0, 1]))
        self.assertEqual(self.dataset_grid_alias.dimension_values('X', expanded=False),
                         np.array([0, 1]))

    def test_dataset_dim_vals_grid_kdims_xs_inv(self):
        self.assertEqual(self.dataset_grid_inv.dimension_values(0, expanded=False),
                         np.array([0, 1]))

    def test_dataset_dim_vals_grid_kdims_expanded_xs_flat(self):
        expanded_xs = np.array([0, 0, 0, 1, 1, 1])
        self.assertEqual(self.dataset_grid.dimension_values(0),
                         expanded_xs)

    def test_dataset_dim_vals_grid_kdims_expanded_xs_flat_inv(self):
        expanded_xs = np.array([0, 0, 0, 1, 1, 1])
        self.assertEqual(self.dataset_grid_inv.dimension_values(0),
                         expanded_xs)

    def test_dataset_dim_vals_grid_kdims_expanded_xs(self):
        expanded_xs = np.array([[0, 0, 0], [1, 1, 1]])
        self.assertEqual(self.dataset_grid.dimension_values(0, flat=False),
                         expanded_xs)

    def test_dataset_dim_vals_grid_kdims_expanded_xs_inv(self):
        expanded_xs = np.array([[0, 0, 0], [1, 1, 1]])
        self.assertEqual(self.dataset_grid_inv.dimension_values(0, flat=False),
                         expanded_xs)

    def test_dataset_dim_vals_grid_kdims_ys(self):
        self.assertEqual(self.dataset_grid.dimension_values(1, expanded=False),
                         np.array([0.1, 0.2, 0.3]))

    def test_dataset_dim_vals_grid_kdims_ys_inv(self):
        self.assertEqual(self.dataset_grid_inv.dimension_values(1, expanded=False),
                         np.array([0.1, 0.2, 0.3]))

    def test_dataset_dim_vals_grid_kdims_expanded_ys_flat(self):
        expanded_ys = np.array([0.1, 0.2, 0.3,
                                0.1, 0.2, 0.3])
        self.assertEqual(self.dataset_grid.dimension_values(1),
                         expanded_ys)

    def test_dataset_dim_vals_grid_kdims_expanded_ys_flat_inv(self):
        expanded_ys = np.array([0.1, 0.2, 0.3,
                                0.1, 0.2, 0.3])
        self.assertEqual(self.dataset_grid_inv.dimension_values(1),
                         expanded_ys)

    def test_dataset_dim_vals_grid_kdims_expanded_ys(self):
        expanded_ys = np.array([[0.1, 0.2, 0.3],
                                [0.1, 0.2, 0.3]])
        self.assertEqual(self.dataset_grid.dimension_values(1, flat=False),
                         expanded_ys)

    def test_dataset_dim_vals_grid_kdims_expanded_ys_inv(self):
        expanded_ys = np.array([[0.1, 0.2, 0.3],
                                [0.1, 0.2, 0.3]])
        self.assertEqual(self.dataset_grid_inv.dimension_values(1, flat=False),
                         expanded_ys)

    def test_dataset_dim_vals_grid_vdims_zs_flat(self):
        expanded_zs = np.array([0, 2, 4, 1, 3, 5])
        self.assertEqual(self.dataset_grid.dimension_values(2),
                         expanded_zs)

    def test_dataset_dim_vals_grid_vdims_zs_flat_inv(self):
        expanded_zs = np.array([5, 3, 1, 4, 2, 0])
        self.assertEqual(self.dataset_grid_inv.dimension_values(2),
                         expanded_zs)

    def test_dataset_dim_vals_grid_vdims_zs(self):
        expanded_zs = np.array([[0, 1], [2, 3], [4, 5]])
        self.assertEqual(self.dataset_grid.dimension_values(2, flat=False),
                         expanded_zs)

    def test_dataset_dim_vals_grid_vdims_zs_inv(self):
        expanded_zs = np.array([[5, 4], [3, 2], [1, 0]])
        self.assertEqual(self.dataset_grid_inv.dimension_values(2, flat=False),
                         expanded_zs)



class GridDatasetTest(GridTests, HomogeneousColumnTypes, ComparisonTestCase):

    def setUp(self):
        self.restore_datatype = Dataset.datatype
        Dataset.datatype = ['grid']
        self.eltype = Dataset
        self.data_instance_type = dict
        self.init_grid_data()
        self.init_column_data()

    def test_dataset_array_init_hm(self):
        "Tests support for arrays (homogeneous)"
        exception = "None of the available storage backends "\
         "were able to support the supplied data format."
        with self.assertRaisesRegexp(Exception, exception):
            Dataset(np.column_stack([self.xs, self.xs_2]),
                    kdims=['x'], vdims=['x2'])

    def test_dataset_dataframe_init_hm(self):
        "Tests support for homogeneous DataFrames"
        if pd is None:
            raise SkipTest("Pandas not available")
        exception = "None of the available storage backends "\
         "were able to support the supplied data format."
        with self.assertRaisesRegexp(Exception, exception):
            Dataset(pd.DataFrame({'x':self.xs, 'x2':self.xs_2}),
                    kdims=['x'], vdims=['x2'])

    def test_dataset_dataframe_init_hm_alias(self):
        "Tests support for homogeneous DataFrames"
        if pd is None:
            raise SkipTest("Pandas not available")
        exception = "None of the available storage backends "\
         "were able to support the supplied data format."
        with self.assertRaisesRegexp(Exception, exception):
            Dataset(pd.DataFrame({'x':self.xs, 'x2':self.xs_2}),
                    kdims=['x'], vdims=['x2'])

    def test_dataset_ndelement_init_hm(self):
        "Tests support for homogeneous NdElement (backwards compatibility)"
        exception = "None of the available storage backends "\
         "were able to support the supplied data format."
        with self.assertRaisesRegexp(Exception, exception):
            Dataset(NdElement(zip(self.xs, self.xs_2),
                              kdims=['x'], vdims=['x2']))

    def test_dataset_sort_vdim_hm(self):
        exception = ('Compressed format cannot be sorted, either instantiate '
                     'in the desired order or use the expanded format.')
        with self.assertRaisesRegexp(Exception, exception):
            self.dataset_hm.sort('y')

    def test_dataset_sort_vdim_hm_alias(self):
        exception = ('Compressed format cannot be sorted, either instantiate '
                     'in the desired order or use the expanded format.')
        with self.assertRaisesRegexp(Exception, exception):
            self.dataset_hm.sort('y')

    def test_dataset_groupby(self):
        self.assertEqual(self.dataset_hm.groupby('x').keys(), list(self.xs))

    def test_dataset_add_dimensions_value_hm(self):
        with self.assertRaisesRegexp(Exception, 'Cannot add key dimension to a dense representation.'):
            self.dataset_hm.add_dimension('z', 1, 0)

    def test_dataset_add_dimensions_values_hm(self):
        table =  self.dataset_hm.add_dimension('z', 1, range(1,12), vdim=True)
        self.assertEqual(table.vdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.array(list(range(1,12))))

    def test_dataset_add_dimensions_values_hm_alias(self):
        table =  self.dataset_hm.add_dimension(('z', 'Z'), 1, range(1,12), vdim=True)
        self.assertEqual(table.vdims[1], 'Z')
        self.compare_arrays(table.dimension_values('Z'), np.array(list(range(1,12))))

    def test_dataset_2D_aggregate_partial_hm(self):
        array = np.random.rand(11, 11)
        dataset = Dataset({'x':self.xs, 'y':self.y_ints, 'z': array},
                          kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(dataset.aggregate(['x'], np.mean),
                         Dataset({'x':self.xs, 'z': np.mean(array, axis=0)},
                                 kdims=['x'], vdims=['z']))

    def test_dataset_2D_aggregate_partial_hm_alias(self):
        array = np.random.rand(11, 11)
        dataset = Dataset({'x':self.xs, 'y':self.y_ints, 'z': array},
                          kdims=[('x', 'X'), ('y', 'Y')], vdims=[('z', 'Z')])
        self.assertEqual(dataset.aggregate(['X'], np.mean),
                         Dataset({'x':self.xs, 'z': np.mean(array, axis=0)},
                                 kdims=[('x', 'X')], vdims=[('z', 'Z')]))

    def test_dataset_2D_reduce_hm(self):
        array = np.random.rand(11, 11)
        dataset = Dataset({'x':self.xs, 'y':self.y_ints, 'z': array},
                          kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(np.array(dataset.reduce(['x', 'y'], np.mean)),
                         np.mean(array))

    def test_dataset_2D_reduce_hm_alias(self):
        array = np.random.rand(11, 11)
        dataset = Dataset({'x':self.xs, 'y':self.y_ints, 'z': array},
                          kdims=[('x', 'X'), ('y', 'Y')], vdims=[('z', 'Z')])
        self.assertEqual(np.array(dataset.reduce(['x', 'y'], np.mean)),
                         np.mean(array))
        self.assertEqual(np.array(dataset.reduce(['X', 'Y'], np.mean)),
                         np.mean(array))

    def test_dataset_groupby_dynamic(self):
        array = np.random.rand(11, 11)
        dataset = Dataset({'x':self.xs, 'y':self.y_ints, 'z': array},
                          kdims=['x', 'y'], vdims=['z'])
        grouped = dataset.groupby('x', dynamic=True)
        first = Dataset({'y': self.y_ints, 'z': array[:, 0]},
                        kdims=['y'], vdims=['z'])
        self.assertEqual(grouped[0], first)

    def test_dataset_groupby_dynamic_alias(self):
        array = np.random.rand(11, 11)
        dataset = Dataset({'x':self.xs, 'y':self.y_ints, 'z': array},
                          kdims=[('x', 'X'), ('y', 'Y')], vdims=[('z', 'Z')])
        grouped = dataset.groupby('X', dynamic=True)
        first = Dataset({'y': self.y_ints, 'z': array[:, 0]},
                        kdims=[('y', 'Y')], vdims=[('z', 'Z')])
        self.assertEqual(grouped[0], first)

    def test_dataset_groupby_multiple_dims(self):
        dataset = Dataset((range(8), range(8), range(8), range(8),
                           np.random.rand(8, 8, 8, 8)),
                          kdims=['a', 'b', 'c', 'd'], vdims=['Value'])
        grouped = dataset.groupby(['c', 'd'])
        keys = list(product(range(8), range(8)))
        self.assertEqual(list(grouped.keys()), keys)
        for c, d in keys:
            self.assertEqual(grouped[c, d], dataset.select(c=c, d=d).reindex(['a', 'b']))

    def test_dataset_groupby_drop_dims(self):
        array = np.random.rand(3, 20, 10)
        ds = Dataset({'x': range(10), 'y': range(20), 'z': range(3), 'Val': array},
                     kdims=['x', 'y', 'z'], vdims=['Val'])
        with DatatypeContext([self.datatype, 'columns', 'dataframe']):
            partial = ds.to(Dataset, kdims=['x'], vdims=['Val'], groupby='y')
        self.assertEqual(partial.last['Val'], array[:, -1, :].T.flatten())

    def test_dataset_groupby_drop_dims_dynamic(self):
        array = np.random.rand(3, 20, 10)
        ds = Dataset({'x': range(10), 'y': range(20), 'z': range(3), 'Val': array},
                     kdims=['x', 'y', 'z'], vdims=['Val'])
        with DatatypeContext([self.datatype, 'columns', 'dataframe']):
            partial = ds.to(Dataset, kdims=['x'], vdims=['Val'], groupby='y', dynamic=True)
            self.assertEqual(partial[19]['Val'], array[:, -1, :].T.flatten())

    def test_dataset_groupby_drop_dims_with_vdim(self):
        array = np.random.rand(3, 20, 10)
        ds = Dataset({'x': range(10), 'y': range(20), 'z': range(3), 'Val': array, 'Val2': array*2},
                     kdims=['x', 'y', 'z'], vdims=['Val', 'Val2'])
        with DatatypeContext([self.datatype, 'columns', 'dataframe']):
            partial = ds.to(Dataset, kdims=['Val'], vdims=['Val2'], groupby='y')
        self.assertEqual(partial.last['Val'], array[:, -1, :].T.flatten())

    def test_dataset_groupby_drop_dims_dynamic_with_vdim(self):
        array = np.random.rand(3, 20, 10)
        ds = Dataset({'x': range(10), 'y': range(20), 'z': range(3), 'Val': array, 'Val2': array*2},
                     kdims=['x', 'y', 'z'], vdims=['Val', 'Val2'])
        with DatatypeContext([self.datatype, 'columns', 'dataframe']):
            partial = ds.to(Dataset, kdims=['Val'], vdims=['Val2'], groupby='y', dynamic=True)
            self.assertEqual(partial[19]['Val'], array[:, -1, :].T.flatten())


class IrisDatasetTest(GridDatasetTest):
    """
    Tests for Iris interface
    """

    datatype = 'cube'

    def setUp(self):
        import iris
        self.restore_datatype = Dataset.datatype
        Dataset.datatype = ['cube']
        self.eltype = Dataset
        self.data_instance_type = iris.cube.Cube
        self.init_column_data()
        self.init_grid_data()

    # Disabled tests for NotImplemented methods
    def test_dataset_add_dimensions_values_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_add_dimensions_values_hm_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_vdim_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_vdim_hm_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_1D_reduce_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_1D_reduce_hm_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_2D_reduce_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_2D_reduce_hm_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_2D_aggregate_partial_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_2D_aggregate_partial_hm_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_sample_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sample_hm_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_groupby_drop_dims_with_vdim(self):
        raise SkipTest("Not supported")

    def test_dataset_groupby_drop_dims_dynamic_with_vdim(self):
        raise SkipTest("Not supported")
    

class XArrayDatasetTest(GridDatasetTest):
    """
    Tests for Iris interface
    """

    datatype = 'xarray'

    def setUp(self):
        import xarray
        self.restore_datatype = Dataset.datatype
        Dataset.datatype = ['xarray']
        self.eltype = Dataset
        self.data_instance_type = xarray.Dataset
        self.init_column_data()
        self.init_grid_data()

    # Disabled tests for NotImplemented methods
    def test_dataset_add_dimensions_values_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_vdim_hm_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_vdim_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sample_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sample_hm_alias(self):
        raise SkipTest("Not supported")


class RasterDatasetTest(GridTests, ComparisonTestCase):
    """
    Tests for Iris interface
    """

    def setUp(self):
        self.restore_datatype = Dataset.datatype
        self.eltype = Image
        Dataset.datatype = ['image']
        self.data_instance_type = dict
        self.init_grid_data()

    def tearDown(self):
        Dataset.datatype = self.restore_datatype
