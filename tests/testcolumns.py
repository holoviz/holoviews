"""
Tests for the Columns Element types.
"""

from unittest import SkipTest
import numpy as np
from holoviews import Columns, Curve, ItemTable, NdElement, HoloMap
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.ndmapping import sorted_context

from collections import OrderedDict
from holoviews.core.dimension import OrderedDict as cyODict

try:
    import pandas as pd
except:
    pd = None



class HomogeneousColumnTypes(object):
    """
    Tests for data formats that require all columns to have the same
    type (e.g numpy arrays)
    """

    def setUp(self):
        self.restore_datatype = Columns.datatype
        self.data_instance_type = None

    def init_data(self):
        self.xs = range(11)
        self.xs_2 = [el**2 for el in self.xs]

        self.y_ints = [i*2 for i in range(11)]
        self.columns_hm = Columns(np.column_stack([self.xs, self.y_ints]),
                                  kdims=['x'], vdims=['y'])

    def tearDown(self):
        Columns.datatype = self.restore_datatype

    # Test the array constructor (homogenous data) to be supported by
    # all interfaces.

    def test_columns_array_init_hm(self):
        "Tests support for arrays (homogeneous)"
        columns = Columns(np.column_stack([self.xs, self.xs_2]),
                          kdims=['x'], vdims=['x2'])
        self.assertTrue(isinstance(columns.data, self.data_instance_type))

    def test_columns_ndelement_init_hm(self):
        "Tests support for homogeneous NdElement (backwards compatibility)"
        columns = Columns(NdElement(zip(self.xs, self.xs_2),
                                    kdims=['x'], vdims=['x2']))
        self.assertTrue(isinstance(columns.data, self.data_instance_type))

    def test_columns_dataframe_init_hm(self):
        "Tests support for homogeneous DataFrames"
        if pd is None:
            raise SkipTest("Pandas not available")
        columns = Columns(pd.DataFrame({'x':self.xs, 'x2':self.xs_2}),
                          kdims=['x'], vdims=[ 'x2'])
        self.assertTrue(isinstance(columns.data, self.data_instance_type))

    # Properties and information

    def test_columns_shape(self):
        self.assertEqual(self.columns_hm.shape, (11, 2))

    def test_columns_range(self):
        self.assertEqual(self.columns_hm.range('y'), (0, 20))

    def test_columns_closest(self):
        closest = self.columns_hm.closest([0.51, 1, 9.9])
        self.assertEqual(closest, [1., 1., 10.])

    # Operations

    def test_columns_sort_vdim_hm(self):
        xs_2 = np.array(self.xs_2)
        columns = Columns(np.column_stack([self.xs, -xs_2]),
                                 kdims=['x'], vdims=['y'])
        columns_sorted = Columns(np.column_stack([self.xs[::-1], -xs_2[::-1]]),
                                 kdims=['x'], vdims=['y'])
        self.assertEqual(columns.sort('y'), columns_sorted)

    def test_columns_sample_hm(self):
        samples = self.columns_hm.sample([0, 5, 10]).dimension_values('y')
        self.assertEqual(samples, np.array([0, 10, 20]))

    def test_columns_array_hm(self):
        self.assertEqual(self.columns_hm.array(),
                         np.column_stack([self.xs, self.y_ints]))

    def test_columns_add_dimensions_value_hm(self):
        table = self.columns_hm.add_dimension('z', 1, 0)
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.zeros(len(table)))

    def test_columns_add_dimensions_values_hm(self):
        table =  self.columns_hm.add_dimension('z', 1, range(1,12))
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.array(list(range(1,12))))


    def test_columns_slice_hm(self):
        columns_slice = Columns({'x':range(5, 9), 'y':[2*i for i in range(5, 9)]},
                                kdims=['x'], vdims=['y'])
        self.assertEqual(self.columns_hm[5:9], columns_slice)

    def test_columns_1D_reduce_hm(self):
        columns = Columns({'x':self.xs, 'y':self.y_ints}, kdims=['x'], vdims=['y'])
        self.assertEqual(columns.reduce('x', np.mean), 10)

    def test_columns_2D_reduce_hm(self):
        columns = Columns({'x':self.xs, 'y':self.y_ints, 'z':[el**2 for el in self.y_ints]},
                          kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(np.array(columns.reduce(['x', 'y'], np.mean)),
                         np.array(140))

    def test_columns_2D_aggregate_partial_hm(self):
        z_ints = [el**2 for el in self.y_ints]
        columns = Columns({'x':self.xs, 'y':self.y_ints, 'z':z_ints},
                          kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(columns.aggregate(['x'], np.mean),
                         Columns({'x':self.xs, 'z':z_ints}, kdims=['x'], vdims=['z']))

    # Indexing

    def test_columns_index_column_idx_hm(self):
        self.assertEqual(self.columns_hm[5], self.y_ints[5])

    def test_columns_index_column_ht(self):
        self.compare_arrays(self.columns_hm['y'], self.y_ints)

    def test_columns_array_ht(self):
        self.assertEqual(self.columns_hm.array(),
                         np.column_stack([self.xs, self.y_ints]))



class HeterogeneousColumnTypes(HomogeneousColumnTypes):
    """
    Tests for data formats that all columns to have varied types
    """

    def init_data(self):
        self.kdims = ['Gender', 'Age']
        self.vdims = ['Weight', 'Height']
        self.gender, self.age = ['M','M','F'], [10,16,12]
        self.weight, self.height = [15,18,10], [0.8,0.6,0.8]
        self.table = Columns({'Gender':self.gender, 'Age':self.age,
                              'Weight':self.weight, 'Height':self.height},
                             kdims=self.kdims, vdims=self.vdims)

        super(HeterogeneousColumnTypes, self).init_data()
        self.ys = np.linspace(0, 1, 11)
        self.zs = np.sin(self.xs)
        self.columns_ht = Columns({'x':self.xs, 'y':self.ys},
                                  kdims=['x'], vdims=['y'])

    # Test the constructor to be supported by all interfaces supporting
    # heterogeneous column types.

    def test_columns_ndelement_init_ht(self):
        "Tests support for heterogeneous NdElement (backwards compatibility)"
        columns = Columns(NdElement(zip(self.xs, self.ys), kdims=['x'], vdims=['y']))
        self.assertTrue(isinstance(columns.data, self.data_instance_type))

    def test_columns_dataframe_init_ht(self):
        "Tests support for heterogeneous DataFrames"
        if pd is None:
            raise SkipTest("Pandas not available")
        columns = Columns(pd.DataFrame({'x':self.xs, 'y':self.ys}), kdims=['x'], vdims=['y'])
        self.assertTrue(isinstance(columns.data, self.data_instance_type))

    # Test literal formats

    def test_columns_implicit_indexing_init(self):
        columns = Columns(self.ys, kdims=['x'], vdims=['y'])
        self.assertTrue(isinstance(columns.data, self.data_instance_type))

    def test_columns_tuple_init(self):
        columns = Columns((self.xs, self.ys), kdims=['x'], vdims=['y'])
        self.assertTrue(isinstance(columns.data, self.data_instance_type))

    def test_columns_simple_zip_init(self):
        columns = Columns(zip(self.xs, self.ys), kdims=['x'], vdims=['y'])
        self.assertTrue(isinstance(columns.data, self.data_instance_type))

    def test_columns_zip_init(self):
        columns = Columns(zip(self.gender, self.age,
                              self.weight, self.height),
                          kdims=self.kdims, vdims=self.vdims)
        self.assertTrue(isinstance(columns.data, self.data_instance_type))

    def test_columns_odict_init(self):
        columns = Columns(OrderedDict(zip(self.xs, self.ys)), kdims=['A'], vdims=['B'])
        self.assertTrue(isinstance(columns.data, self.data_instance_type))

    def test_columns_dict_init(self):
        columns = Columns(dict(zip(self.xs, self.ys)), kdims=['A'], vdims=['B'])
        self.assertTrue(isinstance(columns.data, self.data_instance_type))

    # Operations

    def test_columns_sort_vdim_ht(self):
        columns = Columns({'x':self.xs, 'y':-self.ys},
                          kdims=['x'], vdims=['y'])
        columns_sorted = Columns({'x':self.xs[::-1], 'y':-self.ys[::-1]},
                                 kdims=['x'], vdims=['y'])
        self.assertEqual(columns.sort('y'), columns_sorted)

    def test_columns_sort_string_ht(self):
        columns_sorted = Columns({'Gender':['F','M','M'], 'Age':[12,10,16],
                                  'Weight':[10,15,18], 'Height':[0.8,0.8,0.6]},
                                 kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(self.table.sort(), columns_sorted)

    def test_columns_sample_ht(self):
        samples = self.columns_ht.sample([0, 5, 10]).dimension_values('y')
        self.assertEqual(samples, np.array([0, 0.5, 1]))

    def test_columns_reduce_ht(self):
        reduced = Columns({'Age':self.age, 'Weight':self.weight, 'Height':self.height},
                          kdims=self.kdims[1:], vdims=self.vdims)
        self.assertEqual(self.table.reduce(['Gender'], np.mean), reduced)

    def test_columns_1D_reduce_ht(self):
        self.assertEqual(self.columns_ht.reduce('x', np.mean), np.float64(0.5))

    def test_columns_2D_reduce_ht(self):
        reduced = Columns({'Weight':[14.333333333333334], 'Height':[0.73333333333333339]},
                          kdims=[], vdims=self.vdims)
        self.assertEqual(self.table.reduce(function=np.mean), reduced)

    def test_columns_2D_partial_reduce_ht(self):
        columns = Columns({'x':self.xs, 'y':self.ys, 'z':self.zs},
                          kdims=['x', 'y'], vdims=['z'])
        reduced = Columns({'x':self.xs, 'z':self.zs},
                          kdims=['x'], vdims=['z'])
        self.assertEqual(columns.reduce(['y'], np.mean), reduced)

    def test_column_aggregate_ht(self):
        aggregated = Columns({'Gender':['M','F'], 'Weight':[16.5,10], 'Height':[0.7,0.8]},
                             kdims=self.kdims[:1], vdims=self.vdims)
        self.compare_columns(self.table.aggregate(['Gender'], np.mean), aggregated)

    def test_columns_2D_aggregate_partial_ht(self):
        columns = Columns({'x':self.xs, 'y':self.ys, 'z':self.zs},
                          kdims=['x', 'y'], vdims=['z'])
        reduced = Columns({'x':self.xs, 'z':self.zs},
                          kdims=['x'], vdims=['z'])
        self.assertEqual(columns.aggregate(['x'], np.mean), reduced)


    def test_columns_groupby(self):
        group1 = {'Age':[10,16], 'Weight':[15,18], 'Height':[0.8,0.6]}
        group2 = {'Age':[12], 'Weight':[10], 'Height':[0.8]}
        with sorted_context(False):
            grouped = HoloMap([('M', Columns(group1, kdims=['Age'], vdims=self.vdims)),
                               ('F', Columns(group2, kdims=['Age'], vdims=self.vdims))],
                              kdims=['Gender'])
        self.assertEqual(self.table.groupby(['Gender']), grouped)


    def test_columns_add_dimensions_value_ht(self):
        table = self.columns_ht.add_dimension('z', 1, 0)
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.zeros(len(table)))

    def test_columns_add_dimensions_values_ht(self):
        table = self.columns_ht.add_dimension('z', 1, range(1,12))
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.array(list(range(1,12))))

    # Indexing

    def test_columns_index_row_gender_female(self):
        indexed = Columns({'Gender':['F'],'Age':[12],
                           'Weight':[10], 'Height':[0.8]},
                          kdims=self.kdims, vdims=self.vdims)
        row = self.table['F',:]
        self.assertEquals(row, indexed)

    def test_columns_index_rows_gender_male(self):
        row = self.table['M',:]
        indexed = Columns({'Gender':['M','M'],'Age':[10,16],
                           'Weight':[15,18], 'Height':[0.8,0.6]},
                          kdims=self.kdims, vdims=self.vdims)
        self.assertEquals(row, indexed)

    def test_columns_index_row_age(self):
        indexed = Columns({'Gender':['F'],'Age':[12],
                           'Weight':[10], 'Height':[0.8]},
                          kdims=self.kdims, vdims=self.vdims)
        self.assertEquals(self.table[:, 12], indexed)

    def test_columns_index_item_table(self):
        indexed = Columns({'Gender':['F'],'Age':[12],
                           'Weight':[10], 'Height':[0.8]},
                          kdims=self.kdims, vdims=self.vdims)
        self.assertEquals(self.table['F', 12], indexed)

    def test_columns_index_value1(self):
        self.assertEquals(self.table['F', 12, 'Weight'], 10)

    def test_columns_index_value2(self):
        self.assertEquals(self.table['F', 12, 'Height'], 0.8)

    def test_columns_index_column_ht(self):
        self.compare_arrays(self.columns_ht['y'], self.ys)

    def test_columns_boolean_index(self):
        row = self.table[np.array([True, True, False])]
        indexed = Columns({'Gender':['M','M'],'Age':[10,16],
                           'Weight':[15,18], 'Height':[0.8,0.6]},
                          kdims=self.kdims, vdims=self.vdims)
        self.assertEquals(row, indexed)

    def test_columns_value_dim_index(self):
        row = self.table[:, :, 'Weight']
        indexed = Columns({'Gender':['M','M','F'],'Age':[10,16, 12],
                           'Weight':[15,18, 10]},
                          kdims=self.kdims, vdims=self.vdims[:1])
        self.assertEquals(row, indexed)

    def test_columns_value_dim_scalar_index(self):
        row = self.table['M', 10, 'Weight']
        self.assertEquals(row, 15)

    # Casting

    def test_columns_array_ht(self):
        self.assertEqual(self.columns_ht.array(),
                         np.column_stack([self.xs, self.ys]))


class ArrayColumnsTest(HomogeneousColumnTypes, ComparisonTestCase):
    """
    Test of the ArrayColumns interface.
    """
    def setUp(self):
        self.restore_datatype = Columns.datatype
        Columns.datatype = ['array']
        self.data_instance_type = np.ndarray
        self.init_data()


class DFColumnsTest(HeterogeneousColumnTypes, ComparisonTestCase):
    """
    Test of the pandas DFColumns interface.
    """

    def setUp(self):
        if pd is None:
            raise SkipTest("Pandas not available")
        self.restore_datatype = Columns.datatype
        Columns.datatype = ['dataframe']
        self.data_instance_type = pd.DataFrame
        self.init_data()



class DictColumnsTest(HeterogeneousColumnTypes, ComparisonTestCase):
    """
    Test of the generic dictionary interface.
    """

    def setUp(self):
        self.restore_datatype = Columns.datatype
        Columns.datatype = ['dictionary']
        self.data_instance_type = (dict, cyODict, OrderedDict)
        self.init_data()



class NdColumnsTest(HeterogeneousColumnTypes, ComparisonTestCase):
    """
    Test of the NdColumns interface (mostly for backwards compatibility)
    """

    def setUp(self):
        self.restore_datatype = Columns.datatype
        Columns.datatype = ['ndelement']
        self.data_instance_type = NdElement
        self.init_data()

    # Literal formats that have been previously been supported but
    # currently are only supported via NdElement.

    def test_columns_double_zip_init(self):
        columns = Columns(zip(zip(self.gender, self.age),
                              zip(self.weight, self.height)),
                          kdims=self.kdims, vdims=self.vdims)
        self.assertTrue(isinstance(columns.data, NdElement))

