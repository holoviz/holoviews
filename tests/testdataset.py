"""
Tests for the Dataset Element types.
"""

from unittest import SkipTest
import numpy as np
from holoviews import Dataset, NdElement, HoloMap
from holoviews.element.comparison import ComparisonTestCase

from collections import OrderedDict
from holoviews.core.dimension import OrderedDict as cyODict

try:
    import pandas as pd
except:
    pd = None



class HomogeneousColumnTypes(object):
    """
    Tests for data formats that require all dataset to have the same
    type (e.g numpy arrays)
    """

    def setUp(self):
        self.restore_datatype = Dataset.datatype
        self.data_instance_type = None

    def init_data(self):
        self.xs = range(11)
        self.xs_2 = [el**2 for el in self.xs]

        self.y_ints = [i*2 for i in range(11)]
        self.dataset_hm = Dataset((self.xs, self.y_ints),
                                  kdims=['x'], vdims=['y'])

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

    def test_dataset_sample_hm(self):
        samples = self.dataset_hm.sample([0, 5, 10]).dimension_values('y')
        self.assertEqual(samples, np.array([0, 10, 20]))

    def test_dataset_array_hm(self):
        self.assertEqual(self.dataset_hm.array(),
                         np.column_stack([self.xs, self.y_ints]))

    def test_dataset_add_dimensions_value_hm(self):
        table = self.dataset_hm.add_dimension('z', 1, 0)
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.zeros(len(table)))

    def test_dataset_add_dimensions_values_hm(self):
        table =  self.dataset_hm.add_dimension('z', 1, range(1,12))
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.array(list(range(1,12))))

    def test_dataset_slice_hm(self):
        dataset_slice = Dataset({'x':range(5, 9), 'y':[2 * i for i in range(5, 9)]},
                                kdims=['x'], vdims=['y'])
        self.assertEqual(self.dataset_hm[5:9], dataset_slice)

    def test_dataset_slice_fn_hm(self):
        dataset_slice = Dataset({'x':range(5, 9), 'y':[2 * i for i in range(5, 9)]},
                                kdims=['x'], vdims=['y'])
        self.assertEqual(self.dataset_hm[lambda x: (x >= 5) & (x < 9)], dataset_slice)

    def test_dataset_1D_reduce_hm(self):
        dataset = Dataset({'x':self.xs, 'y':self.y_ints}, kdims=['x'], vdims=['y'])
        self.assertEqual(dataset.reduce('x', np.mean), 10)

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

    def init_data(self):
        self.kdims = ['Gender', 'Age']
        self.vdims = ['Weight', 'Height']
        self.gender, self.age = ['M','M','F'], [10,16,12]
        self.weight, self.height = [15,18,10], [0.8,0.6,0.8]
        self.table = Dataset({'Gender':self.gender, 'Age':self.age,
                              'Weight':self.weight, 'Height':self.height},
                             kdims=self.kdims, vdims=self.vdims)

        super(HeterogeneousColumnTypes, self).init_data()
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

    def test_dataset_simple_zip_init(self):
        dataset = Dataset(zip(self.xs, self.ys), kdims=['x'], vdims=['y'])
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    def test_dataset_zip_init(self):
        dataset = Dataset(zip(self.gender, self.age,
                              self.weight, self.height),
                          kdims=self.kdims, vdims=self.vdims)
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    def test_dataset_odict_init(self):
        dataset = Dataset(OrderedDict(zip(self.xs, self.ys)), kdims=['A'], vdims=['B'])
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    def test_dataset_dict_init(self):
        dataset = Dataset(dict(zip(self.xs, self.ys)), kdims=['A'], vdims=['B'])
        self.assertTrue(isinstance(dataset.data, self.data_instance_type))

    # Operations

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

    def test_column_aggregate_ht(self):
        aggregated = Dataset({'Gender':['M', 'F'], 'Weight':[16.5, 10], 'Height':[0.7, 0.8]},
                             kdims=self.kdims[:1], vdims=self.vdims)
        self.compare_dataset(self.table.aggregate(['Gender'], np.mean), aggregated)

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


    def test_dataset_add_dimensions_value_ht(self):
        table = self.dataset_ht.add_dimension('z', 1, 0)
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.zeros(len(table)))

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
        self.init_data()


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
        self.init_data()



class DictDatasetTest(HeterogeneousColumnTypes, ComparisonTestCase):
    """
    Test of the generic dictionary interface.
    """

    def setUp(self):
        self.restore_datatype = Dataset.datatype
        Dataset.datatype = ['dictionary']
        self.data_instance_type = (dict, cyODict, OrderedDict)
        self.init_data()



class NdDatasetTest(HeterogeneousColumnTypes, ComparisonTestCase):
    """
    Test of the NdDataset interface (mostly for backwards compatibility)
    """

    def setUp(self):
        self.restore_datatype = Dataset.datatype
        Dataset.datatype = ['ndelement']
        self.data_instance_type = NdElement
        self.init_data()

    # Literal formats that have been previously been supported but
    # currently are only supported via NdElement.

    def test_dataset_double_zip_init(self):
        dataset = Dataset(zip(zip(self.gender, self.age),
                              zip(self.weight, self.height)),
                          kdims=self.kdims, vdims=self.vdims)
        self.assertTrue(isinstance(dataset.data, NdElement))


class GridDatasetTest(HomogeneousColumnTypes, ComparisonTestCase):
    """
    Test of the NdDataset interface (mostly for backwards compatibility)
    """

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

    def test_dataset_ndelement_init_hm(self):
        "Tests support for homogeneous NdElement (backwards compatibility)"
        exception = "None of the available storage backends "\
         "were able to support the supplied data format."
        with self.assertRaisesRegexp(Exception, exception):
            Dataset(NdElement(zip(self.xs, self.xs_2),
                              kdims=['x'], vdims=['x2']))

    def test_dataset_2D_aggregate_partial_hm(self):
        array = np.random.rand(11, 11)
        dataset = Dataset({'x':self.xs, 'y':self.y_ints, 'z': array},
                          kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(dataset.aggregate(['x'], np.mean),
                         Dataset({'x':self.xs, 'z': np.mean(array, axis=0)},
                                 kdims=['x'], vdims=['z']))

    def test_dataset_2D_reduce_hm(self):
        array = np.random.rand(11, 11)
        dataset = Dataset({'x':self.xs, 'y':self.y_ints, 'z': array},
                          kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(np.array(dataset.reduce(['x', 'y'], np.mean)),
                         np.mean(array))

    def test_dataset_add_dimensions_value_hm(self):
        with self.assertRaisesRegexp(Exception, 'Cannot add key dimension to a dense representation.'):
            self.dataset_hm.add_dimension('z', 1, 0)

    def test_dataset_add_dimensions_values_hm(self):
        table =  self.dataset_hm.add_dimension('z', 1, range(1,12), vdim=True)
        self.assertEqual(table.vdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.array(list(range(1,12))))

    def test_dataset_sort_vdim_hm(self):
        exception = ('Compressed format cannot be sorted, either instantiate '
                     'in the desired order or use the expanded format.')
        with self.assertRaisesRegexp(Exception, exception):
            self.dataset_hm.sort('y')

    def test_dataset_groupby(self):
        self.assertEqual(self.dataset_hm.groupby('x').keys(), list(self.xs))

