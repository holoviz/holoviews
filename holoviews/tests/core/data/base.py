"""
Tests for the Dataset Element types.
"""

import datetime

import numpy as np

from holoviews import Dataset, HoloMap, Dimension
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Scatter, Curve
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim


import pandas as pd


class DatatypeContext:

    def __init__(self, datatypes, dataset_type=Dataset):
        self.datatypes = datatypes
        self.dataset_type = dataset_type
        self._old_datatypes = {}

    def __enter__(self):
        if isinstance(self.dataset_type, tuple):
            for ds in self.dataset_type:
                self._old_datatypes[ds] = ds.datatype
                ds.datatype = self.datatypes
        else:
            self._old_datatypes = self.dataset_type.datatype
            self.dataset_type.datatype = self.datatypes

    def __exit__(self, *args):
        if isinstance(self.dataset_type, tuple):
            for ds in self.dataset_type:
                ds.datatype = self._old_datatypes[ds]
        else:
            self.dataset_type.datatype = self._old_datatypes



class InterfaceTests(ComparisonTestCase):
    """
    Tests for ImageInterface
    """

    datatype = 'interface'
    data_type = None
    element = Dataset

    def setUp(self):
        self.restore_datatype = self.element.datatype
        self.element.datatype = [self.datatype]
        self.init_column_data()
        self.init_grid_data()
        self.init_data()

    def tearDown(self):
        self.element.datatype = self.restore_datatype

    def init_column_data(self):
        pass

    def init_grid_data(self):
        pass

    def init_data(self):
        pass



class HomogeneousColumnTests:
    """
    Tests for data formats that require all dataset to have the same
    type (e.g. numpy arrays)
    """

    __test__ = False

    def init_column_data(self):
        self.xs = np.array(range(11))
        self.xs_2 = self.xs**2

        self.y_ints = self.xs*2
        self.dataset_hm = Dataset((self.xs, self.y_ints),
                                  kdims=['x'], vdims=['y'])
        self.dataset_hm_alias = Dataset((self.xs, self.y_ints),
                                        kdims=[('x', 'X')], vdims=[('y', 'Y')])

    # Test the array constructor (homogeneous data) to be supported by
    # all interfaces.

    def test_dataset_array_init_hm(self):
        dataset = Dataset(np.column_stack([self.xs, self.xs_2]),
                          kdims=['x'], vdims=['x2'])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_dtypes(self):
        self.assertEqual(self.dataset_hm.interface.dtype(self.dataset_hm, 'x'), np.dtype(int))
        self.assertEqual(self.dataset_hm.interface.dtype(self.dataset_hm, 'y'), np.dtype(int))

    def test_dataset_array_init_hm_tuple_dims(self):
        dataset = Dataset(np.column_stack([self.xs, self.xs_2]),
                          kdims=[('x', 'X')], vdims=[('x2', 'X2')])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_dataframe_init_hm(self):
        "Tests support for homogeneous DataFrames"
        dataset = Dataset(pd.DataFrame({'x':self.xs, 'x2':self.xs_2}),
                          kdims=['x'], vdims=['x2'])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_dataframe_init_hm_alias(self):
        "Tests support for homogeneous DataFrames"
        dataset = Dataset(pd.DataFrame({'x':self.xs, 'x2':self.xs_2}),
                          kdims=[('x', 'X-label')], vdims=[('x2', 'X2-label')])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_empty_list_init(self):
        dataset = Dataset([], kdims=['x'], vdims=['y'])
        for d in 'xy':
            self.assertEqual(dataset.dimension_values(d), np.array([]))

    def test_dataset_dict_dim_not_found_raises_on_array(self):
        with self.assertRaises(ValueError):
            Dataset({'x': np.zeros(5)}, kdims=['Test'], vdims=[])

    def test_dataset_dict_dim_not_found_raises_on_scalar(self):
        with self.assertRaises(ValueError):
            Dataset({'x': 1}, kdims=['Test'], vdims=[])

    # Properties and information

    def test_dataset_shape(self):
        self.assertEqual(self.dataset_hm.shape, (11, 2))

    def test_dataset_range(self):
        self.assertEqual(self.dataset_hm.range('y'), (0, 20))

    def test_dataset_closest(self):
        closest = self.dataset_hm.closest([0.51, 1, 9.9])
        self.assertEqual(closest, [1., 1., 10.])

    # Operations

    def test_dataset_sort_hm(self):
        ds = Dataset(([2, 2, 1], [2,1,2], [0.1, 0.2, 0.3]),
                     kdims=['x', 'y'], vdims=['z']).sort()
        ds_sorted = Dataset(([1, 2, 2], [2, 1, 2], [0.3, 0.2, 0.1]),
                            kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(ds.sort(), ds_sorted)

    def test_dataset_sort_reverse_hm(self):
        ds = Dataset(([2, 1, 2, 1], [2, 2, 1, 1], [0.1, 0.2, 0.3, 0.4]),
                     kdims=['x', 'y'], vdims=['z'])
        ds_sorted = Dataset(([2, 2, 1, 1], [2, 1, 2, 1], [0.1, 0.3, 0.2, 0.4]),
                            kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(ds.sort(reverse=True), ds_sorted)

    def test_dataset_sort_vdim_hm(self):
        xs_2 = np.array(self.xs_2)
        dataset = Dataset(np.column_stack([self.xs, -xs_2]),
                          kdims=['x'], vdims=['y'])
        dataset_sorted = Dataset(np.column_stack([self.xs[::-1], -xs_2[::-1]]),
                                 kdims=['x'], vdims=['y'])
        self.assertEqual(dataset.sort('y'), dataset_sorted)

    def test_dataset_sort_reverse_vdim_hm(self):
        xs_2 = np.array(self.xs_2)
        dataset = Dataset(np.column_stack([self.xs, -xs_2]),
                          kdims=['x'], vdims=['y'])
        dataset_sorted = Dataset(np.column_stack([self.xs, -xs_2]),
                                 kdims=['x'], vdims=['y'])
        self.assertEqual(dataset.sort('y', reverse=True), dataset_sorted)

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

    def test_dataset_redim_hm_kdim_range_aux(self):
        redimmed = self.dataset_hm.redim.range(x=(-100,3))
        self.assertEqual(redimmed.kdims[0].range, (-100,3))

    def test_dataset_redim_hm_kdim_soft_range_aux(self):
        redimmed = self.dataset_hm.redim.soft_range(x=(-100,30))
        self.assertEqual(redimmed.kdims[0].soft_range, (-100,30))

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

    # Tabular indexing

    def test_dataset_iloc_slice_rows(self):
        sliced = self.dataset_hm.iloc[1:4]
        table = Dataset({'x': self.xs[1:4], 'y': self.y_ints[1:4]},
                        kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_slice_rows_slice_cols(self):
        sliced = self.dataset_hm.iloc[1:4, 1:]
        table = Dataset({'y': self.y_ints[1:4]}, kdims=[], vdims=['y'],
                        datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_slice_rows_list_cols(self):
        sliced = self.dataset_hm.iloc[1:4, [0, 1]]
        table = Dataset({'x': self.xs[1:4], 'y': self.y_ints[1:4]},
                        kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_slice_rows_index_cols(self):
        sliced = self.dataset_hm.iloc[1:4, 1]
        table = Dataset({'y': self.y_ints[1:4]}, kdims=[], vdims=['y'],
                        datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_list_rows(self):
        sliced = self.dataset_hm.iloc[[0, 2]]
        table = Dataset({'x': self.xs[[0, 2]], 'y': self.y_ints[[0, 2]]},
                        kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_list_rows_list_cols(self):
        sliced = self.dataset_hm.iloc[[0, 2], [0, 1]]
        table = Dataset({'x': self.xs[[0, 2]], 'y': self.y_ints[[0, 2]]},
                        kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_list_rows_list_cols_by_name(self):
        sliced = self.dataset_hm.iloc[[0, 2], ['x', 'y']]
        table = Dataset({'x': self.xs[[0, 2]], 'y': self.y_ints[[0, 2]]},
                        kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_list_rows_slice_cols(self):
        sliced = self.dataset_hm.iloc[[0, 2], slice(0, 2)]
        table = Dataset({'x': self.xs[[0, 2]], 'y': self.y_ints[[0, 2]]},
                        kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_index_rows_index_cols(self):
        indexed = self.dataset_hm.iloc[1, 1]
        self.assertEqual(indexed, self.y_ints[1])

    def test_dataset_iloc_index_rows_slice_cols(self):
        indexed = self.dataset_hm.iloc[1, :2]
        table = Dataset({'x':self.xs[[1]],  'y':self.y_ints[[1]]},
                        kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(indexed, table)

    def test_dataset_iloc_list_cols(self):
        sliced = self.dataset_hm.iloc[:, [0, 1]]
        table = Dataset({'x':self.xs,  'y':self.y_ints},
                        kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_list_cols_by_name(self):
        sliced = self.dataset_hm.iloc[:, ['x', 'y']]
        table = Dataset({'x':self.xs,  'y':self.y_ints},
                        kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_ellipsis_list_cols(self):
        sliced = self.dataset_hm.iloc[..., [0, 1]]
        table = Dataset({'x':self.xs,  'y':self.y_ints},
                        kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_ellipsis_list_cols_by_name(self):
        sliced = self.dataset_hm.iloc[..., ['x', 'y']]
        table = Dataset({'x':self.xs,  'y':self.y_ints},
                        kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_get_array_by_dimension(self):
        arr = self.dataset_hm.array(['x'])
        self.assertEqual(arr, self.xs[:, np.newaxis])

    def test_dataset_get_dframe(self):
        df = self.dataset_hm.dframe()
        self.assertEqual(df.x.values, self.xs)
        self.assertEqual(df.y.values, self.y_ints)

    def test_dataset_get_dframe_by_dimension(self):
        df = self.dataset_hm.dframe(['x'])
        self.assertEqual(df, pd.DataFrame({'x': self.xs}, dtype=df.dtypes.iloc[0]))

    def test_dataset_transform_replace_hm(self):
        transformed = self.dataset_hm.transform(y=dim('y')*2)
        expected = Dataset((self.xs, self.y_ints*2), 'x', 'y')
        self.assertEqual(transformed, expected)

    def test_dataset_transform_add_hm(self):
        transformed = self.dataset_hm.transform(y2=dim('y')*2)
        expected = Dataset((self.xs, self.y_ints, self.y_ints*2), 'x', ['y', 'y2'])
        self.assertEqual(transformed, expected)



class HeterogeneousColumnTests(HomogeneousColumnTests):
    """
    Tests for data formats that allow dataset to have varied types
    """

    __test__ = False

    def init_column_data(self):
        self.kdims = ['Gender', 'Age']
        self.vdims = ['Weight', 'Height']
        self.gender, self.age = np.array(['M','M','F']), np.array([10,16,12])
        self.weight, self.height = np.array([15,18,10]), np.array([0.8,0.6,0.8])
        self.table = Dataset({'Gender':self.gender, 'Age':self.age,
                              'Weight':self.weight, 'Height':self.height},
                             kdims=self.kdims, vdims=self.vdims)

        self.alias_kdims = [('gender', 'Gender'), ('age', 'Age')]
        self.alias_vdims = [('weight', 'Weight'), ('height', 'Height')]
        self.alias_table = Dataset({'gender':self.gender, 'age':self.age,
                                    'weight':self.weight, 'height':self.height},
                                   kdims=self.alias_kdims, vdims=self.alias_vdims)

        super().init_column_data()
        self.ys = np.linspace(0, 1, 11)
        self.zs = np.sin(self.xs)
        self.dataset_ht = Dataset({'x':self.xs, 'y':self.ys},
                                  kdims=['x'], vdims=['y'])

    # Test the constructor to be supported by all interfaces supporting
    # heterogeneous column types.

    def test_dataset_dataframe_init_ht(self):
        "Tests support for heterogeneous DataFrames"
        dataset = Dataset(pd.DataFrame({'x':self.xs, 'y':self.ys}), kdims=['x'], vdims=['y'])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_dataframe_init_ht_alias(self):
        "Tests support for heterogeneous DataFrames"
        dataset = Dataset(pd.DataFrame({'x':self.xs, 'y':self.ys}),
                          kdims=[('x', 'X')], vdims=[('y', 'Y')])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    # Test dtypes

    def test_dataset_dataset_ht_dtypes(self):
        ds = self.table
        self.assertEqual(ds.interface.dtype(ds, 'Gender'), np.dtype('object'))
        self.assertEqual(ds.interface.dtype(ds, 'Age'), np.dtype(int))
        self.assertEqual(ds.interface.dtype(ds, 'Weight'), np.dtype(int))
        self.assertEqual(ds.interface.dtype(ds, 'Height'), np.dtype('float64'))

    # Test literal formats

    def test_dataset_expanded_dimvals_ht(self):
        # This will run unique(), which for pandas return
        # in order of appearance, but can be sorted for other
        # interfaces like cudf.
        #   pd.Series(["M", "M", "F"]).unique()   -> ["M", "F"]
        #   cudf.Series(["M", "M", "F"]).unique() -> ["F", "M"]
        data = self.table.dimension_values('Gender', expanded=False)
        self.assertEqual(np.sort(data), np.array(['F', 'M']))

    def test_dataset_implicit_indexing_init(self):
        dataset = Scatter(self.ys, kdims=['x'], vdims=['y'])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_tuple_init(self):
        dataset = Dataset((self.xs, self.ys), kdims=['x'], vdims=['y'])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_tuple_init_alias(self):
        dataset = Dataset((self.xs, self.ys), kdims=[('x', 'X')], vdims=[('y', 'Y')])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_simple_zip_init(self):
        dataset = Dataset(zip(self.xs, self.ys), kdims=['x'], vdims=['y'])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_simple_zip_init_alias(self):
        dataset = Dataset(zip(self.xs, self.ys), kdims=[('x', 'X')], vdims=[('y', 'Y')])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_zip_init(self):
        dataset = Dataset(zip(self.gender, self.age,
                              self.weight, self.height),
                          kdims=self.kdims, vdims=self.vdims)
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_zip_init_alias(self):
        dataset = self.alias_table.clone(zip(self.gender, self.age,
                                             self.weight, self.height))
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_odict_init(self):
        dataset = Dataset(dict(zip(self.xs, self.ys)), kdims=['A'], vdims=['B'])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_odict_init_alias(self):
        dataset = Dataset(dict(zip(self.xs, self.ys)),
                          kdims=[('a', 'A')], vdims=[('b', 'B')])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_dict_init(self):
        dataset = Dataset(dict(zip(self.xs, self.ys)), kdims=['A'], vdims=['B'])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_range_with_dimension_range(self):
        dt64 = np.array([np.datetime64(datetime.datetime(2017, 1, i)) for i in range(1, 4)])
        ds = Dataset(dt64, [Dimension('Date', range=(dt64[0], dt64[-1]))])
        self.assertEqual(ds.range('Date'), (dt64[0], dt64[-1]))

    # Operations

    def test_dataset_redim_with_alias_dframe(self):
        test_df = pd.DataFrame({'x': range(10), 'y': range(0,20,2)})
        dataset = Dataset(test_df, kdims=[('x', 'X-label')], vdims=['y'])
        redim_df = pd.DataFrame({'X': range(10), 'y': range(0,20,2)})
        dataset_redim = Dataset(redim_df, kdims=['X'], vdims=['y'])
        self.assertEqual(dataset.redim(**{'X-label':'X'}), dataset_redim)
        self.assertEqual(dataset.redim(x='X'), dataset_redim)

    def test_dataset_mixed_type_range(self):
        ds = Dataset((['A', 'B', 'C', None],), 'A')
        self.assertEqual(ds.range(0), ('A', 'C'))

    def test_dataset_nodata_range(self):
        table = self.table.clone(vdims=[Dimension('Weight', nodata=10), 'Height'])
        self.assertEqual(table.range('Weight'), (15, 18))

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

    def test_dataset_2D_aggregate_spread_fn_with_duplicates(self):
        dataset = Dataset({'x': np.array([0, 0, 1, 1]), 'y': np.array([0, 1, 2, 3]),
                           'z': np.array([1, 2, 3, 4])},
                          kdims=['x', 'y'], vdims=['z'])
        agg = dataset.aggregate('x', function=np.mean, spreadfn=np.var)
        self.assertEqual(agg, Dataset({'x': np.array([0, 1]), 'z': np.array([1.5, 3.5]),
                                       'z_var': np.array([0.25, 0.25])},
                                      kdims=['x'], vdims=['z', 'z_var']))

    def test_dataset_aggregate_ht(self):
        aggregated = Dataset({'Gender':['M', 'F'], 'Weight':[16.5, 10], 'Height':[0.7, 0.8]},
                             kdims=self.kdims[:1], vdims=self.vdims)
        self.compare_dataset(self.table.aggregate(['Gender'], np.mean), aggregated)

    def test_dataset_aggregate_string_types(self):
        ds = Dataset({'Gender':['M', 'M'], 'Weight':[20, 10], 'Name':['Peter', 'Matt']},
                             kdims='Gender', vdims=['Weight', 'Name'])
        aggregated = Dataset({'Gender': ['M'], 'Weight': [15]},
                             kdims='Gender', vdims=['Weight'])
        self.compare_dataset(ds.aggregate(['Gender'], np.mean), aggregated)

    def test_dataset_aggregate_string_types_size(self):
        ds = Dataset({'Gender':['M', 'M'], 'Weight':[20, 10], 'Name':['Peter', 'Matt']},
                             kdims='Gender', vdims=['Weight', 'Name'])
        aggregated = Dataset({'Gender': ['M'], 'Weight': [2], 'Name': [2]},
                             kdims='Gender', vdims=['Weight', 'Name'])
        self.compare_dataset(ds.aggregate(['Gender'], np.size), aggregated)

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

    def test_dataset_empty_aggregate(self):
        dataset = Dataset([], kdims=self.kdims, vdims=self.vdims)
        aggregated = Dataset([], kdims=self.kdims[:1], vdims=self.vdims)
        self.compare_dataset(dataset.aggregate(['Gender'], np.mean), aggregated)

    def test_dataset_empty_aggregate_with_spreadfn(self):
        dataset = Dataset([], kdims=self.kdims, vdims=self.vdims)
        aggregated = Dataset([], kdims=self.kdims[:1], vdims=[d for vd in self.vdims for d in [vd, vd+'_std']])
        self.compare_dataset(dataset.aggregate(['Gender'], np.mean, np.std), aggregated)

    def test_dataset_groupby(self):
        group1 = {'Age':[10,16], 'Weight':[15,18], 'Height':[0.8,0.6]}
        group2 = {'Age':[12], 'Weight':[10], 'Height':[0.8]}
        grouped = HoloMap([('M', Dataset(group1, kdims=['Age'], vdims=self.vdims)),
                           ('F', Dataset(group2, kdims=['Age'], vdims=self.vdims))],
                          kdims=['Gender'], sort=False)
        self.assertEqual(self.table.groupby(['Gender']), grouped)

    def test_dataset_groupby_alias(self):
        group1 = {'age':[10,16], 'weight':[15,18], 'height':[0.8,0.6]}
        group2 = {'age':[12], 'weight':[10], 'height':[0.8]}
        grouped = HoloMap([('M', Dataset(group1, kdims=[('age', 'Age')],
                                         vdims=self.alias_vdims)),
                           ('F', Dataset(group2, kdims=[('age', 'Age')],
                                         vdims=self.alias_vdims))],
                          kdims=[('gender', 'Gender')], sort=False)
        self.assertEqual(self.alias_table.groupby('Gender'), grouped)

    def test_dataset_groupby_second_dim(self):
        group1 = {'Gender':['M'], 'Weight':[15], 'Height':[0.8]}
        group2 = {'Gender':['M'], 'Weight':[18], 'Height':[0.6]}
        group3 = {'Gender':['F'], 'Weight':[10], 'Height':[0.8]}
        grouped = HoloMap([(10, Dataset(group1, kdims=['Gender'], vdims=self.vdims)),
                           (16, Dataset(group2, kdims=['Gender'], vdims=self.vdims)),
                           (12, Dataset(group3, kdims=['Gender'], vdims=self.vdims))],
                          kdims=['Age'], sort=False)
        self.assertEqual(self.table.groupby(['Age']), grouped)

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

    def test_redim_with_extra_dimension(self):
        dataset = self.dataset_ht.add_dimension('Temp', 0, 0).clone(kdims=['x', 'y'], vdims=[])
        redimmed = dataset.redim(x='Time')
        self.assertEqual(redimmed.dimension_values('Time'),
                         self.dataset_ht.dimension_values('x'))

    # Indexing

    def test_dataset_index_row_gender_female(self):
        indexed = Dataset({'Gender':['F'], 'Age':[12],
                           'Weight':[10], 'Height':[0.8]},
                          kdims=self.kdims, vdims=self.vdims)
        row = self.table['F',:]
        self.assertEqual(row, indexed)

    def test_dataset_index_rows_gender_male(self):
        row = self.table['M',:]
        indexed = Dataset({'Gender':['M', 'M'], 'Age':[10, 16],
                           'Weight':[15,18], 'Height':[0.8,0.6]},
                          kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(row, indexed)

    def test_dataset_select_rows_gender_male(self):
        row = self.table.select(Gender='M')
        indexed = Dataset({'Gender':['M', 'M'], 'Age':[10, 16],
                           'Weight':[15,18], 'Height':[0.8,0.6]},
                          kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(row, indexed)

    def test_dataset_select_rows_gender_male_expr(self):
        row = self.table.select(selection_expr=dim('Gender') == 'M')
        indexed = Dataset({'Gender': ['M', 'M'], 'Age': [10, 16],
                           'Weight': [15, 18], 'Height': [0.8,0.6]},
                          kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(row, indexed)

    def test_dataset_select_rows_gender_male_alias(self):
        row = self.alias_table.select(Gender='M')
        alias_row = self.alias_table.select(gender='M')
        indexed = Dataset({'gender':['M', 'M'], 'age':[10, 16],
                           'weight':[15,18], 'height':[0.8,0.6]},
                          kdims=self.alias_kdims, vdims=self.alias_vdims)
        self.assertEqual(row, indexed)
        self.assertEqual(alias_row, indexed)

    def test_dataset_index_row_age(self):
        indexed = Dataset({'Gender':['F'], 'Age':[12],
                           'Weight':[10], 'Height':[0.8]},
                          kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(self.table[:, 12], indexed)

    def test_dataset_index_item_table(self):
        indexed = Dataset({'Gender':['F'], 'Age':[12],
                           'Weight':[10], 'Height':[0.8]},
                          kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(self.table['F', 12], indexed)

    def test_dataset_index_value1(self):
        self.assertEqual(self.table['F', 12, 'Weight'], 10)

    def test_dataset_index_value2(self):
        self.assertEqual(self.table['F', 12, 'Height'], 0.8)

    def test_dataset_index_column_ht(self):
        self.compare_arrays(self.dataset_ht['y'], self.ys)

    def test_dataset_boolean_index(self):
        row = self.table[np.array([True, True, False])]
        indexed = Dataset({'Gender':['M', 'M'], 'Age':[10, 16],
                           'Weight':[15,18], 'Height':[0.8,0.6]},
                          kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(row, indexed)

    def test_dataset_value_dim_index(self):
        row = self.table[:, :, 'Weight']
        indexed = Dataset({'Gender':['M', 'M', 'F'], 'Age':[10, 16, 12],
                           'Weight':[15,18, 10]},
                          kdims=self.kdims, vdims=self.vdims[:1])
        self.assertEqual(row, indexed)

    def test_dataset_value_dim_scalar_index(self):
        row = self.table['M', 10, 'Weight']
        self.assertEqual(row, 15)

    # Tabular indexing

    def test_dataset_iloc_slice_rows(self):
        sliced = self.table.iloc[1:2]
        table = Dataset({'Gender':self.gender[1:2], 'Age':self.age[1:2],
                         'Weight':self.weight[1:2], 'Height':self.height[1:2]},
                        kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(sliced, table)

    def test_dataset_iloc_slice_rows_slice_cols(self):
        sliced = self.table.iloc[1:2, 1:3]
        table = Dataset({'Age':self.age[1:2], 'Weight':self.weight[1:2]},
                        kdims=self.kdims[1:], vdims=self.vdims[:1])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_slice_rows_list_cols(self):
        sliced = self.table.iloc[1:2, [1, 3]]
        table = Dataset({'Age':self.age[1:2], 'Height':self.height[1:2]},
                        kdims=self.kdims[1:], vdims=self.vdims[1:])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_slice_rows_index_cols(self):
        sliced = self.table.iloc[1:2, 2]
        table = Dataset({'Weight':self.weight[1:2]}, kdims=[], vdims=self.vdims[:1])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_list_rows(self):
        sliced = self.table.iloc[[0, 2]]
        table = Dataset({'Gender':self.gender[[0, 2]], 'Age':self.age[[0, 2]],
                         'Weight':self.weight[[0, 2]], 'Height':self.height[[0, 2]]},
                        kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(sliced, table)

    def test_dataset_iloc_list_rows_list_cols(self):
        sliced = self.table.iloc[[0, 2], [0, 2]]
        table = Dataset({'Gender':self.gender[[0, 2]],  'Weight':self.weight[[0, 2]]},
                        kdims=self.kdims[:1], vdims=self.vdims[:1])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_list_rows_list_cols_by_name(self):
        sliced = self.table.iloc[[0, 2], ['Gender', 'Weight']]
        table = Dataset({'Gender':self.gender[[0, 2]],  'Weight':self.weight[[0, 2]]},
                        kdims=self.kdims[:1], vdims=self.vdims[:1])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_list_rows_slice_cols(self):
        sliced = self.table.iloc[[0, 2], slice(1, 3)]
        table = Dataset({'Age':self.age[[0, 2]],  'Weight':self.weight[[0, 2]]},
                        kdims=self.kdims[1:], vdims=self.vdims[:1])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_index_rows_index_cols(self):
        indexed = self.table.iloc[1, 1]
        self.assertEqual(indexed, self.age[1])

    def test_dataset_iloc_index_rows_slice_cols(self):
        indexed = self.table.iloc[1, 1:3]
        table = Dataset({'Age':self.age[[1]],  'Weight':self.weight[[1]]},
                        kdims=self.kdims[1:], vdims=self.vdims[:1])
        self.assertEqual(indexed, table)

    def test_dataset_iloc_list_cols(self):
        sliced = self.table.iloc[:, [0, 2]]
        table = Dataset({'Gender':self.gender,  'Weight':self.weight},
                        kdims=self.kdims[:1], vdims=self.vdims[:1])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_list_cols_by_name(self):
        sliced = self.table.iloc[:, ['Gender', 'Weight']]
        table = Dataset({'Gender':self.gender,  'Weight':self.weight},
                        kdims=self.kdims[:1], vdims=self.vdims[:1])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_ellipsis_list_cols(self):
        sliced = self.table.iloc[..., [0, 2]]
        table = Dataset({'Gender':self.gender,  'Weight':self.weight},
                        kdims=self.kdims[:1], vdims=self.vdims[:1])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_ellipsis_list_cols_by_name(self):
        sliced = self.table.iloc[..., ['Gender', 'Weight']]
        table = Dataset({'Gender':self.gender,  'Weight':self.weight},
                        kdims=self.kdims[:1], vdims=self.vdims[:1])
        self.assertEqual(sliced, table)

    # Casting

    def test_dataset_array_ht(self):
        self.assertEqual(self.dataset_ht.array(),
                         np.column_stack([self.xs, self.ys]))

    # Transforms

    def test_dataset_transform_replace_ht(self):
        transformed = self.table.transform(
            Age=dim('Age')**2, Weight=dim('Weight')*2, Height=dim('Height')/2.
        )
        expected = Dataset({'Gender':self.gender, 'Age':self.age**2,
                            'Weight':self.weight*2, 'Height':self.height/2.},
                           kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(transformed, expected)

    def test_dataset_transform_add_ht(self):
        transformed = self.table.transform(combined=dim('Age')*dim('Weight'))
        expected = Dataset({'Gender':self.gender, 'Age':self.age,
                              'Weight':self.weight, 'Height':self.height,
                              'combined': self.age*self.weight},
                             kdims=self.kdims, vdims=self.vdims+['combined'])
        self.assertEqual(transformed, expected)



class ScalarColumnTests:
    """
    Tests for interfaces that allow on or more columns to be of scalar
    types.
    """

    __test__ = False

    def test_dataset_scalar_constructor(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
        self.assertEqual(ds.dimension_values('A'), np.ones(10))

    def test_dataset_scalar_length(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
        self.assertEqual(len(ds), 10)

    def test_dataset_scalar_array(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
        self.assertEqual(ds.array(), np.column_stack([np.ones(10), np.arange(10)]))

    def test_dataset_scalar_select(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
        self.assertEqual(ds.select(A=1).dimension_values('B'), np.arange(10))

    def test_dataset_scalar_select_expr(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
        self.assertEqual(
            ds.select(selection_expr=dim('A') == 1).dimension_values('B'),
            np.arange(10)
        )

    def test_dataset_scalar_empty_select(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
        self.assertEqual(ds.select(A=0).dimension_values('B'), np.array([]))

    def test_dataset_scalar_empty_select_expr(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
        self.assertEqual(
            ds.select(selection_expr=dim('A') == 0).dimension_values('B'),
            np.array([])
        )

    def test_dataset_scalar_sample(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
        self.assertEqual(ds.sample([(1,)]).dimension_values('B'), np.arange(10))

    def test_dataset_scalar_sort(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)[::-1]}, kdims=['A', 'B'])
        self.assertEqual(ds.sort().dimension_values('B'), np.arange(10))

    def test_dataset_scalar_groupby(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
        groups = ds.groupby('A')
        self.assertEqual(groups, HoloMap({1: Dataset({'B': np.arange(10)}, 'B')}, 'A'))

    def test_dataset_scalar_iloc(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
        self.assertEqual(ds.iloc[:5], Dataset({'A': 1, 'B': np.arange(5)}, kdims=['A', 'B']))



class GriddedInterfaceTests:
    """
    Tests for the grid interfaces
    """

    __test__ = False

    def init_grid_data(self):
        self.grid_xs = np.array([0, 1])
        self.grid_ys = np.array([0.1, 0.2, 0.3])
        self.grid_zs = np.array([[0, 1], [2, 3], [4, 5]])
        self.dataset_grid = self.element(
            (self.grid_xs, self.grid_ys, self.grid_zs), ['x', 'y'], ['z']
        )
        self.dataset_grid_alias = self.element(
            (self.grid_xs, self.grid_ys, self.grid_zs), [('x', 'X'), ('y', 'Y')], [('z', 'Z')]
        )
        self.dataset_grid_inv = self.element(
            (self.grid_xs[::-1], self.grid_ys[::-1], self.grid_zs), ['x', 'y'], ['z']
        )

    def test_canonical_vdim(self):
        x = np.array([ 0.  ,  0.75,  1.5 ])
        y = np.array([ 1.5 ,  0.75,  0.  ])
        z = np.array([[ 0.06925999,  0.05800389,  0.05620127],
                      [ 0.06240918,  0.05800931,  0.04969735],
                      [ 0.05376789,  0.04669417,  0.03880118]])
        dataset = self.element((x, y, z), kdims=['x', 'y'], vdims=['z'])
        canonical = np.array([[ 0.05376789,  0.04669417,  0.03880118],
                              [ 0.06240918,  0.05800931,  0.04969735],
                              [ 0.06925999,  0.05800389,  0.05620127]])
        self.assertEqual(dataset.dimension_values('z', flat=False),
                         canonical)

    def test_gridded_dtypes(self):
        ds = self.dataset_grid
        self.assertEqual(ds.interface.dtype(ds, 'x'), np.dtype(int))
        self.assertEqual(ds.interface.dtype(ds, 'y'), np.float64)
        self.assertEqual(ds.interface.dtype(ds, 'z'), np.dtype(int))

    def test_select_slice(self):
        ds = self.element(
            (self.grid_xs, self.grid_ys[:2], self.grid_zs[:2]), ['x', 'y'], ['z']
        )
        self.assertEqual(self.dataset_grid.select(y=slice(0, 0.25)), ds)

    def test_select_tuple(self):
        ds = self.element(
            (self.grid_xs, self.grid_ys[:2], self.grid_zs[:2]), ['x', 'y'], ['z']
        )
        self.assertEqual(self.dataset_grid.select(y=(0, 0.25)), ds)

    def test_nodata_range(self):
        ds = self.dataset_grid.clone(vdims=[Dimension('z', nodata=0)])
        self.assertEqual(ds.range('z'), (1, 5))

    def test_dataset_ndloc_index(self):
        xs, ys = np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5)
        arr = np.arange(10)*np.arange(5)[np.newaxis].T
        ds = self.element((xs, ys, arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        self.assertEqual(ds.ndloc[0,0], arr[0, 0])

    def test_dataset_ndloc_index_inverted_x(self):
        xs, ys = np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5)
        arr = np.arange(10)*np.arange(5)[np.newaxis].T
        ds = self.element((xs[::-1], ys, arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        self.assertEqual(ds.ndloc[0,0], arr[0, 9])

    def test_dataset_ndloc_index_inverted_y(self):
        xs, ys = np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5)
        arr = np.arange(10)*np.arange(5)[np.newaxis].T
        ds = self.element((xs, ys[::-1], arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        self.assertEqual(ds.ndloc[0,0], arr[4, 0])

    def test_dataset_ndloc_index_inverted_xy(self):
        xs, ys = np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5)
        arr = np.arange(10)*np.arange(5)[np.newaxis].T
        ds = self.element((xs[::-1], ys[::-1], arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        self.assertEqual(ds.ndloc[0,0], arr[4, 9])

    def test_dataset_ndloc_index2(self):
        xs, ys = np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5)
        arr = np.arange(10)*np.arange(5)[np.newaxis].T
        ds = self.element((xs, ys, arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        self.assertEqual(ds.ndloc[4, 9], arr[4, 9])

    def test_dataset_ndloc_slice(self):
        xs, ys = np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5)
        arr = np.arange(10)*np.arange(5)[np.newaxis].T
        ds = self.element((xs, ys, arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        sliced = self.element((xs[2:5], ys[1:], arr[1:, 2:5]), kdims=['x', 'y'], vdims=['z'],
                         datatype=[self.datatype])
        self.assertEqual(ds.ndloc[1:, 2:5], sliced)

    def test_dataset_ndloc_slice_inverted_x(self):
        xs, ys = np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5)
        arr = np.arange(10)*np.arange(5)[np.newaxis].T
        ds = self.element((xs[::-1], ys, arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        sliced = self.element((xs[::-1][5:8], ys[1:], arr[1:, 5:8]), kdims=['x', 'y'], vdims=['z'],
                         datatype=[self.datatype])
        self.assertEqual(ds.ndloc[1:, 2:5], sliced)

    def test_dataset_ndloc_slice_inverted_y(self):
        xs, ys = np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5)
        arr = np.arange(10)*np.arange(5)[np.newaxis].T
        ds = self.element((xs, ys[::-1], arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        sliced = self.element((xs[2:5], ys[::-1][:-1], arr[:-1, 2:5]), kdims=['x', 'y'], vdims=['z'],
                         datatype=[self.datatype])
        self.assertEqual(ds.ndloc[1:, 2:5], sliced)

    def test_dataset_ndloc_slice_inverted_xy(self):
        xs, ys = np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5)
        arr = np.arange(10)*np.arange(5)[np.newaxis].T
        ds = self.element((xs[::-1], ys[::-1], arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        sliced = self.element((xs[::-1][5:8], ys[::-1][:-1], arr[:-1, 5:8]), kdims=['x', 'y'], vdims=['z'],
                         datatype=[self.datatype])
        self.assertEqual(ds.ndloc[1:, 2:5], sliced)

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
        expanded_xs = np.array([[0, 1], [0, 1], [0, 1]])
        self.assertEqual(self.dataset_grid.dimension_values(0, flat=False),
                         expanded_xs)

    def test_dataset_dim_vals_grid_kdims_expanded_xs_inv(self):
        expanded_xs = np.array([[0, 1], [0, 1], [0, 1]])
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
        expanded_ys = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
        self.assertEqual(self.dataset_grid.dimension_values(1, flat=False),
                         expanded_ys)

    def test_dataset_dim_vals_grid_kdims_expanded_ys_inv(self):
        expanded_ys = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
        self.assertEqual(self.dataset_grid_inv.dimension_values(1, flat=False),
                         expanded_ys)

    def test_dataset_dim_vals_dimensions_match_shape(self):
        self.assertEqual(len({self.dataset_grid.dimension_values(i, flat=False).shape
                                 for i in range(3)}), 1)

    def test_dataset_dim_vals_dimensions_match_shape_inv(self):
        self.assertEqual(len({self.dataset_grid_inv.dimension_values(i, flat=False).shape
                                 for i in range(3)}), 1)

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

    def test_dataset_groupby_with_transposed_dimensions(self):
        dat = np.zeros((3,5,7))
        dataset = Dataset((range(7), range(5), range(3), dat), ['z','x','y'], 'value')
        grouped = dataset.groupby('z', kdims=['y', 'x'])
        self.assertEqual(grouped.last.dimension_values(2, flat=False), dat[:, :, -1].T)

    def test_dataset_dynamic_groupby_with_transposed_dimensions(self):
        dat = np.zeros((3,5,7))
        dataset = Dataset((range(7), range(5), range(3), dat), ['z','x','y'], 'value')
        grouped = dataset.groupby('z', kdims=['y', 'x'], dynamic=True)
        self.assertEqual(grouped[2].dimension_values(2, flat=False), dat[:, :, -1].T)

    def test_dataset_slice_inverted_dimension(self):
        xs = np.arange(30)[::-1]
        ys = np.random.rand(30)
        ds = Dataset((xs, ys), 'x', 'y')
        sliced = ds[5:15]
        self.assertEqual(sliced, Dataset((xs[15:25], ys[15:25]), 'x', 'y'))

    def test_sample_2d(self):
        xs = ys = np.linspace(0, 6, 50)
        XS, YS = np.meshgrid(xs, ys)
        values = np.sin(XS)
        sampled = Dataset((xs, ys, values), ['x', 'y'], 'z').sample(y=0)
        self.assertEqual(sampled, Curve((xs, values[0]), vdims='z'))

    def test_aggregate_2d_with_spreadfn(self):
        array = np.random.rand(10, 5)
        ds = Dataset((range(5), range(10), array), ['x', 'y'], 'z')
        agg = ds.aggregate('x', np.mean, np.std)
        example = Dataset((range(5), array.mean(axis=0), array.std(axis=0)), 'x', ['z', 'z_std'])
        self.assertEqual(agg, example)

    def test_concat_grid_3d(self):
        array = np.random.rand(4, 5, 3, 2)
        orig = Dataset((range(2), range(3), range(5), range(4), array), ['A', 'B', 'x', 'y'], 'z')
        hmap = HoloMap({(i, j): self.element((range(5), range(4), array[:, :, j, i]), ['x', 'y'], 'z')
                        for i in range(2) for j in range(3)}, ['A', 'B'])
        ds = concat(hmap)
        self.assertEqual(ds, orig)

    def test_concat_grid_3d_shape_mismatch(self):
        ds1 = Dataset(([0, 1], [1, 2, 3], np.random.rand(3, 2)), ['x', 'y'], 'z')
        ds2 = Dataset(([0, 1, 2], [1, 2], np.random.rand(2, 3)), ['x', 'y'], 'z')
        hmap = HoloMap({1: ds1, 2: ds2})
        with self.assertRaises(DataError):
            concat(hmap)

    def test_grid_3d_groupby_concat_roundtrip(self):
        array = np.random.rand(4, 5, 3, 2)
        orig = Dataset((range(2), range(3), range(5), range(4), array), ['A', 'B', 'x', 'y'], 'z')
        self.assertEqual(concat(orig.groupby(['A', 'B'])), orig)
