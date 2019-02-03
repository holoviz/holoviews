from unittest import SkipTest

import numpy as np

try:
    import pandas as pd
    import dask.dataframe as dd
except:
    raise SkipTest("Could not import dask, skipping DaskInterface tests.")

from holoviews.core.data import Dataset

from .testpandasinterface import PandasInterfaceTests


class DaskDatasetTest(PandasInterfaceTests):
    """
    Test of the pandas DaskDataset interface.
    """

    datatype = 'dask'
    data_type = dd.DataFrame

    # Disabled tests for NotImplemented methods
    def test_dataset_add_dimensions_values_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_add_dimensions_values_ht(self):
        raise SkipTest("Not supported")

    def test_dataset_2D_aggregate_spread_fn_with_duplicates(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_reverse_hm(self):
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

    def test_dataset_aggregate_string_types_size(self):
        raise SkipTest("Not supported")

    def test_dataset_from_multi_index(self):
        df = pd.DataFrame({'x': np.arange(10), 'y': np.arange(10), 'z': np.random.rand(10)})
        ddf = dd.from_pandas(df, 1)
        ds = Dataset(ddf.groupby(['x', 'y']).mean(), ['x', 'y'])
        self.assertEqual(ds, Dataset(df, ['x', 'y']))
    
    def test_dataset_from_multi_index_tuple_dims(self):
        df = pd.DataFrame({'x': np.arange(10), 'y': np.arange(10), 'z': np.random.rand(10)})
        ddf = dd.from_pandas(df, 1)
        ds = Dataset(ddf.groupby(['x', 'y']).mean(), [('x', 'X'), ('y', 'Y')])
        self.assertEqual(ds, Dataset(df, [('x', 'X'), ('y', 'Y')]))

    def test_dataset_range_categorical_dimension(self):
        ddf = dd.from_pandas(pd.DataFrame({'a': ['1', '2', '3']}), 1)
        ds = Dataset(ddf)
        self.assertEqual(ds.range(0), ('1', '3'))

    def test_dataset_range_categorical_dimension_empty(self):
        ddf = dd.from_pandas(pd.DataFrame({'a': ['1', '2', '3']}), 1)
        ds = Dataset(ddf).iloc[:0]
        ds_range = ds.range(0)
        self.assertTrue(np.isnan(ds_range[0]))
        self.assertTrue(np.isnan(ds_range[1]))
