from unittest import SkipTest
import unittest

import numpy as np
import pandas as pd
from packaging.version import Version

try:
    import dask.dataframe as dd
except ImportError:
    raise SkipTest("Could not import dask, skipping DaskInterface tests.")

from holoviews.core.data import Dataset
from holoviews.util.transform import dim
from holoviews.core.util import pandas_version

from .test_pandasinterface import BasePandasInterfaceTests


class DaskDatasetTest(BasePandasInterfaceTests):
    """
    Test of the pandas DaskDataset interface.
    """

    datatype = 'dask'
    data_type = dd.DataFrame

    __test__ = True

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

    def test_dataset_sort_reverse_vdim_hm(self):
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

    def test_dataset_2D_aggregate_partial_hm(self):
        raise SkipTest("Temporarily skipped")

    def test_dataset_2D_aggregate_partial_ht(self):
        raise SkipTest("Temporarily skipped")

    def test_dataset_2D_partial_reduce_ht(self):
        raise SkipTest("Temporarily skipped")

    def test_dataset_aggregate_string_types(self):
        raise SkipTest("Temporarily skipped")

    @unittest.skipIf(
        pandas_version >= Version("2.0"),
        reason="Not supported yet, https://github.com/dask/dask/issues/9913"
    )
    def test_dataset_aggregate_ht(self):
        super().test_dataset_aggregate_ht()

    @unittest.skipIf(
        pandas_version >= Version("2.0"),
        reason="Not supported yet, https://github.com/dask/dask/issues/9913"
    )
    def test_dataset_aggregate_ht_alias(self):
        super().test_dataset_aggregate_ht_alias()

    def test_dataset_from_multi_index(self):
        raise SkipTest("Temporarily skipped")
        df = pd.DataFrame({'x': np.arange(10), 'y': np.arange(10), 'z': np.random.rand(10)})
        ddf = dd.from_pandas(df, 1)
        ds = Dataset(ddf.groupby(['x', 'y']).mean(), ['x', 'y'])
        self.assertEqual(ds, Dataset(df, ['x', 'y']))

    def test_dataset_from_multi_index_tuple_dims(self):
        raise SkipTest("Temporarily skipped")
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

    def test_select_expression_lazy(self):
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 10, 11, 11, 10],
        })
        ddf = dd.from_pandas(df, npartitions=2)
        ds = Dataset(ddf)
        new_ds = ds.select(selection_expr=dim('b') == 10)

        # Make sure that selecting by expression didn't cause evaluation
        self.assertIsInstance(new_ds.data, dd.DataFrame)
        self.assertEqual(new_ds.data.compute(), df[df.b == 10])
