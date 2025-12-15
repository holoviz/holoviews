import numpy as np
import pandas as pd
import pytest

from holoviews.core.data import Dataset
from holoviews.core.util.dependencies import (
    PANDAS_GE_3_0_0,
    PANDAS_VERSION,
    _no_import_version,
)
from holoviews.testing import assert_data_equal, assert_element_equal
from holoviews.util.transform import dim

from ...utils import optional_dependencies
from .test_pandasinterface import BasePandasInterfaceTests

dask, dask_skip = optional_dependencies("dask")

if dask:
    import dask.dataframe as dd

_DASK_CONVERT_STRING = (
    _no_import_version("dask") >= (2023, 7, 1)
    and dask.config.get("dataframe.convert-string") in (True, None)
)


@dask_skip
class DaskDatasetTest(BasePandasInterfaceTests):
    """
    Test of the pandas DaskDataset interface.
    """

    datatype = 'dask'
    force_sort = True
    __test__ = True

    @property
    def data_type(self):
        return dd.DataFrame

    def frame(self, *args, **kwargs):
        df = pd.DataFrame(*args, **kwargs)
        return dd.from_pandas(df, npartitions=2)

    # Disabled tests for NotImplemented methods
    def test_dataset_add_dimensions_values_hm(self):
        pytest.skip("Not supported")

    def test_dataset_add_dimensions_values_ht(self):
        pytest.skip("Not supported")

    def test_dataset_2D_aggregate_spread_fn_with_duplicates(self):
        pytest.skip("Not supported")

    def test_dataset_sort_hm(self):
        pytest.skip("Not supported")

    def test_dataset_sort_reverse_hm(self):
        pytest.skip("Not supported")

    def test_dataset_sort_reverse_vdim_hm(self):
        pytest.skip("Not supported")

    def test_dataset_sort_vdim_ht(self):
        pytest.skip("Not supported")

    def test_dataset_sort_vdim_hm(self):
        pytest.skip("Not supported")

    def test_dataset_sort_vdim_hm_alias(self):
        pytest.skip("Not supported")

    def test_dataset_sort_string_ht(self):
        pytest.skip("Not supported")

    def test_dataset_boolean_index(self):
        pytest.skip("Not supported")

    def test_dataset_aggregate_string_types_size(self):
        pytest.skip("Not supported")

    def test_dataset_2D_aggregate_partial_hm(self):
        pytest.skip("Temporarily skipped")

    def test_dataset_2D_aggregate_partial_ht(self):
        pytest.skip("Temporarily skipped")

    def test_dataset_2D_partial_reduce_ht(self):
        pytest.skip("Temporarily skipped")

    def test_dataset_aggregate_string_types(self):
        pytest.skip("Temporarily skipped")

    @pytest.mark.skipif(
        PANDAS_VERSION >= (2, 0, 0),
        reason="Not supported yet, https://github.com/dask/dask/issues/9913"
    )
    def test_dataset_aggregate_ht(self):
        super().test_dataset_aggregate_ht()

    @pytest.mark.skipif(
        PANDAS_VERSION >= (2, 0, 0),
        reason="Not supported yet, https://github.com/dask/dask/issues/9913"
    )
    def test_dataset_aggregate_ht_alias(self):
        super().test_dataset_aggregate_ht_alias()

    def test_dataset_from_multi_index(self):
        pytest.skip("Temporarily skipped")
        df = pd.DataFrame({'x': np.arange(10), 'y': np.arange(10), 'z': np.random.rand(10)})
        ddf = dd.from_pandas(df, 1)
        ds = Dataset(ddf.groupby(['x', 'y']).mean(), ['x', 'y'])
        assert_element_equal(ds, Dataset(df, ['x', 'y']))

    def test_dataset_from_multi_index_tuple_dims(self):
        pytest.skip("Temporarily skipped")
        df = pd.DataFrame({'x': np.arange(10), 'y': np.arange(10), 'z': np.random.rand(10)})
        ddf = dd.from_pandas(df, 1)
        ds = Dataset(ddf.groupby(['x', 'y']).mean(), [('x', 'X'), ('y', 'Y')])
        assert_element_equal(ds, Dataset(df, [('x', 'X'), ('y', 'Y')]))

    def test_dataset_range_categorical_dimension(self):
        ddf = self.frame({'a': ['1', '2', '3']})
        ds = Dataset(ddf)
        assert ds.range(0) == ('1', '3')

    def test_dataset_range_categorical_dimension_empty(self):
        ddf = self.frame({'a': ['1', '2', '3']})
        ds = Dataset(ddf).iloc[:0]
        ds_range = ds.range(0)
        assert np.isnan(ds_range[0])
        assert np.isnan(ds_range[1])

    def test_select_expression_lazy(self):
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 10, 11, 11, 10],
        })
        ddf = self.frame(df)
        ds = Dataset(ddf)
        new_ds = ds.select(selection_expr=dim('b') == 10)

        # Make sure that selecting by expression didn't cause evaluation
        assert isinstance(new_ds.data, dd.DataFrame)
        assert_data_equal(new_ds.data.compute(), df[df.b == 10])

    def test_dataset_get_dframe_by_dimension(self):
        df = self.dataset_hm.dframe(['x'])
        expected = self.frame({'x': self.xs}, dtype=df.dtypes.iloc[0]).compute()
        assert_data_equal(df, expected)

    def test_dataset_dataset_ht_dtypes(self):
        ds = self.table
        if _DASK_CONVERT_STRING:
            string_dtype = pd.StringDtype(na_value=pd.NA, storage="pyarrow")
        elif PANDAS_GE_3_0_0:
            string_dtype = pd.StringDtype(na_value=np.nan)
        else:
            string_dtype = np.dtype('object')
        assert ds.interface.dtype(ds, 'Gender') == string_dtype
        assert ds.interface.dtype(ds, 'Age') == np.dtype(int)
        assert ds.interface.dtype(ds, 'Weight') == np.dtype(int)
        assert ds.interface.dtype(ds, 'Height') == np.dtype('float64')
