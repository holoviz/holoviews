import re
from datetime import datetime

import narwhals.stable.v2 as nw
import numpy as np
import pytest

from holoviews import Dataset, Dimension
from holoviews.core.data import NarwhalsInterface

from .base import HeterogeneousColumnTests, InterfaceTests


class BaseNarwhalsInterfaceTests(HeterogeneousColumnTests, InterfaceTests):
    """
    Test for the NarwhalsInterface.
    """

    __test__ = False

    datatype = "narwhals"
    data_type = nw.DataFrame
    narwhals_backend = None

    def setUp(self):
        pytest.importorskip(self.narwhals_backend)
        NarwhalsInterface.narwhals_backend = self.narwhals_backend
        super().setUp()

    def tearDown(self):
        NarwhalsInterface.narwhals_backend = None
        super().tearDown()

    def frame(self, *args, **kwargs):
        mod = pytest.importorskip(self.narwhals_backend)
        return mod.DataFrame(*args, **kwargs)

    def test_dataset_dtypes(self):
        assert self.dataset_hm.interface.dtype(self.dataset_hm, "x").dtype == nw.Int64
        assert self.dataset_hm.interface.dtype(self.dataset_hm, "y").dtype == nw.Int64

    def test_dataset_dataset_ht_dtypes(self):
        ds = self.table
        assert isinstance(ds.interface.dtype(ds, "Gender").dtype, nw.String)
        assert isinstance(ds.interface.dtype(ds, "Age").dtype, nw.Int64)
        assert isinstance(ds.interface.dtype(ds, "Weight").dtype, nw.Int64)
        assert isinstance(ds.interface.dtype(ds, "Height").dtype, nw.Float64)

    @pytest.mark.xfail(reason="Doesn't really make sense")
    def test_dataset_dict_init(self):
        super().test_dataset_dict_init()

    @pytest.mark.xfail(reason="Doesn't really make sense")
    def test_dataset_dict_init_alias(self):
        super().test_dataset_dict_init_alias()

    @pytest.mark.xfail(reason="Doesn't really make sense")
    def test_dataset_empty_aggregate(self):
        super().test_dataset_empty_aggregate()

    @pytest.mark.xfail(reason="Doesn't really make sense")
    def test_dataset_empty_aggregate_with_spreadfn(self):
        super().test_dataset_empty_aggregate_with_spreadfn()

    @pytest.mark.xfail(reason="Doesn't really make sense")
    def test_dataset_empty_list_init(self):
        super().test_dataset_empty_list_init()

    @pytest.mark.xfail(reason="Doesn't really make sense")
    def test_dataset_implicit_indexing_init(self):
        super().test_dataset_implicit_indexing_init(self)

    @pytest.mark.xfail(reason="Doesn't really make sense")
    def test_dataset_zip_init(self):
        super().test_dataset_zip_init()

    @pytest.mark.xfail(reason="Doesn't really make sense")
    def test_dataset_zip_init_alias(self):
        super().test_dataset_zip_init_alias()

    @pytest.mark.xfail(reason="Doesn't really make sense")
    def test_dataset_simple_zip_init(self):
        super().test_dataset_simple_zip_init()

    @pytest.mark.xfail(reason="Doesn't really make sense")
    def test_dataset_simple_zip_init_alias(self):
        super().test_dataset_simple_zip_init_alias()

    def test_dataset_get_dframe(self):
        df = self.dataset_hm.dframe()
        if isinstance(df, nw.LazyFrame):
            exp_x = df.select("x").collect()["x"]
            exp_y = df.select("y").collect()["y"]
        else:
            exp_x = df["x"]
            exp_y = df["y"]

        np.testing.assert_array_equal(exp_x, self.xs)
        np.testing.assert_array_equal(exp_y, self.y_ints)

    def test_dataset_get_dframe_by_dimension(self):
        df = self.dataset_hm.dframe(["x"])
        expected = self.frame({"x": self.xs})
        df_lazy, expected_lazy = False, False
        if isinstance(df, nw.LazyFrame):
            df = df.collect()
            df_lazy = True
        if hasattr(expected, "compute"):
            expected = expected.compute()
            expected_lazy = True
        if hasattr(expected, "collect"):
            expected = expected.collect()
            expected_lazy = True
        assert df_lazy == expected_lazy
        np.testing.assert_array_equal(df, expected)

    def test_dataset_range_with_dimension_range(self):
        dt64 = [datetime(2017, 1, i) for i in range(1, 4)]
        ds = Dataset(
            self.frame({"Date": dt64}), [Dimension("Date", range=(dt64[0], dt64[-1]))]
        )
        assert ds.range("Date"), (dt64[0], dt64[-1])

    @pytest.mark.filterwarnings(
        "ignore:Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated:FutureWarning"
    )
    def test_select_with_neighbor(self):
        select = self.table.interface.select_mask(self.table.dataset, {"Weight": 18})
        select_neighbor = self.table.interface._select_mask_neighbor(
            self.table.dataset, {"Weight": 18}
        )

        assert len(self.table.data.filter(select)) == 1
        assert len(self.table.data.filter(select_neighbor)) == 3

    def test_histogram(self):
        df = nw.from_native(self.frame({'values': [1, 1, 2, 2, 2, 3, 3, 4, 4, 5]}))
        bins = [1, 2, 3, 4, 5, 6]

        hist, edges = NarwhalsInterface.histogram(df, bins=bins)

        assert isinstance(hist, np.ndarray)
        assert isinstance(edges, np.ndarray)
        assert len(hist) == len(bins) - 1
        assert len(edges) == len(bins)
        np.testing.assert_array_equal(edges, bins)
        assert np.all(hist >= 0)

    def test_scalar_getitem(self):
        df = nw.from_native(self.frame({
            "day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "count": [1, 2, 3, 4, 5, 6, 7],
        }))
        ds = Dataset(df, kdims=["day"], vdims=["count"])
        assert ds["Mon"] == 1

    def test_non_scalar_getitem(self):
        df = nw.from_native(self.frame({
            "day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "count": [1, 2, 3, 4, 5, 6, 7],
        }))
        ds = Dataset(df, kdims=["day"], vdims=["count"])
        result = ds[["Mon"]].data
        assert isinstance(result, self.data_type)
        if isinstance(result, nw.LazyFrame):
            result = result.collect()
        assert result.shape == (1, 2)


class PandasNarwhalsInterfaceTests(BaseNarwhalsInterfaceTests):
    __test__ = True
    narwhals_backend = "pandas"


class PolarsNarwhalsInterfaceTests(BaseNarwhalsInterfaceTests):
    __test__ = True
    narwhals_backend = "polars"
    force_sort = True


class PyarrowNarwhalsInterfaceTests(BaseNarwhalsInterfaceTests):
    __test__ = True
    narwhals_backend = "pyarrow"

    def frame(self, *args, **kwargs):
        pa = pytest.importorskip("pyarrow")
        return pa.table(*args, **kwargs)


class BaseNarwhalsLazyInterfaceTests(BaseNarwhalsInterfaceTests):
    data_type = (nw.DataFrame, nw.LazyFrame)

    @pytest.mark.xfail(reason="Not supported", raises=NotImplementedError)
    def test_select_with_neighbor(self):
        super().test_select_with_neighbor()


class PolarsNarwhalsLazyInterfaceTests(BaseNarwhalsLazyInterfaceTests):
    __test__ = True
    narwhals_backend = "polars"
    force_sort = True

    def frame(self, *args, **kwargs):
        pl = pytest.importorskip("polars")
        return pl.LazyFrame(*args, **kwargs)


class DaskNarwhalsLazyInterfaceTests(BaseNarwhalsLazyInterfaceTests):
    __test__ = True
    narwhals_backend = "pandas"
    force_sort = True

    def setUp(self):
        pytest.importorskip("dask.dataframe")
        super().setUp()

    def frame(self, *args, **kwargs):
        import dask.dataframe as dd
        import pandas as pd

        return dd.from_pandas(pd.DataFrame(*args, **kwargs), npartitions=2)


class IbisNarwhalsLazyInterfaceTests(BaseNarwhalsLazyInterfaceTests):
    __test__ = True
    narwhals_backend = "pyarrow"
    force_sort = True

    def setUp(self):
        ibis = pytest.importorskip("ibis")
        super().setUp()
        ibis.set_backend("sqlite")

    def tearDown(self):
        import ibis

        ibis.set_backend(None)
        super().tearDown()

    def frame(self, *args, **kwargs):
        import ibis

        return ibis.memtable(*args, **kwargs)

    def test_dataset_get_dframe_by_dimension(self):
        df = self.dataset_hm.dframe(["x"])
        assert isinstance(df, nw.LazyFrame)

        expected = self.frame({"x": self.xs})
        assert df.to_native().to_pyarrow() == expected.to_pyarrow()

    @pytest.mark.xfail(reason="need to investigate failure")
    def test_dataset_aggregate_ht(self):
        # raises AttributeError: 'StringColumn' object has no attribute 'mean'
        return super().test_dataset_aggregate_ht()

    @pytest.mark.xfail(reason="need to investigate failure")
    def test_dataset_aggregate_ht_alias(self):
        # raises AttributeError: 'StringColumn' object has no attribute 'mean'
        return super().test_dataset_aggregate_ht_alias()

    @pytest.mark.xfail(reason="need to investigate failure")
    def test_dataset_aggregate_string_types(self):
        # raises AttributeError: 'StringColumn' object has no attribute 'mean'
        return super().test_dataset_aggregate_string_types()


class DuckdbNarwhalsLazyInterfaceTests(BaseNarwhalsLazyInterfaceTests):
    __test__ = True
    narwhals_backend = "pyarrow"
    force_sort = True

    def setUp(self):
        pytest.importorskip("duckdb")
        super().setUp()

    def frame(self, *args, **kwargs):
        import duckdb
        import pyarrow as pa

        return duckdb.from_arrow(pa.table(*args, **kwargs))

    def test_dataset_get_dframe_by_dimension(self):
        df = self.dataset_hm.dframe(["x"])
        assert isinstance(df, nw.LazyFrame)

        expected = self.frame({"x": self.xs})
        assert df.to_native().to_arrow_table() == expected.to_arrow_table()


@pytest.mark.gpu
class CudfNarwhalsInterfaceTests(BaseNarwhalsInterfaceTests):
    __test__ = True
    narwhals_backend = "cudf"
    force_sort = True

    def frame(self, *args, **kwargs):
        import cudf
        import pandas as pd

        return cudf.from_pandas(pd.DataFrame(*args, **kwargs))

    def test_dataset_get_dframe_by_dimension(self):
        df = self.dataset_hm.dframe(["x"])
        expected = self.frame({"x": self.xs})
        np.testing.assert_array_equal(df.to_numpy(), expected.to_numpy())

    def test_dataset_groupby_dynamic(self):
        msg = "Series object is not iterable."
        with pytest.raises(TypeError, match=re.escape(msg)):
            super().test_dataset_groupby_dynamic()

    def test_dataset_groupby_dynamic_alias(self):
        msg = "Series object is not iterable."
        with pytest.raises(TypeError, match=re.escape(msg)):
            super().test_dataset_groupby_dynamic()

    def test_dataset_nodata_range(self):
        msg = "cudf does not support mixed types, please type-cast the column of dataframe/series and other to same dtypes."
        with pytest.raises(TypeError, match=re.escape(msg)):
            return super().test_dataset_nodata_range()
