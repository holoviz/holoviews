from datetime import datetime

import narwhals as nw
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
    data_type = (nw.DataFrame, nw.Series)
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
        assert self.dataset_hm.interface.dtype(self.dataset_hm, 'x').dtype == nw.Int64
        assert self.dataset_hm.interface.dtype(self.dataset_hm, 'y').dtype == nw.Int64

    def test_dataset_dataset_ht_dtypes(self):
        ds = self.table
        assert isinstance(ds.interface.dtype(ds, 'Gender').dtype, nw.String)
        assert isinstance(ds.interface.dtype(ds, 'Age').dtype, nw.Int64)
        assert isinstance(ds.interface.dtype(ds, 'Weight').dtype, nw.Int64)
        assert isinstance(ds.interface.dtype(ds, 'Height').dtype, nw.Float64)

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
        np.testing.assert_array_equal(df["x"], self.xs)
        np.testing.assert_array_equal(df["y"], self.y_ints)

    def test_dataset_get_dframe_by_dimension(self):
        df = self.dataset_hm.dframe(['x'])
        np.testing.assert_array_equal(df, nw.from_native(self.frame({'x': self.xs})))

    def test_dataset_range_with_dimension_range(self):
        dt64 = [datetime(2017, 1, i) for i in range(1, 4)]
        ds = Dataset(self.frame({"Date": dt64}), [Dimension('Date', range=(dt64[0], dt64[-1]))])
        assert ds.range('Date'), (dt64[0], dt64[-1])

    @pytest.mark.filterwarnings("ignore:Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated:FutureWarning")
    def test_select_with_neighbor(self):
        select = self.table.interface.select_mask(self.table.dataset, {"Weight": 18})
        select_neighbor = self.table.interface._select_mask_neighbor(self.table.dataset, {"Weight": 18})

        assert len(self.table.data.filter(select)) == 1
        assert len(self.table.data.filter(select_neighbor)) == 3


class PandasNarwhalsInterfaceTests(BaseNarwhalsInterfaceTests):
    __test__ = True
    narwhals_backend = "pandas"


# class PolarsNarwhalsInterfaceTests(BaseNarwhalsInterfaceTests):
#     __test__ = True
#     narwhals_backend = "polars"
#
#
# class PyarrowNarwhalsInterfaceTests(BaseNarwhalsInterfaceTests):
#     __test__ = True
#     narwhals_backend = "pyarrow"
#
#     def frame(self, *args, **kwargs):
#         mod = pytest.importorskip(self.narwhals_backend)
#         return mod.table(*args, **kwargs)
