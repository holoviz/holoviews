import narwhals as nw
import pytest

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

    def test_dataset_dataset_ht_dtypes(self):
        ds = self.table
        assert isinstance(ds.interface.dtype(ds, 'Gender').dtype, nw.String)
        assert isinstance(ds.interface.dtype(ds, 'Age').dtype, nw.Int64)
        assert isinstance(ds.interface.dtype(ds, 'Weight').dtype, nw.Int64)
        assert isinstance(ds.interface.dtype(ds, 'Height').dtype, nw.Float64)


class PandasNarwhalsInterfaceTests(BaseNarwhalsInterfaceTests):
    __test__ = True
    narwhals_backend = "pandas"


# class PolarsNarwhalsInterfaceTests(BaseNarwhalsInterfaceTests):
#     __test__ = True
#     narwhals_backend = "polars"
