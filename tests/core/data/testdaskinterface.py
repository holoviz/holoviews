from unittest import SkipTest

try:
    import dask.dataframe as dd
except:
    dd = None

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
