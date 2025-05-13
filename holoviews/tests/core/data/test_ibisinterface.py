import sqlite3
import uuid
import warnings
from tempfile import NamedTemporaryFile
from unittest import SkipTest

try:
    import ibis
    # Getting this Warnings on Python 3.13 and Ibis 9.5
    # DeprecationWarning: Attribute.__init__ missing 1 required positional argument: 'value'.
    # This will become an error in Python 3.15.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ibis import sqlite
except ImportError:
    raise SkipTest("Could not import ibis, skipping IbisInterface tests.")

import numpy as np
import pandas as pd

from holoviews.core.data import Dataset
from holoviews.core.data.ibis import IBIS_VERSION, IbisInterface
from holoviews.core.spaces import HoloMap

from .base import HeterogeneousColumnTests, InterfaceTests


class IbisDatasetTest(HeterogeneousColumnTests, InterfaceTests):
    """
    Test of the generic dictionary interface.
    """

    datatype = "ibis"
    data_type = (ibis.expr.types.Expr,)

    __test__ = True

    def frame(self, *args, **kwargs):
        df = pd.DataFrame(*args, **kwargs)
        with NamedTemporaryFile(delete=False) as my_file:
            filename = my_file.name
        name = uuid.uuid4().hex
        con = sqlite3.Connection(filename)
        df.to_sql(name, con, index=False)
        return sqlite.connect(filename).table(name)

    def test_dataset_array_init_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_dict_dim_not_found_raises_on_scalar(self):
        raise SkipTest("Not supported")

    def test_dataset_array_init_hm_tuple_dims(self):
        raise SkipTest("Not supported")

    def test_dataset_dict_init(self):
        raise SkipTest("Not supported")

    def test_dataset_dict_init_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_simple_zip_init(self):
        raise SkipTest("Not supported")

    def test_dataset_simple_zip_init_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_zip_init(self):
        raise SkipTest("Not supported")

    def test_dataset_zip_init_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_tuple_init(self):
        raise SkipTest("Not supported")

    def test_dataset_tuple_init_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_implicit_indexing_init(self):
        raise SkipTest("Not supported")

    def test_dataset_dataframe_init_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_dataframe_init_hm_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_dataframe_init_ht(self):
        raise SkipTest("Not supported")

    def test_dataset_dataframe_init_ht_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_add_dimensions_values_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_add_dimensions_values_ht(self):
        raise SkipTest("Not supported")

    def test_dataset_dataset_ht_dtypes(self):
        int_dtype = "int64" if IBIS_VERSION >= (9, 0, 0) else "int32"
        ds = self.table
        self.assertEqual(ds.interface.dtype(ds, "Gender"), np.dtype("object"))
        self.assertEqual(ds.interface.dtype(ds, "Age"), np.dtype(int_dtype))
        self.assertEqual(ds.interface.dtype(ds, "Weight"), np.dtype(int_dtype))
        self.assertEqual(ds.interface.dtype(ds, "Height"), np.dtype("float64"))

    def test_dataset_dtypes(self):
        int_dtype = "int64" if IBIS_VERSION >= (9, 0, 0) else "int32"
        self.assertEqual(
            self.dataset_hm.interface.dtype(self.dataset_hm, "x"), np.dtype(int_dtype)
        )
        self.assertEqual(
            self.dataset_hm.interface.dtype(self.dataset_hm, "y"), np.dtype(int_dtype)
        )

    def test_dataset_reduce_ht(self):
        reduced = Dataset(
            self.frame({"Age": self.age, "Weight": self.weight, "Height": self.height}),
            kdims=self.kdims[1:],
            vdims=self.vdims,
        )
        self.assertEqual(self.table.reduce(["Gender"], np.mean).sort(), reduced.sort())

    def test_dataset_aggregate_ht(self):
        aggregated = Dataset(
            self.frame({"Gender": ["M", "F"], "Weight": [16.5, 10], "Height": [0.7, 0.8]}),
            kdims=self.kdims[:1],
            vdims=self.vdims,
        )
        self.compare_dataset(
            self.table.aggregate(["Gender"], np.mean).sort(), aggregated.sort()
        )

    def test_dataset_aggregate_ht_alias(self):
        aggregated = Dataset(
            self.frame({"gender": ["M", "F"], "weight": [16.5, 10], "height": [0.7, 0.8]}),
            kdims=self.alias_kdims[:1],
            vdims=self.alias_vdims,
        )
        self.compare_dataset(
            self.alias_table.aggregate("Gender", np.mean).sort(), aggregated.sort()
        )

    def test_dataset_groupby(self):
        group1 = {"Age": [10, 16], "Weight": [15, 18], "Height": [0.8, 0.6]}
        group2 = {"Age": [12], "Weight": [10], "Height": [0.8]}
        grouped = HoloMap(
            [
                ("M", Dataset(group1, kdims=["Age"], vdims=self.vdims)),
                ("F", Dataset(group2, kdims=["Age"], vdims=self.vdims)),
            ],
            kdims=["Gender"],
        )
        self.assertEqual(
            self.table.groupby(["Gender"]).apply("sort"), grouped.apply("sort")
        )

    def test_dataset_groupby_alias(self):
        group1 = self.frame({"age": [10, 16], "weight": [15, 18], "height": [0.8, 0.6]})
        group2 = self.frame({"age": [12], "weight": [10], "height": [0.8]})
        grouped = HoloMap(
            [
                ("M", Dataset(group1, kdims=[("age", "Age")], vdims=self.alias_vdims)),
                ("F", Dataset(group2, kdims=[("age", "Age")], vdims=self.alias_vdims)),
            ],
            kdims=[("gender", "Gender")],
        )
        self.assertEqual(self.alias_table.groupby("Gender").apply("sort"), grouped)

    def test_dataset_groupby_second_dim(self):
        group1 = self.frame({"Gender": ["M"], "Weight": [15], "Height": [0.8]})
        group2 = self.frame({"Gender": ["M"], "Weight": [18], "Height": [0.6]})
        group3 = self.frame({"Gender": ["F"], "Weight": [10], "Height": [0.8]})
        grouped = HoloMap(
            [
                (10, Dataset(group1, kdims=["Gender"], vdims=self.vdims)),
                (16, Dataset(group2, kdims=["Gender"], vdims=self.vdims)),
                (12, Dataset(group3, kdims=["Gender"], vdims=self.vdims)),
            ],
            kdims=["Age"],
            sort=True,
        )
        self.assertEqual(self.table.groupby(["Age"]), grouped)

    def test_aggregation_operations(self):
        for agg in [
            np.min, np.nanmin, np.max, np.nanmax, np.mean, np.nanmean,
            np.sum, np.nansum, len, np.count_nonzero,
            # TODO: var-based operations failing this test
            # np.std, np.nanstd, np.var, np.nanvar
        ]:
            expected = self.table.clone().aggregate("Gender", agg).sort()
            result = self.table.aggregate("Gender", agg).sort()

            self.compare_dataset(expected, result, msg=str(agg))

    def test_select_with_neighbor(self):
        try:
            # Not currently supported by Ibis
            super().test_select_with_neighbor()
        except NotImplementedError:
            raise SkipTest("Not supported")

    if not IbisInterface.has_rowid():

        def test_dataset_iloc_slice_rows_slice_cols(self):
            raise SkipTest("Not supported")

        def test_dataset_iloc_slice_rows_list_cols(self):
            raise SkipTest("Not supported")

        def test_dataset_iloc_slice_rows_index_cols(self):
            raise SkipTest("Not supported")

        def test_dataset_iloc_slice_rows(self):
            raise SkipTest("Not supported")

        def test_dataset_iloc_list_rows_slice_cols(self):
            raise SkipTest("Not supported")

        def test_dataset_iloc_list_rows_list_cols_by_name(self):
            raise SkipTest("Not supported")

        def test_dataset_iloc_list_rows_list_cols(self):
            raise SkipTest("Not supported")

        def test_dataset_iloc_list_rows(self):
            raise SkipTest("Not supported")

        def test_dataset_iloc_list_cols_by_name(self):
            raise SkipTest("Not supported")

        def test_dataset_iloc_list_cols(self):
            raise SkipTest("Not supported")

        def test_dataset_iloc_index_rows_slice_cols(self):
            raise SkipTest("Not supported")

        def test_dataset_iloc_index_rows_index_cols(self):
            raise SkipTest("Not supported")

        def test_dataset_iloc_ellipsis_list_cols_by_name(self):
            raise SkipTest("Not supported")

        def test_dataset_iloc_ellipsis_list_cols(self):
            raise SkipTest("Not supported")

        def test_dataset_boolean_index(self):
            raise SkipTest("Not supported")
