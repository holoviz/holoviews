import sqlite3
from unittest import SkipTest

from tempfile import NamedTemporaryFile

try:
    import ibis
    from ibis import sqlite
except ImportError:
    raise SkipTest("Could not import ibis, skipping IbisInterface tests.")

import numpy as np
import pandas as pd

from holoviews.core.data import Dataset
from holoviews.core.spaces import HoloMap
from holoviews.core.data.ibis import IbisInterface

from .base import HeterogeneousColumnTests, ScalarColumnTests, InterfaceTests


def create_temp_db(df, name, index=False):
    with NamedTemporaryFile(delete=False) as my_file:
        filename = my_file.name
    con = sqlite3.Connection(filename)
    df.to_sql(name, con, index=index)
    return sqlite.connect(filename)


class IbisDatasetTest(HeterogeneousColumnTests, ScalarColumnTests, InterfaceTests):
    """
    Test of the generic dictionary interface.
    """

    datatype = "ibis"
    data_type = (ibis.expr.types.Expr,)

    __test__ = True

    def setUp(self):
        self.init_column_data()
        self.init_grid_data()
        self.init_data()

    def tearDown(self):
        pass

    def init_column_data(self):
        # Create heterogeneously typed table
        self.kdims = ["Gender", "Age"]
        self.vdims = ["Weight", "Height"]
        self.gender, self.age = np.array(["M", "M", "F"]), np.array([10, 16, 12])
        self.weight, self.height = np.array([15, 18, 10]), np.array([0.8, 0.6, 0.8])

        hetero_df = pd.DataFrame(
            {
                "Gender": self.gender,
                "Age": self.age,
                "Weight": self.weight,
                "Height": self.height,
            },
            columns=["Gender", "Age", "Weight", "Height"],
        )
        hetero_db = create_temp_db(hetero_df, "hetero")
        self.table = Dataset(
            hetero_db.table("hetero"), kdims=self.kdims, vdims=self.vdims
        )

        # Create table with aliased dimension names
        self.alias_kdims = [("gender", "Gender"), ("age", "Age")]
        self.alias_vdims = [("weight", "Weight"), ("height", "Height")]
        alias_df = pd.DataFrame(
            {
                "gender": self.gender,
                "age": self.age,
                "weight": self.weight,
                "height": self.height,
            },
            columns=["gender", "age", "weight", "height"],
        )
        alias_db = create_temp_db(alias_df, "alias")
        self.alias_table = Dataset(
            alias_db.table("alias"), kdims=self.alias_kdims, vdims=self.alias_vdims
        )

        self.xs = np.array(range(11))
        self.xs_2 = self.xs ** 2
        self.y_ints = self.xs * 2
        self.ys = np.linspace(0, 1, 11)
        self.zs = np.sin(self.xs)

        ht_df = pd.DataFrame({"x": self.xs, "y": self.ys}, columns=["x", "y"])
        ht_db = create_temp_db(ht_df, "ht")
        self.dataset_ht = Dataset(ht_db.table("ht"), kdims=["x"], vdims=["y"])

        hm_df = pd.DataFrame({"x": self.xs, "y": self.y_ints}, columns=["x", "y"])
        hm_db = create_temp_db(hm_df, "hm")
        self.dataset_hm = Dataset(hm_db.table("hm"), kdims=["x"], vdims=["y"])
        self.dataset_hm_alias = Dataset(
            hm_db.table("hm"), kdims=[("x", "X")], vdims=[("y", "Y")]
        )

    def test_dataset_array_init_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_dict_dim_not_found_raises_on_scalar(self):
        raise SkipTest("Not supported")

    def test_dataset_array_init_hm_tuple_dims(self):
        raise SkipTest("Not supported")

    def test_dataset_odict_init(self):
        raise SkipTest("Not supported")

    def test_dataset_odict_init_alias(self):
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

    def test_dataset_dict_init(self):
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
        ds = self.table
        self.assertEqual(ds.interface.dtype(ds, "Gender"), np.dtype("object"))
        self.assertEqual(ds.interface.dtype(ds, "Age"), np.dtype("int32"))
        self.assertEqual(ds.interface.dtype(ds, "Weight"), np.dtype("int32"))
        self.assertEqual(ds.interface.dtype(ds, "Height"), np.dtype("float64"))

    def test_dataset_dtypes(self):
        self.assertEqual(
            self.dataset_hm.interface.dtype(self.dataset_hm, "x"), np.dtype("int32")
        )
        self.assertEqual(
            self.dataset_hm.interface.dtype(self.dataset_hm, "y"), np.dtype("int32")
        )

    def test_dataset_reduce_ht(self):
        reduced = Dataset(
            {"Age": self.age, "Weight": self.weight, "Height": self.height},
            kdims=self.kdims[1:],
            vdims=self.vdims,
        )
        self.assertEqual(self.table.reduce(["Gender"], np.mean).sort(), reduced.sort())

    def test_dataset_aggregate_ht(self):
        aggregated = Dataset(
            {"Gender": ["M", "F"], "Weight": [16.5, 10], "Height": [0.7, 0.8]},
            kdims=self.kdims[:1],
            vdims=self.vdims,
        )
        self.compare_dataset(
            self.table.aggregate(["Gender"], np.mean).sort(), aggregated.sort()
        )

    def test_dataset_aggregate_ht_alias(self):
        aggregated = Dataset(
            {"gender": ["M", "F"], "weight": [16.5, 10], "height": [0.7, 0.8]},
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
        group1 = {"age": [10, 16], "weight": [15, 18], "height": [0.8, 0.6]}
        group2 = {"age": [12], "weight": [10], "height": [0.8]}
        grouped = HoloMap(
            [
                ("M", Dataset(group1, kdims=[("age", "Age")], vdims=self.alias_vdims)),
                ("F", Dataset(group2, kdims=[("age", "Age")], vdims=self.alias_vdims)),
            ],
            kdims=[("gender", "Gender")],
        )
        self.assertEqual(self.alias_table.groupby("Gender").apply("sort"), grouped)

    def test_dataset_groupby_second_dim(self):
        group1 = {"Gender": ["M"], "Weight": [15], "Height": [0.8]}
        group2 = {"Gender": ["M"], "Weight": [18], "Height": [0.6]}
        group3 = {"Gender": ["F"], "Weight": [10], "Height": [0.8]}
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
            data = self.table.dframe()
            expected = self.table.clone(
                data=data
            ).aggregate("Gender", agg).sort()

            result = self.table.aggregate("Gender", agg).sort()

            self.compare_dataset(expected, result, msg=str(agg))

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
