import sys
import sqlite3

from unittest import SkipTest

from collections import OrderedDict
from tempfile import NamedTemporaryFile

try:
    import ibis
    from ibis import sqlite
except:
    raise SkipTest("Could not import ibis, skipping IbisInterface tests.")

import numpy as np
import pandas as pd

from holoviews.core.data import Dataset

from .base import HeterogeneousColumnTests, ScalarColumnTests, InterfaceTests


def create_temp_db(df, name, index=False):
    file_obj = NamedTemporaryFile()
    con = sqlite3.Connection(file_obj.name)
    df.to_sql(name, con, index=index)
    return sqlite.connect(file_obj.name)


class IbisDatasetTest(HeterogeneousColumnTests, ScalarColumnTests, InterfaceTests):
    """
    Test of the generic dictionary interface.
    """

    datatype = 'ibis'
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
        self.kdims = ['Gender', 'Age']
        self.vdims = ['Weight', 'Height']
        self.gender, self.age = np.array(['M','M','F']), np.array([10,16,12])
        self.weight, self.height = np.array([15,18,10]), np.array([0.8,0.6,0.8])

        hetero_df  = pd.DataFrame(
            {'Gender':self.gender, 'Age':self.age,
             'Weight':self.weight, 'Height':self.height},
            columns=['Gender', 'Age', 'Weight', 'Height']
        )
        hetero_db = create_temp_db(hetero_df, 'hetero')
        self.table = Dataset(hetero_db.table('hetero'),
                             kdims=self.kdims, vdims=self.vdims)

        # Create table with aliased dimenion names
        self.alias_kdims = [('gender', 'Gender'), ('age', 'Age')]
        self.alias_vdims = [('weight', 'Weight'), ('height', 'Height')]
        alias_df = pd.DataFrame(
            {'gender':self.gender, 'age':self.age,
             'weight':self.weight, 'height':self.height},
            columns=['gender', 'age', 'weight', 'height']
        )
        alias_db = create_temp_db(alias_df, 'alias')
        self.alias_table = Dataset(
            alias_db.table('alias'), kdims=self.alias_kdims, vdims=self.alias_vdims
        )

        self.xs = np.array(range(11))
        self.xs_2 = self.xs**2
        self.y_ints = self.xs*2
        self.ys = np.linspace(0, 1, 11)
        self.zs = np.sin(self.xs)

        ht_df = pd.DataFrame({'x': self.xs, 'y': self.ys}, columns=['x', 'y'])
        ht_db = create_temp_db(ht_df, 'ht')
        self.dataset_ht = Dataset(ht_db.table('ht'), kdims=['x'], vdims=['y'])

        hm_df = pd.DataFrame({'x': self.xs, 'y': self.y_ints}, columns=['x', 'y'])
        hm_db = create_temp_db(hm_df, 'hm')
        self.dataset_hm = Dataset(hm_db.table('hm'), kdims=['x'], vdims=['y'])
        self.dataset_hm_alias = Dataset(hm_db.table('hm'), kdims=[('x', 'X')], vdims=[('y', 'Y')])


    def test_dataset_array_init_hm(self):
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
