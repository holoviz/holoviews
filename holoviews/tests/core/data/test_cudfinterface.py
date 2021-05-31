import logging

from unittest import SkipTest

import numpy as np

try:
    import cudf
except:
    raise SkipTest("Could not import cuDF, skipping cuDFInterface tests.")

from holoviews.core.data import Dataset
from holoviews.core.spaces import HoloMap

from .base import HeterogeneousColumnTests, InterfaceTests


class cuDFInterfaceTests(HeterogeneousColumnTests, InterfaceTests):
    """
    Tests for the cuDFInterface.
    """

    datatype = 'cuDF'
    data_type = cudf.DataFrame

    __test__ = True

    def setUp(self):
        super().setUp()
        logging.getLogger('numba.cuda.cudadrv.driver').setLevel(30)

    def test_dataset_2D_aggregate_spread_fn_with_duplicates(self):
        raise SkipTest("cuDF does not support variance aggregation")

    def test_dataset_mixed_type_range(self):
        ds = Dataset((['A', 'B', 'C', None],), 'A')
        vmin, vmax = ds.range(0)
        self.assertTrue(np.isnan(vmin))
        self.assertTrue(np.isnan(vmax))

    def test_dataset_groupby(self):
        group1 = {'Age':[10,16], 'Weight':[15,18], 'Height':[0.8,0.6]}
        group2 = {'Age':[12], 'Weight':[10], 'Height':[0.8]}
        grouped = HoloMap([('M', Dataset(group1, kdims=['Age'], vdims=self.vdims)),
                           ('F', Dataset(group2, kdims=['Age'], vdims=self.vdims))],
                          kdims=['Gender'])
        self.assertEqual(self.table.groupby(['Gender']).apply('sort'), grouped.apply('sort'))

    def test_dataset_groupby_alias(self):
        group1 = {'age':[10,16], 'weight':[15,18], 'height':[0.8,0.6]}
        group2 = {'age':[12], 'weight':[10], 'height':[0.8]}
        grouped = HoloMap([('M', Dataset(group1, kdims=[('age', 'Age')],
                                         vdims=self.alias_vdims)),
                           ('F', Dataset(group2, kdims=[('age', 'Age')],
                                         vdims=self.alias_vdims))],
                          kdims=[('gender', 'Gender')])
        self.assertEqual(self.alias_table.groupby('Gender').apply('sort'),
                         grouped)

    def test_dataset_aggregate_ht(self):
        aggregated = Dataset({'Gender':['M', 'F'], 'Weight':[16.5, 10], 'Height':[0.7, 0.8]},
                             kdims=self.kdims[:1], vdims=self.vdims)
        self.compare_dataset(self.table.aggregate(['Gender'], np.mean).sort(), aggregated.sort())

    def test_dataset_aggregate_ht_alias(self):
        aggregated = Dataset({'gender':['M', 'F'], 'weight':[16.5, 10], 'height':[0.7, 0.8]},
                             kdims=self.alias_kdims[:1], vdims=self.alias_vdims)
        self.compare_dataset(self.alias_table.aggregate('Gender', np.mean).sort(), aggregated.sort())

    def test_dataset_2D_partial_reduce_ht(self):
        dataset = Dataset({'x':self.xs, 'y':self.ys, 'z':self.zs},
                          kdims=['x', 'y'], vdims=['z'])
        reduced = Dataset({'x':self.xs, 'z':self.zs},
                          kdims=['x'], vdims=['z'])
        self.assertEqual(dataset.reduce(['y'], np.mean).sort(), reduced.sort())

    def test_dataset_2D_aggregate_partial_ht(self):
        dataset = Dataset({'x':self.xs, 'y':self.ys, 'z':self.zs},
                          kdims=['x', 'y'], vdims=['z'])
        reduced = Dataset({'x':self.xs, 'z':self.zs},
                          kdims=['x'], vdims=['z'])
        self.assertEqual(dataset.aggregate(['x'], np.mean).sort(), reduced.sort())

    def test_dataset_2D_aggregate_partial_hm(self):
        z_ints = [el**2 for el in self.y_ints]
        dataset = Dataset({'x':self.xs, 'y':self.y_ints, 'z':z_ints},
                          kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(dataset.aggregate(['x'], np.mean).sort(),
                         Dataset({'x':self.xs, 'z':z_ints}, kdims=['x'], vdims=['z']).sort())

    def test_dataset_reduce_ht(self):
        reduced = Dataset({'Age':self.age, 'Weight':self.weight, 'Height':self.height},
                          kdims=self.kdims[1:], vdims=self.vdims)
        self.assertEqual(self.table.reduce(['Gender'], np.mean).sort(), reduced.sort())

    def test_dataset_groupby_second_dim(self):
        group1 = {'Gender':['M'], 'Weight':[15], 'Height':[0.8]}
        group2 = {'Gender':['M'], 'Weight':[18], 'Height':[0.6]}
        group3 = {'Gender':['F'], 'Weight':[10], 'Height':[0.8]}
        grouped = HoloMap([(10, Dataset(group1, kdims=['Gender'], vdims=self.vdims)),
                           (16, Dataset(group2, kdims=['Gender'], vdims=self.vdims)),
                           (12, Dataset(group3, kdims=['Gender'], vdims=self.vdims))],
                          kdims=['Age'])
        self.assertEqual(self.table.groupby(['Age']).apply('sort'), grouped)

    def test_dataset_aggregate_string_types_size(self):
        raise SkipTest("cuDF does not support variance aggregation")
