from unittest import SkipTest

import numpy as np

from holoviews.core.data import Dataset

from .base import HomogeneousColumnTests, InterfaceTests


class ArrayDatasetTest(HomogeneousColumnTests, InterfaceTests):
    """
    Test of the ArrayDataset interface.
    """

    datatype = 'array'
    data_type = np.ndarray

    __test__ = True

    def test_dataset_empty_list_init_dtypes(self):
        dataset = Dataset([], kdims=['x'], vdims=['y'])
        for d in 'xy':
            self.assertEqual(dataset.dimension_values(d).dtype, np.float64)

    def test_dataset_empty_list_dtypes(self):
        dataset = Dataset([], kdims=['x'], vdims=['y'])
        for d in 'xy':
            self.assertEqual(dataset.interface.dtype(dataset, d), np.float64)

    def test_dataset_simple_dict_sorted(self):
        dataset = Dataset({2: 2, 1: 1, 3: 3}, kdims=['x'], vdims=['y'])
        self.assertEqual(dataset, Dataset([(i, i) for i in range(1, 4)],
                                          kdims=['x'], vdims=['y']))

    def test_dataset_sort_hm(self):
        ds = Dataset(([2, 2, 1], [2,1,2], [1, 2, 3]),
                     kdims=['x', 'y'], vdims=['z']).sort()
        ds_sorted = Dataset(([1, 2, 2], [2, 1, 2], [3, 2, 1]),
                            kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(ds.sort(), ds_sorted)

    def test_dataset_sort_reverse_hm(self):
        ds = Dataset(([2, 1, 2, 1], [2, 2, 1, 1], [0, 1, 2, 3]),
                     kdims=['x', 'y'], vdims=['z'])
        ds_sorted = Dataset(([2, 2, 1, 1], [2, 1, 2, 1], [0, 2, 1, 3]),
                            kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(ds.sort(reverse=True), ds_sorted)

    def test_dataset_transform_replace_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_transform_add_hm(self):
        raise SkipTest("Not supported")
