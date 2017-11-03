"""
Tests for the Dataset Element types.
"""

from unittest import SkipTest

import numpy as np
from holoviews import Dataset
from holoviews.core.data.interface import DataError
from holoviews.element import Path
from holoviews.element.comparison import ComparisonTestCase

try:
    import pandas as pd
except:
    pd = None

try:
    import dask.dataframe as dd
except:
    dd = None

class MultiInterfaceTest(ComparisonTestCase):
    """
    Test of the MultiInterface.
    """

    def test_multi_array_dataset(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=['multitabular'])
        for i, ds in enumerate(mds.split()):
            self.assertEqual(ds, Path(arrays[i], kdims=['x', 'y'], datatype=['array']))

    def test_multi_dict_dataset(self):
        arrays = [{'x': np.arange(i, i+2), 'y': np.arange(i, i+2)} for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=['multitabular'])
        for i, ds in enumerate(mds.split()):
            self.assertEqual(ds, Path(arrays[i], kdims=['x', 'y'], datatype=['dictionary']))

    def test_multi_df_dataset(self):
        if not pd:
            raise SkipTest('Pandas not available')
        arrays = [pd.DataFrame(np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]), columns=['x', 'y'])
                  for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=['multitabular'])
        for i, ds in enumerate(mds.split()):
            self.assertEqual(ds, Path(arrays[i], kdims=['x', 'y'], datatype=['dataframe']))

    def test_multi_dask_df_dataset(self):
        if not dd:
            raise SkipTest('Dask not available')
        arrays = [dd.from_pandas(pd.DataFrame(np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]),
                                              columns=['x', 'y']), npartitions=2)
                  for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=['multitabular'])
        for i, ds in enumerate(mds.split()):
            self.assertEqual(ds, Path(arrays[i], kdims=['x', 'y'], datatype=['dask']))

    def test_multi_array_length(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=['multitabular'])
        self.assertEqual(len(mds), 5)

    def test_multi_empty_length(self):
        mds = Path([], kdims=['x', 'y'], datatype=['multitabular'])
        self.assertEqual(len(mds), 0)

    def test_multi_array_range(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=['multitabular'])
        self.assertEqual(mds.range(0), (0, 2))

    def test_multi_empty_range(self):
        mds = Path([], kdims=['x', 'y'], datatype=['multitabular'])
        low, high = mds.range(0)
        self.assertFalse(np.isfinite(np.NaN))
        self.assertFalse(np.isfinite(np.NaN))

    def test_multi_array_shape(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=['multitabular'])
        self.assertEqual(mds.shape, (5, 2))

    def test_multi_empty_shape(self):
        mds = Path([], kdims=['x', 'y'], datatype=['multitabular'])
        self.assertEqual(mds.shape, (0, 2))

    def test_multi_array_values(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=['multitabular'])
        self.assertEqual(mds.dimension_values(0), np.array([0., 1, np.NaN, 1, 2]))

    def test_multi_empty_array_values(self):
        mds = Path([], kdims=['x', 'y'], datatype=['multitabular'])
        self.assertEqual(mds.dimension_values(0), np.array([]))

    def test_multi_array_values_coordinates_nonexpanded(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=['multitabular'])
        self.assertEqual(mds.dimension_values(0, expanded=False), np.array([0., 1, 1, 2]))

    def test_multi_array_values_coordinates_nonexpanded_constant_kdim(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2), np.ones(2)*i]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], vdims=['z'], datatype=['multitabular'])
        self.assertEqual(mds.dimension_values(2, expanded=False), np.array([0, 1]))

    def test_multi_array_redim(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=['multitabular']).redim(x='x2')
        for i, ds in enumerate(mds.split()):
            self.assertEqual(ds, Path(arrays[i], kdims=['x2', 'y'], datatype=['dask']))

    def test_multi_mixed_interface_raises(self):
        arrays = [np.random.rand(10, 2) if j else {'x': range(10), 'y': range(10)}
                  for i in range(2) for j in range(2)]
        with self.assertRaises(DataError):
            mds = Path(arrays, kdims=['x', 'y'], datatype=['multitabular'])

    def test_multi_mixed_dims_raises(self):
        arrays = [{'x': range(10), 'y' if j else 'z': range(10)}
                  for i in range(2) for j in range(2)]
        with self.assertRaises(DataError):
            mds = Path(arrays, kdims=['x', 'y'], datatype=['multitabular'])

    def test_multi_split(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=['multitabular'])
        for arr1, arr2 in zip(mds.split(datatype='array'), arrays):
            self.assertEqual(arr1, arr2)

    def test_multi_split_empty(self):
        mds = Path([], kdims=['x', 'y'], datatype=['multitabular'])
        self.assertEqual(len(mds.split()), 0)
