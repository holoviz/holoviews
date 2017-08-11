"""
Tests for the Dataset Element types.
"""

from unittest import SkipTest

import numpy as np
from holoviews import Dataset
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
        mds = Path(arrays, kdims=['x', 'y'], datatype=['multi'])
        for i, ds in enumerate(mds.split()):
            self.assertEqual(ds, Path(arrays[i], kdims=['x', 'y'], datatype=['array']))

    def test_multi_dict_dataset(self):
        arrays = [{'x': np.arange(i, i+2), 'y': np.arange(i, i+2)} for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=['multi'])
        for i, ds in enumerate(mds.split()):
            self.assertEqual(ds, Path(arrays[i], kdims=['x', 'y'], datatype=['dictionary']))

    def test_multi_df_dataset(self):
        if not pd:
            raise SkipTest('Pandas not available')
        arrays = [pd.DataFrame(np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]), columns=['x', 'y'])
                  for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=['multi'])
        for i, ds in enumerate(mds.split()):
            self.assertEqual(ds, Path(arrays[i], kdims=['x', 'y'], datatype=['dataframe']))

    def test_multi_dask_df_dataset(self):
        if not dd:
            raise SkipTest('Dask not available')
        arrays = [dd.from_pandas(pd.DataFrame(np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]),
                                              columns=['x', 'y']), npartitions=2)
                  for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=['multi'])
        for i, ds in enumerate(mds.split()):
            self.assertEqual(ds, Path(arrays[i], kdims=['x', 'y'], datatype=['dask']))

    def test_multi_array_length(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=['multi'])
        self.assertEqual(len(mds), 5)

    def test_multi_array_range(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=['multi'])
        self.assertEqual(mds.range(0), (0, 2))

    def test_multi_array_shape(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=['multi'])
        self.assertEqual(mds.shape, (5, 2))
    
    def test_multi_array_values(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=['multi'])
        self.assertEqual(mds.dimension_values(0), np.array([0., 1, np.NaN, 1, 2]))

    def test_multi_array_redim(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=['multi']).redim(x='x2')
        for i, ds in enumerate(mds.split()):
            self.assertEqual(ds, Path(arrays[i], kdims=['x2', 'y'], datatype=['dask']))
