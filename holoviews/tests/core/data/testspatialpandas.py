"""
Tests for the spatialpandas interface.
"""

from unittest import SkipTest

import numpy as np

try:
    import spatialpandas
except:
    spatialpandas = None

from holoviews.core.data import Dataset, SpatialPandasInterface
from holoviews.core.data.interface import DataError

from .testmultiinterface import GeomTests


class SpatialPandasTest(GeomTests):
    """
    Test of the SpatialPandasInterface.
    """

    datatype = 'spatialpandas'

    interface = SpatialPandasInterface

    __test__ = True

    def setUp(self):
        if spatialpandas is None:
            raise SkipTest('SpatialPandasInterface requires spatialpandas, skipping tests')
        super(GeomTests, self).setUp()

    def test_multi_dict_groupby(self):
        arrays = [{'x': np.arange(i, i+2), 'y': i} for i in range(2)]
        mds = Dataset(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        self.assertIs(mds.interface, self.interface)
        with self.assertRaises(DataError):
            mds.groupby('y')

    def test_multi_array_groupby(self):
        arrays = [np.array([(1+i, i), (2+i, i), (3+i, i)]) for i in range(2)]
        mds = Dataset(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        self.assertIs(mds.interface, self.interface)
        with self.assertRaises(DataError):
            mds.groupby('y')

    def test_multi_array_iloc_index_rows_index_cols(self):
        arrays = [np.array([(1+i, i), (2+i, i), (3+i, i)]) for i in range(2)]
        mds = Dataset(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        self.assertIs(mds.interface, self.interface)
        with self.assertRaises(DataError):
            mds.iloc[3, 0]
