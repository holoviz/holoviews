import logging

import numpy as np
import pytest

from holoviews.core.data import Dataset

from .base import HeterogeneousColumnTests, InterfaceTests

pytestmark = pytest.mark.gpu


class cuDFInterfaceTests(HeterogeneousColumnTests, InterfaceTests):
    """
    Tests for the cuDFInterface.
    """

    datatype = 'cuDF'

    __test__ = True
    force_sort = True

    @property
    def data_type(self):
        import cudf
        return cudf.DataFrame

    def frame(self, *args, **kwargs):
        import cudf
        import pandas as pd

        return cudf.from_pandas(pd.DataFrame(*args, **kwargs))

    def test_dataset_get_dframe_by_dimension(self):
        df = self.dataset_hm.dframe(['x'])
        expected = self.frame({'x': self.xs}, dtype=df.dtypes.iloc[0])
        assert isinstance(expected, self.data_type)
        self.assertEqual(df, expected.to_pandas())

    def setUp(self):
        super().setUp()
        logging.getLogger('numba.cuda.cudadrv.driver').setLevel(30)

    @pytest.mark.xfail(reason="cuDF does not support variance aggregation")
    def test_dataset_2D_aggregate_spread_fn_with_duplicates(self):
        super().test_dataset_2D_aggregate_spread_fn_with_duplicates()

    def test_dataset_mixed_type_range(self):
        ds = Dataset((['A', 'B', 'C', None],), 'A')
        vmin, vmax = ds.range(0)
        self.assertTrue(np.isnan(vmin))
        self.assertTrue(np.isnan(vmax))

    @pytest.mark.xfail(reason="cuDF does not support variance aggregation")
    def test_dataset_aggregate_string_types_size(self):
        super().test_dataset_aggregate_string_types_size()

    def test_select_with_neighbor(self):
        import cupy as cp

        select = self.table.interface.select_mask(self.table.dataset, {"Weight": 18})
        select_neighbor = self.table.interface._select_mask_neighbor(self.table.dataset, dict(Weight=18))

        np.testing.assert_almost_equal(cp.asnumpy(select), [False, True, False])
        np.testing.assert_almost_equal(cp.asnumpy(select_neighbor), [True, True, True])
