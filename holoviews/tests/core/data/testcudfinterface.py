from unittest import SkipTest

try:
    import cudf
except:
    raise SkipTest("Could not import cuDF, skipping cuDFInterface tests.")

from .base import HeterogeneousColumnTests, InterfaceTests

import logging



class cuDFInterfaceTests(HeterogeneousColumnTests, InterfaceTests):
    """
    Tests for the cuDFInterface.
    """

    datatype = 'cuDF'
    data_type = cudf.DataFrame

    def setUp(self):
        super(cuDFInterfaceTests, self).setUp()
        logging.getLogger('numba.cuda.cudadrv.driver').setLevel(30)

    def test_dataset_2D_aggregate_spread_fn_with_duplicates(self):
        raise SkipTest("cuDF does not support variance aggregation")

    def test_dataset_reduce_ht(self):
        reduced = Dataset({'Age':self.age, 'Weight':self.weight, 'Height':self.height},
                          kdims=self.kdims[1:], vdims=self.vdims)
        self.assertEqual(self.table.reduce(['Gender'], np.mean), reduced)
