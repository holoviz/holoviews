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

    def test_dataset_sort_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_reverse_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_vdim_ht(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_vdim_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_vdim_hm_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_string_ht(self):
        raise SkipTest("Not supported")


