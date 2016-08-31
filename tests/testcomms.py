from nose.tools import *

from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.comms import Comm, JupyterPushComm


class TestComm(ComparisonTestCase):

    def test_init_comm(self):
        Comm(None)

    def test_init_comm_target(self):
        comm = Comm(None, target='Test')
        self.assertEqual(comm.target, 'Test')

    def test_decode(self):
        msg = 'Test'
        self.assertEqual(Comm.decode(msg), msg)

    def test_on_msg(self):
        def raise_error(msg):
            if msg == 'Error':
                raise Exception()
        comm = Comm(None, target='Test', on_msg=raise_error)
        with self.assertRaises(Exception):
            comm._handle_msg('Error')


class TestJupyterComm(ComparisonTestCase):

    def test_init_comm(self):
        JupyterPushComm(None)

    def test_init_comm_target(self):
        comm = JupyterPushComm(None, target='Test')
        self.assertEqual(comm.target, 'Test')

    def test_decode(self):
        msg = {'content': {'data': 'Test'}}
        decoded = JupyterPushComm.decode(msg)
        self.assertEqual(decoded, 'Test')

    def test_on_msg(self):
        def raise_error(msg):
            if msg == 'Error':
                raise Exception()
        comm = JupyterPushComm(None, target='Test', on_msg=raise_error)
        with self.assertRaises(Exception):
            comm._handle_msg({'content': {'data': 'Error'}})
