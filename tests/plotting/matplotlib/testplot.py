from unittest import SkipTest

import param
import numpy as np

from holoviews.core.element import Element
from holoviews.core.options import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import comms

try:
    import holoviews.plotting.mpl
    mpl_renderer = Store.renderers['matplotlib']
except:
    mpl_renderer = None


class TestMPLPlot(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        if not mpl_renderer:
            raise SkipTest("Matplotlib required to test plot instantiation")
        Store.current_backend = 'matplotlib'
        self.default_comm = mpl_renderer.comms['default']
        mpl_renderer.comms['default'] = (comms.Comm, '')

    def tearDown(self):
        Store.current_backend = self.previous_backend
        mpl_renderer.comms['default'] = self.default_comm
