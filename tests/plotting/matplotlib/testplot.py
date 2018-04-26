from unittest import SkipTest

from holoviews.core.options import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import comms

try:
    import holoviews.plotting.mpl # noqa
    mpl_renderer = Store.renderers['matplotlib']
except:
    mpl_renderer = None

from .. import option_intersections


class TestPlotDefinitions(ComparisonTestCase):

    known_clashes = [(('Arrow',), {'fontsize'})]

    def test_matplotlib_plot_definitions(self):
        self.assertEqual(option_intersections('matplotlib'), self.known_clashes)


class TestMPLPlot(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        self.comm_manager = mpl_renderer.comm_manager
        mpl_renderer.comm_manager = comms.CommManager
        if not mpl_renderer:
            raise SkipTest("Matplotlib required to test plot instantiation")
        Store.current_backend = 'matplotlib'

    def tearDown(self):
        Store.current_backend = self.previous_backend
        mpl_renderer.comm_manager = self.comm_manager
