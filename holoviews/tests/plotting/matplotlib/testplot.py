from unittest import SkipTest

from param import concrete_descendents

from holoviews.core.options import Store
from holoviews.element.comparison import ComparisonTestCase
import pyviz_comms as comms

try:
    from holoviews.plotting.mpl.element import ElementPlot
    import matplotlib.pyplot as plt
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
        Store.set_current_backend('matplotlib')
        self._padding = {}
        for plot in concrete_descendents(ElementPlot).values():
            self._padding[plot] = plot.padding
            plot.padding = 0

    def tearDown(self):
        Store.current_backend = self.previous_backend
        mpl_renderer.comm_manager = self.comm_manager
        plt.close(plt.gcf())
        for plot, padding in self._padding.items():
            plot.padding = padding
