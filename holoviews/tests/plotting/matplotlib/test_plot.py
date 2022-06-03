import matplotlib.pyplot as plt
import pyviz_comms as comms

from holoviews.core.options import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.mpl.element import ElementPlot
from param import concrete_descendents

from .. import option_intersections

mpl_renderer = Store.renderers['matplotlib']


class TestPlotDefinitions(ComparisonTestCase):

    known_clashes = [(('Arrow',), {'fontsize'})]

    def test_matplotlib_plot_definitions(self):
        self.assertEqual(option_intersections('matplotlib'), self.known_clashes)


class TestMPLPlot(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        self.comm_manager = mpl_renderer.comm_manager
        mpl_renderer.comm_manager = comms.CommManager
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
