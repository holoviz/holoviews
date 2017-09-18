from __future__ import absolute_import

from unittest import SkipTest

import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.options import Store
from holoviews.element import Graph, circular_layout
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import comms

# Standardize backend due to random inconsistencies
try:
    from matplotlib import pyplot
    pyplot.switch_backend('agg')
    from holoviews.plotting.mpl import OverlayPlot
    mpl_renderer = Store.renderers['matplotlib']
except:
    mpl_renderer = None


class MplGraphPlotTests(ComparisonTestCase):

    def setUp(self):
        if not mpl_renderer:
            raise SkipTest('Matplotlib tests require matplotlib to be available')
        self.previous_backend = Store.current_backend
        Store.current_backend = 'matplotlib'
        self.default_comm = mpl_renderer.comms['default']
        mpl_renderer.comms['default'] = (comms.Comm, '')

        N = 8
        self.nodes = circular_layout(np.arange(N))
        self.source = np.arange(N)
        self.target = np.zeros(N)
        self.graph = Graph(((self.source, self.target),))
        self.node_info = Dataset(['Output']+['Input']*(N-1), vdims=['Label'])
        self.graph2 = Graph(((self.source, self.target), self.node_info))

    def tearDown(self):
        mpl_renderer.comms['default'] = self.default_comm
        Store.current_backend = self.previous_backend

    def test_plot_simple_graph(self):
        plot = mpl_renderer.get_plot(self.graph)
        nodes = plot.handles['nodes']
        edges = plot.handles['edges']
        self.assertEqual(nodes.get_offsets(), self.graph.nodes.array([0, 1]))
        self.assertEqual([p.vertices for p in edges.get_paths()],
                         [p.array() for p in self.graph.edgepaths.split()])

    def test_plot_graph_colored_nodes(self):
        g = self.graph2.opts(plot=dict(color_index='Label'), style=dict(cmap='Set1'))
        plot = mpl_renderer.get_plot(g)
        nodes = plot.handles['nodes']
        edges = plot.handles['edges']
        self.assertEqual(nodes.get_offsets(), self.graph.nodes.array([0, 1]))
        self.assertEqual([p.vertices for p in edges.get_paths()],
                         [p.array() for p in self.graph.edgepaths.split()])
        self.assertEqual(nodes.get_array(), np.array([1, 0, 0, 0, 0, 0, 0, 0]))
