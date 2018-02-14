from __future__ import absolute_import

from unittest import SkipTest

import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.options import Store, Cycle
from holoviews.element import Graph, TriMesh, circular_layout
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import comms

# Standardize backend due to random inconsistencies
try:
    from matplotlib import pyplot
    pyplot.switch_backend('agg')
    from matplotlib.collections import LineCollection, PolyCollection
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
        self.nodes = circular_layout(np.arange(N, dtype=np.int32))
        self.source = np.arange(N, dtype=np.int32)
        self.target = np.zeros(N, dtype=np.int32)
        self.weights = np.random.rand(N)
        self.graph = Graph(((self.source, self.target),))
        self.node_info = Dataset(['Output']+['Input']*(N-1), vdims=['Label'])
        self.node_info2 = Dataset(self.weights, vdims='Weight')
        self.graph2 = Graph(((self.source, self.target), self.node_info))
        self.graph3 = Graph(((self.source, self.target), self.node_info2))
        self.graph4 = Graph(((self.source, self.target, self.weights),), vdims='Weight')


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

    def test_plot_graph_categorical_colored_nodes(self):
        g = self.graph2.opts(plot=dict(color_index='Label'), style=dict(cmap='Set1'))
        plot = mpl_renderer.get_plot(g)
        nodes = plot.handles['nodes']
        facecolors = np.array([[0.89411765, 0.10196078, 0.10980392, 1.],
                               [0.6       , 0.6       , 0.6       , 1.],
                               [0.6       , 0.6       , 0.6       , 1.],
                               [0.6       , 0.6       , 0.6       , 1.],
                               [0.6       , 0.6       , 0.6       , 1.],
                               [0.6       , 0.6       , 0.6       , 1.],
                               [0.6       , 0.6       , 0.6       , 1.],
                               [0.6       , 0.6       , 0.6       , 1.]])
        self.assertEqual(nodes.get_facecolors(), facecolors)

    def test_plot_graph_numerically_colored_nodes(self):
        g = self.graph3.opts(plot=dict(color_index='Weight'), style=dict(cmap='viridis'))
        plot = mpl_renderer.get_plot(g)
        nodes = plot.handles['nodes']
        self.assertEqual(nodes.get_array(), self.weights)
        self.assertEqual(nodes.get_clim(), (self.weights.min(), self.weights.max()))

    def test_plot_graph_categorical_colored_edges(self):
        g = self.graph3.opts(plot=dict(edge_color_index='start'),
                             style=dict(edge_cmap=['#FFFFFF', '#000000']))
        plot = mpl_renderer.get_plot(g)
        edges = plot.handles['edges']
        colors = np.array([[1., 1., 1., 1.],
                           [0., 0., 0., 1.],
                           [1., 1., 1., 1.],
                           [0., 0., 0., 1.],
                           [1., 1., 1., 1.],
                           [0., 0., 0., 1.],
                           [1., 1., 1., 1.],
                           [0., 0., 0., 1.]])
        self.assertEqual(edges.get_colors(), colors)

    def test_plot_graph_numerically_colored_edges(self):
        g = self.graph4.opts(plot=dict(edge_color_index='Weight'),
                             style=dict(edge_cmap=['#FFFFFF', '#000000']))
        plot = mpl_renderer.get_plot(g)
        edges = plot.handles['edges']
        self.assertEqual(edges.get_array(), self.weights)
        self.assertEqual(edges.get_clim(), (self.weights.min(), self.weights.max()))



class TestMplTriMeshPlots(ComparisonTestCase):

    def setUp(self):
        if not mpl_renderer:
            raise SkipTest('Matplotlib tests require matplotlib to be available')
        self.previous_backend = Store.current_backend
        Store.current_backend = 'matplotlib'
        self.default_comm = mpl_renderer.comms['default']
        mpl_renderer.comms['default'] = (comms.Comm, '')

        self.nodes = [(0, 0, 0), (0.5, 1, 1), (1., 0, 2), (1.5, 1, 3)]
        self.simplices = [(0, 1, 2, 0), (1, 2, 3, 1)]
        self.trimesh = TriMesh((self.simplices, self.nodes))
        self.trimesh_weighted = TriMesh((self.simplices, self.nodes), vdims='weight')

    def tearDown(self):
        mpl_renderer.comms['default'] = self.default_comm
        Store.current_backend = self.previous_backend

    def test_plot_simple_trimesh(self):
        plot = mpl_renderer.get_plot(self.trimesh)
        nodes = plot.handles['nodes']
        edges = plot.handles['edges']
        self.assertIsInstance(edges, LineCollection)
        self.assertEqual(nodes.get_offsets(), self.trimesh.nodes.array([0, 1]))
        self.assertEqual([p.vertices for p in edges.get_paths()],
                         [p.array() for p in self.trimesh._split_edgepaths.split()])

    def test_plot_simple_trimesh_filled(self):
        plot = mpl_renderer.get_plot(self.trimesh.opts(plot=dict(filled=True)))
        nodes = plot.handles['nodes']
        edges = plot.handles['edges']
        self.assertIsInstance(edges, PolyCollection)
        self.assertEqual(nodes.get_offsets(), self.trimesh.nodes.array([0, 1]))
        paths = self.trimesh._split_edgepaths.split(datatype='array')
        self.assertEqual([p.vertices[:4] for p in edges.get_paths()],
                         paths)

    def test_plot_trimesh_colored_edges(self):
        opts = dict(plot=dict(edge_color_index='weight'), style=dict(edge_cmap='Greys'))
        plot = mpl_renderer.get_plot(self.trimesh_weighted.opts(**opts))
        edges = plot.handles['edges']
        colors = np.array([[ 1.,  1.,  1.,  1.],
                           [ 0.,  0.,  0.,  1.]])
        self.assertEqual(edges.get_edgecolors(), colors)

    def test_plot_trimesh_categorically_colored_edges(self):
        opts = dict(plot=dict(edge_color_index='node1'), style=dict(edge_color=Cycle('Set1')))
        plot = mpl_renderer.get_plot(self.trimesh_weighted.opts(**opts))
        edges = plot.handles['edges']
        colors = np.array([[0.894118, 0.101961, 0.109804, 1.],
                           [0.215686, 0.494118, 0.721569, 1.]])
        self.assertEqual(edges.get_edgecolors(), colors)

    def test_plot_trimesh_categorically_colored_edges_filled(self):
        opts = dict(plot=dict(edge_color_index='node1', filled=True),
                    style=dict(edge_color=Cycle('Set1')))
        plot = mpl_renderer.get_plot(self.trimesh_weighted.opts(**opts))
        edges = plot.handles['edges']
        colors = np.array([[0.894118, 0.101961, 0.109804, 1.],
                           [0.215686, 0.494118, 0.721569, 1.]])
        self.assertEqual(edges.get_facecolors(), colors)

