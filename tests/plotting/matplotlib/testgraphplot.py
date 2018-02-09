from __future__ import absolute_import

from unittest import SkipTest

import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.options import Store, Cycle
from holoviews.element import Graph, TriMesh, Chord, circular_layout
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import comms

# Standardize backend due to random inconsistencies
try:
    from holoviews.plotting.mpl import OverlayPlot
    from matplotlib.collections import LineCollection, PolyCollection
except:
    pass

from .testplot import TestMPLPlot, mpl_renderer


class TestMplGraphPlot(TestMPLPlot):

    def setUp(self):
        super(TestMplGraphPlot, self).setUp()

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



class TestMplTriMeshPlot(TestMPLPlot):

    def setUp(self):
        super(TestMplTriMeshPlot, self).setUp()

        self.nodes = [(0, 0, 0), (0.5, 1, 1), (1., 0, 2), (1.5, 1, 3)]
        self.simplices = [(0, 1, 2, 0), (1, 2, 3, 1)]
        self.trimesh = TriMesh((self.simplices, self.nodes))
        self.trimesh_weighted = TriMesh((self.simplices, self.nodes), vdims='weight')

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


class TestMplChordPlot(TestMPLPlot):

    def setUp(self):
        super(TestMplChordPlot, self).setUp()
        self.edges = [(0, 1, 1), (0, 2, 2), (1, 2, 3)]
        self.nodes = Dataset([(0, 'A'), (1, 'B'), (2, 'C')], 'index', 'Label')
        self.chord = Chord((self.edges, self.nodes))

    def test_chord_nodes_label_text(self):
        g = self.chord.opts(plot=dict(label_index='Label'))
        plot = mpl_renderer.get_plot(g)
        labels = plot.handles['labels']
        self.assertEqual([l.get_text() for l in labels], ['A', 'B', 'C'])

    def test_chord_nodes_categorically_colormapped(self):
        g = self.chord.opts(plot=dict(color_index='Label'),
                            style=dict(cmap=['#FFFFFF', '#CCCCCC', '#000000']))
        plot = mpl_renderer.get_plot(g)
        arcs = plot.handles['arcs']
        nodes = plot.handles['nodes']
        colors = np.array([[ 1.,   1.,   1.,   1. ],
                           [ 0.8,  0.8,  0.8,  1. ],
                           [ 0.,   0.,   0.,   1. ]])
        self.assertEqual(arcs.get_colors(), colors)
        self.assertEqual(nodes.get_facecolors(), colors)

    def test_chord_edges_categorically_colormapped(self):
        g = self.chord.opts(plot=dict(edge_color_index='start'),
                            style=dict(edge_cmap=['#FFFFFF', '#000000']))
        plot = mpl_renderer.get_plot(g)
        edges = plot.handles['edges']
        colors = np.array([[ 1., 1., 1., 1. ],
                           [ 1., 1., 1., 1. ],
                           [ 0., 0., 0., 1. ]])
        self.assertEqual(edges.get_edgecolors(), colors)
