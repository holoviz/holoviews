"""
Unit tests of Graph Element.
"""
import numpy as np
from holoviews.element.graphs import Graph, Nodes, circular_layout
from holoviews.element.comparison import ComparisonTestCase


class GraphTests(ComparisonTestCase):

    def setUp(self):
        N = 8
        self.nodes = circular_layout(np.arange(N))
        self.source = np.arange(N)
        self.target = np.zeros(N)
        self.edge_info = np.arange(N)
        self.graph = Graph(((self.source, self.target),))

    def test_basic_constructor(self):
        graph = Graph(((self.source, self.target),))
        nodes = Nodes(self.nodes)
        self.assertEqual(graph.nodes, nodes)

    def test_constructor_with_nodes(self):
        graph = Graph(((self.source, self.target), self.nodes))
        nodes = Nodes(self.nodes)
        self.assertEqual(graph.nodes, nodes)

    def test_constructor_with_nodes_and_paths(self):
        paths = Graph(((self.source, self.target), self.nodes)).edgepaths
        graph = Graph(((self.source, self.target), self.nodes, paths.data))
        nodes = Nodes(self.nodes)
        self.assertEqual(graph._edgepaths, paths)

    def test_constructor_with_nodes_and_paths_dimension_mismatch(self):
        paths = Graph(((self.source, self.target), self.nodes)).edgepaths
        exception = 'Ensure that the first two key dimensions on Nodes and EdgePaths match: x != x2'
        with self.assertRaisesRegexp(ValueError, exception):
            graph = Graph(((self.source, self.target), self.nodes, paths.redim(x='x2')))

    def test_graph_clone_static_plot_id(self):
        self.assertEqual(self.graph.clone()._plot_id, self.graph._plot_id)

    def test_select_by_node_in_edges_selection_mode(self):
        graph = Graph(((self.source, self.target),))
        selection = Graph(([(1, 0), (2, 0)], list(zip(*self.nodes))[0:3]))
        self.assertEqual(graph.select(index=(1, 3)), selection) 

    def test_select_by_node_in_nodes_selection_mode(self):
        graph = Graph(((self.source, self.source+1), self.nodes))
        selection = Graph(([(1, 2)], list(zip(*self.nodes))[1:3]))
        self.assertEqual(graph.select(index=(1, 3), selection_mode='nodes'), selection)

    def test_select_by_source(self):
        graph = Graph(((self.source, self.target),))
        selection = Graph(([(0,0), (1, 0)], list(zip(*self.nodes))[:2]))
        self.assertEqual(graph.select(start=(0, 2)), selection)

    def test_select_by_target(self):
        graph = Graph(((self.source, self.target),))
        selection = Graph(([(0,0), (1, 0)], list(zip(*self.nodes))[:2]))
        self.assertEqual(graph.select(start=(0, 2)), selection)

    def test_select_by_source_and_target(self):
        graph = Graph(((self.source, self.source+1), self.nodes))
        selection = Graph(([(0,1)], list(zip(*self.nodes))[:2]))
        self.assertEqual(graph.select(start=(0, 3), end=1), selection)

    def test_select_by_edge_data(self):
        graph = Graph(((self.target, self.source, self.edge_info),), vdims=['info'])
        selection = Graph(([(0, 0, 0), (0, 1, 1)], list(zip(*self.nodes))[:2]), vdims=['info'])
        self.assertEqual(graph.select(info=(0, 2)), selection) 

    def test_graph_node_range(self):
        graph = Graph(((self.target, self.source),))
        self.assertEqual(graph.range('x'), (-1, 1))
        self.assertEqual(graph.range('y'), (-1, 1))

    def test_graph_redim_nodes(self):
        graph = Graph(((self.target, self.source),))
        redimmed = graph.redim(x='x2', y='y2')
        self.assertEqual(redimmed.nodes, graph.nodes.redim(x='x2', y='y2'))
        self.assertEqual(redimmed.edgepaths, graph.edgepaths.redim(x='x2', y='y2'))
