"""
Unit tests of Graph Element.
"""
import numpy as np
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.graphs import (
    Graph, Nodes, EdgePaths, TriMesh, circular_layout, connect_edges,
    connect_edges_pd)
from holoviews.element.comparison import ComparisonTestCase



class GraphTests(ComparisonTestCase):

    def setUp(self):
        N = 8
        self.nodes = circular_layout(np.arange(N))
        self.source = np.arange(N, dtype=np.int32)
        self.target = np.zeros(N, dtype=np.int32)
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

    def test_graph_edge_segments(self):
        segments = connect_edges(self.graph)
        paths = []
        nodes = np.column_stack(self.nodes)
        for start, end in zip(nodes[self.source], nodes[self.target]):
            paths.append(np.array([start[:2], end[:2]]))
        self.assertEqual(segments, paths)

    def test_graph_node_info_no_index(self):
        node_info = Dataset(np.arange(8), vdims=['Label'])
        graph = Graph(((self.source, self.target), node_info))
        self.assertEqual(graph.nodes.dimension_values(3),
                         node_info.dimension_values(0))

    def test_graph_node_info_no_index_mismatch(self):
        node_info = Dataset(np.arange(6), vdims=['Label'])
        with self.assertRaises(ValueError):
            Graph(((self.source, self.target), node_info))

    def test_graph_node_info_merge_on_index(self):
        node_info = Dataset((np.arange(8), np.arange(1,9)), 'index', 'label')
        graph = Graph(((self.source, self.target), node_info))
        self.assertEqual(graph.nodes.dimension_values(3),
                         node_info.dimension_values(1))

    def test_graph_node_info_merge_on_index_partial(self):
        node_info = Dataset((np.arange(5), np.arange(1,6)), 'index', 'label')
        graph = Graph(((self.source, self.target), node_info))
        expected = np.array([1., 2., 3., 4., 5., np.NaN, np.NaN, np.NaN])
        self.assertEqual(graph.nodes.dimension_values(3), expected)

    def test_graph_edge_segments_pd(self):
        segments = connect_edges_pd(self.graph)
        paths = []
        nodes = np.column_stack(self.nodes)
        for start, end in zip(nodes[self.source], nodes[self.target]):
            paths.append(np.array([start[:2], end[:2]]))
        self.assertEqual(segments, paths)

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


class TriMeshTests(ComparisonTestCase):

    def setUp(self):
        self.nodes = [(0, 0, 0), (0.5, 1, 1), (1., 0, 2), (1.5, 1, 3)]
        self.simplices = [(0, 1, 2), (1, 2, 3)]

    def test_trimesh_constructor(self):
        trimesh = TriMesh((self.simplices, self.nodes))
        self.assertEqual(trimesh.array(), np.array(self.simplices))
        self.assertEqual(trimesh.nodes.array(), np.array(self.nodes))

    def test_trimesh_empty(self):
        trimesh = TriMesh([])
        self.assertEqual(trimesh.array(), np.empty((0, 3)))
        self.assertEqual(trimesh.nodes.array(), np.empty((0, 3)))

    def test_trimesh_empty_clone(self):
        trimesh = TriMesh([]).clone()
        self.assertEqual(trimesh.array(), np.empty((0, 3)))
        self.assertEqual(trimesh.nodes.array(), np.empty((0, 3)))

    def test_trimesh_constructor_tuple_nodes(self):
        nodes = tuple(zip(*self.nodes))[:2]
        trimesh = TriMesh((self.simplices, nodes))
        self.assertEqual(trimesh.array(), np.array(self.simplices))
        self.assertEqual(trimesh.nodes.array(), np.array(self.nodes))

    def test_trimesh_constructor_point_nodes(self):
        trimesh = TriMesh((self.simplices, Points([n[:2] for n in self.nodes])))
        self.assertEqual(trimesh.array(), np.array(self.simplices))
        self.assertEqual(trimesh.nodes.array(), np.array(self.nodes))

    def test_trimesh_edgepaths(self):
        trimesh = TriMesh((self.simplices, self.nodes))
        paths = [np.array([(0, 0), (0.5, 1), (1, 0), (0, 0), (np.NaN, np.NaN),
                 (0.5, 1), (1, 0), (1.5, 1), (0.5, 1)])]
        for p1, p2 in zip(trimesh.edgepaths.split(datatype='array'), paths):
            self.assertEqual(p1, p2)

    def test_trimesh_select(self):
        trimesh = TriMesh((self.simplices, self.nodes)).select(x=(0.1, None))
        self.assertEqual(trimesh.array(), np.array(self.simplices[1:]))
