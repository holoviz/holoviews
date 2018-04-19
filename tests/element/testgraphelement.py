"""
Unit tests of Graph Element.
"""
from unittest import SkipTest

import numpy as np
from holoviews.core.data import Dataset
from holoviews.core import util
from holoviews.element.chart import Points
from holoviews.element.graphs import (
    Graph, Nodes, TriMesh, Chord, circular_layout, connect_edges,
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
        self.assertEqual(graph._edgepaths, paths)

    def test_constructor_with_nodes_and_paths_dimension_mismatch(self):
        paths = Graph(((self.source, self.target), self.nodes)).edgepaths
        exception = 'Ensure that the first two key dimensions on Nodes and EdgePaths match: x != x2'
        with self.assertRaisesRegexp(ValueError, exception):
            Graph(((self.source, self.target), self.nodes, paths.redim(x='x2')))

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



class ChordTests(ComparisonTestCase):

    def setUp(self):
        self.simplices = [(0, 1, 2), (1, 2, 3), (2, 3, 4)]

    def test_chord_constructor_no_vdims(self):
        chord = Chord(self.simplices)
        nodes = np.array([[0.8660254037844387, 0.49999999999999994, 0],
                          [-0.4999999999999998, 0.8660254037844388, 1],
                          [-0.5000000000000004, -0.8660254037844384, 2],
                          [0.8660254037844379, -0.5000000000000012, 3]])
        self.assertEqual(chord.nodes, Nodes(nodes))
        self.assertEqual(chord.array(), np.array([s[:2] for s in self.simplices]))

    def test_chord_constructor_with_vdims(self):
        chord = Chord(self.simplices, vdims=['z'])
        nodes = np.array(
            [[0.9396926207859084, 0.3420201433256687, 0],
             [6.123233995736766e-17, 1.0, 1],
             [-0.8660254037844388, -0.4999999999999998, 2],
             [0.7660444431189779, -0.6427876096865396, 3]]
        )
        self.assertEqual(chord.nodes, Nodes(nodes))
        self.assertEqual(chord.array(), np.array(self.simplices))

    def test_chord_constructor_self_reference(self):
        chord = Chord([('A', 'B', 2), ('B', 'A', 3), ('A', 'A', 2)])
        nodes = np.array(
            [[-0.5, 0.866025, 0],
             [0.5, -0.866025, 1]]
        ) 
        self.assertEqual(chord.nodes, Nodes(nodes))



class TriMeshTests(ComparisonTestCase):

    def setUp(self):
        self.nodes = [(0, 0, 0), (0.5, 1, 1), (1., 0, 2), (1.5, 1, 4)]
        self.simplices = [(0, 1, 2), (1, 2, 3)]

    def test_trimesh_constructor(self):
        nodes = [n[:2] for n in self.nodes]
        trimesh = TriMesh((self.simplices, nodes))
        self.assertEqual(trimesh.array(), np.array(self.simplices))
        self.assertEqual(trimesh.nodes.array([0, 1]), np.array(nodes))
        self.assertEqual(trimesh.nodes.dimension_values(2), np.arange(4))

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
        self.assertEqual(trimesh.nodes.array([0, 1]), np.array(nodes).T)
        self.assertEqual(trimesh.nodes.dimension_values(2), np.arange(4))

    def test_trimesh_constructor_df_nodes(self):
        if util.pd is None:
            raise SkipTest('Test requires pandas')
        nodes_df = util.pd.DataFrame(self.nodes, columns=['x', 'y', 'z'])
        trimesh = TriMesh((self.simplices, nodes_df))
        nodes = Nodes([(0, 0, 0, 0), (0.5, 1, 1, 1),
                       (1., 0, 2, 2), (1.5, 1, 3, 4)], vdims='z')
        self.assertEqual(trimesh.array(), np.array(self.simplices))
        self.assertEqual(trimesh.nodes, nodes)

    def test_trimesh_constructor_point_nodes(self):
        nodes = [n[:2] for n in self.nodes]
        trimesh = TriMesh((self.simplices, Points(self.nodes)))
        self.assertEqual(trimesh.array(), np.array(self.simplices))
        self.assertEqual(trimesh.nodes.array([0, 1]), np.array(nodes))
        self.assertEqual(trimesh.nodes.dimension_values(2), np.arange(4))

    def test_trimesh_edgepaths(self):
        trimesh = TriMesh((self.simplices, self.nodes))
        paths = [np.array([(0, 0), (0.5, 1), (1, 0), (0, 0), (np.NaN, np.NaN),
                 (0.5, 1), (1, 0), (1.5, 1), (0.5, 1)])]
        for p1, p2 in zip(trimesh.edgepaths.split(datatype='array'), paths):
            self.assertEqual(p1, p2)

    def test_trimesh_select(self):
        trimesh = TriMesh((self.simplices, self.nodes)).select(x=(0.1, None))
        self.assertEqual(trimesh.array(), np.array(self.simplices[1:]))
