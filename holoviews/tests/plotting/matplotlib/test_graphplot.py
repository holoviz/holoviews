import numpy as np
import pytest
from matplotlib.collections import LineCollection, PolyCollection

from holoviews.core.data import Dataset
from holoviews.core.options import AbbreviatedException, Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Chord, Graph, Nodes, TriMesh, circular_layout
from holoviews.util.transform import dim

from .test_plot import MPL_GE_3_4_0, TestMPLPlot, mpl_renderer


class TestMplGraphPlot(TestMPLPlot):

    def setUp(self):
        super().setUp()

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
        self.assertEqual(np.asarray(nodes.get_offsets()), self.graph.nodes.array([0, 1]))
        self.assertEqual([p.vertices for p in edges.get_paths()],
                         [p.array() for p in self.graph.edgepaths.split()])

    def test_plot_graph_categorical_colored_nodes(self):
        g = self.graph2.opts(color_index='Label', cmap='Set1')
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
        g = self.graph3.opts(color_index='Weight', cmap='viridis')
        plot = mpl_renderer.get_plot(g)
        nodes = plot.handles['nodes']
        self.assertEqual(np.asarray(nodes.get_array()), self.weights)
        self.assertEqual(nodes.get_clim(), (self.weights.min(), self.weights.max()))

    def test_plot_graph_categorical_colored_edges(self):
        g = self.graph3.opts(
            edge_color_index='start', edge_cmap=['#FFFFFF', '#000000']
        )
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
        g = self.graph4.opts(
            edge_color_index='Weight', edge_cmap=['#FFFFFF', '#000000']
        )
        plot = mpl_renderer.get_plot(g)
        edges = plot.handles['edges']
        self.assertEqual(np.asarray(edges.get_array()), self.weights)
        self.assertEqual(edges.get_clim(), (self.weights.min(), self.weights.max()))

    ###########################
    #    Styling mapping      #
    ###########################

    def test_graph_op_node_color(self):
        edges = [(0, 1), (0, 2)]
        nodes = Nodes([(0, 0, 0, '#000000'), (0, 1, 1, '#FF0000'), (1, 1, 2, '#00FF00')],
                      vdims='color')
        graph = Graph((edges, nodes)).opts(node_color='color')
        plot = mpl_renderer.get_plot(graph)
        artist = plot.handles['nodes']
        self.assertEqual(artist.get_facecolors(),
                         np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1]]))

    def test_graph_op_node_color_update(self):
        edges = [(0, 1), (0, 2)]
        def get_graph(i):
            c1, c2, c3 = {0: ('#00FF00', '#0000FF', '#FF0000'),
                          1: ('#FF0000', '#00FF00', '#0000FF')}[i]
            nodes = Nodes([(0, 0, 0, c1), (0, 1, 1, c2), (1, 1, 2, c3)],
                      vdims='color')
            return Graph((edges, nodes))
        graph = HoloMap({0: get_graph(0), 1: get_graph(1)}).opts(node_color='color')
        plot = mpl_renderer.get_plot(graph)
        artist = plot.handles['nodes']
        self.assertEqual(artist.get_facecolors(),
                         np.array([[0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1]]))
        plot.update((1,))
        self.assertEqual(artist.get_facecolors(),
                         np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]))

    def test_graph_op_node_color_linear(self):
        edges = [(0, 1), (0, 2)]
        nodes = Nodes([(0, 0, 0, 0.5), (0, 1, 1, 1.5), (1, 1, 2, 2.5)],
                      vdims='color')
        graph = Graph((edges, nodes)).opts(node_color='color')
        plot = mpl_renderer.get_plot(graph)
        artist = plot.handles['nodes']
        self.assertEqual(np.asarray(artist.get_array()), np.array([0.5, 1.5, 2.5]))
        self.assertEqual(artist.get_clim(), (0.5, 2.5))

    def test_graph_op_node_color_linear_update(self):
        edges = [(0, 1), (0, 2)]
        def get_graph(i):
            c1, c2, c3 = {0: (0.5, 1.5, 2.5),
                          1: (3, 2, 1)}[i]
            nodes = Nodes([(0, 0, 0, c1), (0, 1, 1, c2), (1, 1, 2, c3)],
                      vdims='color')
            return Graph((edges, nodes))
        graph = HoloMap({0: get_graph(0), 1: get_graph(1)}).opts(node_color='color', framewise=True)
        plot = mpl_renderer.get_plot(graph)
        artist = plot.handles['nodes']
        self.assertEqual(np.asarray(artist.get_array()), np.array([0.5, 1.5, 2.5]))
        self.assertEqual(artist.get_clim(), (0.5, 2.5))
        plot.update((1,))
        self.assertEqual(np.asarray(artist.get_array()), np.array([3, 2, 1]))
        self.assertEqual(artist.get_clim(), (1, 3))

    def test_graph_op_node_color_categorical(self):
        edges = [(0, 1), (0, 2)]
        nodes = Nodes([(0, 0, 0, 'A'), (0, 1, 1, 'B'), (1, 1, 2, 'A')],
                      vdims='color')
        graph = Graph((edges, nodes)).opts(node_color='color')
        plot = mpl_renderer.get_plot(graph)
        artist = plot.handles['nodes']
        self.assertEqual(np.asarray(artist.get_array()), np.array([0, 1, 0]))

    def test_graph_op_node_size(self):
        edges = [(0, 1), (0, 2)]
        nodes = Nodes([(0, 0, 0, 2), (0, 1, 1, 4), (1, 1, 2, 6)],
                      vdims='size')
        graph = Graph((edges, nodes)).opts(node_size='size')
        plot = mpl_renderer.get_plot(graph)
        artist = plot.handles['nodes']
        self.assertEqual(artist.get_sizes(), np.array([4, 16, 36]))

    def test_graph_op_node_size_update(self):
        edges = [(0, 1), (0, 2)]
        def get_graph(i):
            c1, c2, c3 = {0: (2, 4, 6),
                          1: (12, 3, 5)}[i]
            nodes = Nodes([(0, 0, 0, c1), (0, 1, 1, c2), (1, 1, 2, c3)],
                      vdims='size')
            return Graph((edges, nodes))
        graph = HoloMap({0: get_graph(0), 1: get_graph(1)}).opts(node_size='size')
        plot = mpl_renderer.get_plot(graph)
        artist = plot.handles['nodes']
        self.assertEqual(artist.get_sizes(), np.array([4, 16, 36]))
        plot.update((1,))
        self.assertEqual(artist.get_sizes(), np.array([144, 9, 25]))

    def test_graph_op_node_linewidth(self):
        edges = [(0, 1), (0, 2)]
        nodes = Nodes([(0, 0, 0, 2), (0, 1, 1, 4), (1, 1, 2, 3.5)], vdims='line_width')
        graph = Graph((edges, nodes)).opts(node_linewidth='line_width')
        plot = mpl_renderer.get_plot(graph)
        artist = plot.handles['nodes']
        self.assertEqual(artist.get_linewidths(), [2, 4, 3.5])

    def test_graph_op_node_linewidth_update(self):
        edges = [(0, 1), (0, 2)]
        def get_graph(i):
            c1, c2, c3 = {0: (2, 4, 6),
                          1: (12, 3, 5)}[i]
            nodes = Nodes([(0, 0, 0, c1), (0, 1, 1, c2), (1, 1, 2, c3)],
                      vdims='line_width')
            return Graph((edges, nodes))
        graph = HoloMap({0: get_graph(0), 1: get_graph(1)}).opts(node_linewidth='line_width')
        plot = mpl_renderer.get_plot(graph)
        artist = plot.handles['nodes']
        self.assertEqual(artist.get_linewidths(), [2, 4, 6])
        plot.update((1,))
        self.assertEqual(artist.get_linewidths(), [12, 3, 5])

    def test_graph_op_node_alpha(self):
        edges = [(0, 1), (0, 2)]
        nodes = Nodes([(0, 0, 0, 0.2), (0, 1, 1, 0.6), (1, 1, 2, 1)], vdims='alpha')
        graph = Graph((edges, nodes)).opts(node_alpha='alpha')

        if MPL_GE_3_4_0:
            plot = mpl_renderer.get_plot(graph)
            artist = plot.handles['nodes']
            self.assertEqual(artist.get_alpha(), np.array([0.2, 0.6, 1]))
        else:
            # Python 3.6 only support up to matplotlib 3.3
            msg = 'TypeError: alpha must be a float or None'
            with pytest.raises(AbbreviatedException, match=msg):
                mpl_renderer.get_plot(graph)

    def test_graph_op_edge_color(self):
        edges = [(0, 1, 'red'), (0, 2, 'green'), (1, 3, 'blue')]
        graph = Graph(edges, vdims='color').opts(edge_color='color')
        plot = mpl_renderer.get_plot(graph)
        edges = plot.handles['edges']
        self.assertEqual(edges.get_edgecolors(), np.array([
            [1. , 0. , 0. , 1. ], [0. , 0.50196078, 0. , 1. ],
            [0. , 0. , 1. , 1. ]]
        ))

    def test_graph_op_edge_color_update(self):
        graph = HoloMap({
            0: Graph([(0, 1, 'red'), (0, 2, 'green'), (1, 3, 'blue')],
                    vdims='color'),
            1: Graph([(0, 1, 'green'), (0, 2, 'blue'), (1, 3, 'red')],
                     vdims='color')}).opts(edge_color='color')
        plot = mpl_renderer.get_plot(graph)
        edges = plot.handles['edges']
        self.assertEqual(edges.get_edgecolors(), np.array([
            [1. , 0. , 0. , 1. ], [0. , 0.50196078, 0. , 1. ],
            [0. , 0. , 1. , 1. ]]
        ))
        plot.update((1,))
        self.assertEqual(edges.get_edgecolors(), np.array([
            [0. , 0.50196078, 0. , 1. ], [0. , 0. , 1. , 1. ],
            [1. , 0. , 0. , 1. ]]
        ))

    def test_graph_op_edge_color_linear(self):
        edges = [(0, 1, 2), (0, 2, 0.5), (1, 3, 3)]
        graph = Graph(edges, vdims='color').opts(edge_color='color')
        plot = mpl_renderer.get_plot(graph)
        edges = plot.handles['edges']
        self.assertEqual(np.asarray(edges.get_array()), np.array([2, 0.5, 3]))
        self.assertEqual(edges.get_clim(), (0.5, 3))

    def test_graph_op_edge_color_linear_update(self):
        graph = HoloMap({
            0: Graph([(0, 1, 2), (0, 2, 0.5), (1, 3, 3)],
                    vdims='color'),
            1: Graph([(0, 1, 4.3), (0, 2, 1.4), (1, 3, 2.6)],
                     vdims='color')}).opts(edge_color='color', framewise=True)
        plot = mpl_renderer.get_plot(graph)
        edges = plot.handles['edges']
        self.assertEqual(np.asarray(edges.get_array()), np.array([2, 0.5, 3]))
        self.assertEqual(edges.get_clim(), (0.5, 3))
        plot.update((1,))
        self.assertEqual(np.asarray(edges.get_array()), np.array([4.3, 1.4, 2.6]))
        self.assertEqual(edges.get_clim(), (1.4, 4.3))

    def test_graph_op_edge_color_categorical(self):
        edges = [(0, 1, 'C'), (0, 2, 'B'), (1, 3, 'A')]
        graph = Graph(edges, vdims='color').opts(edge_color='color')
        plot = mpl_renderer.get_plot(graph)
        edges = plot.handles['edges']
        self.assertEqual(np.asarray(edges.get_array()), np.array([0, 1, 2]))
        self.assertEqual(edges.get_clim(), (0, 2))

    def test_graph_op_edge_alpha(self):
        edges = [(0, 1, 0.1), (0, 2, 0.5), (1, 3, 0.3)]
        graph = Graph(edges, vdims='alpha').opts(edge_alpha='alpha')
        msg = 'ValueError: Mapping a dimension to the "edge_alpha" style'
        with pytest.raises(AbbreviatedException, match=msg):
            mpl_renderer.get_plot(graph)

    def test_graph_op_edge_linewidth(self):
        edges = [(0, 1, 2), (0, 2, 10), (1, 3, 6)]
        graph = Graph(edges, vdims='line_width').opts(edge_linewidth='line_width')
        plot = mpl_renderer.get_plot(graph)
        edges = plot.handles['edges']
        self.assertEqual(edges.get_linewidths(), [2, 10, 6])

    def test_graph_op_edge_line_width_update(self):
        graph = HoloMap({
            0: Graph([(0, 1, 2), (0, 2, 0.5), (1, 3, 3)],
                    vdims='line_width'),
            1: Graph([(0, 1, 4.3), (0, 2, 1.4), (1, 3, 2.6)],
                     vdims='line_width')}).opts(edge_linewidth='line_width')
        plot = mpl_renderer.get_plot(graph)
        edges = plot.handles['edges']
        self.assertEqual(edges.get_linewidths(), [2, 0.5, 3])
        plot.update((1,))
        self.assertEqual(edges.get_linewidths(), [4.3, 1.4, 2.6])



class TestMplTriMeshPlot(TestMPLPlot):

    def setUp(self):
        super().setUp()

        self.nodes = [(0, 0, 0), (0.5, 1, 1), (1., 0, 2), (1.5, 1, 3)]
        self.simplices = [(0, 1, 2, 0), (1, 2, 3, 1)]
        self.trimesh = TriMesh((self.simplices, self.nodes))
        self.trimesh_weighted = TriMesh((self.simplices, self.nodes), vdims='weight')

    def test_plot_simple_trimesh(self):
        plot = mpl_renderer.get_plot(self.trimesh)
        nodes = plot.handles['nodes']
        edges = plot.handles['edges']
        self.assertIsInstance(edges, LineCollection)
        self.assertEqual(np.asarray(nodes.get_offsets()), self.trimesh.nodes.array([0, 1]))
        self.assertEqual([p.vertices for p in edges.get_paths()],
                         [p.array() for p in self.trimesh._split_edgepaths.split()])

    def test_plot_simple_trimesh_filled(self):
        plot = mpl_renderer.get_plot(self.trimesh.opts(filled=True))
        nodes = plot.handles['nodes']
        edges = plot.handles['edges']
        self.assertIsInstance(edges, PolyCollection)
        self.assertEqual(np.asarray(nodes.get_offsets()), self.trimesh.nodes.array([0, 1]))
        paths = self.trimesh._split_edgepaths.split(datatype='array')
        self.assertEqual([p.vertices[:4] for p in edges.get_paths()],
                         paths)

    def test_plot_trimesh_colored_edges(self):
        opts = dict(edge_color_index='weight', edge_cmap='Greys')
        plot = mpl_renderer.get_plot(self.trimesh_weighted.opts(**opts))
        edges = plot.handles['edges']
        colors = np.array([[ 1.,  1.,  1.,  1.],
                           [ 0.,  0.,  0.,  1.]])
        self.assertEqual(edges.get_edgecolors(), colors)

    def test_plot_trimesh_categorically_colored_edges(self):
        opts = dict(edge_color_index='node1', edge_color=Cycle('Set1'))
        plot = mpl_renderer.get_plot(self.trimesh_weighted.opts(**opts))
        edges = plot.handles['edges']
        colors = np.array([[0.894118, 0.101961, 0.109804, 1.],
                           [0.215686, 0.494118, 0.721569, 1.]])
        self.assertEqual(edges.get_edgecolors(), colors)

    def test_plot_trimesh_categorically_colored_edges_filled(self):
        opts = dict(edge_color_index='node1', filled=True, edge_color=Cycle('Set1'))
        plot = mpl_renderer.get_plot(self.trimesh_weighted.opts(**opts))
        edges = plot.handles['edges']
        colors = np.array([[0.894118, 0.101961, 0.109804, 1.],
                           [0.215686, 0.494118, 0.721569, 1.]])
        self.assertEqual(edges.get_facecolors(), colors)

    ###########################
    #    Styling mapping      #
    ###########################

    def test_trimesh_op_node_color(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 'red'), (0, 0, 1, 'green'), (0, 1, 2, 'blue'), (1, 0, 3, 'black')]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='color'))).opts(node_color='color')
        plot = mpl_renderer.get_plot(trimesh)
        artist = plot.handles['nodes']
        self.assertEqual(artist.get_facecolors(),
                         np.array([[1, 0, 0, 1], [0, 0.501961, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]]))

    def test_trimesh_op_node_color_linear(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 2), (0, 0, 1, 1), (0, 1, 2, 3), (1, 0, 3, 4)]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='color'))).opts(node_color='color')
        plot = mpl_renderer.get_plot(trimesh)
        artist = plot.handles['nodes']
        self.assertEqual(np.asarray(artist.get_array()), np.array([2, 1, 3, 4]))
        self.assertEqual(artist.get_clim(), (1, 4))

    def test_trimesh_op_node_color_categorical(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 'B'), (0, 0, 1, 'C'), (0, 1, 2, 'A'), (1, 0, 3, 'B')]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='color'))).opts(node_color='color')
        plot = mpl_renderer.get_plot(trimesh)
        artist = plot.handles['nodes']
        self.assertEqual(np.asarray(artist.get_array()), np.array([0, 1, 2, 0]))
        self.assertEqual(artist.get_clim(), (0, 2))

    def test_trimesh_op_node_size(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 3), (0, 0, 1, 2), (0, 1, 2, 8), (1, 0, 3, 4)]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='size'))).opts(node_size='size')
        plot = mpl_renderer.get_plot(trimesh)
        artist = plot.handles['nodes']
        self.assertEqual(artist.get_sizes(), np.array([9, 4, 64, 16]))

    def test_trimesh_op_node_alpha(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 0.2), (0, 0, 1, 0.6), (0, 1, 2, 1), (1, 0, 3, 0.3)]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='alpha'))).opts(node_alpha='alpha')

        if MPL_GE_3_4_0:
            plot = mpl_renderer.get_plot(trimesh)
            artist = plot.handles['nodes']
            self.assertEqual(artist.get_alpha(), np.array([0.2, 0.6, 1, 0.3]))
        else:
            # Python 3.6 only support up to matplotlib 3.3
            msg = "TypeError: alpha must be a float or None"
            with pytest.raises(AbbreviatedException, match=msg):
                mpl_renderer.get_plot(trimesh)

    def test_trimesh_op_node_line_width(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 0.2), (0, 0, 1, 0.6), (0, 1, 2, 1), (1, 0, 3, 0.3)]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='line_width'))).opts(node_linewidth='line_width')
        plot = mpl_renderer.get_plot(trimesh)
        artist = plot.handles['nodes']
        self.assertEqual(artist.get_linewidths(), [0.2, 0.6, 1, 0.3])

    def test_trimesh_op_edge_color_linear_mean_node(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 2), (0, 0, 1, 1), (0, 1, 2, 3), (1, 0, 3, 4)]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='color'))).opts(edge_color='color')
        plot = mpl_renderer.get_plot(trimesh)
        artist = plot.handles['edges']
        self.assertEqual(np.asarray(artist.get_array()), np.array([2, 8/3.]))
        self.assertEqual(artist.get_clim(), (1, 4))

    def test_trimesh_op_edge_color(self):
        edges = [(0, 1, 2, 'red'), (1, 2, 3, 'blue')]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='color').opts(edge_color='color')
        plot = mpl_renderer.get_plot(trimesh)
        artist = plot.handles['edges']
        self.assertEqual(artist.get_edgecolors(), np.array([
            [1, 0, 0, 1], [0, 0, 1, 1]]))

    def test_trimesh_op_edge_color_linear(self):
        edges = [(0, 1, 2, 2.4), (1, 2, 3, 3.6)]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='color').opts(edge_color='color')
        plot = mpl_renderer.get_plot(trimesh)
        artist = plot.handles['edges']
        self.assertEqual(np.asarray(artist.get_array()), np.array([2.4, 3.6]))
        self.assertEqual(artist.get_clim(), (2.4, 3.6))

    def test_trimesh_op_edge_color_categorical(self):
        edges = [(0, 1, 2, 'A'), (1, 2, 3, 'B')]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='color').opts(edge_color='color')
        plot = mpl_renderer.get_plot(trimesh)
        artist = plot.handles['edges']
        self.assertEqual(np.asarray(artist.get_array()), np.array([0, 1]))
        self.assertEqual(artist.get_clim(), (0, 1))

    def test_trimesh_op_edge_alpha(self):
        edges = [(0, 1, 2, 0.7), (1, 2, 3, 0.3)]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='alpha').opts(edge_alpha='alpha')
        msg = 'ValueError: Mapping a dimension to the "edge_alpha" style'
        with pytest.raises(AbbreviatedException, match=msg):
            mpl_renderer.get_plot(trimesh)

    def test_trimesh_op_edge_line_width(self):
        edges = [(0, 1, 2, 7), (1, 2, 3, 3)]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='line_width').opts(edge_linewidth='line_width')
        plot = mpl_renderer.get_plot(trimesh)
        artist = plot.handles['edges']
        self.assertEqual(artist.get_linewidths(), [7, 3])



class TestMplChordPlot(TestMPLPlot):

    def setUp(self):
        super().setUp()
        self.edges = [(0, 1, 1), (0, 2, 2), (1, 2, 3)]
        self.nodes = Dataset([(0, 'A'), (1, 'B'), (2, 'C')], 'index', 'Label')
        self.chord = Chord((self.edges, self.nodes))

    def make_chord(self, i):
        edges = [(0, 1, 1+i), (0, 2, 2+i), (1, 2, 3+i)]
        nodes = Dataset([(0, 0+i), (1, 1+i), (2, 2+i)], 'index', 'Label')
        return Chord((edges, nodes), vdims='weight')

    def test_chord_nodes_label_text(self):
        g = self.chord.opts(label_index='Label')
        plot = mpl_renderer.get_plot(g)
        labels = plot.handles['labels']
        self.assertEqual([l.get_text() for l in labels], ['A', 'B', 'C'])

    def test_chord_nodes_labels_mapping(self):
        g = self.chord.opts(labels='Label')
        plot = mpl_renderer.get_plot(g)
        labels = plot.handles['labels']
        self.assertEqual([l.get_text() for l in labels], ['A', 'B', 'C'])

    def test_chord_nodes_categorically_colormapped(self):
        g = self.chord.opts(color_index='Label', cmap=['#FFFFFF', '#CCCCCC', '#000000'])
        plot = mpl_renderer.get_plot(g)
        arcs = plot.handles['arcs']
        nodes = plot.handles['nodes']
        colors = np.array([[ 1.,   1.,   1.,   1. ],
                           [ 0.8,  0.8,  0.8,  1. ],
                           [ 0.,   0.,   0.,   1. ]])
        self.assertEqual(arcs.get_colors(), colors)
        self.assertEqual(nodes.get_facecolors(), colors)

    def test_chord_node_color_style_mapping(self):
        g = self.chord.opts(node_color='Label', cmap=['#FFFFFF', '#CCCCCC', '#000000'])
        plot = mpl_renderer.get_plot(g)
        arcs = plot.handles['arcs']
        nodes = plot.handles['nodes']
        self.assertEqual(np.asarray(nodes.get_array()), np.array([0, 1, 2]))
        self.assertEqual(np.asarray(arcs.get_array()), np.array([0, 1, 2]))
        self.assertEqual(nodes.get_clim(), (0, 2))
        self.assertEqual(arcs.get_clim(), (0, 2))

    def test_chord_edges_categorically_colormapped(self):
        g = self.chord.opts(edge_color_index='start', edge_cmap=['#FFFFFF', '#000000'])
        plot = mpl_renderer.get_plot(g)
        edges = plot.handles['edges']
        colors = np.array([[ 1., 1., 1., 1. ],
                           [ 1., 1., 1., 1. ],
                           [ 0., 0., 0., 1. ]])
        self.assertEqual(edges.get_edgecolors(), colors)

    def test_chord_edge_color_style_mapping(self):
        g = self.chord.opts(edge_color=dim('start').astype(str), edge_cmap=['#FFFFFF', '#000000'])
        plot = mpl_renderer.get_plot(g)
        edges = plot.handles['edges']
        self.assertEqual(np.asarray(edges.get_array()), np.array([0, 0, 1]))
        self.assertEqual(edges.get_clim(), (0, 2))

    def test_chord_edge_color_linear_style_mapping_update(self):
        hmap = HoloMap({0: self.make_chord(0), 1: self.make_chord(1)}).opts(edge_color='weight', framewise=True)
        plot = mpl_renderer.get_plot(hmap)
        edges = plot.handles['edges']
        self.assertEqual(np.asarray(edges.get_array()), np.array([1, 2, 3]))
        self.assertEqual(edges.get_clim(), (1, 3))
        plot.update((1,))
        self.assertEqual(np.asarray(edges.get_array()), np.array([2, 3, 4]))
        self.assertEqual(edges.get_clim(), (2, 4))

    def test_chord_node_color_linear_style_mapping_update(self):
        hmap = HoloMap({0: self.make_chord(0), 1: self.make_chord(1)}).opts(node_color='Label', framewise=True)
        plot = mpl_renderer.get_plot(hmap)
        arcs = plot.handles['arcs']
        nodes = plot.handles['nodes']
        self.assertEqual(np.asarray(nodes.get_array()), np.array([0, 1, 2]))
        self.assertEqual(np.asarray(arcs.get_array()), np.array([0, 1, 2]))
        self.assertEqual(nodes.get_clim(), (0, 2))
        self.assertEqual(arcs.get_clim(), (0, 2))
        plot.update((1,))
        self.assertEqual(np.asarray(nodes.get_array()), np.array([1, 2, 3]))
        self.assertEqual(np.asarray(arcs.get_array()), np.array([1, 2, 3]))
        self.assertEqual(nodes.get_clim(), (1, 3))
        self.assertEqual(arcs.get_clim(), (1, 3))

    def test_chord_edge_color_style_mapping_update(self):
        hmap = HoloMap({0: self.make_chord(0), 1: self.make_chord(1)}).opts(
            edge_color=dim('weight').categorize({1: 'red', 2: 'green', 3: 'blue', 4: 'black'})
        )
        plot = mpl_renderer.get_plot(hmap)
        edges = plot.handles['edges']
        self.assertEqual(edges.get_edgecolors(), np.array([
            [1, 0, 0, 1], [0, 0.501961, 0, 1], [0, 0, 1, 1]
        ]))
        plot.update((1,))
        self.assertEqual(edges.get_edgecolors(), np.array([
            [0, 0.501961, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]
        ]))

    def test_chord_node_color_style_mapping_update(self):
        hmap = HoloMap({0: self.make_chord(0), 1: self.make_chord(1)}).opts(
            node_color=dim('Label').categorize({0: 'red', 1: 'green', 2: 'blue', 3: 'black'})
        )
        plot = mpl_renderer.get_plot(hmap)
        arcs = plot.handles['arcs']
        nodes = plot.handles['nodes']
        colors = np.array([
            [1, 0, 0, 1], [0, 0.501961, 0, 1], [0, 0, 1, 1]
        ])
        self.assertEqual(arcs.get_edgecolors(), colors)
        self.assertEqual(nodes.get_facecolors(), colors)
        plot.update((1,))
        colors = np.array([
            [0, 0.501961, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]
        ])
        self.assertEqual(arcs.get_edgecolors(), colors)
        self.assertEqual(nodes.get_facecolors(), colors)
