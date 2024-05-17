import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper

from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim

from .test_plot import TestBokehPlot, bokeh_renderer


class TestBokehGraphPlot(TestBokehPlot):

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
        plot = bokeh_renderer.get_plot(self.graph)
        node_source = plot.handles['scatter_1_source']
        edge_source = plot.handles['multi_line_1_source']
        layout_source = plot.handles['layout_source']
        self.assertEqual(node_source.data['index'], self.source)
        self.assertEqual(edge_source.data['start'], self.source)
        self.assertEqual(edge_source.data['end'], self.target)
        layout = {z: (x, y) for x, y, z in self.graph.nodes.array()}

        self.assertEqual(layout_source.graph_layout, layout)

    def test_plot_graph_annotation_overlay(self):
        plot = bokeh_renderer.get_plot(VLine(0) * self.graph)
        x_range = plot.handles['x_range']
        y_range = plot.handles['x_range']
        self.assertEqual(x_range.start, -1)
        self.assertEqual(x_range.end, 1)
        self.assertEqual(y_range.start, -1)
        self.assertEqual(y_range.end, 1)

    def test_plot_graph_with_paths(self):
        graph = self.graph.clone((self.graph.data, self.graph.nodes, self.graph.edgepaths))
        plot = bokeh_renderer.get_plot(graph)
        node_source = plot.handles['scatter_1_source']
        edge_source = plot.handles['multi_line_1_source']
        layout_source = plot.handles['layout_source']
        self.assertEqual(node_source.data['index'], self.source)
        self.assertEqual(edge_source.data['start'], self.source)
        self.assertEqual(edge_source.data['end'], self.target)
        edges = graph.edgepaths.split()
        self.assertEqual(edge_source.data['xs'], [path.dimension_values(0) for path in edges])
        self.assertEqual(edge_source.data['ys'], [path.dimension_values(1) for path in edges])
        layout = {z: (x, y) for x, y, z in self.graph.nodes.array()}
        self.assertEqual(layout_source.graph_layout, layout)

    def test_graph_inspection_policy_nodes(self):
        plot = bokeh_renderer.get_plot(self.graph)
        renderer = plot.handles['glyph_renderer']
        hover = plot.handles['hover']
        self.assertIsInstance(renderer.inspection_policy, NodesAndLinkedEdges)
        self.assertEqual(hover.tooltips, [('index', '@{index_hover}')])
        self.assertIn(renderer, hover.renderers)

    def test_graph_inspection_policy_edges(self):
        plot = bokeh_renderer.get_plot(self.graph.opts(inspection_policy='edges'))
        renderer = plot.handles['glyph_renderer']
        hover = plot.handles['hover']
        self.assertIsInstance(renderer.inspection_policy, EdgesAndLinkedNodes)
        self.assertEqual(hover.tooltips, [('start', '@{start_values}'), ('end', '@{end_values}')])
        self.assertIn(renderer, hover.renderers)

    def test_graph_inspection_policy_edges_non_default_names(self):
        graph = self.graph.redim(start='source', end='target')
        plot = bokeh_renderer.get_plot(graph.opts(inspection_policy='edges'))
        renderer = plot.handles['glyph_renderer']
        hover = plot.handles['hover']
        self.assertIsInstance(renderer.inspection_policy, EdgesAndLinkedNodes)
        self.assertEqual(hover.tooltips, [('source', '@{source}'), ('target', '@{target}')])
        self.assertIn(renderer, hover.renderers)

    def test_graph_inspection_policy_none(self):
        plot = bokeh_renderer.get_plot(self.graph.opts(inspection_policy=None))
        renderer = plot.handles['glyph_renderer']
        self.assertIsInstance(renderer.inspection_policy, NodesOnly)

    def test_graph_selection_policy_nodes(self):
        plot = bokeh_renderer.get_plot(self.graph)
        renderer = plot.handles['glyph_renderer']
        hover = plot.handles['hover']
        self.assertIsInstance(renderer.selection_policy, NodesAndLinkedEdges)
        self.assertIn(renderer, hover.renderers)

    def test_graph_selection_policy_edges(self):
        plot = bokeh_renderer.get_plot(self.graph.opts(selection_policy='edges'))
        renderer = plot.handles['glyph_renderer']
        hover = plot.handles['hover']
        self.assertIsInstance(renderer.selection_policy, EdgesAndLinkedNodes)
        self.assertIn(renderer, hover.renderers)

    def test_graph_selection_policy_none(self):
        plot = bokeh_renderer.get_plot(self.graph.opts(selection_policy=None))
        renderer = plot.handles['glyph_renderer']
        self.assertIsInstance(renderer.selection_policy, NodesOnly)

    def test_graph_nodes_categorical_colormapped(self):
        g = self.graph2.opts(color_index='Label', cmap='Set1')
        plot = bokeh_renderer.get_plot(g)
        cmapper = plot.handles['color_mapper']
        node_source = plot.handles['scatter_1_source']
        glyph = plot.handles['scatter_1_glyph']
        self.assertIsInstance(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, ['Output', 'Input'])
        self.assertEqual(node_source.data['Label'], self.node_info['Label'])
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'Label', 'transform': cmapper})

    def test_graph_nodes_numerically_colormapped(self):
        g = self.graph3.opts(color_index='Weight', cmap='viridis')
        plot = bokeh_renderer.get_plot(g)
        cmapper = plot.handles['color_mapper']
        node_source = plot.handles['scatter_1_source']
        glyph = plot.handles['scatter_1_glyph']
        self.assertIsInstance(cmapper, LinearColorMapper)
        self.assertEqual(cmapper.low, self.weights.min())
        self.assertEqual(cmapper.high, self.weights.max())
        self.assertEqual(node_source.data['Weight'], self.node_info2['Weight'])
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'Weight', 'transform': cmapper})

    def test_graph_edges_categorical_colormapped(self):
        g = self.graph3.opts(edge_color_index='start', edge_cmap=['#FFFFFF', '#000000'])
        plot = bokeh_renderer.get_plot(g)
        cmapper = plot.handles['edge_colormapper']
        edge_source = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        self.assertIsInstance(cmapper, CategoricalColorMapper)
        factors = ['0', '1', '2', '3', '4', '5', '6', '7']
        self.assertEqual(cmapper.factors, factors)
        self.assertEqual(edge_source.data['start_str__'], factors)
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'start_str__', 'transform': cmapper})

    def test_graph_edges_numerically_colormapped(self):
        g = self.graph4.opts(edge_color_index='Weight', edge_cmap=['#FFFFFF', '#000000'])
        plot = bokeh_renderer.get_plot(g)
        cmapper = plot.handles['edge_colormapper']
        edge_source = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        self.assertIsInstance(cmapper, LinearColorMapper)
        self.assertEqual(cmapper.low, self.weights.min())
        self.assertEqual(cmapper.high, self.weights.max())
        self.assertEqual(edge_source.data['Weight'], self.node_info2['Weight'])
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'Weight', 'transform': cmapper})

    ###########################
    #    Styling mapping      #
    ###########################

    def test_graph_op_node_color(self):
        edges = [(0, 1), (0, 2)]
        nodes = Nodes([(0, 0, 0, 'red'), (0, 1, 1, 'green'), (1, 1, 2, 'blue')],
                      vdims='color')
        graph = Graph((edges, nodes)).opts(node_color='color')
        plot = bokeh_renderer.get_plot(graph)
        cds = plot.handles['scatter_1_source']
        glyph = plot.handles['scatter_1_glyph']
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'node_color'})
        self.assertEqual(glyph.line_color, 'black')
        self.assertEqual(cds.data['node_color'], np.array(['red', 'green', 'blue']))

    def test_graph_op_node_color_linear(self):
        edges = [(0, 1), (0, 2)]
        nodes = Nodes([(0, 0, 0, 0.5), (0, 1, 1, 1.5), (1, 1, 2, 2.5)],
                      vdims='color')
        graph = Graph((edges, nodes)).opts(node_color='color')
        plot = bokeh_renderer.get_plot(graph)
        cds = plot.handles['scatter_1_source']
        glyph = plot.handles['scatter_1_glyph']
        cmapper = plot.handles['node_color_color_mapper']
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'node_color', 'transform': cmapper})
        self.assertEqual(glyph.line_color, 'black')
        self.assertEqual(cds.data['node_color'], np.array([0.5, 1.5, 2.5]))

    def test_graph_op_node_color_colorbar(self):
        edges = [(0, 1), (0, 2)]
        nodes = Nodes([(0, 0, 0, 0.5), (0, 1, 1, 1.5), (1, 1, 2, 2.5)],
                      vdims='color')
        graph = Graph((edges, nodes)).opts(node_color='color', colorbar=True)
        plot = bokeh_renderer.get_plot(graph)
        assert 'node_color_colorbar' in plot.handles
        assert plot.handles['node_color_colorbar'].color_mapper is plot.handles['node_color_color_mapper']

    def test_graph_op_node_color_categorical(self):
        edges = [(0, 1), (0, 2)]
        nodes = Nodes([(0, 0, 0, 'A'), (0, 1, 1, 'B'), (1, 1, 2, 'C')],
                      vdims='color')
        graph = Graph((edges, nodes)).opts(node_color='color')
        plot = bokeh_renderer.get_plot(graph)
        cds = plot.handles['scatter_1_source']
        glyph = plot.handles['scatter_1_glyph']
        cmapper = plot.handles['node_color_color_mapper']
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'node_color', 'transform': cmapper})
        self.assertEqual(glyph.line_color, 'black')
        self.assertEqual(cds.data['node_color'], np.array(['A', 'B', 'C']))

    def test_graph_op_node_size(self):
        edges = [(0, 1), (0, 2)]
        nodes = Nodes([(0, 0, 0, 2), (0, 1, 1, 4), (1, 1, 2, 6)],
                      vdims='size')
        graph = Graph((edges, nodes)).opts(node_size='size')
        plot = bokeh_renderer.get_plot(graph)
        cds = plot.handles['scatter_1_source']
        glyph = plot.handles['scatter_1_glyph']
        self.assertEqual(property_to_dict(glyph.size), {'field': 'node_size'})
        self.assertEqual(cds.data['node_size'], np.array([2, 4, 6]))

    def test_graph_op_node_alpha(self):
        edges = [(0, 1), (0, 2)]
        nodes = Nodes([(0, 0, 0, 0.2), (0, 1, 1, 0.6), (1, 1, 2, 1)], vdims='alpha')
        graph = Graph((edges, nodes)).opts(node_alpha='alpha')
        plot = bokeh_renderer.get_plot(graph)
        cds = plot.handles['scatter_1_source']
        glyph = plot.handles['scatter_1_glyph']
        self.assertEqual(property_to_dict(glyph.fill_alpha), {'field': 'node_alpha'})
        self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'node_alpha'})
        self.assertEqual(cds.data['node_alpha'], np.array([0.2, 0.6, 1]))

    def test_graph_op_node_line_width(self):
        edges = [(0, 1), (0, 2)]
        nodes = Nodes([(0, 0, 0, 2), (0, 1, 1, 4), (1, 1, 2, 6)], vdims='line_width')
        graph = Graph((edges, nodes)).opts(node_line_width='line_width')
        plot = bokeh_renderer.get_plot(graph)
        cds = plot.handles['scatter_1_source']
        glyph = plot.handles['scatter_1_glyph']
        self.assertEqual(property_to_dict(glyph.line_width), {'field': 'node_line_width'})
        self.assertEqual(cds.data['node_line_width'], np.array([2, 4, 6]))

    def test_graph_op_edge_color(self):
        edges = [(0, 1, 'red'), (0, 2, 'green'), (1, 3, 'blue')]
        graph = Graph(edges, vdims='color').opts(edge_color='color')
        plot = bokeh_renderer.get_plot(graph)
        cds = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'edge_color'})
        self.assertEqual(cds.data['edge_color'], np.array(['red', 'green', 'blue']))

    def test_graph_op_edge_color_linear(self):
        edges = [(0, 1, 2), (0, 2, 0.5), (1, 3, 3)]
        graph = Graph(edges, vdims='color').opts(edge_color='color')
        plot = bokeh_renderer.get_plot(graph)
        cds = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        cmapper = plot.handles['edge_color_color_mapper']
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'edge_color', 'transform': cmapper})
        self.assertEqual(cds.data['edge_color'], np.array([2, 0.5, 3]))

    def test_graph_op_edge_color_colorbar(self):
        edges = [(0, 1, 2), (0, 2, 0.5), (1, 3, 3)]
        graph = Graph(edges, vdims='color').opts(edge_color='color', colorbar=True)
        plot = bokeh_renderer.get_plot(graph)
        assert 'edge_color_colorbar' in plot.handles
        assert plot.handles['edge_color_colorbar'].color_mapper is plot.handles['edge_color_color_mapper']

    def test_graph_op_edge_color_categorical(self):
        edges = [(0, 1, 'C'), (0, 2, 'B'), (1, 3, 'A')]
        graph = Graph(edges, vdims='color').opts(edge_color='color')
        plot = bokeh_renderer.get_plot(graph)
        cds = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        cmapper = plot.handles['edge_color_color_mapper']
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'edge_color', 'transform': cmapper})
        self.assertEqual(cds.data['edge_color'], np.array(['C', 'B', 'A']))

    def test_graph_op_edge_alpha(self):
        edges = [(0, 1, 0.1), (0, 2, 0.5), (1, 3, 0.3)]
        graph = Graph(edges, vdims='alpha').opts(edge_alpha='alpha')
        plot = bokeh_renderer.get_plot(graph)
        cds = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'edge_alpha'})
        self.assertEqual(cds.data['edge_alpha'], np.array([0.1, 0.5, 0.3]))

    def test_graph_op_edge_line_width(self):
        edges = [(0, 1, 2), (0, 2, 10), (1, 3, 6)]
        graph = Graph(edges, vdims='line_width').opts(edge_line_width='line_width')
        plot = bokeh_renderer.get_plot(graph)
        cds = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        self.assertEqual(property_to_dict(glyph.line_width), {'field': 'edge_line_width'})
        self.assertEqual(cds.data['edge_line_width'], np.array([2, 10, 6]))


class TestBokehTriMeshPlot(TestBokehPlot):

    def setUp(self):
        super().setUp()

        self.nodes = [(0, 0, 0), (0.5, 1, 1), (1., 0, 2), (1.5, 1, 3)]
        self.simplices = [(0, 1, 2, 0), (1, 2, 3, 1)]
        self.trimesh = TriMesh((self.simplices, self.nodes))
        self.trimesh_weighted = TriMesh((self.simplices, self.nodes), vdims='weight')

    def test_plot_simple_trimesh(self):
        plot = bokeh_renderer.get_plot(self.trimesh)
        node_source = plot.handles['scatter_1_source']
        edge_source = plot.handles['multi_line_1_source']
        layout_source = plot.handles['layout_source']
        self.assertEqual(node_source.data['index'], np.arange(4))
        self.assertEqual(edge_source.data['start'], np.arange(2))
        self.assertEqual(edge_source.data['end'], np.arange(1, 3))
        layout = {z: (x, y) for x, y, z in self.trimesh.nodes.array()}
        self.assertEqual(layout_source.graph_layout, layout)

    def test_plot_simple_trimesh_filled(self):
        plot = bokeh_renderer.get_plot(self.trimesh.opts(filled=True))
        node_source = plot.handles['scatter_1_source']
        edge_source = plot.handles['patches_1_source']
        layout_source = plot.handles['layout_source']
        self.assertIsInstance(plot.handles['patches_1_glyph'], Patches)
        self.assertEqual(node_source.data['index'], np.arange(4))
        self.assertEqual(edge_source.data['start'], np.arange(2))
        self.assertEqual(edge_source.data['end'], np.arange(1, 3))
        layout = {z: (x, y) for x, y, z in self.trimesh.nodes.array()}
        self.assertEqual(layout_source.graph_layout, layout)

    def test_trimesh_edges_categorical_colormapped(self):
        g = self.trimesh.opts(
            edge_color_index='node1', edge_cmap=['#FFFFFF', '#000000']
        )
        plot = bokeh_renderer.get_plot(g)
        cmapper = plot.handles['edge_colormapper']
        edge_source = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        self.assertIsInstance(cmapper, CategoricalColorMapper)
        factors = ['0', '1', '2', '3']
        self.assertEqual(cmapper.factors, factors)
        self.assertEqual(edge_source.data['node1_str__'], ['0', '1'])
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'node1_str__', 'transform': cmapper})

    def test_trimesh_nodes_numerically_colormapped(self):
        g = self.trimesh_weighted.opts(
            edge_color_index='weight', edge_cmap=['#FFFFFF', '#000000']
        )
        plot = bokeh_renderer.get_plot(g)
        cmapper = plot.handles['edge_colormapper']
        edge_source = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        self.assertIsInstance(cmapper, LinearColorMapper)
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 1)
        self.assertEqual(edge_source.data['weight'], np.array([0, 1]))
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'weight', 'transform': cmapper})

    ###########################
    #    Styling mapping      #
    ###########################

    def test_trimesh_op_node_color(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 'red'), (0, 0, 1, 'green'), (0, 1, 2, 'blue'), (1, 0, 3, 'black')]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='color'))).opts(node_color='color')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['scatter_1_source']
        glyph = plot.handles['scatter_1_glyph']
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'node_color'})
        self.assertEqual(glyph.line_color, 'black')
        self.assertEqual(cds.data['node_color'], np.array(['red', 'green', 'blue', 'black']))

    def test_trimesh_op_node_color_linear(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 2), (0, 0, 1, 1), (0, 1, 2, 3), (1, 0, 3, 4)]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='color'))).opts(node_color='color')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['scatter_1_source']
        glyph = plot.handles['scatter_1_glyph']
        cmapper = plot.handles['node_color_color_mapper']
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'node_color', 'transform': cmapper})
        self.assertEqual(glyph.line_color, 'black')
        self.assertEqual(cds.data['node_color'], np.array([2, 1, 3, 4]))
        self.assertEqual(cmapper.low, 1)
        self.assertEqual(cmapper.high, 4)

    def test_trimesh_op_node_color_categorical(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 'B'), (0, 0, 1, 'C'), (0, 1, 2, 'A'), (1, 0, 3, 'B')]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='color'))).opts(node_color='color')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['scatter_1_source']
        glyph = plot.handles['scatter_1_glyph']
        cmapper = plot.handles['node_color_color_mapper']
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'node_color', 'transform': cmapper})
        self.assertEqual(glyph.line_color, 'black')
        self.assertEqual(cds.data['node_color'], np.array(['B', 'C', 'A', 'B']))

    def test_trimesh_op_node_size(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 3), (0, 0, 1, 2), (0, 1, 2, 8), (1, 0, 3, 4)]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='size'))).opts(node_size='size')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['scatter_1_source']
        glyph = plot.handles['scatter_1_glyph']
        self.assertEqual(property_to_dict(glyph.size), {'field': 'node_size'})
        self.assertEqual(cds.data['node_size'], np.array([3, 2, 8, 4]))

    def test_trimesh_op_node_alpha(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 0.2), (0, 0, 1, 0.6), (0, 1, 2, 1), (1, 0, 3, 0.3)]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='alpha'))).opts(node_alpha='alpha')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['scatter_1_source']
        glyph = plot.handles['scatter_1_glyph']
        self.assertEqual(property_to_dict(glyph.fill_alpha), {'field': 'node_alpha'})
        self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'node_alpha'})
        self.assertEqual(cds.data['node_alpha'], np.array([0.2, 0.6, 1, 0.3]))

    def test_trimesh_op_node_line_width(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 0.2), (0, 0, 1, 0.6), (0, 1, 2, 1), (1, 0, 3, 0.3)]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='line_width'))).opts(node_line_width='line_width')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['scatter_1_source']
        glyph = plot.handles['scatter_1_glyph']
        self.assertEqual(property_to_dict(glyph.line_width), {'field': 'node_line_width'})
        self.assertEqual(cds.data['node_line_width'], np.array([0.2, 0.6, 1, 0.3]))

    def test_trimesh_op_edge_color_linear_mean_node(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 2), (0, 0, 1, 1), (0, 1, 2, 3), (1, 0, 3, 4)]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='color'))).opts(edge_color='color')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        cmapper = plot.handles['edge_color_color_mapper']
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'edge_color', 'transform': cmapper})
        self.assertEqual(cds.data['edge_color'], np.array([2, 8/3.]))
        self.assertEqual(cmapper.low, 1)
        self.assertEqual(cmapper.high, 4)

    def test_trimesh_op_edge_color(self):
        edges = [(0, 1, 2, 'red'), (1, 2, 3, 'blue')]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='color').opts(edge_color='color')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'edge_color'})
        self.assertEqual(cds.data['edge_color'], np.array(['red', 'blue']))

    def test_trimesh_op_edge_color_linear(self):
        edges = [(0, 1, 2, 2.4), (1, 2, 3, 3.6)]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='color').opts(edge_color='color')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        cmapper = plot.handles['edge_color_color_mapper']
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'edge_color', 'transform': cmapper})
        self.assertEqual(cds.data['edge_color'], np.array([2.4, 3.6]))
        self.assertEqual(cmapper.low, 2.4)
        self.assertEqual(cmapper.high, 3.6)

    def test_trimesh_op_edge_color_linear_filled(self):
        edges = [(0, 1, 2, 2.4), (1, 2, 3, 3.6)]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='color').opts(edge_color='color', filled=True)
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['patches_1_source']
        glyph = plot.handles['patches_1_glyph']
        cmapper = plot.handles['edge_color_color_mapper']
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'edge_color', 'transform': cmapper})
        self.assertEqual(glyph.line_color, 'black')
        self.assertEqual(cds.data['edge_color'], np.array([2.4, 3.6]))
        self.assertEqual(cmapper.low, 2.4)
        self.assertEqual(cmapper.high, 3.6)

    def test_trimesh_op_edge_color_categorical(self):
        edges = [(0, 1, 2, 'A'), (1, 2, 3, 'B')]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='color').opts(edge_color='color')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        cmapper = plot.handles['edge_color_color_mapper']
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'edge_color', 'transform': cmapper})
        self.assertEqual(cds.data['edge_color'], np.array(['A', 'B']))
        self.assertEqual(cmapper.factors, ['A', 'B'])

    def test_trimesh_op_edge_alpha(self):
        edges = [(0, 1, 2, 0.7), (1, 2, 3, 0.3)]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='alpha').opts(edge_alpha='alpha')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'edge_alpha'})
        self.assertEqual(cds.data['edge_alpha'], np.array([0.7, 0.3]))

    def test_trimesh_op_edge_line_width(self):
        edges = [(0, 1, 2, 7), (1, 2, 3, 3)]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='line_width').opts(edge_line_width='line_width')
        plot = bokeh_renderer.get_plot(trimesh)
        cds = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        self.assertEqual(property_to_dict(glyph.line_width), {'field': 'edge_line_width'})
        self.assertEqual(cds.data['edge_line_width'], np.array([7, 3]))


class TestBokehChordPlot(TestBokehPlot):

    def setUp(self):
        super().setUp()
        self.edges = [(0, 1, 1), (0, 2, 2), (1, 2, 3)]
        self.nodes = Dataset([(0, 'A'), (1, 'B'), (2, 'C')], 'index', 'Label')
        self.chord = Chord((self.edges, self.nodes))

    def test_chord_draw_order(self):
        plot = bokeh_renderer.get_plot(self.chord)
        renderers = plot.state.renderers
        graph_renderer = plot.handles['glyph_renderer']
        arc_renderer = plot.handles['multi_line_2_glyph_renderer']
        self.assertTrue(renderers.index(arc_renderer)<renderers.index(graph_renderer))

    def test_chord_label_draw_order(self):
        g = self.chord.opts(labels='Label')
        plot = bokeh_renderer.get_plot(g)
        renderers = plot.state.renderers
        graph_renderer = plot.handles['glyph_renderer']
        label_renderer = plot.handles['text_1_glyph_renderer']
        self.assertTrue(renderers.index(graph_renderer)<renderers.index(label_renderer))

    def test_chord_nodes_label_text(self):
        g = self.chord.opts(label_index='Label')
        plot = bokeh_renderer.get_plot(g)
        source = plot.handles['text_1_source']
        self.assertEqual(source.data['text'], ['A', 'B', 'C'])

    def test_chord_nodes_labels_mapping(self):
        g = self.chord.opts(labels='Label')
        plot = bokeh_renderer.get_plot(g)
        source = plot.handles['text_1_source']
        self.assertEqual(source.data['text'], ['A', 'B', 'C'])

    def test_chord_nodes_categorically_colormapped(self):
        g = self.chord.opts(
            color_index='Label', label_index='Label', cmap=['#FFFFFF', '#888888', '#000000']
        )
        plot = bokeh_renderer.get_plot(g)
        cmapper = plot.handles['color_mapper']
        source = plot.handles['scatter_1_source']
        arc_source = plot.handles['multi_line_2_source']
        glyph = plot.handles['scatter_1_glyph']
        self.assertIsInstance(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, ['A', 'B', 'C'])
        self.assertEqual(cmapper.palette, ['#FFFFFF', '#888888', '#000000'])
        self.assertEqual(source.data['Label'], np.array(['A', 'B', 'C']))
        self.assertEqual(arc_source.data['Label'], np.array(['A', 'B', 'C']))
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'Label', 'transform': cmapper})

    def test_chord_nodes_style_map_node_color_colormapped(self):
        g = self.chord.opts(
            labels='Label', node_color='Label', cmap=['#FFFFFF', '#888888', '#000000']
        )
        plot = bokeh_renderer.get_plot(g)
        cmapper = plot.handles['node_color_color_mapper']
        source = plot.handles['scatter_1_source']
        arc_source = plot.handles['multi_line_2_source']
        glyph = plot.handles['scatter_1_glyph']
        arc_glyph = plot.handles['multi_line_2_glyph']
        self.assertIsInstance(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, ['A', 'B', 'C'])
        self.assertEqual(cmapper.palette, ['#FFFFFF', '#888888', '#000000'])
        self.assertEqual(source.data['Label'], np.array(['A', 'B', 'C']))
        self.assertEqual(arc_source.data['Label'], np.array(['A', 'B', 'C']))
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'node_color', 'transform': cmapper})
        self.assertEqual(property_to_dict(arc_glyph.line_color), {'field': 'node_color', 'transform': cmapper})

    def test_chord_edges_categorically_colormapped(self):
        g = self.chord.opts(
            edge_color_index='start', edge_cmap=['#FFFFFF', '#000000']
        )
        plot = bokeh_renderer.get_plot(g)
        cmapper = plot.handles['edge_colormapper']
        edge_source = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        self.assertIsInstance(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.palette, ['#FFFFFF', '#000000', '#FFFFFF'])
        self.assertEqual(cmapper.factors, ['0', '1', '2'])
        self.assertEqual(edge_source.data['start_str__'], ['0', '0', '1'])
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'start_str__', 'transform': cmapper})

    def test_chord_edge_color_style_mapping(self):
        g = self.chord.opts(
            edge_color=dim('start').astype(str), edge_cmap=['#FFFFFF', '#000000']
        )
        plot = bokeh_renderer.get_plot(g)
        cmapper = plot.handles['edge_color_color_mapper']
        edge_source = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        self.assertIsInstance(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.palette, ['#FFFFFF', '#000000', '#FFFFFF'])
        self.assertEqual(cmapper.factors, ['0', '1', '2'])
        self.assertEqual(edge_source.data['edge_color'], np.array(['0', '0', '1']))
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'edge_color', 'transform': cmapper})
