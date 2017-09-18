from __future__ import absolute_import

from unittest import SkipTest

import numpy as np
from holoviews.core.options import Store
from holoviews.element import Graph, circular_layout
from holoviews.element.comparison import ComparisonTestCase

try:
    from holoviews.plotting.bokeh.util import bokeh_version
    bokeh_renderer = Store.renderers['bokeh']
    from bokeh.models import (NodesAndLinkedEdges, EdgesAndLinkedNodes)
except :
    bokeh_renderer = None


class BokehGraphPlotTests(ComparisonTestCase):

    def setUp(self):
        if bokeh_version < str('0.12.9'):
            raise SkipTest("Bokeh >= 0.12.9 required to test graphs")
        N = 8
        self.nodes = circular_layout(np.arange(N))
        self.source = np.arange(N)
        self.target = np.zeros(N)
        self.graph = Graph(((self.source, self.target),))

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
        self.assertEqual(hover.tooltips, [('index', '@{index}')])
        self.assertIn(renderer, hover.renderers)

    def test_graph_inspection_policy_edges(self):
        plot = bokeh_renderer.get_plot(self.graph.opts(plot=dict(inspection_policy='edges')))
        renderer = plot.handles['glyph_renderer']
        hover = plot.handles['hover']
        self.assertIsInstance(renderer.inspection_policy, EdgesAndLinkedNodes)
        self.assertEqual(hover.tooltips, [('start', '@{start}'), ('end', '@{end}')])
        self.assertIn(renderer, hover.renderers)

    def test_graph_inspection_policy_none(self):
        plot = bokeh_renderer.get_plot(self.graph.opts(plot=dict(inspection_policy=None)))
        renderer = plot.handles['glyph_renderer']
        hover = plot.handles['hover']
        self.assertIs(renderer.inspection_policy, None)

    def test_graph_selection_policy_nodes(self):
        plot = bokeh_renderer.get_plot(self.graph)
        renderer = plot.handles['glyph_renderer']
        hover = plot.handles['hover']
        self.assertIsInstance(renderer.selection_policy, NodesAndLinkedEdges)
        self.assertIn(renderer, hover.renderers)

    def test_graph_selection_policy_edges(self):
        plot = bokeh_renderer.get_plot(self.graph.opts(plot=dict(selection_policy='edges')))
        renderer = plot.handles['glyph_renderer']
        hover = plot.handles['hover']
        self.assertIsInstance(renderer.selection_policy, EdgesAndLinkedNodes)
        self.assertIn(renderer, hover.renderers)

    def test_graph_selection_policy_none(self):
        plot = bokeh_renderer.get_plot(self.graph.opts(plot=dict(selection_policy=None)))
        renderer = plot.handles['glyph_renderer']
        hover = plot.handles['hover']
        self.assertIs(renderer.selection_policy, None)
