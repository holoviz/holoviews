import param

from bokeh.models import DataRange1d, Circle, MultiLine

try:
    from bokeh.models import (StaticLayoutProvider, GraphRenderer, NodesAndLinkedEdges,
                              EdgesAndLinkedNodes)
except:
    pass

from ...core.options import abbreviated_exception
from .element import CompositeElementPlot, line_properties, fill_properties, property_prefixes
from .util import mpl_to_bokeh


class GraphPlot(CompositeElementPlot):

    selection_policy = param.ObjectSelector(default='nodes', objects=['edges', 'nodes', None], doc="""
        Determines policy for inspection of graph components, i.e. whether to highlight
        nodes or edges when selecting connected edges and nodes respectively.""")

    inspection_policy = param.ObjectSelector(default='nodes', objects=['edges', 'nodes', None], doc="""
        Determines policy for inspection of graph components, i.e. whether to highlight
        nodes or edges when hovering over connected edges and nodes respectively.""")

    # X-axis is categorical
    _x_range_type = DataRange1d

    # Declare that y-range should auto-range if not bounded
    _y_range_type = DataRange1d

        # Map each glyph to a style group
    _style_groups = {'scatter': 'node', 'multi_line': 'edge'}

    # Define all the glyph handles to update
    _update_handles = ([glyph+'_'+model for model in ['glyph', 'glyph_renderer', 'source']
                        for glyph in ['scatter_1', 'multi_line_1']] +
                       ['color_mapper', 'colorbar'])

    style_opts = (['edge_'+p for p in line_properties] +\
                  ['node_'+p for p in fill_properties+line_properties]+['node_size'])

    def get_extents(self, element, ranges):
        """
        Extents are set to '' and None because x-axis is categorical and
        y-axis auto-ranges.
        """
        x0, x1 = element.nodes.range(0)
        y0, y1 = element.nodes.range(1)
        return (x0, y0, x1, y1)

    def _get_axis_labels(self, *args, **kwargs):
        """
        Override axis labels to group all key dimensions together.
        """
        element = self.current_frame
        xlabel, ylabel = [kd.pprint_label for kd in element.nodes.kdims[:2]]
        return xlabel, ylabel, None

    def get_data(self, element, ranges=None, empty=False):
        point_data = {'index': element.nodes.dimension_values(2).astype(int)}
        point_mapping = {'index': 'index'}

        xidx, yidx = (1, 0) if self.invert_axes else (0, 1)
        xs, ys = (element.dimension_values(i) for i in range(2))
        path_data = dict(start=xs, end=ys)
        path_mapping = dict(start='start', end='end')

        data = {'scatter_1': point_data, 'multi_line_1': path_data}
        mapping = {'scatter_1': point_mapping, 'multi_line_1': path_mapping}
        return data, mapping

    def _init_glyphs(self, plot, element, ranges, source):
        # Get data and initialize data source
        data, mapping = self.get_data(element, ranges, False)
        self.handles['previous_id'] = element._plot_id
        for key in dict(mapping, **data):
            source = self._init_datasource(data.get(key, {}))
            self.handles[key+'_source'] = source
            properties = self._glyph_properties(plot, element, source, ranges)
            properties = self._process_properties(key, properties)
            properties = {p: v for p, v in properties.items() if p not in ('legend', 'source')}
            for prefix in [''] + property_prefixes:
                glyph_key = prefix+'_'+key if prefix else key
                other_prefixes = [p for p in property_prefixes if p != prefix]
                gprops = {p[len(prefix)+1:] if prefix and prefix in p else p: v for p, v in properties.items()
                          if not any(pre in p for pre in other_prefixes)}
                with abbreviated_exception():
                    renderer, glyph = self._init_glyph(plot, mapping.get(key, {}), gprops, key)
                self.handles[glyph_key+'_glyph'] = glyph

        # Define static layout
        layout_dict = {int(z): (x, y) for x, y, z in element.nodes.array([0, 1, 2])}
        layout = StaticLayoutProvider(graph_layout=layout_dict)

        # Initialize GraphRenderer
        graph = GraphRenderer(layout_provider=layout)
        plot.renderers.append(graph)
        for prefix in [''] + property_prefixes:
            node_key = 'scatter_1_glyph'
            edge_key = 'multi_line_1_glyph'
            glyph_key = 'glyph'
            if prefix:
                glyph_key = '_'.join([prefix, glyph_key])
                node_key = '_'.join([prefix, node_key])
                edge_key = '_'.join([prefix, edge_key])
            if node_key not in self.handles:
                continue
            setattr(graph.node_renderer, glyph_key, self.handles[node_key])
            setattr(graph.edge_renderer, glyph_key, self.handles[edge_key])
        graph.node_renderer.data_source.data = self.handles['scatter_1_source'].data
        graph.edge_renderer.data_source.data = self.handles['multi_line_1_source'].data
        if self.selection_policy == 'nodes':
            graph.selection_policy = NodesAndLinkedEdges()
        elif self.selection_policy == 'edges':
            graph.selection_policy = EdgesAndLinkedNodes()
        else:
            graph.selection_policy = None

        if self.inspection_policy == 'nodes':
            graph.inspection_policy = NodesAndLinkedEdges()
        elif self.inspection_policy == 'edges':
            graph.inspection_policy = EdgesAndLinkedNodes()
        else:
            graph.inspection_policy = None
        self.handles['renderer'] = graph
        self.handles['scatter_1_renderer'] = graph.node_renderer
        self.handles['multi_line_1_renderer'] = graph.edge_renderer

    def _init_glyph(self, plot, mapping, properties, key):
        """
        Returns a Bokeh glyph object.
        """
        properties = mpl_to_bokeh(properties)
        plot_method = '_'.join(key.split('_')[:-1])
        glyph = MultiLine if plot_method == 'multi_line' else Circle
        glyph = glyph(**properties)
        return None, glyph
