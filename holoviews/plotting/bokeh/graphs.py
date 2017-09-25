import param
import numpy as np

from bokeh.models import Range1d, Circle, MultiLine, HoverTool, ColumnDataSource

try:
    from bokeh.models import (StaticLayoutProvider, GraphRenderer, NodesAndLinkedEdges,
                              EdgesAndLinkedNodes)
except:
    pass

from ...core.options import abbreviated_exception, SkipRendering
from ...core.util import basestring, dimension_sanitizer
from .chart import ColorbarPlot
from .element import CompositeElementPlot, line_properties, fill_properties, property_prefixes
from .util import mpl_to_bokeh, bokeh_version


class GraphPlot(CompositeElementPlot, ColorbarPlot):

    color_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    selection_policy = param.ObjectSelector(default='nodes', objects=['edges', 'nodes', None], doc="""
        Determines policy for inspection of graph components, i.e. whether to highlight
        nodes or edges when selecting connected edges and nodes respectively.""")

    inspection_policy = param.ObjectSelector(default='nodes', objects=['edges', 'nodes', None], doc="""
        Determines policy for inspection of graph components, i.e. whether to highlight
        nodes or edges when hovering over connected edges and nodes respectively.""")

    tools = param.List(default=['hover', 'tap'], doc="""
        A list of plugin tools to use on the plot.""")

    # X-axis is categorical
    _x_range_type = Range1d

    # Declare that y-range should auto-range if not bounded
    _y_range_type = Range1d

        # Map each glyph to a style group
    _style_groups = {'scatter': 'node', 'multi_line': 'edge'}

    # Define all the glyph handles to update
    _update_handles = ([glyph+'_'+model for model in ['glyph', 'glyph_renderer', 'source']
                        for glyph in ['scatter_1', 'multi_line_1']] +
                       ['color_mapper', 'colorbar', 'layout_source'])

    style_opts = (['edge_'+p for p in line_properties] +\
                  ['node_'+p for p in fill_properties+line_properties]+['node_size', 'cmap'])

    def initialize_plot(self, ranges=None, plot=None, plots=None):
        if bokeh_version < '0.12.7':
            raise SkipRendering('Graph rendering requires bokeh version >=0.12.7.')
        return super(GraphPlot, self).initialize_plot(ranges, plot, plots)

    def _hover_opts(self, element):
        if self.inspection_policy == 'nodes':
            dims = element.nodes.dimensions()[2:]
        elif self.inspection_policy == 'edges':
            kdims = [(kd.pprint_label, '@{%s}' % ref)
                     for kd, ref in zip(element.kdims, ['start', 'end'])]
            dims = kdims+element.vdims
        else:
            dims = []
        return dims, {}

    def get_extents(self, element, ranges):
        """
        Extents are set to '' and None because x-axis is categorical and
        y-axis auto-ranges.
        """
        xdim, ydim = element.nodes.kdims[:2]
        x0, x1 = ranges[xdim.name]
        y0, y1 = ranges[ydim.name]
        return (x0, y0, x1, y1)

    def _get_axis_labels(self, *args, **kwargs):
        """
        Override axis labels to group all key dimensions together.
        """
        element = self.current_frame
        xlabel, ylabel = [kd.pprint_label for kd in element.nodes.kdims[:2]]
        return xlabel, ylabel, None

    def get_data(self, element, ranges=None, empty=False):
        style = self.style[self.cyclic_index]
        xidx, yidx = (1, 0) if self.invert_axes else (0, 1)

        # Get node data
        nodes = element.nodes.dimension_values(2)
        node_positions = element.nodes.array([0, 1, 2])
        # Map node indices to integers
        if nodes.dtype.kind not in 'if':
            node_indices = {v: i for i, v in enumerate(nodes)}
            index = np.array([node_indices[n] for n in nodes], dtype=np.int32)
            layout = {node_indices[z]: (y, x) if self.invert_axes else (x, y)
                      for x, y, z in node_positions}
        else:
            index = nodes.astype(np.int32)
            layout = {int(z): (y, x) if self.invert_axes else (x, y)
                      for x, y, z in node_positions}
        point_data = {'index': index}
        cdata, cmapping = self._get_color_data(element.nodes, ranges, style, 'node_fill_color')
        point_data.update(cdata)
        point_mapping = cmapping
        if 'node_fill_color' in point_mapping:
            point_mapping['node_nonselection_fill_color'] = point_mapping['node_fill_color']

        # Get edge data
        nan_node = index.max()+1
        start, end = (element.dimension_values(i) for i in range(2))
        if nodes.dtype.kind not in 'if':
            start = np.array([node_indices.get(x, nan_node) for x in start], dtype=np.int32)
            end = np.array([node_indices.get(y, nan_node) for y in end], dtype=np.int32)
        path_data = dict(start=start, end=end)
        if element._edgepaths:
            edges = element.edgepaths.split()
            if len(edges) == len(start):
                path_data['xs'] = [path.dimension_values(xidx) for path in edges]
                path_data['ys'] = [path.dimension_values(yidx) for path in edges]
            else:
                self.warning('Graph edge paths do not match the number of abstract edges '
                             'and will be skipped')

        # Get hover data
        if any(isinstance(t, HoverTool) for t in self.state.tools):
            if self.inspection_policy == 'nodes':
                for d in element.nodes.dimensions()[2:]:
                    point_data[dimension_sanitizer(d.name)] = element.nodes.dimension_values(d)
            elif self.inspection_policy == 'edges':
                for d in element.vdims:
                    path_data[dimension_sanitizer(d.name)] = element.dimension_values(d)

        data = {'scatter_1': point_data, 'multi_line_1': path_data, 'layout': layout}
        mapping = {'scatter_1': point_mapping, 'multi_line_1': {}}
        return data, mapping


    def _update_datasource(self, source, data):
        """
        Update datasource with data for a new frame.
        """
        if isinstance(source, ColumnDataSource):
            source.data.update(data)
        else:
            source.graph_layout.update(data)


    def _init_glyphs(self, plot, element, ranges, source):
        # Get data and initialize data source
        data, mapping = self.get_data(element, ranges, False)
        self.handles['previous_id'] = element._plot_id
        properties = {}
        mappings = {}
        for key in mapping:
            source = self._init_datasource(data.get(key, {}))
            self.handles[key+'_source'] = source
            glyph_props = self._glyph_properties(plot, element, source, ranges)
            properties.update(glyph_props)
            mappings.update(mapping.get(key, {}))
        properties = {p: v for p, v in properties.items() if p not in ('legend', 'source')}
        properties.update(mappings)

        # Define static layout
        layout = StaticLayoutProvider(graph_layout=data['layout'])
        node_source = self.handles['scatter_1_source']
        edge_source = self.handles['multi_line_1_source']
        renderer = plot.graph(node_source, edge_source, layout, **properties)

        # Initialize GraphRenderer
        if self.selection_policy == 'nodes':
            renderer.selection_policy = NodesAndLinkedEdges()
        elif self.selection_policy == 'edges':
            renderer.selection_policy = EdgesAndLinkedNodes()
        else:
            renderer.selection_policy = None

        if self.inspection_policy == 'nodes':
            renderer.inspection_policy = NodesAndLinkedEdges()
        elif self.inspection_policy == 'edges':
            renderer.inspection_policy = EdgesAndLinkedNodes()
        else:
            renderer.inspection_policy = None

        self.handles['layout_source'] = layout
        self.handles['glyph_renderer'] = renderer
        self.handles['scatter_1_glyph_renderer'] = renderer.node_renderer
        self.handles['multi_line_1_glyph_renderer'] = renderer.edge_renderer
        self.handles['scatter_1_glyph'] = renderer.node_renderer.glyph
        self.handles['multi_line_1_glyph'] = renderer.edge_renderer.glyph
        if 'hover' in self.handles:
            self.handles['hover'].renderers.append(renderer)

