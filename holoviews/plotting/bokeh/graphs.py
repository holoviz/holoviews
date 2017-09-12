import param

from bokeh.models import Range1d, Circle, MultiLine

try:
    from bokeh.models import (StaticLayoutProvider, GraphRenderer, NodesAndLinkedEdges,
                              EdgesAndLinkedNodes)
except:
    pass

from ...core.options import abbreviated_exception, SkipRendering
from ...core.util import basestring
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

    # X-axis is categorical
    _x_range_type = Range1d

    # Declare that y-range should auto-range if not bounded
    _y_range_type = Range1d

        # Map each glyph to a style group
    _style_groups = {'scatter': 'node', 'multi_line': 'edge'}

    # Define all the glyph handles to update
    _update_handles = ([glyph+'_'+model for model in ['glyph', 'glyph_renderer', 'source']
                        for glyph in ['scatter_1', 'multi_line_1']] +
                       ['color_mapper', 'colorbar'])

    style_opts = (['edge_'+p for p in line_properties] +\
                  ['node_'+p for p in fill_properties+line_properties]+['node_size', 'cmap'])

    def initialize_plot(self, ranges=None, plot=None, plots=None):
        if bokeh_version < '0.12.7':
            raise SkipRendering('Graph rendering requires bokeh version >=0.12.7.')
        super(GraphPlot, self).initialize_plot(ranges, plot, plots)

    def _hover_opts(self, element):
        dims = element.nodes.dimensions()[3:]
        return dims, {}

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
        style = self.style[self.cyclic_index]
        point_data = {'index': element.nodes.dimension_values(2).astype(int)}
        for d in element.nodes.dimensions()[2:]:
            point_data[d.name] = element.nodes.dimension_values(d)

        cdata, cmapping = self._get_color_data(element.nodes, ranges, style, 'node_fill_color')
        point_data.update(cdata)
        point_mapping = cmapping

        xidx, yidx = (1, 0) if self.invert_axes else (0, 1)
        xs, ys = (element.dimension_values(i) for i in range(2))
        path_data = dict(start=xs, end=ys)
        if element._nodepaths:
            edges = element.nodepaths
            path_data['xs'] = [path[:, xidx] for path in edges.data]
            path_data['ys'] = [path[:, yidx] for path in edges.data]

        data = {'scatter_1': point_data, 'multi_line_1': path_data}
        mapping = {'scatter_1': point_mapping, 'multi_line_1': {}}
        return data, mapping

    def _init_glyphs(self, plot, element, ranges, source):
        # Get data and initialize data source
        data, mapping = self.get_data(element, ranges, False)
        self.handles['previous_id'] = element._plot_id
        properties = {}
        mappings = {}
        for key in dict(mapping, **data):
            source = self._init_datasource(data.get(key, {}))
            self.handles[key+'_source'] = source
            glyph_props = self._glyph_properties(plot, element, source, ranges)
            properties.update(glyph_props)
            mappings.update(mapping.get(key, {}))
        properties = {p: v for p, v in properties.items() if p not in ('legend', 'source')}
        properties.update(mappings)

        # Define static layout
        layout_dict = {int(z): (x, y) for x, y, z in element.nodes.array([0, 1, 2])}
        layout = StaticLayoutProvider(graph_layout=layout_dict)
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

        self.handles['renderer'] = renderer
        self.handles['scatter_1_renderer'] = renderer.node_renderer
        self.handles['multi_line_1_renderer'] = renderer.edge_renderer
