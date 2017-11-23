from collections import defaultdict

import param
import numpy as np
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.models import (StaticLayoutProvider, NodesAndLinkedEdges,
                          EdgesAndLinkedNodes, Patches, Bezier)

from ...core.data import Dataset
from ...core.util import basestring, dimension_sanitizer, unique_array, max_range
from ...core.options import Cycle
from .chart import ColorbarPlot, PointPlot
from .element import (CompositeElementPlot, LegendPlot, line_properties,
                      fill_properties, text_properties)
from ..util import process_cmap


class GraphPlot(CompositeElementPlot, ColorbarPlot, LegendPlot):

    color_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    edge_color_index = param.ClassSelector(default=None, class_=(basestring, int),
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

    # Map each glyph to a style group
    _style_groups = {'scatter': 'node', 'multi_line': 'edge', 'patches': 'edge', 'bezier': 'edge'}

    style_opts = (['edge_'+p for p in line_properties] +
                  ['node_'+p for p in fill_properties+line_properties] +
                  ['node_size', 'cmap', 'edge_cmap'])

    # Filled is only supported for subclasses
    filled = False

    # Bezier paths
    bezier = False

    # Declares which columns in the data refer to node indices
    _node_columns = [0, 1]

    @property
    def edge_glyph(self):
        if self.filled:
            return 'patches_1'
        elif self.bezier:
            return 'bezier_1'
        else:
            return 'multi_line_1'

    def _hover_opts(self, element):
        if self.inspection_policy == 'nodes':
            dims = element.nodes.dimensions()
            dims = [(dims[2].pprint_label, '@{index_hover}')]+dims[3:]
        elif self.inspection_policy == 'edges':
            dims = element.kdims+element.vdims
        else:
            dims = []
        return dims, {}

    def get_extents(self, element, ranges):
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


    def _get_edge_colors(self, element, ranges, edge_data, edge_mapping, style):
        cdim = element.get_dimension(self.edge_color_index)
        if not cdim:
            return
        elstyle = self.lookup_options(element, 'style')
        cycle = elstyle.kwargs.get('edge_color')

        idx = element.get_dimension_index(cdim)
        field = dimension_sanitizer(cdim.name)
        cvals = element.dimension_values(cdim)
        if idx in self._node_columns:
            factors = element.nodes.dimension_values(2, expanded=False)
        elif idx == 2 and cvals.dtype.kind in 'if':
            factors = None
        else:
            factors = unique_array(cvals)

        default_cmap = 'viridis' if factors is None else 'tab20'
        cmap = style.get('edge_cmap', style.get('cmap', default_cmap))
        if factors is None or (factors.dtype.kind in 'if' and idx not in self._node_columns):
            colors, factors = None, None
        else:
            if factors.dtype.kind == 'f':
                cvals = cvals.astype(np.int32)
                factors = factors.astype(np.int32)
            if factors.dtype.kind not in 'SU':
                field += '_str'
                cvals = [str(f) for f in cvals]
                factors = (str(f) for f in factors)
            factors = list(factors)
            colors = process_cmap(cycle or cmap, len(factors))

        if field not in edge_data:
            edge_data[field] = cvals
        edge_style = dict(style, cmap=cmap)
        mapper = self._get_colormapper(cdim, element, ranges, edge_style,
                                       factors, colors, 'edge_colormapper')
        transform = {'field': field, 'transform': mapper}
        color_type = 'fill_color' if self.filled else 'line_color'
        edge_mapping['edge_'+color_type] = transform
        edge_mapping['edge_nonselection_'+color_type] = transform
        edge_mapping['edge_selection_'+color_type] = transform


    def _get_edge_paths(self, element):
        path_data, mapping = {}, {}
        xidx, yidx = (1, 0) if self.invert_axes else (0, 1)
        if element._edgepaths is not None:
            edges = element._split_edgepaths.split(datatype='array', dimensions=element.edgepaths.kdims)
            if len(edges) == len(element):
                path_data['xs'] = [path[:, xidx] for path in edges]
                path_data['ys'] = [path[:, yidx] for path in edges]
                mapping = {'xs': 'xs', 'ys': 'ys'}
            else:
                raise ValueError("Edge paths do not match the number of supplied edges."
                                 "Expected %d, found %d paths." % (len(element), len(edges)))
        return path_data, mapping


    def get_data(self, element, ranges, style):
        # Force static source to False
        static = self.static_source
        self.handles['static_source'] = static
        self.static_source = False

        # Get node data
        nodes = element.nodes.dimension_values(2)
        node_positions = element.nodes.array([0, 1])
        # Map node indices to integers
        if nodes.dtype.kind not in 'if':
            node_indices = {v: i for i, v in enumerate(nodes)}
            index = np.array([node_indices[n] for n in nodes], dtype=np.int32)
            layout = {str(node_indices[k]): (y, x) if self.invert_axes else (x, y)
                      for k, (x, y) in zip(nodes, node_positions)}
        else:
            index = nodes.astype(np.int32)
            layout = {str(k): (y, x) if self.invert_axes else (x, y)
                      for k, (x, y) in zip(index, node_positions)}
        point_data = {'index': index}
        cycle = self.lookup_options(element, 'style').kwargs.get('node_color')
        if isinstance(cycle, Cycle):
            style.pop('node_color', None)
            colors = cycle
        else:
            colors = None
        cdata, cmapping = self._get_color_data(
            element.nodes, ranges, style, name='node_fill_color',
            colors=colors, int_categories=True
        )
        point_data.update(cdata)
        point_mapping = cmapping
        if 'node_fill_color' in point_mapping:
            style = {k: v for k, v in style.items() if k not in
                     ['node_fill_color', 'node_nonselection_fill_color']}
            point_mapping['node_nonselection_fill_color'] = point_mapping['node_fill_color']

        edge_mapping = {}
        nan_node = index.max()+1 if len(index) else 0
        start, end = (element.dimension_values(i) for i in range(2))
        if nodes.dtype.kind == 'f':
            start, end = start.astype(np.int32), end.astype(np.int32)
        elif nodes.dtype.kind != 'i':
            start = np.array([node_indices.get(x, nan_node) for x in start], dtype=np.int32)
            end = np.array([node_indices.get(y, nan_node) for y in end], dtype=np.int32)
        path_data = dict(start=start, end=end)
        self._get_edge_colors(element, ranges, path_data, edge_mapping, style)
        if not static:
            pdata, pmapping = self._get_edge_paths(element)
            path_data.update(pdata)
            edge_mapping.update(pmapping)

        # Get hover data
        if any(isinstance(t, HoverTool) for t in self.state.tools):
            if self.inspection_policy == 'nodes':
                index_dim = element.nodes.get_dimension(2)
                point_data['index_hover'] = [index_dim.pprint_value(v) for v in element.nodes.dimension_values(2)]
                for d in element.nodes.dimensions()[3:]:
                    point_data[dimension_sanitizer(d.name)] = element.nodes.dimension_values(d)
            elif self.inspection_policy == 'edges':
                for d in element.dimensions():
                    path_data[dimension_sanitizer(d.name)] = element.dimension_values(d)
        data = {'scatter_1': point_data, self.edge_glyph: path_data, 'layout': layout}
        mapping = {'scatter_1': point_mapping, self.edge_glyph: edge_mapping}
        return data, mapping, style


    def _update_datasource(self, source, data):
        """
        Update datasource with data for a new frame.
        """
        if isinstance(source, ColumnDataSource):
            if self.handles['static_source']:
                source.trigger('data')
            else:
                source.data.update(data)
        else:
            source.graph_layout = data


    def _init_glyphs(self, plot, element, ranges, source):
        # Get data and initialize data source
        style = self.style[self.cyclic_index]
        data, mapping, style = self.get_data(element, ranges, style)
        edge_mapping = {k: v for k, v in mapping[self.edge_glyph].items()
                        if 'color' not in k}
        self.handles['previous_id'] = element._plot_id

        properties = {}
        mappings = {}
        for key in list(mapping):
            if not any(glyph in key for glyph in ('scatter_1', self.edge_glyph)):
                continue
            source = self._init_datasource(data.pop(key, {}))
            self.handles[key+'_source'] = source
            glyph_props = self._glyph_properties(plot, element, source, ranges, style)
            properties.update(glyph_props)
            mappings.update(mapping.pop(key, {}))
        properties = {p: v for p, v in properties.items() if p not in ('legend', 'source')}
        properties.update(mappings)

        layout = data.pop('layout', {})
        if data and mapping:
            CompositeElementPlot._init_glyphs(self, plot, element, ranges, source,
                                              data, mapping, style)

        # Define static layout
        layout = StaticLayoutProvider(graph_layout=layout)
        node_source = self.handles['scatter_1_source']
        edge_source = self.handles[self.edge_glyph+'_source']
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
        self.handles[self.edge_glyph+'_glyph_renderer'] = renderer.edge_renderer
        self.handles['scatter_1_glyph'] = renderer.node_renderer.glyph
        if self.filled or self.bezier:
            glyph_model = Patches if self.filled else Bezier
            allowed_properties = glyph_model.properties()
            for glyph_type in ('', 'selection_', 'nonselection_', 'hover_', 'muted_'):
                glyph = getattr(renderer.edge_renderer, glyph_type+'glyph', None)
                if glyph is None:
                    continue
                props = self._process_properties(self.edge_glyph, properties, mappings)
                filtered = self._filter_properties(props, glyph_type, allowed_properties)
                new_glyph = glyph_model(**dict(filtered, **edge_mapping))
                setattr(renderer.edge_renderer, glyph_type+'glyph', new_glyph)
        self.handles[self.edge_glyph+'_glyph'] = renderer.edge_renderer.glyph
        if 'hover' in self.handles:
            self.handles['hover'].renderers.append(renderer)



class ChordPlot(GraphPlot):

    label_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the node labels will be drawn""")

    show_frame = param.Boolean(default=False, doc="""
        Whether or not to show a complete frame around the plot.""")

    # Map each glyph to a style group
    _style_groups = {'scatter': 'node', 'multi_line': 'edge', 'text': 'label',
                     'arc': 'arc'}

    style_opts = (GraphPlot.style_opts + ['label_'+p for p in text_properties])

    _draw_order = ['scatter', 'multi_line', 'layout']

    def get_extents(self, element, ranges):
        """
        A Chord plot is always drawn on a unit circle.
        """
        xdim, ydim = element.nodes.kdims[:2]
        rng = 1.1 if element.nodes.get_dimension(self.label_index) is None else 1.4
        x0, x1 = max_range([xdim.range, (-rng, rng)])
        y0, y1 = max_range([ydim.range, (-rng, rng)])
        return (x0, y0, x1, y1)

    def get_data(self, element, ranges, style):
        offset = style.pop('label_offset', 1.05)
        data, mapping, style = super(ChordPlot, self).get_data(element, ranges, style)
        if 'node_fill_color' in mapping['scatter_1']:
            angles = element._angles
            arcs = defaultdict(list)
            for i in range(len(element.nodes)):
                start, end = angles[i:i+2]
                vals = np.linspace(start, end, 20)
                xs, ys = np.cos(vals), np.sin(vals)
                arcs['arc_xs'].append(xs)
                arcs['arc_ys'].append(ys)
            data['scatter_1'].update(arcs)
            data['multi_line_2'] = data['scatter_1']
            mapping['multi_line_2'] = {'xs': 'arc_xs', 'ys': 'arc_ys', 'line_width': 10}
            mapping['multi_line_2']['line_color'] = mapping['scatter_1']['node_fill_color']
            mapping['multi_line_2']['nonselection_line_color'] = mapping['scatter_1']['node_fill_color']
            mapping['multi_line_2']['selection_line_color'] = mapping['scatter_1']['node_fill_color']

        lidx = element.nodes.get_dimension(self.label_index)
        if lidx is None:
            if self.label_index is not None:
                dims = element.nodes.dimensions()[2:]
                self.warning("label_index supplied to Chord not found, "
                             "expected one of %s, got %s." %
                             (dims, self.label_index))
            return data, mapping, style
        if element.vdims:
            edges = Dataset(element)[element[element.vdims[0].name]>0]
            nodes = list(np.unique([edges.dimension_values(i) for i in range(2)]))
            nodes = element.nodes.select(**{element.nodes.kdims[2].name: nodes})
        else:
            nodes = element
        xs, ys = (nodes.dimension_values(i)*offset for i in range(2))
        labels = [lidx.pprint_value(v) for v in nodes.dimension_values(lidx)]
        angles = np.arctan2(ys, xs)
        data['text_1'] = dict(x=xs, y=ys, text=[str(l) for l in labels], angle=angles)
        mapping['text_1'] = dict(text='text', x='x', y='y', angle='angle', text_baseline='middle')
        return data, mapping, style



class NodePlot(PointPlot):
    """
    Simple subclass of PointPlot which hides x, y position on hover.
    """

    def _hover_opts(self, element):
        return element.dimensions()[2:], {}



class TriMeshPlot(GraphPlot):

    filled = param.Boolean(default=False, doc="""
        Whether the triangles should be drawn as filled.""")

    style_opts = (['edge_'+p for p in line_properties+fill_properties] +
                  ['node_'+p for p in fill_properties+line_properties] +
                  ['node_size', 'cmap', 'edge_cmap'])

    # Declares that three columns in TriMesh refer to edges
    _node_columns = [0, 1, 2]

    def get_data(self, element, ranges, style):
        # Ensure the edgepaths for the triangles are generated before plotting
        simplex_dim = element.get_dimension(self.edge_color_index)
        vertex_dim = element.nodes.get_dimension(self.edge_color_index)
        if not isinstance(self.edge_color_index, int) and vertex_dim and not simplex_dim:
            simplices = element.array([0, 1, 2])
            z = element.nodes.dimension_values(vertex_dim)
            z = z[simplices].mean(axis=1)
            element = element.add_dimension(vertex_dim, len(element.vdims), z, vdim=True)
        element.edgepaths
        return super(TriMeshPlot, self).get_data(element, ranges, style)

