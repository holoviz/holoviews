import param
import numpy as np

from bokeh.models import Patches

from ...core.data import Dataset
from ...core.util import basestring, max_range, dimension_sanitizer
from .graphs import GraphPlot



class SankeyPlot(GraphPlot):

    color_index = param.ClassSelector(default=2, class_=(basestring, int),
                                      allow_None=True, doc="""
        Index of the dimension from which the node colors will be drawn""")

    label_index = param.ClassSelector(default=2, class_=(basestring, int),
                                      allow_None=True, doc="""
        Index of the dimension from which the node labels will be drawn""")

    label_position = param.ObjectSelector(default='right', objects=['left', 'right'],
                                          doc="""
        Whether node labels should be placed to the left or right.""")

    show_values = param.Boolean(default=True, doc="""
        Whether to show the values.""")

    node_width = param.Number(default=15, doc="""
        Width of the nodes.""")

    node_padding = param.Integer(default=10, doc="""
        Number of pixels of padding relative to the bounds.""")

    iterations = param.Integer(default=32, doc="""
        Number of iterations to run the layout algorithm.""")

    _style_groups = dict(GraphPlot._style_groups, quad='nodes', text='label')

    _draw_order = ['patches', 'quad', 'text']

    style_opts = GraphPlot.style_opts + ['edge_fill_alpha', 'nodes_line_color',
                                         'label_text_font_size']

    filled = True

    def _init_glyphs(self, plot, element, ranges, source):
        ret = super(SankeyPlot, self)._init_glyphs(plot, element, ranges, source)
        renderer = plot.renderers.pop(plot.renderers.index(self.handles['glyph_renderer']))
        plot.renderers = [renderer] + plot.renderers
        return ret

    def get_data(self, element, ranges, style):
        data, mapping, style = super(SankeyPlot, self).get_data(element, ranges, style)
        self._compute_quads(element, data, mapping)
        style['nodes_line_color'] = 'black'

        lidx = element.nodes.get_dimension(self.label_index)
        if lidx is None:
            if self.label_index is not None:
                dims = element.nodes.dimensions()[2:]
                self.warning("label_index supplied to Sankey not found, "
                             "expected one of %s, got %s." %
                             (dims, self.label_index))
            return data, mapping, style

        self._compute_labels(element, data, mapping)
        self._patch_hover(element, data)
        return data, mapping, style

    def _compute_quads(self, element, data, mapping):
        """
        Computes the node quad glyph data.x
        """
        quad_mapping = {'left': 'x0', 'right': 'x1', 'bottom': 'y0', 'top': 'y1'}
        quad_data = dict(data['scatter_1'])
        quad_data.update({'x0': [], 'x1': [], 'y0': [], 'y1': []})
        for node in element._sankey['nodes']:
            quad_data['x0'].append(node['x0'])
            quad_data['y0'].append(node['y0'])
            quad_data['x1'].append(node['x1'])
            quad_data['y1'].append(node['y1'])
        data['quad_1'] = quad_data
        if 'node_fill_color' in mapping['scatter_1']:
            quad_mapping['fill_color'] = mapping['scatter_1']['node_fill_color']
        mapping['quad_1'] = quad_mapping

    def _compute_labels(self, element, data, mapping):
        """
        Computes labels for the nodes and adds it to the data.
        """
        lidx = element.nodes.get_dimension(self.label_index)
        if element.vdims:
            edges = Dataset(element)[element[element.vdims[0].name]>0]
            nodes = list(np.unique([edges.dimension_values(i) for i in range(2)]))
            nodes = element.nodes.select(**{element.nodes.kdims[2].name: nodes})
        else:
            nodes = element

        value_dim = element.vdims[0]
        labels = [lidx.pprint_value(v) for v in nodes.dimension_values(lidx)]
        if self.show_values:
            value_labels = []
            for i, node in enumerate(element._sankey['nodes']):
                value = value_dim.pprint_value(node['value'])
                label = '%s - %s' % (labels[i], value)
                if value_dim.unit:
                    label += ' %s' % value_dim.unit
                value_labels.append(label)
            labels = value_labels

        ys = nodes.dimension_values(1)
        nodes = element._sankey['nodes']
        offset = (nodes[0]['x1']-nodes[0]['x0'])/4.
        if self.label_position == 'right':
            xs = np.array([node['x1'] for node in nodes])+offset
        else:
            xs = np.array([node['x0'] for node in nodes])-offset
        data['text_1'] = dict(x=xs, y=ys, text=[str(l) for l in labels])
        align = 'left' if self.label_position == 'right' else 'right'
        mapping['text_1'] = dict(text='text', x='x', y='y', text_baseline='middle', text_align=align)

    def _patch_hover(self, element, data):
        """
        Replace edge start and end hover data with label_index data.
        """
        if not (self.inspection_policy == 'edges' and 'hover' in self.handles):
            return
        lidx = element.nodes.get_dimension(self.label_index)
        src, tgt = [dimension_sanitizer(kd.name) for kd in element.kdims[:2]]
        if src == 'start': src += '_values'
        if tgt == 'end':   tgt += '_values'
        lookup = dict(zip(*(element.nodes.dimension_values(d) for d in (2, lidx))))
        src_vals = data['patches_1'][src]
        tgt_vals = data['patches_1'][tgt]
        data['patches_1'][src] = [lookup.get(v, v) for v in src_vals]
        data['patches_1'][tgt] = [lookup.get(v, v) for v in tgt_vals]

    def get_extents(self, element, ranges, range_type='combined'):
        if range_type == 'extents':
            return element.nodes.extents
        xdim, ydim = element.nodes.kdims[:2]
        xpad = .05 if self.label_index is None else 0.25
        x0, x1 = ranges[xdim.name][range_type]
        y0, y1 = ranges[ydim.name][range_type]
        xdiff = (x1-x0)
        ydiff = (y1-y0)
        if self.label_position == 'right':
            x0, x1 = x0-(0.05*xdiff), x1+xpad*xdiff
        else:
            x0, x1 = x0-xpad*xdiff, x1+(0.05*xdiff)
        x0, x1 = max_range([xdim.range, (x0, x1)])
        y0, y1 = max_range([ydim.range, (y0-(0.05*ydiff), y1+(0.05*ydiff))])
        return (x0, y0, x1, y1)

    def _postprocess_hover(self, renderer, source):
        if self.inspection_policy == 'edges':
            if not isinstance(renderer.glyph, Patches):
                return
        else:
            if isinstance(renderer.glyph, Patches):
                return
        super(SankeyPlot, self)._postprocess_hover(renderer, source)
