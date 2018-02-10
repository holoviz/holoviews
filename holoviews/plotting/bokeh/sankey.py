import param
import numpy as np

from ...core.data import Dataset
from ...core.util import basestring, max_range
from .graphs import GraphPlot
from bokeh.models import Patches


class SankeyPlot(GraphPlot):

    color_index = param.ClassSelector(default=2, class_=(basestring, int),
                                      allow_None=True, doc="""
        Index of the dimension from which the node labels will be drawn""")

    label_index = param.ClassSelector(default=2, class_=(basestring, int),
                                      allow_None=True, doc="""
        Index of the dimension from which the node labels will be drawn""")

    label_position = param.ObjectSelector(default='right', objects=['left', 'right'],
                                          doc="""
        Whether node labels should be placed to the left or right.""")

    show_values = param.Boolean(default=True, doc="""
        Whether to show the values.""")
    
    _style_groups = dict(GraphPlot._style_groups, quad='nodes', text='label')

    _draw_order = ['patches', 'multi_line', 'text', 'quad']

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
        quad_mapping = {'left': 'x0', 'right': 'x1', 'bottom': 'y0', 'top': 'y1'}
        quad_data = data['scatter_1']
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
        style['nodes_line_color'] = 'black'

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

        value_dim = element.vdims[0]
        labels = [lidx.pprint_value(v) for v in nodes.dimension_values(lidx)]
        if self.show_values:
            value_labels = []
            for i, node in enumerate(element._sankey['nodes']):
                value = value_dim.pprint_value(node['value'])
                label = '%s - %s' % (labels[i], value)
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
        return data, mapping, style

    def get_extents(self, element, ranges):
        """
        A Chord plot is always drawn on a unit circle.
        """
        xdim, ydim = element.nodes.kdims[:2]
        xpad = .05 if self.label_index is None else 0.25
        x0, x1 = ranges[xdim.name]
        y0, y1 = ranges[ydim.name]
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
