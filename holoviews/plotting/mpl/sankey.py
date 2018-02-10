import param
import numpy as np

from ...core.util import basestring, max_range
from .graphs import GraphPlot
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection


class SankeyPlot(GraphPlot):
    
    color_index = param.ClassSelector(default=2, class_=(basestring, int),
                                      allow_None=True, doc="""
        Index of the dimension from which the node labels will be drawn""")
    
    label_index = param.ClassSelector(default=2, class_=(basestring, int),
                                      allow_None=True, doc="""
        Index of the dimension from which the node labels will be drawn""")

    show_values = param.Boolean(default=True, doc="""
        Whether to show the values.""")

    label_position = param.ObjectSelector(default='right', objects=['left', 'right'],
                                          doc="""
        Whether node labels should be placed to the left or right.""")

    filled = True
    
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
    
    def get_data(self, element, ranges, style):
        data, style, axis_kwargs = super(SankeyPlot, self).get_data(element, ranges, style)
        rects, labels, value_labels = [], [], []
        label_dim = element.nodes.get_dimension(self.label_index)
        value_dim = element.vdims[0]
        values = [] if label_dim is None else element.nodes.dimension_values(label_dim)
        for i, node in enumerate(element._sankey['nodes']):
            x0, x1, y0, y1 = (node[a+i] for a in 'xy' for i in '01')
            rect = {'height': y1-y0, 'width': x1-x0, 'xy': (x0, y0)}
            rects.append(rect)
            if len(values):
                label = label_dim.pprint_value(values[i])
                if self.show_values:
                    value = value_dim.pprint_value(node['value'])
                    label = '%s - %s' % (label, value)
                x = x1+(x1-x0)/3. if self.label_position == 'right' else x0-(x1-x0)/3.
                labels.append((label, (x, (y0+y1)/2.)))
        data['rects'] = rects
        if labels:
            data['text'] = labels
        return data, style, axis_kwargs
    
    def init_artists(self, ax, plot_args, plot_kwargs):
        artists = super(SankeyPlot, self).init_artists(ax, plot_args, plot_kwargs)
        groups = [g for g in self._style_groups if g != 'node']
        node_opts = {k[5:] if 'node_' in k else k: v
                     for k, v in plot_kwargs.items()
                     if not (any(k.startswith(p) for p in groups) or 'size' in k)}
        rects = [Rectangle(**rect) for rect in plot_args['rects']]
        artists['rects'] = ax.add_collection(PatchCollection(rects, **node_opts))
        if 'text' in plot_args:
            fontsize = plot_kwargs.get('text_font_size', 8)
            align = 'left' if self.label_position == 'right' else 'right'
            labels = []
            for text in plot_args['text']:
                label = ax.annotate(*text, xycoords='data',
                                    horizontalalignment=align, fontsize=fontsize,
                                    verticalalignment='center', rotation_mode='anchor')
            labels.append(label)
            artists['labels'] = labels
        return artists
