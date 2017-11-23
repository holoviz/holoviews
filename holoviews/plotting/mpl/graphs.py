import param
import numpy as np

from matplotlib.collections import LineCollection, PolyCollection

from ...core.data import Dataset
from ...core.options import Cycle
from ...core.util import basestring, unique_array, search_indices, max_range
from ..util import process_cmap
from .element import ColorbarPlot


class GraphPlot(ColorbarPlot):

    color_index = param.ClassSelector(default=None, class_=(basestring, int),
                                  allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    edge_color_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    style_opts = ['edge_alpha', 'edge_color', 'edge_linestyle', 'edge_linewidth',
                  'node_alpha', 'node_color', 'node_edgecolors', 'node_facecolors',
                  'node_linewidth', 'node_marker', 'node_size', 'visible', 'cmap',
                  'edge_cmap']

    _style_groups = ['node', 'edge']

    filled = False

    def _compute_styles(self, element, ranges, style):
        elstyle = self.lookup_options(element, 'style')
        color = elstyle.kwargs.get('node_color')
        cdim = element.nodes.get_dimension(self.color_index)
        cmap = elstyle.kwargs.get('cmap', 'tab20')
        if cdim:
            cs = element.nodes.dimension_values(self.color_index)
            # Check if numeric otherwise treat as categorical
            if cs.dtype.kind == 'f':
                style['c'] = cs
            else:
                factors = unique_array(cs)
                cmap = color if isinstance(color, Cycle) else cmap
                colors = process_cmap(cmap, len(factors))
                cs = search_indices(cs, factors)
                style['node_facecolors'] = [colors[v%len(colors)] for v in cs]
                style.pop('node_color', None)
            if 'c' in style:
                self._norm_kwargs(element.nodes, ranges, style, cdim)
        elif color:
            style['c'] = style.pop('node_color')
        style['node_edgecolors'] = style.pop('node_edgecolors', 'none')

        edge_cdim = element.get_dimension(self.edge_color_index)
        if not edge_cdim:
            return style

        elstyle = self.lookup_options(element, 'style')
        cycle = elstyle.kwargs.get('edge_color')
        idx = element.get_dimension_index(edge_cdim)
        cvals = element.dimension_values(edge_cdim)
        if idx in [0, 1]:
            factors = element.nodes.dimension_values(2, expanded=False)
        elif idx == 2 and cvals.dtype.kind in 'if':
            factors = None
        else:
            factors = unique_array(cvals)
        if factors is None or (factors.dtype.kind == 'f' and idx not in [0, 1]):
            style['edge_array'] = cvals
        else:
            cvals = search_indices(cvals, factors)
            factors = list(factors)
            cmap = elstyle.kwargs.get('edge_cmap', 'tab20')
            cmap = cycle if isinstance(cycle, Cycle) else cmap
            colors = process_cmap(cmap, len(factors))
            style['edge_colors'] = [colors[v%len(colors)] for v in cvals]
            style.pop('edge_color', None)
        if 'edge_array' in style:
            self._norm_kwargs(element, ranges, style, edge_cdim, 'edge_')
        else:
            style.pop('edge_cmap', None)
        if 'edge_vmin' in style:
            style['edge_clim'] = (style.pop('edge_vmin'), style.pop('edge_vmax'))
        return style


    def get_data(self, element, ranges, style):
        xidx, yidx = (1, 0) if self.invert_axes else (0, 1)
        pxs, pys = (element.nodes.dimension_values(i) for i in range(2))
        dims = element.nodes.dimensions()
        self._compute_styles(element, ranges, style)

        paths = element._split_edgepaths.split(datatype='array', dimensions=element.edgepaths.kdims)
        if self.invert_axes:
            paths = [p[:, ::-1] for p in paths]
        return {'nodes': (pxs, pys), 'edges': paths}, style, {'dimensions': dims}


    def get_extents(self, element, ranges):
        """
        Extents are set to '' and None because x-axis is categorical and
        y-axis auto-ranges.
        """
        x0, x1 = element.nodes.range(0)
        y0, y1 = element.nodes.range(1)
        return (x0, y0, x1, y1)


    def init_artists(self, ax, plot_args, plot_kwargs):
        # Draw edges
        color_opts = ['c', 'cmap', 'vmin', 'vmax', 'norm']
        groups = [g for g in self._style_groups if g != 'edge']
        edge_opts = {k[5:] if 'edge_' in k else k: v
                     for k, v in plot_kwargs.items()
                     if not any(k.startswith(p) for p in groups)
                     and k not in color_opts}
        paths = plot_args['edges']
        if self.filled:
            coll = PolyCollection
            if 'colors' in edge_opts:
                edge_opts['facecolors'] = edge_opts.pop('colors')
        else:
            coll = LineCollection
        edges = coll(paths, **edge_opts)
        ax.add_collection(edges)

        # Draw nodes
        xs, ys = plot_args['nodes']
        groups = [g for g in self._style_groups if g != 'node']
        node_opts = {k[5:] if 'node_' in k else k: v
                     for k, v in plot_kwargs.items()
                     if not any(k.startswith(p) for p in groups)}
        if 'size' in node_opts: node_opts['s'] = node_opts.pop('size')**2
        nodes = ax.scatter(xs, ys, **node_opts)

        return {'nodes': nodes, 'edges': edges}


    def _update_nodes(self, element, data, style):
        nodes = self.handles['nodes']
        xs, ys = data['nodes']
        nodes.set_offsets(np.column_stack([xs, ys]))
        cdim = element.nodes.get_dimension(self.color_index)
        if cdim and 'c' in style:
            nodes.set_clim((style['vmin'], style['vmax']))
            nodes.set_array(style['c'])
            if 'norm' in style:
                nodes.norm = style['norm']


    def _update_edges(self, element, data, style):
        edges = self.handles['edges']
        paths = data['edges']
        edges.set_paths(paths)
        edges.set_visible(style.get('visible', True))
        cdim = element.get_dimension(self.edge_color_index)
        if cdim:
            if 'edge_array' in style:
                edges.set_clim(style['edge_clim'])
                edges.set_array(style['edge_array'])
                if 'norm' in style:
                    edges.norm = style['edge_norm']
            elif 'edge_colors' in style:
                if self.filled:
                    edges.set_facecolors(style['edge_colors'])
                else:
                    edges.set_edgecolors(style['edge_colors'])


    def update_handles(self, key, axis, element, ranges, style):
        data, style, axis_kwargs = self.get_data(element, ranges, style)
        self._update_nodes(element, data, style)
        self._update_edges(element, data, style)
        return axis_kwargs



class TriMeshPlot(GraphPlot):

    filled = param.Boolean(default=False, doc="""
        Whether the triangles should be drawn as filled.""")

    style_opts = GraphPlot.style_opts + ['edge_facecolors']

    def get_data(self, element, ranges, style):
        simplex_dim = element.get_dimension(self.edge_color_index)
        vertex_dim = element.nodes.get_dimension(self.edge_color_index)
        if not isinstance(self.edge_color_index, int) and vertex_dim and not simplex_dim:
            simplices = element.array([0, 1, 2])
            z = element.nodes.dimension_values(vertex_dim)
            z = z[simplices].mean(axis=1)
            element = element.add_dimension(vertex_dim, len(element.vdims), z, vdim=True)
        # Ensure the edgepaths for the triangles are generated before plotting
        element.edgepaths
        return super(TriMeshPlot, self).get_data(element, ranges, style)



class ChordPlot(GraphPlot):

    label_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the node labels will be drawn""")

    style_opts = GraphPlot.style_opts + ['text_font_size', 'label_offset']

    _style_groups = ['edge', 'node', 'arc']

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
        data, style, plot_kwargs = super(ChordPlot, self).get_data(element, ranges, style)
        if isinstance(style.get('node_facecolors'), list):
            angles = element._angles
            paths = []
            for i in range(len(element.nodes)):
                start, end = angles[i:i+2]
                vals = np.linspace(start, end, 20)
                paths.append(np.column_stack([np.cos(vals), np.sin(vals)]))
            data['arcs'] = paths
            style['arc_colors'] = style['node_facecolors']
            style['arc_linewidth'] = 10

        lidx = element.nodes.get_dimension(self.label_index)
        if lidx is None:
            if self.label_index is not None:
                dims = element.nodes.dimensions()[2:]
                self.warning("label_index supplied to Chord not found, "
                             "expected one of %s, got %s." %
                             (dims, self.label_index))
            return data, style, plot_kwargs
        if element.vdims:
            edges = Dataset(element)[element[element.vdims[0].name]>0]
            nodes = list(np.unique([edges.dimension_values(i) for i in range(2)]))
            nodes = element.nodes.select(**{element.nodes.kdims[2].name: nodes})
        else:
            nodes = element
        offset = style.get('label_offset', 1.05)
        xs, ys = (nodes.dimension_values(i)*offset for i in range(2))
        labels = [lidx.pprint_value(v) for v in nodes.dimension_values(lidx)]
        angles = np.rad2deg(np.arctan2(ys, xs))
        data['text'] = (xs, ys, labels, angles)
        return data, style, plot_kwargs


    def init_artists(self, ax, plot_args, plot_kwargs):
        artists = {}
        if 'arcs' in plot_args:
            color_opts = ['c', 'cmap', 'vmin', 'vmax', 'norm']
            groups = [g for g in self._style_groups if g != 'arc']
            edge_opts = {k[4:] if 'arc_' in k else k: v
                         for k, v in plot_kwargs.items()
                         if not any(k.startswith(p) for p in groups)
                         and k not in color_opts}
            paths = plot_args['arcs']
            edges = LineCollection(paths, **edge_opts)
            ax.add_collection(edges)
            artists['arcs'] = edges

        artists.update(super(ChordPlot, self).init_artists(ax, plot_args, plot_kwargs))
        if 'text' in plot_args:
            fontsize = plot_kwargs.get('text_font_size', 8)
            for (x, y, l, a) in zip(*plot_args['text']):
                ax.annotate(l, xy=(x, y), xycoords='data', rotation=a,
                            horizontalalignment='left', fontsize=fontsize,
                            verticalalignment='center', rotation_mode='anchor')

        return artists
