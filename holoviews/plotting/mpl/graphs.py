from matplotlib.collections import LineCollection

from .chart import ChartPlot

class GraphPlot(ChartPlot):
    """
    GraphPlot
    """

    style_opts = ['edge_alpha', 'edge_color', 'edge_linestyle', 'edge_linewidth',
                  'node_alpha', 'node_color', 'node_edgecolors', 'node_facecolors',
                  'node_linewidth', 'node_marker', 'node_size', 'visible']

    def get_data(self, element, ranges, style):
        xidx, yidx = (1, 0) if self.invert_axes else (0, 1)
        pxs, pys = (element.nodes.dimension_values(i) for i in range(2))
        dims = element.nodes.dimensions()

        paths = element.nodepaths.data
        if self.invert_axes:
            paths = [p[:, ::-1] for p in paths]
        return {'points': (pxs, pys), 'paths': paths}, style, {'dimensions': dims}

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
        edge_opts = {k[5:] if 'edge_' in k else k: v
                     for k, v in plot_kwargs.items()
                     if 'node_' not in k}
        paths = plot_args['paths']
        edges = LineCollection(paths, **edge_opts)
        ax.add_collection(edges)

        # Draw nodes
        xs, ys = plot_args['points']
        node_opts = {k[5:] if 'node_' in k else k: v
                     for k, v in plot_kwargs.items()
                     if 'edge_' not in k}
        if 'size' in node_opts: node_opts['s'] = node_opts.pop('size')**2
        nodes = ax.scatter(xs, ys, **node_opts)
        
        return {'nodes': nodes, 'edges': edges}

    def update_handles(self, key, axis, element, ranges, style):
        artist = self.handles['nodes']
        data, style, axis_kwargs = self.get_data(element, ranges, style)
        xs, ys = data['nodes']
        artist.set_xdata(xs)
        artist.set_ydata(ys)
        
        edges = self.handles['edges']
        paths = data['edges']
        artist.set_paths(paths)
        artist.set_visible(style.get('visible', True))
        
        return axis_kwargs
