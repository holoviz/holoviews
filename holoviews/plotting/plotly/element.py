import numpy as np
import plotly.graph_objs as go
import param

from ...core.util import basestring
from .plot import PlotlyPlot
from ..plot import GenericElementPlot, GenericOverlayPlot
from .. import util


class ElementPlot(PlotlyPlot, GenericElementPlot):

    aspect = param.Parameter(default='cube', doc="""
        The aspect ratio mode of the plot. By default, a plot may
        select its own appropriate aspect ratio but sometimes it may
        be necessary to force a square aspect ratio (e.g. to display
        the plot as an element of a grid). The modes 'auto' and
        'equal' correspond to the axis modes of the same name in
        matplotlib, a numeric value may also be passed.""")

    bgcolor = param.ClassSelector(class_=(str, tuple), default=None, doc="""
        If set bgcolor overrides the background color of the axis.""")

    invert_axes = param.ObjectSelector(default=False, doc="""
        Inverts the axes of the plot. Note that this parameter may not
        always be respected by all plots but should be respected by
        adjoined plots when appropriate.""")

    invert_xaxis = param.Boolean(default=False, doc="""
        Whether to invert the plot x-axis.""")

    invert_yaxis = param.Boolean(default=False, doc="""
        Whether to invert the plot y-axis.""")

    invert_zaxis = param.Boolean(default=False, doc="""
        Whether to invert the plot z-axis.""")

    labelled = param.List(default=['x', 'y'], doc="""
        Whether to plot the 'x' and 'y' labels.""")

    logx = param.Boolean(default=False, doc="""
         Whether to apply log scaling to the x-axis of the Chart.""")

    logy  = param.Boolean(default=False, doc="""
         Whether to apply log scaling to the y-axis of the Chart.""")

    logz  = param.Boolean(default=False, doc="""
         Whether to apply log scaling to the y-axis of the Chart.""")

    margins = param.NumericTuple(default=(50, 50, 50, 50), doc="""
         Margins in pixel values specified as a tuple of the form
         (left, bottom, right, top).""")

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    xaxis = param.ObjectSelector(default='bottom',
                                 objects=['top', 'bottom', 'bare', 'top-bare',
                                          'bottom-bare', None], doc="""
        Whether and where to display the xaxis, bare options allow suppressing
        all axis labels including ticks and xlabel. Valid options are 'top',
        'bottom', 'bare', 'top-bare' and 'bottom-bare'.""")

    xticks = param.Parameter(default=None, doc="""
        Ticks along x-axis specified as an integer, explicit list of
        tick locations, list of tuples containing the locations and
        labels or a matplotlib tick locator object. If set to None
        default matplotlib ticking behavior is applied.""")

    yaxis = param.ObjectSelector(default='left',
                                      objects=['left', 'right', 'bare', 'left-bare',
                                               'right-bare', None], doc="""
        Whether and where to display the yaxis, bare options allow suppressing
        all axis labels including ticks and ylabel. Valid options are 'left',
        'right', 'bare' 'left-bare' and 'right-bare'.""")

    yticks = param.Parameter(default=None, doc="""
        Ticks along y-axis specified as an integer, explicit list of
        tick locations, list of tuples containing the locations and
        labels or a matplotlib tick locator object. If set to None
        default matplotlib ticking behavior is applied.""")

    graph_obj = None

    def initialize_plot(self, ranges=None):
        """
        Initializes a new plot object with the last available frame.
        """
        # Get element key and ranges for frame
        fig = self.generate_plot(self.keys[-1], ranges)
        self.drawn = True
        return fig


    def generate_plot(self, key, ranges):
        element = self._get_frame(key)
        if element is None:
            return self.handles['fig']
        plot_opts = self.lookup_options(element, 'plot').options
        self.set_param(**{k: v for k, v in plot_opts.items()
                          if k in self.params()})
        self.style = self.lookup_options(element, 'style')

        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = util.match_spec(element, ranges)

        data_args, data_kwargs = self.get_data(element, ranges)
        opts = self.graph_options(element, ranges)
        graph = self.init_graph(data_args, dict(opts, **data_kwargs))
        self.handles['graph'] = graph

        layout = self.init_layout(key, element, ranges)
        self.handles['layout'] = layout

        if isinstance(graph, go.Figure):
            graph.update({'layout': layout})
            self.handles['fig'] = graph
        else:
            if not isinstance(graph, list):
                graph = [graph]
            fig = go.Figure(data=graph, layout=layout)
            self.handles['fig'] = fig
            return fig


    def graph_options(self, element, ranges):
        if self.overlay_dims:
            legend = ', '.join([d.pprint_value_string(v) for d, v in
                                self.overlay_dims.items()])
        else:
            legend = element.label

        opts = dict(showlegend=self.show_legend,
                    legendgroup=element.group,
                    name=legend)

        if self.layout_num:
            opts['xaxis'] = 'x' + str(self.layout_num)
            opts['yaxis'] = 'y' + str(self.layout_num)

        return opts


    def init_graph(self, plot_args, plot_kwargs):
        return self.graph_obj(*plot_args, **plot_kwargs)


    def get_data(self, element, ranges):
        return {}


    def init_layout(self, key, element, ranges, xdim=None, ydim=None):
        l, b, r, t = self.get_extents(element, ranges)

        options = {}

        xdim = element.get_dimension(0) if xdim is None else xdim
        ydim = element.get_dimension(1) if ydim is None else ydim
        xlabel, ylabel, zlabel = self._get_axis_labels([xdim, ydim])

        if self.invert_axes:
            xlabel, ylabel = ylabel, xlabel
            l, b, r, t = b, l, t, r

        if 'x' not in self.labelled:
            xlabel = ''
        if 'y' not in self.labelled:
            ylabel = ''

        if xdim:
            xaxis = dict(range=[l, r], title=xlabel)
            if self.logx:
                xaxis['type'] = 'log'
            options['xaxis'] = xaxis

        if ydim:
            yaxis = dict(range=[b, t], title=ylabel)
            if self.logy:
                yaxis['type'] = 'log'
            options['yaxis'] = yaxis

        l, b, r, t = self.margins
        margin = go.Margin(l=l, r=r,b=b, t=t, pad=4)
        return go.Layout(width=self.width, height=self.height,
                         title=self._format_title(key, separator=' '),
                         plot_bgcolor=self.bgcolor, margin=margin,
                         **options)


    def update_frame(self, key, ranges=None):
        """
        Updates an existing plot with data corresponding
        to the key.
        """
        self.generate_plot(key, ranges)


class ColorbarPlot(ElementPlot):

    colorbar = param.Boolean(default=False, doc="""
        Whether to display a colorbar.""")

    colorbar_opts = param.Dict(default={}, doc="""
        Allows setting including borderwidth, showexponent, nticks,
        outlinecolor, thickness, bgcolor, outlinewidth, bordercolor,
        ticklen, xpad, ypad, tickangle...""")

    def get_color_opts(self, dim, element, ranges, style):
        opts = {}
        if self.colorbar:
            opts['colorbar'] = dict(title=dim.pprint_label,
                                    **self.colorbar_opts)
        else:
            opts['showscale'] = False

        cmap = style.pop('cmap', 'viridis')
        if cmap == 'fire':
            values = np.linspace(0, 1, len(util.fire_colors))
            cmap = [(v, 'rgb(%d, %d, %d)' % tuple(c))
                    for v, c in zip(values, np.array(util.fire_colors)*255)]
        elif isinstance(cmap, basestring):
            if cmap[0] == cmap[0].lower():
                cmap = cmap[0].upper() + cmap[1:]
            if cmap.endswith('_r'):
                cmap = cmap[:-2]
                opts['reversescale'] = True
        opts['colorscale'] = cmap
        if dim:
            if dim.name in ranges:
                cmin, cmax = ranges[dim.name]['combined']
            else:
                cmin, cmax = element.range(dim.name)
            opts['cmin'] = cmin
            opts['cmax'] = cmax
            opts['cauto'] = False
        return dict(style, **opts)


class OverlayPlot(GenericOverlayPlot, ElementPlot):


    def initialize_plot(self, ranges=None):
        """
        Initializes a new plot object with the last available frame.
        """
        # Get element key and ranges for frame
        return self.generate_plot(list(self.hmap.data.keys())[0], ranges)


    def generate_plot(self, key, ranges):
        element = self._get_frame(key)

        ranges = self.compute_ranges(self.hmap, key, ranges)
        figure = None
        for okey, subplot in self.subplots.items():
            fig = subplot.generate_plot(key, ranges)
            if figure is None:
                figure = fig
            else:
                figure['data'].extend(fig['data'])

        layout = self.init_layout(key, element, ranges)
        figure['layout'].update(layout)
        self.handles['fig'] = figure
        return figure
