from __future__ import absolute_import, division, unicode_literals

import uuid
import numpy as np
import param
import re

from ...core import util
from ...core.element import Element
from ...core.spaces import DynamicMap
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import dim_range_key, dynamic_update
from .plot import PlotlyPlot
from .renderer import PlotlyRenderer
from .util import (
    STYLE_ALIASES, get_colorscale, merge_figure, legend_trace_types)


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

    labelled = param.List(default=['x', 'y', 'z'], doc="""
        Whether to label the 'x' and 'y' axes.""")

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
        tick locations, list of tuples containing the locations.""")

    yaxis = param.ObjectSelector(default='left',
                                      objects=['left', 'right', 'bare', 'left-bare',
                                               'right-bare', None], doc="""
        Whether and where to display the yaxis, bare options allow suppressing
        all axis labels including ticks and ylabel. Valid options are 'left',
        'right', 'bare' 'left-bare' and 'right-bare'.""")

    yticks = param.Parameter(default=None, doc="""
        Ticks along y-axis specified as an integer, explicit list of
        tick locations, list of tuples containing the locations.""")

    zlabel = param.String(default=None, doc="""
        An explicit override of the z-axis label, if set takes precedence
        over the dimension label.""")

    zticks = param.Parameter(default=None, doc="""
        Ticks along z-axis specified as an integer, explicit list of
        tick locations, list of tuples containing the locations.""")

    trace_kwargs = {}

    _style_key = None

    # Whether vectorized styles are applied per trace
    _per_trace = False

    # Declare which styles cannot be mapped to a non-scalar dimension
    _nonvectorized_styles = []

    def __init__(self, element, plot=None, **params):
        super(ElementPlot, self).__init__(element, **params)
        self.trace_uid = str(uuid.uuid4())
        self.static = len(self.hmap) == 1 and len(self.keys) == len(self.hmap)
        self.callbacks = self._construct_callbacks()


    def initialize_plot(self, ranges=None):
        """
        Initializes a new plot object with the last available frame.
        """
        # Get element key and ranges for frame
        fig = self.generate_plot(self.keys[-1], ranges)
        self.drawn = True
        return fig


    def generate_plot(self, key, ranges, element=None):
        if element is None:
            element = self._get_frame(key)

        if element is None:
            return self.handles['fig']

        # Set plot options
        plot_opts = self.lookup_options(element, 'plot').options
        self.param.set_param(**{k: v for k, v in plot_opts.items()
                                if k in self.params()})

        # Get ranges
        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = util.match_spec(element, ranges)

        # Get style
        self.style = self.lookup_options(element, 'style')
        style = self.style[self.cyclic_index]

        # Get data and options and merge them
        data = self.get_data(element, ranges, style)
        opts = self.graph_options(element, ranges, style)

        components = {
            'traces': [],
            'images': [],
            'annotations': [],
            'shapes': [],
        }

        for i, d in enumerate(data):
            # Initialize traces
            datum_components = self.init_graph(d, opts, index=i)

            # Handle traces
            traces = datum_components.get('traces', [])
            components['traces'].extend(traces)

            if i == 0 and traces:
                # Associate element with trace.uid property of the first
                # plotly trace that is used to render the element. This is
                # used to associate the element with the trace during callbacks
                traces[0]['uid'] = self.trace_uid

            # Handle images, shapes, annotations
            for k in ['images', 'shapes', 'annotations']:
                components[k].extend(datum_components.get(k, []))

        self.handles['components'] = components

        # Initialize layout
        layout = self.init_layout(key, element, ranges)
        for k in ['images', 'shapes', 'annotations']:
            layout.setdefault(k, [])
            layout[k].extend(components.get(k, []))

        self.handles['layout'] = layout

        # Create figure and return it
        self.drawn = True
        fig = dict(data=components['traces'], layout=layout)


        self.handles['fig'] = fig
        return fig


    def graph_options(self, element, ranges, style):
        if self.overlay_dims:
            legend = ', '.join([d.pprint_value_string(v) for d, v in
                                self.overlay_dims.items()])
        else:
            legend = element.label

        opts = dict(
            name=legend, **self.trace_kwargs)

        if self.trace_kwargs.get('type', None) in legend_trace_types:
            opts.update(
                showlegend=self.show_legend, legendgroup=element.group)

        if self._style_key is not None:
            styles = self._apply_transforms(element, ranges, style)

            # If style starts with '{_style_key}_', remove the prefix.  This way
            # a line_color property with self._style_key of 'line' doesn't end up
            # as `line_line_color`
            key_prefix_re = re.compile('^' + self._style_key + '_')
            styles = {key_prefix_re.sub('', k): v for k, v in styles.items()}

            opts[self._style_key] = {STYLE_ALIASES.get(k, k): v
                                     for k, v in styles.items()}

            # Move selectedpoints from style key back to root
            if 'selectedpoints' in opts.get(self._style_key, {}):
                opts['selectedpoints'] = opts[self._style_key].pop('selectedpoints')
        else:
            opts.update({STYLE_ALIASES.get(k, k): v
                         for k, v in style.items() if k != 'cmap'})

        return opts

    def init_graph(self, datum, options, index=0):
        """
        Initialize the plotly components that will represent the element

        Parameters
        ----------
        datum: dict
            An element of the data list returned by the get_data method
        options: dict
            Graph options that were returned by the graph_options method
        index: int
            Index of datum in the original list returned by the get_data method

        Returns
        -------
        dict
            Dictionary of the plotly components that represent the element.
            Keys may include:
             - 'traces': List of trace dicts
             - 'annotations': List of annotations dicts
             - 'images': List of image dicts
             - 'shapes': List of shape dicts
        """
        trace = dict(options)
        for k, v in datum.items():
            if k in trace and isinstance(trace[k], dict):
                trace[k].update(v)
            else:
                trace[k] = v

        if self._style_key and self._per_trace:
            vectorized = {k: v for k, v in options[self._style_key].items()
                          if isinstance(v, np.ndarray)}
            trace[self._style_key] = dict(trace[self._style_key])
            for s, val in vectorized.items():
                trace[self._style_key][s] = val[index]
        return {'traces': [trace]}


    def get_data(self, element, ranges, style):
        return []


    def get_aspect(self, xspan, yspan):
        """
        Computes the aspect ratio of the plot
        """
        return self.width/self.height


    def _get_axis_dims(self, element):
        """Returns the dimensions corresponding to each axis.

        Should return a list of dimensions or list of lists of
        dimensions, which will be formatted to label the axis
        and to link axes.
        """
        dims = element.dimensions()[:3]
        pad = [None]*max(3-len(dims), 0)
        return dims + pad


    def _apply_transforms(self, element, ranges, style):
        new_style = dict(style)
        for k, v in dict(style).items():
            if isinstance(v, util.basestring):
                if k == 'marker' and v in 'xsdo':
                    continue
                elif v in element:
                    v = dim(v)
                elif any(d==v for d in self.overlay_dims):
                    v = dim([d for d in self.overlay_dims if d==v][0])

            if not isinstance(v, dim):
                continue
            elif (not v.applies(element) and v.dimension not in self.overlay_dims):
                new_style.pop(k)
                self.warning('Specified %s dim transform %r could not be applied, as not all '
                             'dimensions could be resolved.' % (k, v))
                continue

            if len(v.ops) == 0 and v.dimension in self.overlay_dims:
                val = self.overlay_dims[v.dimension]
            else:
                val = v.apply(element, ranges=ranges, flat=True)

            if (not util.isscalar(val) and len(util.unique_array(val)) == 1
                and not 'color' in k):
                val = val[0]

            if not util.isscalar(val):
                if k in self._nonvectorized_styles:
                    element = type(element).__name__
                    raise ValueError('Mapping a dimension to the "{style}" '
                                     'style option is not supported by the '
                                     '{element} element using the {backend} '
                                     'backend. To map the "{dim}" dimension '
                                     'to the {style} use a groupby operation '
                                     'to overlay your data along the dimension.'.format(
                                         style=k, dim=v.dimension, element=element,
                                         backend=self.renderer.backend))

            # If color is not valid colorspec add colormapper
            numeric = isinstance(val, np.ndarray) and val.dtype.kind in 'uifMm'
            if ('color' in k and isinstance(val, np.ndarray) and numeric):
                copts = self.get_color_opts(v, element, ranges, style)
                new_style.pop('cmap', None)
                new_style.update(copts)
            new_style[k] = val
        return new_style


    def init_layout(self, key, element, ranges):
        el = element.traverse(lambda x: x, [Element])
        el = el[0] if el else element

        extent = self.get_extents(element, ranges)

        if len(extent) == 4:
            l, b, r, t = extent
        else:
            l, b, z0, r, t, z1 = extent

        options = {'uirevision': True}

        dims = self._get_axis_dims(el)
        if len(dims) > 2:
            xdim, ydim, zdim = dims
        else:
            xdim, ydim = dims
            zdim = None
        xlabel, ylabel, zlabel = self._get_axis_labels(dims)

        if self.invert_axes:
            xlabel, ylabel = ylabel, xlabel
            ydim, xdim = xdim, ydim
            l, b, r, t = b, l, t, r

        if 'x' not in self.labelled:
            xlabel = ''
        if 'y' not in self.labelled:
            ylabel = ''
        if 'z' not in self.labelled:
            zlabel = ''

        if xdim:
            try:
                if any(np.isnan([r, l])):
                    r, l = 0, 1
            except TypeError:
                # r and l not numeric, don't change anything
                pass

            xrange = [r, l] if self.invert_xaxis else [l, r]
            xaxis = dict(range=xrange, title=xlabel)
            if self.logx:
                xaxis['type'] = 'log'
            self._get_ticks(xaxis, self.xticks)

            if self.projection != '3d' and self.xaxis:
                xaxis['automargin'] = False

                # Create dimension string used to compute matching axes
                if isinstance(xdim, (list, tuple)):
                    dim_str = "-".join(["%s^%s^%s" % (d.name, d.label, d.unit)
                                        for d in xdim])
                else:
                    dim_str = "%s^%s^%s" % (xdim.name, xdim.label, xdim.unit)

                xaxis['_dim'] = dim_str

                if 'bare' in self.xaxis:
                    xaxis['ticks'] = ''
                    xaxis['showticklabels'] = False
                    xaxis['title'] = ''

                if 'top' in self.xaxis:
                    xaxis['side'] = 'top'
                else:
                    xaxis['side'] = 'bottom'
        else:
            xaxis = {}

        if ydim:
            try:
                if any(np.isnan([b, t])):
                    b, t = 0, 1
            except TypeError:
                # b and t not numeric, don't change anything
                pass

            yrange = [t, b] if self.invert_yaxis else [b, t]
            yaxis = dict(range=yrange, title=ylabel)
            if self.logy:
                yaxis['type'] = 'log'
            self._get_ticks(yaxis, self.yticks)

            if self.projection != '3d' and self.yaxis:
                yaxis['automargin'] = False

                # Create dimension string used to compute matching axes
                if isinstance(ydim, (list, tuple)):
                    dim_str = "-".join(["%s^%s^%s" % (d.name, d.label, d.unit)
                                        for d in ydim])
                else:
                    dim_str = "%s^%s^%s" % (ydim.name, ydim.label, ydim.unit)

                yaxis['_dim'] = dim_str,
                if 'bare' in self.yaxis:
                    yaxis['ticks'] = ''
                    yaxis['showticklabels'] = False
                    yaxis['title'] = ''

                if 'right' in self.yaxis:
                    yaxis['side'] = 'right'
                else:
                    yaxis['side'] = 'left'

        else:
            yaxis = {}

        if self.projection == '3d':
            scene = dict(xaxis=xaxis, yaxis=yaxis)
            if zdim:
                zrange = [z1, z0] if self.invert_zaxis else [z0, z1]
                zaxis = dict(range=zrange, title=zlabel)
                if self.logz:
                    zaxis['type'] = 'log'
                self._get_ticks(zaxis, self.zticks)
                scene['zaxis'] = zaxis
            if self.aspect == 'cube':
                scene['aspectmode'] = 'cube'
            else:
                scene['aspectmode'] = 'manual'
                scene['aspectratio'] = self.aspect
            options['scene'] = scene
        else:
            l, b, r, t = self.margins
            options['xaxis'] = xaxis
            options['yaxis'] = yaxis
            options['margin'] = dict(l=l, r=r, b=b, t=t, pad=4)

        return dict(width=self.width, height=self.height,
                    title=self._format_title(key, separator=' '),
                    plot_bgcolor=self.bgcolor, **options)

    def _get_ticks(self, axis, ticker):
        axis_props = {}
        if isinstance(ticker, (tuple, list)):
            if all(isinstance(t, tuple) for t in ticker):
                ticks, labels = zip(*ticker)
                labels = [l if isinstance(l, util.basestring) else str(l)
                              for l in labels]
                axis_props['tickvals'] = ticks
                axis_props['ticktext'] = labels
            else:
                axis_props['tickvals'] = ticker
            axis.update(axis_props)

    def update_frame(self, key, ranges=None, element=None):
        """
        Updates an existing plot with data corresponding
        to the key.
        """
        self.generate_plot(key, ranges, element)
        PlotlyRenderer.trigger_plot_pane(self.id, self.state)


class ColorbarPlot(ElementPlot):

    clim = param.NumericTuple(default=(np.nan, np.nan), length=2, doc="""
       User-specified colorbar axis range limits for the plot, as a tuple (low,high).
       If specified, takes precedence over data and dimension ranges.""")

    colorbar = param.Boolean(default=False, doc="""
        Whether to display a colorbar.""")

    color_levels = param.ClassSelector(default=None, class_=(int, list), doc="""
        Number of discrete colors to use when colormapping or a set of color
        intervals defining the range of values to map each color to.""")

    colorbar_opts = param.Dict(default={}, doc="""
        Allows setting including borderwidth, showexponent, nticks,
        outlinecolor, thickness, bgcolor, outlinewidth, bordercolor,
        ticklen, xpad, ypad, tickangle...""")

    symmetric = param.Boolean(default=False, doc="""
        Whether to make the colormap symmetric around zero.""")

    def get_color_opts(self, eldim, element, ranges, style):
        opts = {}
        dim_name = dim_range_key(eldim)
        if self.colorbar:
            if isinstance(eldim, dim):
                title = str(eldim) if eldim.ops else str(eldim)[1:-1]
            else:
                title = eldim.pprint_label
            opts['colorbar'] = dict(title=title, **self.colorbar_opts)
            opts['showscale'] = True
        else:
            opts['showscale'] = False

        if eldim:
            auto = False
            if util.isfinite(self.clim).all():
                cmin, cmax = self.clim
            elif dim_name in ranges:
                cmin, cmax = ranges[dim_name]['combined']
            elif isinstance(eldim, dim):
                cmin, cmax = np.nan, np.nan
                auto = True
            else:
                cmin, cmax = element.range(dim_name)
            if self.symmetric:
                cabs = np.abs([cmin, cmax])
                cmin, cmax = -cabs.max(), cabs.max()
        else:
            auto = True
            cmin, cmax = None, None

        cmap = style.pop('cmap', 'viridis')
        colorscale = get_colorscale(cmap, self.color_levels, cmin, cmax)

        # Reduce colorscale length to <= 255 to work around
        # https://github.com/plotly/plotly.js/issues/3699. Plotly.js performs
        # colorscale interpolation internally so reducing the number of colors
        # here makes very little difference to the displayed colorscale.
        #
        # Note that we need to be careful to make sure the first and last
        # colorscale pairs, colorscale[0] and colorscale[-1], are preserved
        # as the first and last in the subsampled colorscale
        if isinstance(colorscale, list) and len(colorscale) > 255:
            last_clr_pair = colorscale[-1]
            step = int(np.ceil(len(colorscale) / 255))
            colorscale = colorscale[0::step]
            colorscale[-1] = last_clr_pair

        if cmin is not None:
            opts['cmin'] = cmin
        if cmax is not None:
            opts['cmax'] = cmax
        opts['cauto'] = auto
        opts['colorscale'] = colorscale
        return opts


class OverlayPlot(GenericOverlayPlot, ElementPlot):

    _propagate_options = [
        'width', 'height', 'xaxis', 'yaxis', 'labelled', 'bgcolor',
        'invert_axes', 'show_frame', 'show_grid', 'logx', 'logy',
        'xticks', 'toolbar', 'yticks', 'xrotation', 'yrotation',
        'invert_xaxis', 'invert_yaxis', 'sizing_mode', 'title', 'title_format',
        'padding', 'xlabel', 'ylabel', 'zlabel', 'xlim', 'ylim', 'zlim']

    def initialize_plot(self, ranges=None):
        """
        Initializes a new plot object with the last available frame.
        """
        # Get element key and ranges for frame
        return self.generate_plot(list(self.hmap.data.keys())[0], ranges)


    def generate_plot(self, key, ranges, element=None):
        if element is None:
            element = self._get_frame(key)
        items = [] if element is None else list(element.data.items())

        # Update plot options
        plot_opts = self.lookup_options(element, 'plot').options
        inherited = self._traverse_options(element, 'plot',
                                           self._propagate_options,
                                           defaults=False)
        plot_opts.update(**{k: v[0] for k, v in inherited.items() if k not in plot_opts})
        self.set_param(**plot_opts)

        ranges = self.compute_ranges(self.hmap, key, ranges)
        figure = None
        for okey, subplot in self.subplots.items():
            if element is not None and subplot.drawn:
                idx, spec, exact = dynamic_update(self, subplot, okey, element, items)
                if idx is not None:
                    _, el = items.pop(idx)
                else:
                    el = None
            else:
                el = None

            fig = subplot.generate_plot(key, ranges, el)
            if figure is None:
                figure = fig
            else:
                merge_figure(figure, fig)

        layout = self.init_layout(key, element, ranges)
        figure['layout'].update(layout)
        self.drawn = True

        self.handles['fig'] = figure
        return figure

    def update_frame(self, key, ranges=None, element=None):
        reused = isinstance(self.hmap, DynamicMap) and self.overlaid
        if not reused and element is None:
            element = self._get_frame(key)
        elif element is not None:
            self.current_frame = element
            self.current_key = key
        items = [] if element is None else list(element.data.items())

        # Instantiate dynamically added subplots
        for k, subplot in self.subplots.items():
            # If in Dynamic mode propagate elements to subplots
            if not (isinstance(self.hmap, DynamicMap) and element is not None):
                continue
            idx, _, _ = dynamic_update(self, subplot, k, element, items)
            if idx is not None:
                items.pop(idx)
        if isinstance(self.hmap, DynamicMap) and items:
            self._create_dynamic_subplots(key, items, ranges)

        self.generate_plot(key, ranges, element)
        PlotlyRenderer.trigger_plot_pane(self.id, self.state)
