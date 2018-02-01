import warnings

import param
import numpy as np
import bokeh
import bokeh.plotting
from bokeh.core.properties import value
from bokeh.models import (HoverTool, Renderer, Range1d, DataRange1d, Title,
                          FactorRange, FuncTickFormatter, Tool, Legend)
from bokeh.models.tickers import Ticker, BasicTicker, FixedTicker, LogTicker
from bokeh.models.widgets import Panel, Tabs
from bokeh.models.mappers import LinearColorMapper
try:
    from bokeh.models import ColorBar
    from bokeh.models.mappers import LogColorMapper, CategoricalColorMapper
except ImportError:
    LogColorMapper, ColorBar = None, None
from bokeh.plotting.helpers import _known_tools as known_tools

from ...core import DynamicMap, CompositeOverlay, Element, Dimension
from ...core.options import abbreviated_exception, SkipRendering
from ...core import util
from ...streams import Buffer
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import dynamic_update, process_cmap, color_intervals
from .plot import BokehPlot, TOOLS
from .util import (mpl_to_bokeh, get_tab_title,  py2js_tickformatter,
                   rgba_tuple, recursive_model_update, glyph_order,
                   decode_bytes, bokeh_version)

property_prefixes = ['selection', 'nonselection', 'muted', 'hover']

# Define shared style properties for bokeh plots
line_properties = ['line_color', 'line_alpha', 'color', 'alpha', 'line_width',
                   'line_join', 'line_cap', 'line_dash']
line_properties += ['_'.join([prefix, prop]) for prop in line_properties[:4]
                    for prefix in property_prefixes]

fill_properties = ['fill_color', 'fill_alpha']
fill_properties += ['_'.join([prefix, prop]) for prop in fill_properties
                    for prefix in property_prefixes]

text_properties = ['text_font', 'text_font_size', 'text_font_style', 'text_color',
                   'text_alpha', 'text_align', 'text_baseline']

legend_dimensions = ['label_standoff', 'label_width', 'label_height', 'glyph_width',
                     'glyph_height', 'legend_padding', 'legend_spacing', 'click_policy']


class ElementPlot(BokehPlot, GenericElementPlot):

    border = param.Number(default=10, doc="""
        Minimum border around plot.""")

    finalize_hooks = param.HookList(default=[], doc="""
        Optional list of hooks called when finalizing an axis.
        The hook is passed the plot object and the displayed
        object, other plotting handles can be accessed via plot.handles.""")

    fontsize = param.Parameter(default={'title': '12pt'}, allow_None=True,  doc="""
       Specifies various fontsizes of the displayed text.

       Finer control is available by supplying a dictionary where any
       unmentioned keys reverts to the default sizes, e.g:

          {'ticks': '20pt', 'title': '15pt', 'ylabel': '5px', 'xlabel': '5px'}""")

    gridstyle = param.Dict(default={}, doc="""
        Allows customizing the grid style, e.g. grid_line_color defines
        the line color for both grids while xgrid_line_color exclusively
        customizes the x-axis grid lines.""")

    labelled = param.List(default=['x', 'y'], doc="""
        Whether to plot the 'x' and 'y' labels.""")

    lod = param.Dict(default={'factor': 10, 'interval': 300,
                              'threshold': 2000, 'timeout': 500}, doc="""
        Bokeh plots offer "Level of Detail" (LOD) capability to
        accommodate large (but not huge) amounts of data. The available
        options are:

          * factor    - Decimation factor to use when applying
                        decimation.
          * interval  - Interval (in ms) downsampling will be enabled
                        after an interactive event.
          * threshold - Number of samples before downsampling is enabled.
          * timeout   - Timeout (in ms) for checking whether interactive
                        tool events are still occurring.""")
    show_frame = param.Boolean(default=True, doc="""
        Whether or not to show a complete frame around the plot.""")

    shared_axes = param.Boolean(default=True, doc="""
        Whether to invert the share axes across plots
        for linked panning and zooming.""")

    default_tools = param.List(default=['save', 'pan', 'wheel_zoom',
                                        'box_zoom', 'reset'],
        doc="A list of plugin tools to use on the plot.")

    tools = param.List(default=[], doc="""
        A list of plugin tools to use on the plot.""")

    toolbar = param.ObjectSelector(default='right',
                                   objects=["above", "below",
                                            "left", "right", None],
                                   doc="""
        The toolbar location, must be one of 'above', 'below',
        'left', 'right', None.""")

    _categorical = False

    # Declares the default types for continuous x- and y-axes
    _x_range_type = Range1d
    _y_range_type = Range1d

    # Whether the plot supports streaming data
    _stream_data = True

    def __init__(self, element, plot=None, **params):
        self.current_ranges = None
        super(ElementPlot, self).__init__(element, **params)
        self.handles = {} if plot is None else self.handles['plot']
        self.static = len(self.hmap) == 1 and len(self.keys) == len(self.hmap)
        self.callbacks = self._construct_callbacks()
        self.static_source = False
        self.streaming = [s for s in self.streams if isinstance(s, Buffer)]

        # Whether axes are shared between plots
        self._shared = {'x': False, 'y': False}


    def _hover_opts(self, element):
        if self.batched:
            dims = list(self.hmap.last.kdims)
        else:
            dims = list(self.overlay_dims.keys())
        dims += element.dimensions()
        return list(util.unique_iterator(dims)), {}


    def _init_tools(self, element, callbacks=[]):
        """
        Processes the list of tools to be supplied to the plot.
        """
        tooltips, hover_opts = self._hover_opts(element)
        tooltips = [(ttp.pprint_label, '@{%s}' % util.dimension_sanitizer(ttp.name))
                    if isinstance(ttp, Dimension) else ttp for ttp in tooltips]
        if not tooltips: tooltips = None

        callbacks = callbacks+self.callbacks
        cb_tools, tool_names = [], []
        hover = False
        for cb in callbacks:
            for handle in cb.models+cb.extra_models:
                if handle and handle in known_tools:
                    tool_names.append(handle)
                    if handle == 'hover':
                        tool = HoverTool(tooltips=tooltips, **hover_opts)
                        hover = tool
                    else:
                        tool = known_tools[handle]()
                    cb_tools.append(tool)
                    self.handles[handle] = tool

        tools = [t for t in cb_tools + self.default_tools + self.tools
                 if t not in tool_names]

        copied_tools = []
        for tool in tools:
            if isinstance(tool, Tool):
                properties = tool.properties_with_values(include_defaults=False)
                tool = type(tool)(**properties)
            copied_tools.append(tool)

        hover_tools = [t for t in copied_tools if isinstance(t, HoverTool)]
        if 'hover' in copied_tools:
            hover = HoverTool(tooltips=tooltips, **hover_opts)
            copied_tools[copied_tools.index('hover')] = hover
        elif any(hover_tools):
            hover = hover_tools[0]
        if hover:
            self.handles['hover'] = hover
        return copied_tools


    def _get_hover_data(self, data, element, dimensions=None):
        """
        Initializes hover data based on Element dimension values.
        If empty initializes with no data.
        """
        if 'hover' not in self.handles or self.static_source:
            return

        for d in (dimensions or element.dimensions()):
            dim = util.dimension_sanitizer(d.name)
            if dim not in data:
                data[dim] = element.dimension_values(d)
            elif isinstance(data[dim], np.ndarray) and data[dim].dtype.kind == 'M':
                data[dim+'_dt_strings'] = [d.pprint_value(v) for v in data[dim]]

        for k, v in self.overlay_dims.items():
            dim = util.dimension_sanitizer(k.name)
            if dim not in data:
                data[dim] = [v for _ in range(len(list(data.values())[0]))]


    def _merge_ranges(self, plots, xlabel, ylabel):
        """
        Given a list of other plots return axes that are shared
        with another plot by matching the axes labels
        """
        plot_ranges = {}
        for plot in plots:
            if plot is None:
                continue
            if hasattr(plot, 'xaxis'):
                if plot.xaxis[0].axis_label == xlabel:
                    plot_ranges['x_range'] = plot.x_range
                if plot.xaxis[0].axis_label == ylabel:
                    plot_ranges['y_range'] = plot.x_range
            if hasattr(plot, 'yaxis'):
                if plot.yaxis[0].axis_label == ylabel:
                    plot_ranges['y_range'] = plot.y_range
                if plot.yaxis[0].axis_label == xlabel:
                    plot_ranges['x_range'] = plot.y_range
        return plot_ranges


    def _axes_props(self, plots, subplots, element, ranges):
        # Get the bottom layer and range element
        el = element.traverse(lambda x: x, [Element])
        el = el[0] if el else element

        dims = el.dimensions()
        xlabel, ylabel, zlabel = self._get_axis_labels(dims)
        if self.invert_axes:
            xlabel, ylabel = ylabel, xlabel

        plot_ranges = {}
        # Try finding shared ranges in other plots in the same Layout
        norm_opts = self.lookup_options(el, 'norm').options
        if plots and self.shared_axes and not norm_opts.get('axiswise', False):
            plot_ranges = self._merge_ranges(plots, xlabel, ylabel)

        # Get the Element that determines the range and get_extents
        range_el = el if self.batched and not isinstance(self, OverlayPlot) else element
        l, b, r, t = self.get_extents(range_el, ranges)
        if self.invert_axes:
            l, b, r, t = b, l, t, r

        xtype = el.get_dimension_type(0)
        if ((xtype is np.object_ and type(l) in util.datetime_types) or
            xtype in util.datetime_types):
            x_axis_type = 'datetime'
        else:
            x_axis_type = 'log' if self.logx else 'auto'

        y_axis_type = 'log' if self.logy else 'auto'
        if len(dims) > 1:
            ytype = el.get_dimension_type(1)
            if ((ytype is np.object_ and type(b) in util.datetime_types)
                or ytype in util.datetime_types):
                y_axis_type = 'datetime'

        # Declare shared axes
        if 'x_range' in plot_ranges:
            self._shared['x'] = True
        if 'y_range' in plot_ranges:
            self._shared['y'] = True

        categorical = any(self.traverse(lambda x: x._categorical))
        categorical_x = any(isinstance(x, util.basestring) for x in (l, r))
        categorical_y = any(isinstance(y, util.basestring) for y in (b, t))

        range_types = (self._x_range_type, self._y_range_type)
        if self.invert_axes: range_types = range_types[::-1]
        x_range_type, y_range_type = range_types
        if categorical or categorical_x:
            x_axis_type = 'auto'
            plot_ranges['x_range'] = FactorRange()
        elif 'x_range' not in plot_ranges:
            plot_ranges['x_range'] = x_range_type()

        if categorical or categorical_y:
            y_axis_type = 'auto'
            plot_ranges['y_range'] = FactorRange()
        elif 'y_range' not in plot_ranges:
            plot_ranges['y_range'] = y_range_type()

        return (x_axis_type, y_axis_type), (xlabel, ylabel, zlabel), plot_ranges


    def _init_plot(self, key, element, plots, ranges=None):
        """
        Initializes Bokeh figure to draw Element into and sets basic
        figure and axis attributes including axes types, labels,
        titles and plot height and width.
        """
        subplots = list(self.subplots.values()) if self.subplots else []

        axis_types, labels, plot_ranges = self._axes_props(plots, subplots, element, ranges)
        xlabel, ylabel, _ = labels
        x_axis_type, y_axis_type = axis_types
        properties = dict(plot_ranges)
        properties['x_axis_label'] = xlabel if 'x' in self.labelled else ' '
        properties['y_axis_label'] = ylabel if 'y' in self.labelled else ' '

        if not self.show_frame:
            properties['outline_line_alpha'] = 0

        if self.show_title and self.adjoined is None:
            title = self._format_title(key, separator=' ')
        else:
            title = ''

        if self.toolbar:
            tools = self._init_tools(element)
            properties['tools'] = tools
        properties['toolbar_location'] = self.toolbar

        if self.renderer.webgl:
            properties['output_backend'] = 'webgl'

        with warnings.catch_warnings():
            # Bokeh raises warnings about duplicate tools but these
            # are not really an issue
            warnings.simplefilter('ignore', UserWarning)
            return bokeh.plotting.Figure(x_axis_type=x_axis_type,
                                         y_axis_type=y_axis_type, title=title,
                                         **properties)


    def _plot_properties(self, key, plot, element):
        """
        Returns a dictionary of plot properties.
        """
        size_multiplier = self.renderer.size/100.
        plot_props = dict(plot_height=int(self.height*size_multiplier),
                          plot_width=int(self.width*size_multiplier),
                          sizing_mode=self.sizing_mode)
        if self.bgcolor:
            plot_props['background_fill_color'] = self.bgcolor
        if self.border is not None:
            for p in ['left', 'right', 'top', 'bottom']:
                plot_props['min_border_'+p] = self.border
        lod = dict(self.defaults().get('lod', {}), **self.lod)
        for lod_prop, v in lod.items():
            plot_props['lod_'+lod_prop] = v
        return plot_props


    def _title_properties(self, key, plot, element):
        if self.show_title and self.adjoined is None:
            title = self._format_title(key, separator=' ')
        else:
            title = ''

        opts = dict(text=title, text_color='black')
        title_font = self._fontsize('title').get('fontsize')
        if title_font:
            opts['text_font_size'] = value(title_font)
        return opts

    def _init_axes(self, plot):
        if self.xaxis is None:
            plot.xaxis.visible = False
        elif 'top' in self.xaxis:
            plot.above = plot.below
            plot.below = []
            plot.xaxis[:] = plot.above
        self.handles['xaxis'] = plot.xaxis[0]
        self.handles['x_range'] = plot.x_range

        if self.yaxis is None:
            plot.yaxis.visible = False
        elif 'right' in self.yaxis:
            plot.right = plot.left
            plot.left = []
            plot.yaxis[:] = plot.right
        self.handles['yaxis'] = plot.yaxis[0]
        self.handles['y_range'] = plot.y_range


    def _axis_properties(self, axis, key, plot, dimension=None,
                         ax_mapping={'x': 0, 'y': 1}):
        """
        Returns a dictionary of axis properties depending
        on the specified axis.
        """
        axis_props = {}
        if ((axis == 'x' and self.xaxis in ['bottom-bare', 'top-bare']) or
            (axis == 'y' and self.yaxis in ['left-bare', 'right-bare'])):
            axis_props['axis_label_text_font_size'] = value('0pt')
            axis_props['major_label_text_font_size'] = value('0pt')
            axis_props['major_tick_line_color'] = None
            axis_props['minor_tick_line_color'] = None
        else:
            labelsize = self._fontsize('%slabel' % axis).get('fontsize')
            if labelsize:
                axis_props['axis_label_text_font_size'] = labelsize
            ticksize = self._fontsize('%sticks' % axis, common=False).get('fontsize')
            if ticksize:
                axis_props['major_label_text_font_size'] = value(ticksize)
            rotation = self.xrotation if axis == 'x' else self.yrotation
            if rotation:
                axis_props['major_label_orientation'] = np.radians(rotation)
            ticker = self.xticks if axis == 'x' else self.yticks
            if isinstance(ticker, Ticker):
                axis_props['ticker'] = ticker
            elif isinstance(ticker, int):
                axis_props['ticker'] = BasicTicker(desired_num_ticks=ticker)
            elif isinstance(ticker, (tuple, list)):
                if all(isinstance(t, tuple) for t in ticker):
                    ticks, labels = zip(*ticker)
                    labels = [l if isinstance(l, util.basestring) else str(l)
                              for l in labels]
                    axis_props['ticker'] = FixedTicker(ticks=ticks)
                    axis_props['major_label_overrides'] = dict(zip(ticks, labels))
                else:
                    axis_props['ticker'] = FixedTicker(ticks=ticker)

        if FuncTickFormatter is not None and ax_mapping and dimension:
            formatter = None
            if dimension.value_format:
                formatter = dimension.value_format
            elif dimension.type in dimension.type_formatters:
                formatter = dimension.type_formatters[dimension.type]
            if formatter:
                msg = ('%s dimension formatter could not be '
                       'converted to tick formatter. ' % dimension.name)
                jsfunc = py2js_tickformatter(formatter, msg)
                if jsfunc:
                    formatter = FuncTickFormatter(code=jsfunc)
                    axis_props['formatter'] = formatter
        return axis_props


    def _update_plot(self, key, plot, element=None):
        """
        Updates plot parameters on every frame
        """
        el = element.traverse(lambda x: x, [Element])
        dimensions = el[0].dimensions() if el else el.dimensions()
        if not len(dimensions) >= 2:
            dimensions = dimensions+[None]
        plot.update(**self._plot_properties(key, plot, element))

        props = {axis: self._axis_properties(axis, key, plot, dim)
                 for axis, dim in zip(['x', 'y'], dimensions)}
        xlabel, ylabel, zlabel = self._get_axis_labels(dimensions)
        if self.invert_axes: xlabel, ylabel = ylabel, xlabel
        props['x']['axis_label'] = xlabel if 'x' in self.labelled else ''
        props['y']['axis_label'] = ylabel if 'y' in self.labelled else ''
        recursive_model_update(plot.xaxis[0], props.get('x', {}))
        recursive_model_update(plot.yaxis[0], props.get('y', {}))

        if plot.title:
            plot.title.update(**self._title_properties(key, plot, element))
        else:
            plot.title = Title(**self._title_properties(key, plot, element))

        if not self.show_grid:
            plot.xgrid.grid_line_color = None
            plot.ygrid.grid_line_color = None
        else:
            replace = ['bounds', 'bands']
            style_items = list(self.gridstyle.items())
            both = {k: v for k, v in style_items if k.startswith('grid_') or k.startswith('minor_grid')}
            xgrid = {k.replace('xgrid', 'grid'): v for k, v in style_items if 'xgrid' in k}
            ygrid = {k.replace('ygrid', 'grid'): v for k, v in style_items if 'ygrid' in k}
            xopts = {k.replace('grid_', '') if any(r in k for r in replace) else k: v
                     for k, v in dict(both, **xgrid).items()}
            yopts = {k.replace('grid_', '') if any(r in k for r in replace) else k: v
                     for k, v in dict(both, **ygrid).items()}
            plot.xgrid[0].update(**xopts)
            plot.ygrid[0].update(**yopts)


    def _update_ranges(self, element, ranges):
        x_range = self.handles['x_range']
        y_range = self.handles['y_range']

        l, b, r, t = None, None, None, None
        if any(isinstance(r, (Range1d, DataRange1d)) for r in [x_range, y_range]):
            l, b, r, t = self.get_extents(element, ranges)
            if self.invert_axes:
                l, b, r, t = b, l, t, r

        xfactors, yfactors = None, None
        if any(isinstance(ax_range, FactorRange) for ax_range in [x_range, y_range]):
            xfactors, yfactors = self._get_factors(element)
        framewise = self.framewise
        streaming = (self.streaming and any(stream._triggering for stream in self.streaming))
        xupdate = ((not self.model_changed(x_range) and (framewise or streaming))
                   or xfactors is not None)
        yupdate = ((not self.model_changed(y_range) and (framewise or streaming))
                   or yfactors is not None)
        if not self.drawn or xupdate:
            self._update_range(x_range, l, r, xfactors, self.invert_xaxis,
                               self._shared['x'], self.logx, streaming)
        if not self.drawn or yupdate:
            self._update_range(y_range, b, t, yfactors, self.invert_yaxis,
                               self._shared['y'], self.logy, streaming)


    def _update_range(self, axis_range, low, high, factors, invert, shared, log, streaming=False):
        if isinstance(axis_range, (Range1d, DataRange1d)) and self.apply_ranges:
            if (low == high and low is not None):
                if isinstance(low, util.datetime_types):
                    offset = np.timedelta64(500, 'ms')
                    low -= offset
                    high += offset
                else:
                    offset = abs(low*0.1 if low else 0.5)
                    low -= offset
                    high += offset
            if invert: low, high = high, low
            if shared:
                shared = (axis_range.start, axis_range.end)
                low, high = util.max_range([(low, high), shared])
            if log and (low is None or low <= 0):
                low = 0.01 if high < 0.01 else 10**(np.log10(high)-2)
                self.warning("Logarithmic axis range encountered value less than or equal to zero, "
                             "please supply explicit lower-bound to override default of %.3f." % low)
            updates = {}
            reset_supported = bokeh_version > '0.12.16'
            if util.isfinite(low):
                updates['start'] = (axis_range.start, low)
                if reset_supported:
                    updates['reset_start'] = updates['start']
            if util.isfinite(high):
                updates['end'] = (axis_range.end, high)
                if reset_supported:
                    updates['reset_end'] = updates['end']
            for k, (old, new) in updates.items():
                axis_range.update(**{k:new})
                if streaming and not k.startswith('reset_'):
                    axis_range.trigger(k, old, new)
        elif isinstance(axis_range, FactorRange):
            factors = list(decode_bytes(factors))
            if invert: factors = factors[::-1]
            axis_range.factors = factors


    def _categorize_data(self, data, cols, dims):
        """
        Transforms non-string or integer types in datasource if the
        axis to be plotted on is categorical. Accepts the column data
        source data, the columns corresponding to the axes and the
        dimensions for each axis, changing the data inplace.
        """
        if self.invert_axes:
            cols = cols[::-1]
            dims = dims[:2][::-1]
        ranges = [self.handles['%s_range' % ax] for ax in 'xy']
        for i, col in enumerate(cols):
            column = data[col]
            if (isinstance(ranges[i], FactorRange) and
                (isinstance(column, list) or column.dtype.kind not in 'SU')):
                data[col] = [dims[i].pprint_value(v) for v in column]

    def _get_factors(self, element):
        """
        Get factors for categorical axes.
        """
        xdim, ydim = element.dimensions()[:2]
        xvals, yvals = [element.dimension_values(i, False)
                        for i in range(2)]
        coords = tuple([v if vals.dtype.kind in 'SU' else dim.pprint_value(v) for v in vals]
                  for dim, vals in [(xdim, xvals), (ydim, yvals)])
        if self.invert_axes: coords = coords[::-1]
        return coords


    def _process_legend(self):
        """
        Disables legends if show_legend is disabled.
        """
        for l in self.handles['plot'].legend:
            l.items[:] = []
            l.border_line_alpha = 0
            l.background_fill_alpha = 0


    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object.
        """
        properties = mpl_to_bokeh(properties)
        plot_method = self._plot_methods.get('batched' if self.batched else 'single')
        if isinstance(plot_method, tuple):
            # Handle alternative plot method for flipped axes
            plot_method = plot_method[int(self.invert_axes)]
        renderer = getattr(plot, plot_method)(**dict(properties, **mapping))
        return renderer, renderer.glyph


    def _glyph_properties(self, plot, element, source, ranges, style):
        properties = dict(style, source=source)
        if self.show_legend:
            if self.overlay_dims:
                legend = ', '.join([d.pprint_value(v) for d, v in
                                    self.overlay_dims.items()])
            else:
                legend = element.label
            if legend:
                properties['legend'] = value(legend)
        return properties


    def _filter_properties(self, properties, glyph_type, allowed):
        glyph_props = dict(properties)
        for gtype in ((glyph_type, '') if glyph_type else ('',)):
            for prop in ('color', 'alpha'):
                glyph_prop = properties.get(gtype+prop)
                if glyph_prop and ('line_'+prop not in glyph_props or gtype):
                    glyph_props['line_'+prop] = glyph_prop
                if glyph_prop and ('fill_'+prop not in glyph_props or gtype):
                    glyph_props['fill_'+prop] = glyph_prop

            props = {k[len(gtype):]: v for k, v in glyph_props.items()
                     if k.startswith(gtype)}
            if self.batched:
                glyph_props = dict(props, **glyph_props)
            else:
                glyph_props.update(props)
        return {k: v for k, v in glyph_props.items() if k in allowed}


    def _update_glyph(self, renderer, properties, mapping, glyph):
        allowed_properties = glyph.properties()
        properties = mpl_to_bokeh(properties)
        merged = dict(properties, **mapping)
        legend = merged.pop('legend', None)
        for glyph_type in ('', 'selection_', 'nonselection_', 'hover_', 'muted_'):
            if renderer:
                glyph = getattr(renderer, glyph_type+'glyph', None)
            if not glyph or (not renderer and glyph_type):
                continue
            filtered = self._filter_properties(merged, glyph_type, allowed_properties)
            glyph.update(**filtered)

        if legend is not None:
            for leg in self.state.legend:
                for item in leg.items:
                    if renderer in item.renderers:
                        item.label = legend


    def _postprocess_hover(self, renderer, source):
        """
        Attaches renderer to hover tool and processes tooltips to
        ensure datetime data is displayed correctly.
        """
        hover = self.handles.get('hover')
        if hover is None:
            return
        if hover.renderers == 'auto':
            hover.renderers = []
        hover.renderers.append(renderer)

        # If datetime column is in the data replace hover formatter
        for k, v in source.data.items():
            if k+'_dt_strings' in source.data:
                tooltips = []
                for name, formatter in hover.tooltips:
                    if formatter == '@{%s}' % k:
                        formatter = '@{%s_dt_strings}' % k
                    tooltips.append((name, formatter))
                hover.tooltips = tooltips


    def _init_glyphs(self, plot, element, ranges, source):
        style_element = element.last if self.batched else element

        # Get data and initialize data source
        if self.batched:
            current_id = tuple(element.traverse(lambda x: x._plot_id, [Element]))
            data, mapping, style = self.get_batched_data(element, ranges)
        else:
            style = self.style[self.cyclic_index]
            data, mapping, style = self.get_data(element, ranges, style)
            current_id = element._plot_id
        if source is None:
            source = self._init_datasource(data)
        self.handles['previous_id'] = current_id
        self.handles['source'] = self.handles['cds'] = source

        properties = self._glyph_properties(plot, style_element, source, ranges, style)
        with abbreviated_exception():
            renderer, glyph = self._init_glyph(plot, mapping, properties)
        self.handles['glyph'] = glyph
        if isinstance(renderer, Renderer):
            self.handles['glyph_renderer'] = renderer

        self._postprocess_hover(renderer, source)

        # Update plot, source and glyph
        with abbreviated_exception():
            self._update_glyph(renderer, properties, mapping, glyph)


    def initialize_plot(self, ranges=None, plot=None, plots=None, source=None):
        """
        Initializes a new plot object with the last available frame.
        """
        # Get element key and ranges for frame
        if self.batched:
            element = [el for el in self.hmap.data.values() if el][-1]
        else:
            element = self.hmap.last
        key = util.wrap_tuple(self.hmap.last_key)
        ranges = self.compute_ranges(self.hmap, key, ranges)
        self.current_ranges = ranges
        self.current_frame = element
        self.current_key = key
        style_element = element.last if self.batched else element
        ranges = util.match_spec(style_element, ranges)
        # Initialize plot, source and glyph
        if plot is None:
            plot = self._init_plot(key, style_element, ranges=ranges, plots=plots)
            self._init_axes(plot)
        else:
            self.handles['xaxis'] = plot.xaxis[0]
            self.handles['x_range'] = plot.x_range
            self.handles['yaxis'] = plot.yaxis[0]
            self.handles['y_range'] = plot.y_range
        self.handles['plot'] = plot

        self._init_glyphs(plot, element, ranges, source)
        if not self.overlaid:
            self._update_plot(key, plot, style_element)
            self._update_ranges(style_element, ranges)

        for cb in self.callbacks:
            cb.initialize()

        if not self.overlaid:
            self._process_legend()
        self._execute_hooks(element)

        self.drawn = True

        return plot


    def _update_glyphs(self, element, ranges):
        plot = self.handles['plot']
        glyph = self.handles.get('glyph')
        source = self.handles['source']
        mapping = {}

        # Cache frame object id to skip updating data if unchanged
        previous_id = self.handles.get('previous_id', None)
        if self.batched:
            current_id = tuple(element.traverse(lambda x: x._plot_id, [Element]))
        else:
            current_id = element._plot_id
        self.handles['previous_id'] = current_id
        self.static_source = (self.dynamic and (current_id == previous_id))
        style = self.style[self.cyclic_index]
        if self.batched:
            data, mapping, style = self.get_batched_data(element, ranges)
        else:
            data, mapping, style = self.get_data(element, ranges, style)

        if not self.static_source:
            self._update_datasource(source, data)

        if glyph:
            properties = self._glyph_properties(plot, element, source, ranges, style)
            renderer = self.handles.get('glyph_renderer')
            with abbreviated_exception():
                self._update_glyph(renderer, properties, mapping, glyph)


    def update_frame(self, key, ranges=None, plot=None, element=None):
        """
        Updates an existing plot with data corresponding
        to the key.
        """
        reused = isinstance(self.hmap, DynamicMap) and (self.overlaid or self.batched)
        if not reused and element is None:
            element = self._get_frame(key)
        elif element is not None:
            self.current_key = key
            self.current_frame = element

        renderer = self.handles.get('glyph_renderer', None)
        glyph = self.handles.get('glyph', None)
        visible = element is not None
        if hasattr(renderer, 'visible'):
            renderer.visible = visible
        if hasattr(glyph, 'visible'):
            glyph.visible = visible

        if ((self.batched and not element) or element is None or (not self.dynamic and self.static) or
            (self.streaming and self.streaming[0].data is self.current_frame.data and not self.streaming[0]._triggering)):
            return

        if self.batched:
            style_element = element.last
            max_cycles = None
        else:
            style_element = element
            max_cycles = self.style._max_cycles
        style = self.lookup_options(style_element, 'style')
        self.style = style.max_cycles(max_cycles) if max_cycles else style

        ranges = self.compute_ranges(self.hmap, key, ranges)
        self.set_param(**self.lookup_options(style_element, 'plot').options)
        ranges = util.match_spec(style_element, ranges)
        self.current_ranges = ranges
        plot = self.handles['plot']
        if not self.overlaid:
            self._update_ranges(style_element, ranges)
            self._update_plot(key, plot, style_element)

        self._update_glyphs(element, ranges)
        self._execute_hooks(element)


    def model_changed(self, model):
        """
        Determines if the bokeh model was just changed on the frontend.
        Useful to suppress boomeranging events, e.g. when the frontend
        just sent an update to the x_range this should not trigger an
        update on the backend.
        """
        callbacks = [cb for cbs in self.traverse(lambda x: x.callbacks)
                             for cb in cbs]
        stream_metadata = [stream._metadata for cb in callbacks
                           for stream in cb.streams if stream._metadata]
        return any(md['id'] == model.ref['id'] for models in stream_metadata
                   for md in models.values())


    @property
    def framewise(self):
        """
        Property to determine whether the current frame should have
        framewise normalization enabled. Required for bokeh plotting
        classes to determine whether to send updated ranges for each
        frame.
        """
        current_frames = [el for f in self.traverse(lambda x: x.current_frame)
                          for el in (f.traverse(lambda x: x, [Element])
                                     if f else [])]
        current_frames = util.unique_iterator(current_frames)
        return any(self.lookup_options(frame, 'norm').options.get('framewise')
                   for frame in current_frames)


class CompositeElementPlot(ElementPlot):
    """
    A CompositeElementPlot is an Element plot type that coordinates
    drawing of multiple glyphs.
    """

    # Mapping between glyph names and style groups
    _style_groups = {}

    # Defines the order in which glyphs are drawn, defined by glyph name
    _draw_order = []

    def _init_glyphs(self, plot, element, ranges, source, data=None, mapping=None, style=None):
        # Get data and initialize data source
        if None in (data, mapping):
            style = self.style[self.cyclic_index]
            data, mapping, style = self.get_data(element, ranges, style)

        keys = glyph_order(dict(data, **mapping), self._draw_order)

        source_cache = {}
        current_id = element._plot_id
        self.handles['previous_id'] = current_id
        for key in keys:
            ds_data = data.get(key, {})
            if id(ds_data) in source_cache:
                source = source_cache[id(ds_data)]
            else:
                source = self._init_datasource(ds_data)
                source_cache[id(ds_data)] = source
            self.handles[key+'_source'] = source
            properties = self._glyph_properties(plot, element, source, ranges, style)
            properties = self._process_properties(key, properties, mapping.get(key, {}))
            with abbreviated_exception():
                renderer, glyph = self._init_glyph(plot, mapping.get(key, {}), properties, key)
            self.handles[key+'_glyph'] = glyph
            if isinstance(renderer, Renderer):
                self.handles[key+'_glyph_renderer'] = renderer

            self._postprocess_hover(renderer, source)

            # Update plot, source and glyph
            with abbreviated_exception():
                self._update_glyph(renderer, properties, mapping.get(key, {}), glyph)


    def _process_properties(self, key, properties, mapping):
        key = '_'.join(key.split('_')[:-1]) if '_' in key else key
        style_group = self._style_groups[key]
        group_props = {}
        for k, v in properties.items():
            if k in self.style_opts:
                group = k.split('_')[0]
                if group == style_group:
                    if k in mapping:
                        v = mapping[k]
                    k = '_'.join(k.split('_')[1:])
                else:
                    continue
            group_props[k] = v
        return group_props


    def _update_glyphs(self, element, ranges):
        plot = self.handles['plot']

        # Cache frame object id to skip updating data if unchanged
        previous_id = self.handles.get('previous_id', None)
        if self.batched:
            current_id = tuple(element.traverse(lambda x: x._plot_id, [Element]))
        else:
            current_id = element._plot_id
        self.handles['previous_id'] = current_id
        self.static_source = (self.dynamic and (current_id == previous_id))
        style = self.style[self.cyclic_index]
        data, mapping, style = self.get_data(element, ranges, style)

        keys = glyph_order(dict(data, **mapping), self._draw_order)
        for key in keys:
            gdata = data.get(key)
            source = self.handles[key+'_source']
            glyph = self.handles.get(key+'_glyph')
            if not self.static_source and gdata is not None:
                self._update_datasource(source, gdata)

            if glyph:
                properties = self._glyph_properties(plot, element, source, ranges, style)
                properties = self._process_properties(key, properties, mapping[key])
                renderer = self.handles.get(key+'_glyph_renderer')
                with abbreviated_exception():
                    self._update_glyph(renderer, properties, mapping[key], glyph)


    def _init_glyph(self, plot, mapping, properties, key):
        """
        Returns a Bokeh glyph object.
        """
        properties = mpl_to_bokeh(properties)
        plot_method = '_'.join(key.split('_')[:-1])
        renderer = getattr(plot, plot_method)(**dict(properties, **mapping))
        return renderer, renderer.glyph



class ColorbarPlot(ElementPlot):
    """
    ColorbarPlot provides methods to create colormappers and colorbar
    models which can be added to a glyph. Additionally it provides
    parameters to control the position and other styling options of
    the colorbar. The default colorbar_position options are defined
    by the colorbar_specs, but may be overridden by the colorbar_opts.
    """

    colorbar_specs = {'right':     {'pos': 'right',
                                    'opts': {'location': (0, 0)}},
                      'left':      {'pos': 'left',
                                    'opts':{'location':(0, 0)}},
                      'bottom':    {'pos': 'below',
                                    'opts': {'location': (0, 0),
                                             'orientation':'horizontal'}},
                      'top':       {'pos': 'above',
                                    'opts': {'location':(0, 0),
                                             'orientation':'horizontal'}},
                      'top_right':   {'pos': 'center',
                                      'opts': {'location': 'top_right'}},
                      'top_left':    {'pos': 'center',
                                      'opts': {'location': 'top_left'}},
                      'bottom_left': {'pos': 'center',
                                      'opts': {'location': 'bottom_left',
                                               'orientation': 'horizontal'}},
                      'bottom_right': {'pos': 'center',
                                      'opts': {'location': 'bottom_right',
                                               'orientation': 'horizontal'}}}

    color_levels = param.ClassSelector(default=None, class_=(int, list), doc="""
        Number of discrete colors to use when colormapping or a set of color
        intervals defining the range of values to map each color to.""")

    colorbar = param.Boolean(default=False, doc="""
        Whether to display a colorbar.""")

    colorbar_position = param.ObjectSelector(objects=list(colorbar_specs),
                                             default="right", doc="""
        Allows selecting between a number of predefined colorbar position
        options. The predefined options may be customized in the
        colorbar_specs class attribute.""")

    colorbar_opts = param.Dict(default={}, doc="""
        Allows setting specific styling options for the colorbar overriding
        the options defined in the colorbar_specs class attribute. Includes
        location, orientation, height, width, scale_alpha, title, title_props,
        margin, padding, background_fill_color and more.""")

    clipping_colors = param.Dict(default={}, doc="""
        Dictionary to specify colors for clipped values, allows
        setting color for NaN values and for values above and below
        the min and max value. The min, max or NaN color may specify
        an RGB(A) color as a color hex string of the form #FFFFFF or
        #FFFFFFFF or a length 3 or length 4 tuple specifying values in
        the range 0-1 or a named HTML color.""")

    logz  = param.Boolean(default=False, doc="""
         Whether to apply log scaling to the z-axis.""")

    symmetric = param.Boolean(default=False, doc="""
        Whether to make the colormap symmetric around zero.""")

    _colorbar_defaults = dict(bar_line_color='black', label_standoff=8,
                              major_tick_line_color='black')

    _default_nan = '#8b8b8b'

    def _draw_colorbar(self, plot, color_mapper):
        if CategoricalColorMapper and isinstance(color_mapper, CategoricalColorMapper):
            return
        if LogColorMapper and isinstance(color_mapper, LogColorMapper) and color_mapper.low > 0:
            ticker = LogTicker()
        else:
            ticker = BasicTicker()
        cbar_opts = dict(self.colorbar_specs[self.colorbar_position])

        # Check if there is a colorbar in the same position
        pos = cbar_opts['pos']
        if any(isinstance(model, ColorBar) for model in getattr(plot, pos, [])):
            return

        opts = dict(cbar_opts['opts'], **self._colorbar_defaults)
        color_bar = ColorBar(color_mapper=color_mapper, ticker=ticker,
                             **dict(opts, **self.colorbar_opts))

        plot.add_layout(color_bar, pos)
        self.handles['colorbar'] = color_bar


    def _get_colormapper(self, dim, element, ranges, style, factors=None, colors=None,
                         name='color_mapper'):
        # The initial colormapper instance is cached the first time
        # and then only updated
        if dim is None and colors is None:
            return None
        if self.adjoined:
            cmappers = self.adjoined.traverse(lambda x: (x.handles.get('color_dim'),
                                                         x.handles.get(name)))
            cmappers = [cmap for cdim, cmap in cmappers if cdim == dim]
            if cmappers:
                cmapper = cmappers[0]
                self.handles['color_mapper'] = cmapper
                return cmapper
            else:
                return None

        ncolors = None if factors is None else len(factors)
        if dim:
            if dim.name in ranges:
                low, high = ranges[dim.name]['combined']
            else:
                low, high = element.range(dim.name)
            if self.symmetric:
                sym_max = max(abs(low), high)
                low, high = -sym_max, sym_max
        else:
            low, high = None, None

        cmap = colors or style.pop('cmap', 'viridis')
        nan_colors = {k: rgba_tuple(v) for k, v in self.clipping_colors.items()}
        if isinstance(cmap, dict) and factors:
            palette = [cmap.get(f, nan_colors.get('NaN', self._default_nan)) for f in factors]
        else:
            categorical = ncolors is not None
            if isinstance(self.color_levels, int):
                ncolors = self.color_levels
            elif isinstance(self.color_levels, list):
                ncolors = len(self.color_levels) - 1
                if isinstance(cmap, list) and len(cmap) != ncolors:
                    raise ValueError('The number of colors in the colormap '
                                     'must match the intervals defined in the '
                                     'color_levels, expected %d colors found %d.'
                                     % (ncolors, len(cmap)))
            palette = process_cmap(cmap, ncolors, categorical=categorical)
            if isinstance(self.color_levels, list):
                palette = color_intervals(palette, self.color_levels, clip=(low, high))
        colormapper, opts = self._get_cmapper_opts(low, high, factors, nan_colors)

        cmapper = self.handles.get(name)
        if cmapper is not None:
            if cmapper.palette != palette:
                cmapper.palette = palette
            opts = {k: opt for k, opt in opts.items()
                    if getattr(cmapper, k) != opt}
            if opts:
                cmapper.update(**opts)
        else:
            cmapper = colormapper(palette=palette, **opts)
            self.handles[name] = cmapper
            self.handles['color_dim'] = dim
        return cmapper


    def _get_color_data(self, element, ranges, style, name='color', factors=None, colors=None,
                        int_categories=False):
        data, mapping = {}, {}
        cdim = element.get_dimension(self.color_index)
        if not cdim:
            return data, mapping

        cdata = element.dimension_values(cdim)
        field = util.dimension_sanitizer(cdim.name)
        dtypes = 'iOSU' if int_categories else 'OSU'
        if factors is None and (isinstance(cdata, list) or cdata.dtype.kind in dtypes):
            factors = list(util.unique_array(cdata))
        if factors and int_categories and cdata.dtype.kind == 'i':
            field += '_str'
            cdata = [str(f) for f in cdata]
            factors = [str(f) for f in factors]

        mapper = self._get_colormapper(cdim, element, ranges, style,
                                       factors, colors)
        data[field] = cdata
        if factors is not None and self.show_legend:
            mapping['legend'] = {'field': field}
        mapping[name] = {'field': field, 'transform': mapper}
        return data, mapping


    def _get_cmapper_opts(self, low, high, factors, colors):
        if factors is None:
            colormapper = LogColorMapper if self.logz else LinearColorMapper
            if isinstance(low, (bool, np.bool_)): low = int(low)
            if isinstance(high, (bool, np.bool_)): high = int(high)
            opts = {}
            if util.isfinite(low):
                opts['low'] = low
            if util.isfinite(high):
                opts['high'] = high
            color_opts = [('NaN', 'nan_color'), ('max', 'high_color'), ('min', 'low_color')]
            opts.update({opt: colors[name] for name, opt in color_opts if name in colors})
        else:
            colormapper = CategoricalColorMapper
            factors = decode_bytes(factors)
            opts = dict(factors=factors)
            if 'NaN' in colors:
                opts['nan_color'] = colors['NaN']
        return colormapper, opts


    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object and optionally creates a colorbar.
        """
        ret = super(ColorbarPlot, self)._init_glyph(plot, mapping, properties)
        if self.colorbar and 'color_mapper' in self.handles:
            self._draw_colorbar(plot, self.handles['color_mapper'])
        return ret


class LegendPlot(ElementPlot):

    legend_position = param.ObjectSelector(objects=["top_right",
                                                    "top_left",
                                                    "bottom_left",
                                                    "bottom_right",
                                                    'right', 'left',
                                                    'top', 'bottom'],
                                                    default="top_right",
                                                    doc="""
        Allows selecting between a number of predefined legend position
        options. The predefined options may be customized in the
        legend_specs class attribute.""")

    legend_muted = param.Boolean(default=False, doc="""
        Controls whether the legend entries are muted by default.""")

    legend_offset = param.NumericTuple(default=(0, 0), doc="""
        If legend is placed outside the axis, this determines the
        (width, height) offset in pixels from the original position.""")

    legend_cols = param.Integer(default=False, doc="""
       Whether to lay out the legend as columns.""")

    legend_specs = {'right': 'right', 'left': 'left', 'top': 'above',
                    'bottom': 'below'}

    def _process_legend(self, plot=None):
        plot = plot or self.handles['plot']
        if not plot.legend:
            return
        legend = plot.legend[0]
        cmapper = self.handles.get('color_mapper')
        if cmapper:
            categorical = isinstance(cmapper, CategoricalColorMapper)
        else:
            categorical = False

        if (not categorical and  not self.overlaid and len(legend.items) == 1) or not self.show_legend:
            legend.items[:] = []
        else:
            plot.legend.orientation = 'horizontal' if self.legend_cols else 'vertical'
            pos = self.legend_position
            if pos in self.legend_specs:
                plot.legend[:] = []
                legend.plot = None
                legend.location = self.legend_offset
                if pos in ['top', 'bottom']:
                    plot.legend.orientation = 'horizontal'
                plot.add_layout(legend, self.legend_specs[pos])
            else:
                legend.location = pos

            # Apply muting
            for leg in plot.legend:
                for item in leg.items:
                    for r in item.renderers:
                        r.muted = self.legend_muted



class OverlayPlot(GenericOverlayPlot, LegendPlot):

    tabs = param.Boolean(default=False, doc="""
        Whether to display overlaid plots in separate panes""")

    style_opts = (legend_dimensions + ['border_'+p for p in line_properties] +
                  text_properties + ['background_fill_color', 'background_fill_alpha'])

    multiple_legends = param.Boolean(default=False, doc="""
        Whether to split the legend for subplots into multiple legends.""")

    _propagate_options = ['width', 'height', 'xaxis', 'yaxis', 'labelled',
                          'bgcolor', 'fontsize', 'invert_axes', 'show_frame',
                          'show_grid', 'logx', 'logy', 'xticks', 'toolbar',
                          'yticks', 'xrotation', 'yrotation', 'lod',
                          'border', 'invert_xaxis', 'invert_yaxis', 'sizing_mode',
                          'title_format', 'legend_position', 'legend_offset',
                          'legend_cols', 'gridstyle', 'legend_muted']


    def _process_legend(self):
        plot = self.handles['plot']
        if not self.show_legend or len(plot.legend) == 0:
            return super(OverlayPlot, self)._process_legend()

        options = {}
        properties = self.lookup_options(self.hmap.last, 'style')[self.cyclic_index]
        for k, v in properties.items():
            if k in line_properties and 'line' not in k:
                ksplit = k.split('_')
                k = '_'.join(ksplit[:1]+'line'+ksplit[1:])
            if k in text_properties:
                k = 'label_' + k
            if k.startswith('legend_'):
                k = k[7:]
            options[k] = v

        if not plot.legend:
            return

        pos = self.legend_position
        orientation = 'horizontal' if self.legend_cols else 'vertical'
        if pos in ['top', 'bottom']:
            orientation = 'horizontal'

        legend_fontsize = self._fontsize('legend', 'size').get('size',False)
        legend = plot.legend[0]
        legend.update(**options)
        if legend_fontsize:
            legend.label_text_font_size = value(legend_fontsize)

        if pos in self.legend_specs:
            pos = self.legend_specs[pos]
        else:
            legend.location = pos

        legend.orientation = orientation

        legend_items = []
        legend_labels = {}
        for item in legend.items:
            label = tuple(item.label.items()) if isinstance(item.label, dict) else item.label
            if not label or (isinstance(item.label, dict) and not item.label.get('value', True)):
                continue
            if label in legend_labels:
                prev_item = legend_labels[label]
                prev_item.renderers += item.renderers
            else:
                legend_labels[label] = item
                legend_items.append(item)
        legend.items[:] = legend_items

        if self.multiple_legends:
            plot.legend.pop(plot.legend.index(legend))
            legend.plot = None
            properties = legend.properties_with_values(include_defaults=False)
            legend_group = []
            for item in legend.items:
                if not isinstance(item.label, dict) or 'value'  in item.label:
                    legend_group.append(item)
                    continue
                new_legend = Legend(**dict(properties, items=[item]))
                new_legend.location = self.legend_offset
                plot.add_layout(new_legend, pos)
            if legend_group:
                new_legend = Legend(**dict(properties, items=legend_group))
                new_legend.location = self.legend_offset
                plot.add_layout(new_legend, pos)
            legend.items[:] = []
        elif pos in ['above', 'below', 'right', 'left']:
            plot.legend.pop(plot.legend.index(legend))
            legend.plot = None
            legend.location = self.legend_offset
            plot.add_layout(legend, pos)

        # Apply muting
        for leg in plot.legend:
            for item in leg.items:
                for r in item.renderers:
                    r.muted = self.legend_muted


    def _init_tools(self, element, callbacks=[]):
        """
        Processes the list of tools to be supplied to the plot.
        """
        tools = []
        hover_tools = {}
        tool_types = []
        for key, subplot in self.subplots.items():
            el = element.get(key)
            if el is not None:
                el_tools = subplot._init_tools(el, self.callbacks)
                for tool in el_tools:
                    if isinstance(tool, util.basestring):
                        tool_type = TOOLS.get(tool)
                    else:
                        tool_type = type(tool)
                    if isinstance(tool, HoverTool):
                        tooltips = tuple(tool.tooltips) if tool.tooltips else ()
                        if tooltips in hover_tools:
                            continue
                        else:
                            hover_tools[tooltips] = tool
                    elif tool_type in tool_types:
                        continue
                    else:
                        tool_types.append(tool_type)
                    tools.append(tool)
        self.handles['hover_tools'] = hover_tools
        return tools


    def _merge_tools(self, subplot):
        """
        Merges tools on the overlay with those on the subplots.
        """
        if self.batched and 'hover' in subplot.handles:
            self.handles['hover'] = subplot.handles['hover']
        elif 'hover' in subplot.handles and 'hover_tools' in self.handles:
            hover = subplot.handles['hover']
            # Datetime formatter may have been applied, remove _dt_strings
            # to match on the hover tooltips, then merge tool renderers
            if hover.tooltips and not isinstance(hover.tooltips, util.basestring):
                tooltips = tuple((name, spec.replace('_dt_strings', ''))
                                  for name, spec in hover.tooltips)
            else:
                tooltips = ()
            tool = self.handles['hover_tools'].get(tooltips)
            if tool:
                tool_renderers = [] if tool.renderers == 'auto' else tool.renderers
                hover_renderers = [] if hover.renderers == 'auto' else hover.renderers
                renderers = tool_renderers + hover_renderers
                tool.renderers = list(util.unique_iterator(renderers))
            if 'hover' not in self.handles:
                self.handles['hover'] = tool


    def _get_factors(self, overlay):
        xfactors, yfactors = [], []
        for k, sp in self.subplots.items():
            el = overlay.data.get(k)
            if el is not None:
                xfs, yfs = sp._get_factors(el)
                xfactors.append(xfs)
                yfactors.append(yfs)
        if xfactors:
            xfactors = np.concatenate(xfactors)
        if yfactors:
            yfactors = np.concatenate(yfactors)
        return util.unique_array(xfactors), util.unique_array(yfactors)


    def initialize_plot(self, ranges=None, plot=None, plots=None):
        key = util.wrap_tuple(self.hmap.last_key)
        nonempty = [el for el in self.hmap.data.values() if el]
        if not nonempty:
            raise SkipRendering('All Overlays empty, cannot initialize plot.')
        element = nonempty[-1]
        ranges = self.compute_ranges(self.hmap, key, ranges)
        if plot is None and not self.tabs and not self.batched:
            plot = self._init_plot(key, element, ranges=ranges, plots=plots)
            self._init_axes(plot)
        self.handles['plot'] = plot

        if plot and not self.overlaid:
            self._update_plot(key, plot, element)
            self._update_ranges(element, ranges)

        panels = []
        for key, subplot in self.subplots.items():
            frame = None
            if self.tabs:
                subplot.overlaid = False
            child = subplot.initialize_plot(ranges, plot, plots)
            if isinstance(element, CompositeOverlay):
                frame = element.get(key, None)
                subplot.current_frame = frame
            if self.batched:
                self.handles['plot'] = child
            if self.tabs:
                title = subplot._format_title(key, dimensions=False)
                if not title:
                    title = get_tab_title(key, frame, self.hmap.last)
                panels.append(Panel(child=child, title=title))
            self._merge_tools(subplot)

        if self.tabs:
            self.handles['plot'] = Tabs(tabs=panels)
        elif not self.overlaid:
            self._process_legend()
        self.drawn = True
        self.handles['plots'] = plots

        self._update_callbacks(self.handles['plot'])
        if 'plot' in self.handles and not self.tabs:
            plot = self.handles['plot']
            self.handles['xaxis'] = plot.xaxis[0]
            self.handles['yaxis'] = plot.yaxis[0]
            self.handles['x_range'] = plot.x_range
            self.handles['y_range'] = plot.y_range
        for cb in self.callbacks:
            cb.initialize()

        if self.top_level:
            self.init_links()

        self._execute_hooks(element)

        return self.handles['plot']


    def update_frame(self, key, ranges=None, element=None):
        """
        Update the internal state of the Plot to represent the given
        key tuple (where integers represent frames). Returns this
        state.
        """
        reused = isinstance(self.hmap, DynamicMap) and self.overlaid
        if not reused and element is None:
            element = self._get_frame(key)
        elif element is not None:
            self.current_frame = element
            self.current_key = key
        items = [] if element is None else list(element.data.items())

        if isinstance(self.hmap, DynamicMap):
            range_obj = element
        else:
            range_obj = self.hmap

        if element is not None:
            ranges = self.compute_ranges(range_obj, key, ranges)

        if element and not self.overlaid and not self.tabs and not self.batched:
            self._update_ranges(element, ranges)

        # Determine which stream (if any) triggered the update
        triggering = [stream for stream in self.streams if stream._triggering]

        for k, subplot in self.subplots.items():
            el = None

            # If in Dynamic mode propagate elements to subplots
            if isinstance(self.hmap, DynamicMap) and element:
                # In batched mode NdOverlay is passed to subplot directly
                if self.batched:
                    el = element
                # If not batched get the Element matching the subplot
                elif element is not None:
                    idx, spec, exact = dynamic_update(self, subplot, k, element, items)
                    if idx is not None:
                        _, el = items.pop(idx)
                        if not exact:
                            self._update_subplot(subplot, spec)

                # Skip updates to subplots when its streams is not one of
                # the streams that initiated the update
                if triggering and all(s not in triggering for s in subplot.streams):
                    continue
            subplot.update_frame(key, ranges, element=el)

        if not self.batched and isinstance(self.hmap, DynamicMap) and items:
            init_kwargs = {'plots': self.handles['plots']}
            if not self.tabs:
                init_kwargs['plot'] = self.handles['plot']
            self._create_dynamic_subplots(key, items, ranges, **init_kwargs)
            if not self.overlaid and not self.tabs:
                self._process_legend()

        if element and not self.overlaid and not self.tabs and not self.batched:
            self._update_plot(key, self.handles['plot'], element)

        self._execute_hooks(element)
