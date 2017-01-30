from io import BytesIO
from itertools import groupby

import numpy as np
import bokeh
import bokeh.plotting
from bokeh.core.properties import value
from bokeh.models import Range, HoverTool, Renderer, Range1d, FactorRange
from bokeh.models.tickers import Ticker, BasicTicker, FixedTicker
from bokeh.models.widgets import Panel, Tabs

from bokeh.models.mappers import LinearColorMapper
try:
    from bokeh.models import ColorBar
    from bokeh.models.mappers import LogColorMapper
except ImportError:
    LogColorMapper, ColorBar = None, None
from bokeh.models import LogTicker, BasicTicker
from bokeh.plotting.helpers import _known_tools as known_tools

try:
    from bokeh import mpl
except ImportError:
    mpl = None
import param

from ...core import (Store, HoloMap, Overlay, DynamicMap,
                     CompositeOverlay, Element)
from ...core.options import abbreviated_exception, SkipRendering
from ...core import util
from ...element import RGB
from ...streams import Stream, RangeXY, RangeX, RangeY
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import dynamic_update, get_sources
from .plot import BokehPlot
from .util import (mpl_to_bokeh, convert_datetime, update_plot, get_tab_title,
                   bokeh_version, mplcmap_to_palette, py2js_tickformatter)

if bokeh_version >= '0.12':
    from bokeh.models import FuncTickFormatter
else:
    FuncTickFormatter = None


# Define shared style properties for bokeh plots
line_properties = ['line_width', 'line_color', 'line_alpha',
                   'line_join', 'line_cap', 'line_dash']

fill_properties = ['fill_color', 'fill_alpha']

text_properties = ['text_font', 'text_font_size', 'text_font_style', 'text_color',
                   'text_alpha', 'text_align', 'text_baseline']

legend_dimensions = ['label_standoff', 'label_width', 'label_height', 'glyph_width',
                     'glyph_height', 'legend_padding', 'legend_spacing']



class ElementPlot(BokehPlot, GenericElementPlot):

    bgcolor = param.Parameter(default='white', doc="""
        Background color of the plot.""")

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

    invert_axes = param.Boolean(default=False, doc="""
        Whether to invert the x- and y-axis""")

    invert_xaxis = param.Boolean(default=False, doc="""
        Whether to invert the plot x-axis.""")

    invert_yaxis = param.Boolean(default=False, doc="""
        Whether to invert the plot y-axis.""")

    labelled = param.List(default=['x', 'y'], doc="""
        Whether to plot the 'x' and 'y' labels.""")

    lod = param.Dict(default={'factor': 10, 'interval': 300,
                              'threshold': 2000, 'timeout': 500}, doc="""
        Bokeh plots offer "Level of Detail" (LOD) capability to
        accomodate large (but not huge) amounts of data. The available
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

    show_grid = param.Boolean(default=True, doc="""
        Whether to show a Cartesian grid on the plot.""")

    show_legend = param.Boolean(default=True, doc="""
        Whether to show legend for the plot.""")

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

    xaxis = param.ObjectSelector(default='bottom',
                                 objects=['top', 'bottom', 'bare', 'top-bare',
                                          'bottom-bare', None], doc="""
        Whether and where to display the xaxis, bare options allow suppressing
        all axis labels including ticks and xlabel. Valid options are 'top',
        'bottom', 'bare', 'top-bare' and 'bottom-bare'.""")

    logx = param.Boolean(default=False, doc="""
        Whether the x-axis of the plot will be a log axis.""")

    xrotation = param.Integer(default=None, bounds=(0, 360), doc="""
        Rotation angle of the xticks.""")

    xticks = param.Parameter(default=None, doc="""
        Ticks along x-axis specified as an integer, explicit list of
        tick locations or bokeh Ticker object. If set to None default
        bokeh ticking behavior is applied.""")

    yaxis = param.ObjectSelector(default='left',
                                      objects=['left', 'right', 'bare', 'left-bare',
                                               'right-bare', None], doc="""
        Whether and where to display the yaxis, bare options allow suppressing
        all axis labels including ticks and ylabel. Valid options are 'left',
        'right', 'bare' 'left-bare' and 'right-bare'.""")

    logy = param.Boolean(default=False, doc="""
        Whether the y-axis of the plot will be a log axis.""")

    yrotation = param.Integer(default=None, bounds=(0, 360), doc="""
        Rotation angle of the yticks.""")

    yticks = param.Parameter(default=None, doc="""
        Ticks along y-axis specified as an integer, explicit list of
        tick locations or bokeh Ticker object. If set to None
        default bokeh ticking behavior is applied.""")

    # The plot objects to be updated on each frame
    # Any entries should be existing keys in the handles
    # instance attribute.
    _update_handles = ['source', 'glyph']
    _categorical = False

    def __init__(self, element, plot=None, **params):
        self.current_ranges = None
        super(ElementPlot, self).__init__(element, **params)
        self.handles = {} if plot is None else self.handles['plot']
        self.static = len(self.hmap) == 1 and len(self.keys) == len(self.hmap)
        self.callbacks = self._construct_callbacks()
        self.static_source = False


    def _construct_callbacks(self):
        """
        Initializes any callbacks for streams which have defined
        the plotted object as a source.
        """
        if not self.static or isinstance(self.hmap, DynamicMap):
            sources = [(i, o) for i, o in get_sources(self.hmap)
                       if i in [None, self.zorder]]
        else:
            sources = [(self.zorder, self.hmap.last)]
        cb_classes = set()
        for _, source in sources:
            streams = Stream.registry.get(id(source), [])
            registry = Stream._callbacks['bokeh']
            cb_classes |= {(registry[type(stream)], stream) for stream in streams
                           if type(stream) in registry and streams}
        cbs = []
        sorted_cbs = sorted(cb_classes, key=lambda x: id(x[0]))
        for cb, group in groupby(sorted_cbs, lambda x: x[0]):
            cb_streams = [s for _, s in group]
            cbs.append(cb(self, cb_streams, source))
        return cbs

    def _hover_tooltips(self, element):
        if self.batched:
            dims = list(self.hmap.last.kdims)
        else:
            dims = list(self.overlay_dims.keys())
        dims += element.dimensions()
        return dims

    def _init_tools(self, element, callbacks=[]):
        """
        Processes the list of tools to be supplied to the plot.
        """
        tooltip_dims = self._hover_tooltips(element)
        tooltips = [(d.pprint_label, '@'+util.dimension_sanitizer(d.name))
                    for d in tooltip_dims]

        callbacks = callbacks+self.callbacks
        cb_tools, tool_names = [], []
        for cb in callbacks:
            for handle in cb.handles:
                if handle and handle in known_tools:
                    tool_names.append(handle)
                    if handle == 'hover':
                        tool = HoverTool(tooltips=tooltips)
                    else:
                        tool = known_tools[handle]()
                    cb_tools.append(tool)
                    self.handles[handle] = tool

        tools = [t for t in cb_tools + self.default_tools + self.tools
                 if t not in tool_names]
        if 'hover' in tools:
            tools[tools.index('hover')] = HoverTool(tooltips=tooltips)
        return tools


    def _get_hover_data(self, data, element, empty=False):
        """
        Initializes hover data based on Element dimension values.
        If empty initializes with no data.
        """
        if 'hover' not in self.default_tools + self.tools:
            return

        for d in element.dimensions(label=True):
            sanitized = util.dimension_sanitizer(d)
            data[sanitized] = [] if empty else element.dimension_values(d)
        for k, v in self.overlay_dims.items():
            dim = util.dimension_sanitizer(k.name)
            data[dim] = [v for _ in range(len(data.values()[0]))]


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
        if plots and self.shared_axes:
            for plot in plots:
                if plot is None or not hasattr(plot, 'xaxis'): continue
                if plot.xaxis[0].axis_label == xlabel:
                    plot_ranges['x_range'] = plot.x_range
                if plot.xaxis[0].axis_label == ylabel:
                    plot_ranges['y_range'] = plot.x_range
                if plot.yaxis[0].axis_label == ylabel:
                    plot_ranges['y_range'] = plot.y_range
                if plot.yaxis[0].axis_label == xlabel:
                    plot_ranges['x_range'] = plot.y_range

        if el.get_dimension_type(0) is np.datetime64:
            x_axis_type = 'datetime'
        else:
            x_axis_type = 'log' if self.logx else 'auto'

        if len(dims) > 1 and el.get_dimension_type(1) is np.datetime64:
            y_axis_type = 'datetime'
        else:
            y_axis_type = 'log' if self.logy else 'auto'

        # Get the Element that determines the range and get_extents
        range_el = el if self.batched and not isinstance(self, OverlayPlot) else element
        l, b, r, t = self.get_extents(range_el, ranges)

        categorical = False
        if not 'x_range' in plot_ranges:
            if 'x_range' in ranges:
                plot_ranges['x_range'] = ranges['x_range']
            else:
                low, high = (b, t) if self.invert_axes else (l, r)
                if x_axis_type == 'datetime':
                    low = convert_datetime(low)
                    high = convert_datetime(high)
                elif any(isinstance(x, util.basestring) for x in (low, high)):
                    plot_ranges['x_range'] = FactorRange()
                    categorical = True
                elif low == high and low is not None:
                    offset = low*0.1 if low else 0.5
                    low -= offset
                    high += offset
                if not categorical and all(x is not None and np.isfinite(x) for x in (low, high)):
                    plot_ranges['x_range'] = [low, high]

        if self.invert_xaxis:
            x_range = plot_ranges['x_range']
            if isinstance(x_range, Range1d):
                plot_ranges['x_range'] = x_range.__class__(start=x_range.end,
                                                           end=x_range.start)
            elif not isinstance(x_range, (Range, FactorRange)):
                plot_ranges['x_range'] = x_range[::-1]

        categorical = False
        if not 'y_range' in plot_ranges:
            if 'y_range' in ranges:
                plot_ranges['y_range'] = ranges['y_range']
            else:
                low, high = (l, r) if self.invert_axes else (b, t)
                if y_axis_type == 'datetime':
                    low = convert_datetime(low)
                    high = convert_datetime(high)
                elif any(isinstance(y, util.basestring) for y in (low, high)):
                    plot_ranges['y_range'] = FactorRange()
                    categorical = True
                elif low == high and low is not None:
                    offset = low*0.1 if low else 0.5
                    low -= offset
                    high += offset
                if not categorical and all(y is not None and np.isfinite(y) for y in (low, high)):
                    plot_ranges['y_range'] = [low, high]

        if self.invert_yaxis:
            yrange = plot_ranges['y_range']
            if isinstance(yrange, Range1d):
                plot_ranges['y_range'] = yrange.__class__(start=yrange.end,
                                                          end=yrange.start)
            elif not isinstance(yrange, (Range, FactorRange)):
                plot_ranges['y_range'] = yrange[::-1]

        categorical = any(self.traverse(lambda x: x._categorical))
        if categorical:
            x_axis_type, y_axis_type = 'auto', 'auto'
            plot_ranges['x_range'] = FactorRange()
            plot_ranges['y_range'] = FactorRange()
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

        if self.show_title:
            title = self._format_title(key, separator=' ')
        else:
            title = ''

        if self.toolbar:
            tools = self._init_tools(element)
            properties['tools'] = tools
            properties['toolbar_location'] = self.toolbar

        properties['webgl'] = Store.renderers[self.renderer.backend].webgl
        return bokeh.plotting.Figure(x_axis_type=x_axis_type,
                                     y_axis_type=y_axis_type, title=title,
                                     **properties)


    def _plot_properties(self, key, plot, element):
        """
        Returns a dictionary of plot properties.
        """
        size_multiplier = Store.renderers[self.renderer.backend].size/100.
        plot_props = dict(plot_height=int(self.height*size_multiplier),
                          plot_width=int(self.width*size_multiplier))
        if bokeh_version < '0.12':
            plot_props.update(self._title_properties(key, plot, element))
        if self.bgcolor:
            plot_props['background_fill_color'] = self.bgcolor
        if self.border is not None:
            for p in ['left', 'right', 'top', 'bottom']:
                plot_props['min_border_'+p] = self.border
        lod = dict(self.defaults()['lod'], **self.lod)
        for lod_prop, v in lod.items():
            plot_props['lod_'+lod_prop] = v
        return plot_props


    def _title_properties(self, key, plot, element):
        if self.show_title:
            title = self._format_title(key, separator=' ')
        else:
            title = ''

        if bokeh_version < '0.12':
            title_font = self._fontsize('title', 'title_text_font_size')
            return dict(title=title, title_text_color='black', **title_font)
        else:
            title_font = self._fontsize('title', 'text_font_size')
            title_font['text_font_size'] = value(title_font['text_font_size'])
            return dict(text=title, text_color='black', **title_font)


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
            axis_props['axis_label'] = ''
            axis_props['major_label_text_font_size'] = value('0pt')
            axis_props['major_tick_line_color'] = None
            axis_props['minor_tick_line_color'] = None
        else:
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
                    pass
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
        plot.xaxis[0].update(**props.get('x', {}))
        plot.yaxis[0].update(**props.get('y', {}))

        if bokeh_version >= '0.12' and not self.overlaid:
            plot.title.update(**self._title_properties(key, plot, element))

        if not self.show_grid:
            plot.xgrid.grid_line_color = None
            plot.ygrid.grid_line_color = None


    def _update_ranges(self, element, ranges):
        plot = self.handles['plot']
        x_range = self.handles['x_range']
        y_range = self.handles['y_range']

        if any(isinstance(r, Range1d) for r in [x_range, y_range]):
            l, b, r, t = self.get_extents(element, ranges)
            if self.invert_axes:
                l, b, r, t = b, l, t, r

        if any(isinstance(r, FactorRange) for r in [x_range, y_range]):
            xfactors, yfactors = self._get_factors(element)

        if isinstance(x_range, Range1d):
            if l == r and l is not None:
                offset = abs(l*0.1 if l else 0.5)
                l -= offset
                r += offset

            if self.invert_xaxis: l, r = r, l
            if l is not None and (isinstance(l, np.datetime64) or np.isfinite(l)):
                plot.x_range.start = l
            if r is not None and (isinstance(r, np.datetime64) or np.isfinite(r)):
                plot.x_range.end   = r
        elif isinstance(x_range, FactorRange):
            xfactors = list(xfactors)
            if self.invert_xaxis: xfactors = xfactors[::-1]
            x_range.factors = xfactors

        if isinstance(plot.y_range, Range1d):
            if b == t and b is not None:
                offset = abs(b*0.1 if b else 0.5)
                b -= offset
                t += offset
            if self.invert_yaxis: b, t = t, b
            if b is not None and (isinstance(l, np.datetime64) or np.isfinite(b)):
                plot.y_range.start = b
            if t is not None and (isinstance(l, np.datetime64) or np.isfinite(t)):
                plot.y_range.end   = t
        elif isinstance(y_range, FactorRange):
            yfactors = list(yfactors)
            if self.invert_yaxis: yfactors = yfactors[::-1]
            y_range.factors = yfactors


    def _categorize_data(self, data, cols, dims):
        """
        Transforms non-string or integer types in datasource if the
        axis to be plotted on is categorical. Accepts the column data
        sourcec data, the columns corresponding to the axes and the
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
        coords = ([x if xvals.dtype.kind in 'SU' else xdim.pprint_value(x) for x in xvals],
                  [y if yvals.dtype.kind in 'SU' else ydim.pprint_value(y) for y in yvals])
        if self.invert_axes: coords = coords[::-1]
        return coords


    def _process_legend(self):
        """
        Disables legends if show_legend is disabled.
        """
        for l in self.handles['plot'].legend:
            if bokeh_version > '0.12.2':
                l.items[:] = []
            else:
                l.legends[:] = []
            l.border_line_alpha = 0
            l.background_fill_alpha = 0


    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object.
        """
        properties = mpl_to_bokeh(properties)
        plot_method = self._plot_methods.get('batched' if self.batched else 'single')
        renderer = getattr(plot, plot_method)(**dict(properties, **mapping))
        return renderer, renderer.glyph


    def _glyph_properties(self, plot, element, source, ranges):
        properties = self.style[self.cyclic_index]

        if self.show_legend:
            if self.overlay_dims:
                legend = ', '.join([d.pprint_value(v) for d, v in
                                    self.overlay_dims.items()])
            else:
                legend = element.label
            properties['legend'] = legend
        properties['source'] = source
        return properties


    def _update_glyph(self, glyph, properties, mapping):
        allowed_properties = glyph.properties()
        properties = mpl_to_bokeh(properties)
        merged = dict(properties, **mapping)
        glyph.update(**{k: v for k, v in merged.items()
                        if k in allowed_properties})

    def _execute_hooks(self, element):
        """
        Executes finalize hooks
        """
        for hook in self.finalize_hooks:
            try:
                hook(self, element)
            except Exception as e:
                self.warning("Plotting hook %r could not be applied:\n\n %s" % (hook, e))


    def initialize_plot(self, ranges=None, plot=None, plots=None, source=None):
        """
        Initializes a new plot object with the last available frame.
        """
        # Get element key and ranges for frame
        if self.batched:
            element = [el for el in self.hmap.data.values() if len(el)][-1]
        else:
            element = self.hmap.last
        key = self.keys[-1]
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
            self.handles['y_axis'] = plot.yaxis[0]
            self.handles['y_range'] = plot.y_range
        self.handles['plot'] = plot

        # Get data and initialize data source
        empty = False
        if self.batched:
            data, mapping = self.get_batched_data(element, ranges, empty)
        else:
            data, mapping = self.get_data(element, ranges, empty)
        if source is None:
            source = self._init_datasource(data)
        self.handles['source'] = source

        properties = self._glyph_properties(plot, style_element, source, ranges)
        with abbreviated_exception():
            renderer, glyph = self._init_glyph(plot, mapping, properties)
        self.handles['glyph'] = glyph
        if isinstance(renderer, Renderer):
            self.handles['glyph_renderer'] = renderer

        # Update plot, source and glyph
        with abbreviated_exception():
            self._update_glyph(glyph, properties, mapping)
        if not self.overlaid:
            self._update_plot(key, plot, style_element)
            self._update_ranges(style_element, ranges)

        if not self.batched:
            for cb in self.callbacks:
                cb.initialize()

        if not self.overlaid:
            self._process_legend()
        self._execute_hooks(element)

        self.drawn = True

        return plot


    def update_frame(self, key, ranges=None, plot=None, element=None, empty=False):
        """
        Updates an existing plot with data corresponding
        to the key.
        """
        reused = isinstance(self.hmap, DynamicMap) and self.overlaid
        if not reused and element is None:
            element = self._get_frame(key)
        else:
            self.current_key = key
            self.current_frame = element

        glyph = self.handles.get('glyph', None)
        if hasattr(glyph, 'visible'):
            glyph.visible = bool(element)

        if not element or (not self.dynamic and self.static):
            return

        style_element = element.last if self.batched else element
        self.style = self.lookup_options(style_element, 'style')

        ranges = self.compute_ranges(self.hmap, key, ranges)
        self.set_param(**self.lookup_options(style_element, 'plot').options)
        ranges = util.match_spec(style_element, ranges)
        self.current_ranges = ranges

        plot = self.handles['plot']
        source = self.handles['source']
        empty = False
        mapping = {}

        # Cache frame object id to skip updating data if unchanged
        previous_id = self.handles.get('previous_id', None)
        if self.batched:
            current_id = sum(element.traverse(lambda x: id(x.data), [Element]))
        else:
            current_id = id(element.data)
        self.handles['previous_id'] = current_id
        self.static_source = self.dynamic and (current_id == previous_id)
        if not self.static_source:
            if self.batched:
                data, mapping = self.get_batched_data(element, ranges, empty)
            else:
                data, mapping = self.get_data(element, ranges, empty)
            self._update_datasource(source, data)

        if glyph:
            properties = self._glyph_properties(plot, element, source, ranges)
            with abbreviated_exception():
                self._update_glyph(self.handles['glyph'], properties, mapping)
        if not self.overlaid:
            self._update_ranges(style_element, ranges)
            self._update_plot(key, plot, style_element)

        self._execute_hooks(element)


    @property
    def current_handles(self):
        """
        Returns a list of the plot objects to update.
        """
        handles = []
        if self.static and not self.dynamic:
            return handles

        for handle in self._update_handles:
            if (handle == 'source' and self.static_source):
                continue
            if handle in self.handles:
                handles.append(self.handles[handle])

        if self.overlaid:
            return handles

        plot = self.state
        handles.append(plot)
        if bokeh_version >= '0.12':
            handles.append(plot.title)

        for ax in 'xy':
            key = '%s_range' % ax
            if isinstance(self.handles[key], FactorRange):
                handles.append(self.handles[key])

        if self.current_frame:
            if not self.apply_ranges:
                rangex, rangey = False, False
            elif self.framewise:
                rangex, rangey = True, True
            elif isinstance(self.hmap, DynamicMap):
                rangex, rangey = True, True
                callbacks = [cb for cbs in self.traverse(lambda x: x.callbacks)
                             for cb in cbs]
                streams = [s for cb in callbacks for s in cb.streams]
                for stream in streams:
                    if isinstance(stream, RangeXY):
                        rangex, rangey = False, False
                        break
                    elif isinstance(stream, RangeX):
                        rangex = False
                    elif isinstance(stream, RangeY):
                        rangey = False
            else:
                rangex, rangey = False, False
            if rangex:
                handles += [plot.x_range]
            if rangey:
                handles += [plot.y_range]
        return handles


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
        return any(self.lookup_options(frame, 'norm').options.get('framewise')
                   for frame in current_frames)



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

    _update_handles = ['color_mapper', 'source', 'glyph', 'colorbar']

    _colorbar_defaults = dict(bar_line_color='black', label_standoff=8,
                              major_tick_line_color='black')

    def _draw_colorbar(self, plot, color_mapper):
        if LogColorMapper and isinstance(color_mapper, LogColorMapper):
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


    def _get_colormapper(self, dim, element, ranges, style):
        # The initial colormapper instance is cached the first time
        # and then only updated
        if dim is None:
            return None
        low, high = ranges.get(dim.name, element.range(dim.name))
        palette = mplcmap_to_palette(style.pop('cmap', 'viridis'))
        if self.adjoined:
            cmappers = self.adjoined.traverse(lambda x: (x.handles.get('color_dim'),
                                                         x.handles.get('color_mapper')))
            cmappers = [cmap for cdim, cmap in cmappers if cdim == dim]
            if cmappers:
                cmapper = cmappers[0]
                self.handles['color_mapper'] = cmapper
                return cmapper
            else:
                return None
        colors = self.clipping_colors
        if isinstance(low, (bool, np.bool_)): low = int(low)
        if isinstance(high, (bool, np.bool_)): high = int(high)
        opts = {'low': low, 'high': high}
        color_opts = [('NaN', 'nan_color'), ('max', 'high_color'), ('min', 'low_color')]
        for name, opt in color_opts:
            color = colors.get(name)
            if not color:
                continue
            elif isinstance(color, tuple):
                color = [int(c*255) if i<3 else c for i, c in enumerate(color)]
            opts[opt] = color
        if 'color_mapper' in self.handles:
            cmapper = self.handles['color_mapper']
            cmapper.palette = palette
            cmapper.update(**opts)
        else:
            colormapper = LogColorMapper if self.logz else LinearColorMapper
            cmapper = colormapper(palette, **opts)
            self.handles['color_mapper'] = cmapper
            self.handles['color_dim'] = dim
        return cmapper


    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object and optionally creates a colorbar.
        """
        ret = super(ColorbarPlot, self)._init_glyph(plot, mapping, properties)
        if self.colorbar and 'color_mapper' in self.handles:
            self._draw_colorbar(plot, self.handles['color_mapper'])
        return ret


    def _update_glyph(self, glyph, properties, mapping):
        allowed_properties = glyph.properties()
        merged = dict(properties, **mapping)
        glyph.update(**{k: v for k, v in merged.items()
                        if k in allowed_properties})


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


    legend_cols = param.Integer(default=False, doc="""
       Whether to lay out the legend as columns.""")


    legend_specs = {'right': dict(pos='right', loc=(5, -40)),
                    'left': dict(pos='left', loc=(0, -40)),
                    'top': dict(pos='above', loc=(120, 5)),
                    'bottom': dict(pos='below', loc=(60, 0))}



class BokehMPLWrapper(ElementPlot):
    """
    Wraps an existing HoloViews matplotlib plot and converts
    it to bokeh.
    """

    def __init__(self, element, plot=None, **params):
        super(ElementPlot, self).__init__(element, **params)
        if isinstance(element, HoloMap):
            etype = element.type
        else:
            etype = type(element)
        plot = Store.registry['matplotlib'][etype]
        params = dict({k: v.default for k, v in self.params().items()
                       if k in ['bgcolor']})
        params = dict(params, **self.lookup_options(element, 'plot').options)
        style = self.lookup_options(element, 'style')
        self.mplplot = plot(element, style=style, **params)


    def initialize_plot(self, ranges=None, plot=None, plots=None):
        self.mplplot.initialize_plot(ranges)

        plot = plot if plot else self.handles.get('plot')
        new_plot = mpl.to_bokeh(self.mplplot.state)
        if plot:
            update_plot(plot, new_plot)
        else:
            plot = new_plot

        self.handles['plot'] = plot
        if not self.overlaid:
            self._update_plot(self.keys[-1], plot, self.hmap.last)
        return plot


    def _update_plot(self, key, plot, element=None):
        """
        Updates plot parameters on every frame
        """
        plot.update(**self._plot_properties(key, plot, element))

    def update_frame(self, key, ranges=None, plot=None, element=None, empty=False):
        self.mplplot.update_frame(key, ranges)

        reused = isinstance(self.hmap, DynamicMap) and self.overlaid
        if not reused and element is None:
            element = self._get_frame(key)
        else:
            self.current_key = key
            self.current_frame = element

        plot = mpl.to_bokeh(self.mplplot.state)
        update_plot(self.handles['plot'], plot)
        if not self.overlaid:
            self._update_plot(key, self.handles['plot'], element)


class BokehMPLRawWrapper(BokehMPLWrapper):
    """
    Wraps an existing HoloViews matplotlib plot, renders it as
    an image and displays it as a HoloViews object.
    """

    def initialize_plot(self, ranges=None, plot=None, plots=None):
        element = self.hmap.last
        self.mplplot.initialize_plot(ranges)
        plot = self._render_plot(element, plot)
        self.handles['plot'] = plot
        return plot

    def _render_plot(self, element, plot=None):
        from .raster import RGBPlot
        bytestream = BytesIO()
        renderer = self.mplplot.renderer.instance(dpi=120)
        renderer.save(self.mplplot, bytestream, fmt='png')
        group = ('RGB' if element.group == type(element).__name__ else
                 element.group)
        rgb = RGB.load_image(bytestream, bare=True, group=group,
                             label=element.label)
        plot_opts = self.lookup_options(element, 'plot').options
        rgbplot = RGBPlot(rgb, **plot_opts)
        return rgbplot.initialize_plot(plot=plot)


    def update_frame(self, key, ranges=None, element=None):
        element = self.get_frame(key)
        if key in self.hmap:
            self.mplplot.update_frame(key, ranges)
            self.handles['plot'] = self._render_plot(element)


class OverlayPlot(GenericOverlayPlot, LegendPlot):

    tabs = param.Boolean(default=False, doc="""
        Whether to display overlaid plots in separate panes""")

    style_opts = legend_dimensions + line_properties + text_properties

    _update_handles = ['source']

    _propagate_options = ['width', 'height', 'xaxis', 'yaxis', 'labelled',
                          'bgcolor', 'fontsize', 'invert_axes', 'show_frame',
                          'show_grid', 'logx', 'logy', 'xticks',
                          'yticks', 'xrotation', 'yrotation', 'lod',
                          'border', 'invert_xaxis', 'invert_yaxis']

    def _process_legend(self):
        plot = self.handles['plot']
        if not self.show_legend or len(plot.legend) == 0:
            return super(OverlayPlot, self)._process_legend()

        options = {}
        properties = self.lookup_options(self.hmap.last, 'style')[self.cyclic_index]
        for k, v in properties.items():
            if k in line_properties:
                k = 'border_' + k
            elif k in text_properties:
                k = 'label_' + k
            options[k] = v

        legend_labels = []
        if not plot.legend:
            return
        plot.legend[0].update(**options)
        legend_fontsize = self._fontsize('legend', 'size').get('size',False)
        if legend_fontsize:
            plot.legend[0].label_text_font_size = value(legend_fontsize)

        if self.legend_position not in self.legend_specs:
            plot.legend.location = self.legend_position
        plot.legend.orientation = 'horizontal' if self.legend_cols else 'vertical'
        new_legends = []
        if bokeh_version > '0.12.2':
            legends = plot.legend[0].items
            for item in legends:
                if item.label in legend_labels:
                    continue
                legend_labels.append(item.label)
                new_legends.append(item)
            plot.legend[0].items[:] = new_legends
        else:
            legends = plot.legend[0].legends
            for label, l in legends:
                if label in legend_labels:
                    continue
                legend_labels.append(label)
                new_legends.append((label, l))
            plot.legend[0].legends[:] = new_legends
        if self.legend_position in self.legend_specs:
            legend = plot.legend[0]
            plot.legend[:] = []
            legend.plot = None
            leg_opts = self.legend_specs[self.legend_position]
            legend.location = leg_opts['loc']
            plot.add_layout(plot.legend[0], leg_opts['pos'])


    def _init_tools(self, element, callbacks=[]):
        """
        Processes the list of tools to be supplied to the plot.
        """
        tools = []
        hover = False
        for key, subplot in self.subplots.items():
            el = element.get(key)
            if el is not None:
                el_tools = subplot._init_tools(el, self.callbacks)
                el_tools = [t for t in el_tools
                            if not (isinstance(t, HoverTool) and hover)]
                tools += el_tools
                if any(isinstance(t, HoverTool) for t in el_tools):
                    hover = True
        return list(set(tools))


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
        key = self.keys[-1]
        nonempty = [el for el in self.hmap.data.values() if len(el)]
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
            if self.tabs: subplot.overlaid = False
            child = subplot.initialize_plot(ranges, plot, plots)
            if isinstance(element, CompositeOverlay):
                frame = element.get(key, None)
                subplot.current_frame = frame
            if self.batched:
                self.handles['plot'] = child
            if self.tabs:
                title = get_tab_title(key, frame, self.hmap.last)
                panels.append(Panel(child=child, title=title))

        if self.tabs:
            self.handles['plot'] = Tabs(tabs=panels)
        elif not self.overlaid:
            self._process_legend()
        self.drawn = True

        for cb in self.callbacks:
            cb.initialize()

        self._execute_hooks(element)

        return self.handles['plot']


    def update_frame(self, key, ranges=None, element=None, empty=False):
        """
        Update the internal state of the Plot to represent the given
        key tuple (where integers represent frames). Returns this
        state.
        """
        if element is None:
            element = self._get_frame(key)
        else:
            self.current_frame = element
            self.current_key = key

        items = element.items() if element else []
        if isinstance(self.hmap, DynamicMap):
            range_obj = element
        else:
            range_obj = self.hmap

        all_empty = empty
        ranges = self.compute_ranges(range_obj, key, ranges)
        for k, subplot in self.subplots.items():
            empty, el = False, None
            # If in Dynamic mode propagate elements to subplots
            if isinstance(self.hmap, DynamicMap) and element:
                # In batched mode NdOverlay is passed to subplot directly
                if self.batched:
                    el = element
                    empty = False
                # If not batched get the Element matching the subplot
                else:
                    idx = dynamic_update(self, subplot, k, element, items)
                    empty = idx is None
                    if not empty:
                        _, el = items.pop(idx)
            subplot.update_frame(key, ranges, element=el, empty=(empty or all_empty))

        if not self.batched and isinstance(self.hmap, DynamicMap) and items:
            self.warning("Some Elements returned by the dynamic callback "
                         "were not initialized correctly and could not be "
                         "rendered.")

        if element and not self.overlaid and not self.tabs and not self.batched:
            self._update_ranges(element, ranges)
            self._update_plot(key, self.handles['plot'], element)

        self._execute_hooks(element)
