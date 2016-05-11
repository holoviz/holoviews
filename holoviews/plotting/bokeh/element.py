from io import BytesIO

import numpy as np
import bokeh
import bokeh.plotting
from bokeh.models import Range, HoverTool, Renderer
from bokeh.models.tickers import Ticker, BasicTicker, FixedTicker
from bokeh.models.widgets import Panel, Tabs
from distutils.version import LooseVersion

try:
    from bokeh import mpl
except ImportError:
    mpl = None
import param

from ...core import (Store, HoloMap, Overlay, DynamicMap,
                     CompositeOverlay, Element)
from ...core.options import abbreviated_exception
from ...core import util
from ...element import RGB
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import dynamic_update
from .callbacks import Callbacks
from .plot import BokehPlot
from .renderer import bokeh_lt_011
from .util import mpl_to_bokeh, convert_datetime, update_plot


# Define shared style properties for bokeh plots
line_properties = ['line_width', 'line_color', 'line_alpha',
                   'line_join', 'line_cap', 'line_dash']

fill_properties = ['fill_color', 'fill_alpha']

text_properties = ['text_font', 'text_font_size', 'text_font_style', 'text_color',
                   'text_alpha', 'text_align', 'text_baseline']

legend_dimensions = ['label_standoff', 'label_width', 'label_height', 'glyph_width',
                     'glyph_height', 'legend_padding', 'legend_spacing']



class ElementPlot(BokehPlot, GenericElementPlot):

    callbacks = param.ClassSelector(class_=Callbacks, doc="""
        Callbacks object defining any javascript callbacks applied
        to the plot.""")

    bgcolor = param.Parameter(default='white', doc="""
        Background color of the plot.""")

    border = param.Number(default=10, doc="""
        Minimum border around plot.""")

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

    show_grid = param.Boolean(default=True, doc="""
        Whether to show a Cartesian grid on the plot.""")

    show_legend = param.Boolean(default=True, doc="""
        Whether to show legend for the plot.""")

    shared_axes = param.Boolean(default=True, doc="""
        Whether to invert the share axes across plots
        for linked panning and zooming.""")

    default_tools = param.List(default=['save', 'pan', 'wheel_zoom',
                                        'box_zoom', 'resize', 'reset'],
        doc="A list of plugin tools to use on the plot.")

    tools = param.List(default=[], doc="""
        A list of plugin tools to use on the plot.""")

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

    # A string corresponding to the glyph being drawn by the
    # ElementPlot
    _plot_method = None

    # The plot objects to be updated on each frame
    # Any entries should be existing keys in the handles
    # instance attribute.
    _update_handles = ['source', 'glyph']

    def __init__(self, element, plot=None, show_labels=['x', 'y'], **params):
        self.show_labels = show_labels
        self.current_ranges = None
        super(ElementPlot, self).__init__(element, **params)
        self.handles = {} if plot is None else self.handles['plot']
        element_ids = self.hmap.traverse(lambda x: id(x), [Element])
        self.static = len(set(element_ids)) == 1 and len(self.keys) == len(self.hmap)


    def _init_tools(self, element):
        """
        Processes the list of tools to be supplied to the plot.
        """
        tools = self.default_tools + self.tools
        if 'hover' in tools:
            tooltips = [(d.pprint_label, '@'+util.dimension_sanitizer(d.name))
                        for d in element.dimensions()]
            tools[tools.index('hover')] = HoverTool(tooltips=tooltips)
        return tools


    def _get_hover_data(self, data, element, empty=False):
        """
        Initializes hover data based on Element dimension values.
        If empty initializes with no data.
        """
        if 'hover' in self.default_tools + self.tools:
            for d in element.dimensions(label=True):
                sanitized = util.dimension_sanitizer(d)
                data[sanitized] = [] if empty else element.dimension_values(d)


    def _axes_props(self, plots, subplots, element, ranges):
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

        if not 'x_range' in plot_ranges:
            if 'x_range' in ranges:
                plot_ranges['x_range'] = ranges['x_range']
            else:
                l, b, r, t = self.get_extents(element, ranges)
                low, high = (b, t) if self.invert_axes else (l, r)
                if x_axis_type == 'datetime':
                    low = convert_datetime(low)
                    high = convert_datetime(high)
                elif low == high and low is not None:
                    offset = low*0.1 if low else 0.5
                    low -= offset
                    high += offset
                if all(x is not None for x in (low, high)):
                    plot_ranges['x_range'] = [low, high]

        if self.invert_xaxis:
            plot_ranges['x_ranges'] = plot_ranges['x_ranges'][::-1]

        if not 'y_range' in plot_ranges:
            if 'y_range' in ranges:
                plot_ranges['y_range'] = ranges['y_range']
            else:
                l, b, r, t = self.get_extents(element, ranges)
                low, high = (l, r) if self.invert_axes else (b, t)
                if y_axis_type == 'datetime':
                    low = convert_datetime(low)
                    high = convert_datetime(high)
                elif low == high and low is not None:
                    offset = low*0.1 if low else 0.5
                    low -= offset
                    high += offset
                if all(y is not None for y in (low, high)):
                    plot_ranges['y_range'] = [low, high]
        if self.invert_yaxis:
            yrange = plot_ranges['y_range']
            if isinstance(yrange, Range):
                plot_ranges['y_range'] = yrange.__class__(start=yrange.end,
                                                          end=yrange.start)
            else:
                plot_ranges['y_range'] = yrange[::-1]
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
        tools = self._init_tools(element)
        properties = dict(plot_ranges)
        properties['x_axis_label'] = xlabel if 'x' in self.show_labels else ' '
        properties['y_axis_label'] = ylabel if 'y' in self.show_labels else ' '

        if LooseVersion(bokeh.__version__) >= LooseVersion('0.10'):
            properties['webgl'] = self.renderer.webgl
        return bokeh.plotting.Figure(x_axis_type=x_axis_type,
                                     y_axis_type=y_axis_type,
                                     tools=tools, **properties)


    def _plot_properties(self, key, plot, element):
        """
        Returns a dictionary of plot properties.
        """
        title_font = self._fontsize('title', 'title_text_font_size')
        plot_props = dict(plot_height=self.height, plot_width=self.width,
                          title_text_color='black', **title_font)
        if self.show_title:
            plot_props['title'] = self._format_title(key, separator=' ')
        if self.bgcolor:
            bg_attr = 'background_fill'
            if not bokeh_lt_011: bg_attr += '_color'
            plot_props[bg_attr] = self.bgcolor
        if self.border is not None:
            for p in ['left', 'right', 'top', 'bottom']:
                plot_props['min_border_'+p] = self.border
        lod = dict(self.defaults()['lod'], **self.lod)
        for lod_prop, v in lod.items():
            plot_props['lod_'+lod_prop] = v
        return plot_props


    def _init_axes(self, plot):
        if self.xaxis is None:
            plot.xaxis.visible = False
        elif self.xaxis == 'top':
            plot.above = plot.below
            plot.below = []
            plot.xaxis[:] = plot.above

        if self.yaxis is None:
            plot.yaxis.visible = False
        elif self.yaxis == 'right':
            plot.right = plot.left
            plot.left = []
            plot.yaxis[:] = plot.right


    def _axis_properties(self, axis, key, plot, element):
        """
        Returns a dictionary of axis properties depending
        on the specified axis.
        """
        axis_props = {}
        if ((axis == 'x' and self.xaxis in ['bottom-bare', 'top-bare']) or
            (axis == 'y' and self.yaxis in ['left-bare', 'right-bare'])):
            axis_props['axis_label'] = ''
            axis_props['major_label_text_font_size'] = '0pt'
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
        return axis_props


    def _update_plot(self, key, plot, element=None):
        """
        Updates plot parameters on every frame
        """
        plot.set(**self._plot_properties(key, plot, element))
        props = {axis: self._axis_properties(axis, key, plot, element)
                 for axis in ['x', 'y']}
        plot.xaxis[0].set(**props['x'])
        plot.yaxis[0].set(**props['y'])

        if not self.show_grid:
            plot.xgrid.grid_line_color = None
            plot.ygrid.grid_line_color = None


    def _update_ranges(self, element, ranges):
        framewise = self.lookup_options(element, 'norm').options.get('framewise')
        l, b, r, t = self.get_extents(element, ranges)
        if not framewise and not self.dynamic:
            return
        plot = self.handles['plot']
        if self.invert_axes:
            l, b, r, t = b, l, t, r
        if l == r:
            offset = abs(l*0.1 if l else 0.5)
            l -= offset
            r += offset
        if b == t:
            offset = abs(b*0.1 if b else 0.5)
            b -= offset
            t += offset
        plot.x_range.start = l
        plot.x_range.end   = r
        plot.y_range.start = b
        plot.y_range.end   = t


    def _process_legend(self):
        """
        Disables legends if show_legend is disabled.
        """
        if not self.overlaid:
            for l in self.handles['plot'].legend:
                l.legends[:] = []
                l.border_line_alpha = 0
                l.background_fill_alpha = 0


    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object.
        """
        properties = mpl_to_bokeh(properties)
        renderer = getattr(plot, self._plot_method)(**dict(properties, **mapping))
        return renderer, renderer.glyph


    def _glyph_properties(self, plot, element, source, ranges):
        properties = self.style[self.cyclic_index]

        if self.show_legend:
            if self.overlay_dims:
                legend = ', '.join([d.pprint_value_string(v) for d, v in
                                    self.overlay_dims.items()])
            else:
                legend = element.label
            properties['legend'] = legend
        properties['source'] = source
        return properties


    def _update_glyph(self, glyph, properties, mapping):
        allowed_properties = glyph.properties()
        merged = dict(properties, **mapping)
        glyph.set(**{k: v for k, v in merged.items()
                     if k in allowed_properties})


    def initialize_plot(self, ranges=None, plot=None, plots=None, source=None):
        """
        Initializes a new plot object with the last available frame.
        """
        # Get element key and ranges for frame
        element = self.hmap.last
        key = self.keys[-1]
        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = util.match_spec(element, ranges)
        self.current_ranges = ranges
        self.current_frame = element
        self.current_key = key

        # Initialize plot, source and glyph
        if plot is None:
            plot = self._init_plot(key, element, ranges=ranges, plots=plots)
            self._init_axes(plot)
        self.handles['plot'] = plot

        # Get data and initialize data source
        empty = self.callbacks and self.callbacks.downsample
        data, mapping = self.get_data(element, ranges, empty)
        if source is None:
            source = self._init_datasource(data)
        self.handles['source'] = source

        properties = self._glyph_properties(plot, element, source, ranges)
        with abbreviated_exception():
            renderer, glyph = self._init_glyph(plot, mapping, properties)
        self.handles['glyph'] = glyph
        if isinstance(renderer, Renderer):
            self.handles['glyph_renderer'] = renderer

        # Update plot, source and glyph
        with abbreviated_exception():
            self._update_glyph(glyph, properties, mapping)
        if not self.overlaid:
            self._update_plot(key, plot, element)
        if self.callbacks:
            self.callbacks(self)
            self.callbacks.update(self)
        self._process_legend()
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
        if not element:
            return

        if isinstance(self.hmap, DynamicMap):
            ranges = self.compute_ranges(self.hmap, key, ranges)
        else:
            ranges = self.compute_ranges(element, key, ranges)

        self.set_param(**self.lookup_options(element, 'plot').options)
        ranges = util.match_spec(element, ranges)
        self.current_ranges = ranges

        plot = self.handles['plot']
        source = self.handles['source']
        empty = (self.callbacks and self.callbacks.downsample) or empty
        data, mapping = self.get_data(element, ranges, empty)
        self._update_datasource(source, data)

        self.style = self.lookup_options(element, 'style')
        if glyph:
            properties = self._glyph_properties(plot, element, source, ranges)
            with abbreviated_exception():
                self._update_glyph(self.handles['glyph'], properties, mapping)
        if not self.overlaid:
            self._update_ranges(element, ranges)
            self._update_plot(key, plot, element)
        if self.callbacks:
            self.callbacks.update(self)


    @property
    def current_handles(self):
        """
        Returns a list of the plot objects to update.
        """
        handles = []
        if self.static and not self.dynamic:
            return handles
        for handle in self._update_handles:
            if handle in self.handles:
                handles.append(self.handles[handle])

        if self.overlaid:
            return handles

        plot = self.state
        handles.append(plot)
        if self.current_frame:
            framewise = self.lookup_options(self.current_frame, 'norm').options.get('framewise')
            if framewise or isinstance(self.hmap, DynamicMap):
                handles += [plot.x_range, plot.y_range]
        return handles



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
        plot.set(**self._plot_properties(key, plot, element))

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


class OverlayPlot(GenericOverlayPlot, ElementPlot):

    show_legend = param.Boolean(default=True, doc="""
        Whether to show legend for the plot.""")

    legend_position = param.ObjectSelector(objects=["top_right",
                                                    "top_left",
                                                    "bottom_left",
                                                    "bottom_right"],
                                                    default="top_right",
                                                    doc="""
        Allows selecting between a number of predefined legend position
        options. The predefined options may be customized in the
        legend_specs class attribute.""")

    tabs = param.Boolean(default=False, doc="""
        Whether to display overlaid plots in separate panes""")

    style_opts = legend_dimensions + line_properties + text_properties

    _update_handles = ['source']

    def _process_legend(self):
        plot = self.handles['plot']
        if not self.show_legend or len(plot.legend) == 0:
            for l in plot.legend:
                l.legends[:] = []
                l.border_line_alpha = 0
                l.background_fill_alpha = 0
            return

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
        plot.legend[0].set(**options)
        legend_fontsize = self._fontsize('legend', 'size').get('size',False)
        if legend_fontsize:
            plot.legend[0].label_text_font_size = legend_fontsize

        if bokeh_lt_011:
            plot.legend.orientation = self.legend_position
        else:
            plot.legend.location = self.legend_position
        legends = plot.legend[0].legends
        new_legends = []
        for label, l in legends:
            if label in legend_labels:
               continue
            legend_labels.append(label)
            new_legends.append((label, l))
        plot.legend[0].legends[:] = new_legends


    def _init_tools(self, element):
        """
        Processes the list of tools to be supplied to the plot.
        """
        tools = []
        for key, subplot in self.subplots.items():
            try:
                el = element.get(key)
                if el:
                    tools.extend(subplot._init_tools(el))
            except:
                pass
        return list(set(tools))


    def initialize_plot(self, ranges=None, plot=None, plots=None):
        key = self.keys[-1]
        element = self._get_frame(key)
        ranges = self.compute_ranges(self.hmap, key, ranges)
        if plot is None and not self.tabs:
            plot = self._init_plot(key, element, ranges=ranges, plots=plots)
            self._init_axes(plot)
        if plot and not self.overlaid:
            self._update_plot(key, plot, self.hmap.last)
        self.handles['plot'] = plot

        panels = []
        for key, subplot in self.subplots.items():
            if self.tabs: subplot.overlaid = False
            child = subplot.initialize_plot(ranges, plot, plots)
            if self.tabs:
                if self.hmap.type is Overlay:
                    title = ' '.join(key)
                else:
                    title = ', '.join([d.pprint_value_string(k) for d, k in
                                       zip(self.hmap.last.kdims, key)])
                panels.append(Panel(child=child, title=title))
            if isinstance(element, CompositeOverlay):
                frame = element.get(key, None)
                subplot.current_frame = frame

        if self.tabs:
            self.handles['plot'] = Tabs(tabs=panels)
        else:
            self._process_legend()

        self.drawn = True

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

        if isinstance(self.hmap, DynamicMap):
            range_obj = element
            items = element.items()
        else:
            range_obj = self.hmap
            items = element.items()

        all_empty = empty
        ranges = self.compute_ranges(range_obj, key, ranges)
        for k, subplot in self.subplots.items():
            empty, el = False, None
            if isinstance(self.hmap, DynamicMap):
                idx = dynamic_update(self, subplot, k, element, items)
                empty = idx is None
                if not empty:
                    _, el = items.pop(idx)
            subplot.update_frame(key, ranges, element=el, empty=(empty or all_empty))

        if isinstance(self.hmap, DynamicMap) and items:
            raise Exception("Some Elements returned by the dynamic callback "
                            "were not initialized correctly and could not be "
                            "rendered.")

        if not self.overlaid and not self.tabs:
            self._update_ranges(element, ranges)
            self._update_plot(key, self.handles['plot'], element)
