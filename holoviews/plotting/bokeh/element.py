import numpy as np
import bokeh.plotting
from bokeh.models import Range, HoverTool
from bokeh.models.tickers import Ticker, FixedTicker
from bokeh.models.widgets import Panel, Tabs

try:
    from bokeh import mpl
except ImportError:
    mpl = None
import param

from ...core import Store, HoloMap, Overlay
from ...core import util
from ..plot import GenericElementPlot, GenericOverlayPlot
from .plot import BokehPlot
from .util import mpl_to_bokeh


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

    border = param.Number(default=2, doc="""
        Minimum border around plot.""")

    fontsize = param.Parameter(default={'title': '12pt'}, allow_None=True,  doc="""
       Specifies various fontsizes of the displayed text.

       Finer control is available by supplying a dictionary where any
       unmentioned keys reverts to the default sizes, e.g:

          {'ticks': '20pt', 'title': '15pt', 'ylabel': '5px', 'xlabel': '5px'}""")

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

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    shared_axes = param.Boolean(default=True, doc="""
        Whether to invert the share axes across plots
        for linked panning and zooming.""")

    default_tools = param.List(default=['save', 'pan', 'wheel_zoom',
                                        'box_zoom', 'resize', 'reset'],
        doc="A list of plugin tools to use on the plot.")

    tools = param.List(default=[], doc="""
        A list of plugin tools to use on the plot.""")

    xaxis = param.ObjectSelector(default='left',
                                 objects=['left', 'right', 'bare',
                                          'left-bare', 'right-bare',
                                          None], doc="""
        Whether and where to display the xaxis, bare options allow
        suppressing all axis labels including ticks and xlabel.""")

    logx = param.Boolean(default=False, doc="""
        Whether the x-axis of the plot will be a log axis.""")

    xrotation = param.Integer(default=None, bounds=(0, 360), doc="""
        Rotation angle of the xticks.""")

    xticks = param.Parameter(default=None, doc="""
        Ticks along x-axis specified as an integer, explicit list of
        tick locations or bokeh Ticker object. If set to None default
        bokeh ticking behavior is applied.""")

    yaxis = param.ObjectSelector(default='bottom',
                                 objects=['top', 'bottom',
                                          'bare', 'top-bare',
                                          'bottom-bare', None],
                                 doc="""
        Whether and where to display the yaxis, bare options allow
        suppressing all axis labels including ticks and ylabel.""")

    logy = param.Boolean(default=False, doc="""
        Whether the y-axis of the plot will be a log axis.""")

    yrotation = param.Integer(default=None, bounds=(0, 360), doc="""
        Rotation angle of the xticks.""")

    yticks = param.Parameter(default=None, doc="""
        Ticks along y-axis specified as an integer, explicit list of
        tick locations or bokeh Ticker object. If set to None
        default bokeh ticking behavior is applied.""")

    # A string corresponding to the glyph being drawn by the
    # ElementPlot
    _plot_method = None

    def __init__(self, element, plot=None, **params):
        super(ElementPlot, self).__init__(element, **params)
        self.handles = {} if plot is None else self.handles['plot']


    def _init_tools(self, element):
        """
        Processes the list of tools to be supplied to the plot.
        """
        tools = self.default_tools + self.tools
        if 'hover' in tools:
            tooltips = [(d, '@'+d) for d in element.dimensions(label=True)]
            tools[tools.index('hover')] = HoverTool(tooltips=tooltips)
        return tools


    def _init_axes(self, plots, element, ranges):
        xlabel, ylabel, zlabel = self._axis_labels(element, plots)
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

        if not 'x_range' in plot_ranges:
            if 'x_range' in ranges:
                plot_ranges['x_range'] = ranges['x_range']
            else:
                l, _, r, _ = self.get_extents(element, ranges)
                if all(x is not None for x in (l, r)):
                    plot_ranges['x_range'] = [l, r]

        if self.invert_xaxis:
            plot_ranges['x_ranges'] = plot_ranges['x_ranges'][::-1]

        if not 'y_range' in plot_ranges:
            if 'y_range' in ranges:
                plot_ranges['y_range'] = ranges['y_range']
            else:
                _, b, _, t = self.get_extents(element, ranges)
                if all(y is not None for y in (b, t)):
                    plot_ranges['y_range'] = [b, t]
        if self.invert_yaxis:
            yrange = plot_ranges['y_range']
            if isinstance(yrange, Range):
                plot_ranges['y_range'] = yrange.__class__(start=yrange.end,
                                                          end=yrange.start)
            else:
                plot_ranges['y_range'] = yrange[::-1]
        x_axis_type = 'log' if self.logx else 'auto'
        y_axis_type = 'log' if self.logy else 'auto'
        return (x_axis_type, y_axis_type), (xlabel, ylabel, zlabel), plot_ranges


    def _init_plot(self, key, plots, ranges=None):
        """
        Initializes Bokeh figure to draw Element into and sets basic
        figure and axis attributes including axes types, labels,
        titles and plot height and width.
        """

        element = self._get_frame(key)
        subplots = list(self.subplots.values()) if self.subplots else []

        axis_types, labels, plot_ranges = self._init_axes(plots, element, ranges)
        xlabel, ylabel, zlabel = labels
        x_axis_type, y_axis_type = axis_types
        tools = self._init_tools(element)

        return bokeh.plotting.figure(x_axis_type=x_axis_type,
                                     x_axis_label=xlabel,
                                     y_axis_type=y_axis_type,
                                     y_axis_label=ylabel,
                                     tools=tools, **plot_ranges)


    def _plot_properties(self, key, plot, element):
        """
        Returns a dictionary of plot properties.
        """
        title_font = self._fontsize('title', 'title_text_font_size')
        plot_props = dict(plot_height=self.height, plot_width=self.width,
                          title_text_color='black', **title_font)
        if self.show_title:
            plot_props['title'] = self._format_title(key, separator='')
        if self.bgcolor:
            plot_props['background_fill'] = self.bgcolor
        if self.border is not None:
            plot_props['min_border'] = self.border
        lod = dict(self.defaults()['lod'], **self.lod)
        for lod_prop, v in lod.items():
            plot_props['lod_'+lod_prop] = v
        return plot_props


    def _axis_properties(self, axis, key, plot, element):
        """
        Returns a dictionary of axis properties depending
        on the specified axis.
        """
        axis_props = {}
        if ((axis == 'x' and self.xaxis in ['left-bare', None]) or
            (axis == 'y' and self.yaxis in ['bottom-bare', None])):
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
                axis_props['ticker'] = Ticker(desired_num_ticks=ticker)
            elif isinstance(ticker, list):
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
        plot.xaxis[0].set(**self._axis_properties('x', key, plot, element))
        plot.yaxis[0].set(**self._axis_properties('y', key, plot, element))


    def _process_legend(self):
        """
        Disables legends if show_legend is disabled.
        """
        if not self.overlaid and not self.show_legend:
            for l in self.handles['plot'].legend:
                l.legends[:] = []
                l.border_line_alpha = 0


    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object.
        """
        properties = mpl_to_bokeh(properties)
        getattr(plot, self._plot_method)(**dict(properties, **mapping))


    def _glyph_properties(self, plot, element, source, ranges):
        properties = self.style[self.cyclic_index]
        properties['legend'] = element.label
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

        # Initialize plot, source and glyph
        if plot is None:
            plot = self._init_plot(key, ranges=ranges, plots=plots)
        self.handles['plot'] = plot

        data, mapping = self.get_data(element, ranges)
        if source is None:
            source = self._init_datasource(data)
        self.handles['source'] = source

        properties = self._glyph_properties(plot, element, source, ranges)
        self._init_glyph(plot, mapping, properties)
        glyph = plot.renderers[-1].glyph
        self.handles['glyph']  = glyph

        # Update plot, source and glyph
        if not self.overlaid:
            self._update_plot(key, plot, element)
        self._process_legend()
        self.drawn = True

        return plot


    def update_frame(self, key, ranges=None, plot=None):
        """
        Updates an existing plot with data corresponding
        to the key.
        """
        element = self._get_frame(key)
        if not element:
            return

        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = util.match_spec(element, ranges)

        plot = self.handles['plot']
        source = self.handles['source']
        data, mapping = self.get_data(element, ranges)
        self._update_datasource(source, data)
        if not self.overlaid:
            self._update_plot(key, plot, element)



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
        self.mplplot = plot(element, **params)


    def initialize_plot(self, ranges=None, plot=None, plots=None):
        element = self.hmap.last
        key = self.keys[-1]

        self.mplplot.initialize_plot(ranges)
        plot = mpl.to_bokeh(self.mplplot.state)
        self.handles['plot'] = plot
        return plot


    def update_frame(self, key, ranges=None):
        if key in self.hmap:
            self.mplplot.update_frame(key, ranges)
            self.handles['plot'] = mpl.to_bokeh(self.mplplot.state)



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

    def _process_legend(self):
        plot = self.handles['plot']
        if not self.show_legend or len(plot.legend) >= 1:
            for l in plot.legend:
                l.legends[:] = []
                l.border_line_alpha = 0
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
        plot.legend.orientation = self.legend_position
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
        for i, subplot in enumerate(self.subplots.values()):
            tools.extend(subplot._init_tools(element.get(i)))
        return list(set(tools))


    def initialize_plot(self, ranges=None, plot=None, plots=None):
        key = self.keys[-1]
        ranges = self.compute_ranges(self.hmap, key, ranges)
        if plot is None and not self.tabs:
            plot = self._init_plot(key, ranges=ranges, plots=plots)
        if plot and not self.overlaid:
            self._update_plot(key, plot, self.hmap.last)
        self.handles['plot'] = plot

        panels = []
        for key, subplot in self.subplots.items():
            child = subplot.initialize_plot(ranges, plot, plots)
            if self.tabs:
                if self.hmap.type is Overlay:
                    title = ' '.join(key)
                else:
                    title = ', '.join([d.pprint_value_string(k) for d, k in
                                       zip(self.hmap.last.kdims, key)])
                panels.append(Panel(child=child, title=title))

        if self.tabs:
            self.handles['plot'] = Tabs(tabs=panels)
        else:
            self._process_legend()

        self.drawn = True

        return self.handles['plot']


    def update_frame(self, key, ranges=None):
        """
        Update the internal state of the Plot to represent the given
        key tuple (where integers represent frames). Returns this
        state.
        """
        overlay = self._get_frame(key)
        for subplot in self.subplots.values():
            subplot.update_frame(key, ranges)
        if not self.overlaid and not self.tabs:
            self._update_plot(key, self.handles['plot'], overlay)
