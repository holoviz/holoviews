from __future__ import absolute_import, division, unicode_literals

import sys
import warnings

from itertools import chain
from types import FunctionType

import param
import numpy as np
import bokeh
import bokeh.plotting

from bokeh.core.properties import value
from bokeh.document.events import ModelChangedEvent
from bokeh.models import (
    ColorBar, ColorMapper, Legend, Renderer, Title, tools
)
from bokeh.models.axes import CategoricalAxis, DatetimeAxis
from bokeh.models.formatters import (
    FuncTickFormatter, TickFormatter, MercatorTickFormatter
)
from bokeh.models.mappers import (
    LinearColorMapper, LogColorMapper, CategoricalColorMapper
)
from bokeh.models.ranges import Range1d, DataRange1d, FactorRange
from bokeh.models.tickers import (
    Ticker, BasicTicker, FixedTicker, LogTicker, MercatorTicker
)
from bokeh.models.tools import Tool
from bokeh.models.widgets import Panel, Tabs

from ...core import DynamicMap, CompositeOverlay, Element, Dimension, Dataset
from ...core.options import abbreviated_exception, SkipRendering
from ...core import util
from ...element import (
    Annotation, Contours, Graph, Path, Tiles, VectorField
)
from ...streams import Buffer, RangeXY, PlotSize
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import process_cmap, color_intervals, dim_range_key
from .callbacks import PlotSizeCallback
from .plot import BokehPlot
from .styles import (
    base_properties, legend_dimensions, line_properties, mpl_to_bokeh,
    property_prefixes, rgba_tuple, text_properties, validate
)
from .tabular import TablePlot
from .util import (
    LooseVersion, TOOL_TYPES, bokeh_version, date_to_integer, decode_bytes, get_tab_title,
    glyph_order, py2js_tickformatter, recursive_model_update,
    theme_attr_json, cds_column_replace, hold_policy, match_dim_specs,
    compute_layout_properties, wrap_formatter, match_ax_type,
    prop_is_none, remove_legend
)

try:
    from bokeh.models import EqHistColorMapper
except ImportError:
    EqHistColorMapper = None

try:
    from bokeh.models import BinnedTicker
except ImportError:
    BinnedTicker = None

if bokeh_version >= LooseVersion('2.0.1'):
    try:
        TOOLS_MAP = Tool._known_aliases
    except Exception:
        TOOLS_MAP = TOOL_TYPES
elif bokeh_version >= LooseVersion('2.0.0'):
    from bokeh.plotting._tools import TOOLS_MAP
else:
    from bokeh.plotting.helpers import _known_tools as TOOLS_MAP



class ElementPlot(BokehPlot, GenericElementPlot):

    active_tools = param.List(default=[], doc="""
        Allows specifying which tools are active by default. Note
        that only one tool per gesture type can be active, e.g.
        both 'pan' and 'box_zoom' are drag tools, so if both are
        listed only the last one will be active.""")

    align = param.ObjectSelector(default='start', objects=['start', 'center', 'end'], doc="""
        Alignment (vertical or horizontal) of the plot in a layout.""")

    border = param.Number(default=10, doc="""
        Minimum border around plot.""")

    aspect = param.Parameter(default=None, doc="""
        The aspect ratio mode of the plot. By default, a plot may
        select its own appropriate aspect ratio but sometimes it may
        be necessary to force a square aspect ratio (e.g. to display
        the plot as an element of a grid). The modes 'auto' and
        'equal' correspond to the axis modes of the same name in
        matplotlib, a numeric value specifying the ratio between plot
        width and height may also be passed. To control the aspect
        ratio between the axis scales use the data_aspect option
        instead.""")

    data_aspect = param.Number(default=None, doc="""
        Defines the aspect of the axis scaling, i.e. the ratio of
        y-unit to x-unit.""")

    width = param.Integer(default=300, allow_None=True, bounds=(0, None), doc="""
        The width of the component (in pixels). This can be either
        fixed or preferred width, depending on width sizing policy.""")

    height = param.Integer(default=300, allow_None=True, bounds=(0, None), doc="""
        The height of the component (in pixels).  This can be either
        fixed or preferred height, depending on height sizing policy.""")

    frame_width = param.Integer(default=None, allow_None=True, bounds=(0, None), doc="""
        The width of the component (in pixels). This can be either
        fixed or preferred width, depending on width sizing policy.""")

    frame_height = param.Integer(default=None, allow_None=True, bounds=(0, None), doc="""
        The height of the component (in pixels).  This can be either
        fixed or preferred height, depending on height sizing policy.""")

    min_width = param.Integer(default=None, bounds=(0, None), doc="""
        Minimal width of the component (in pixels) if width is adjustable.""")

    min_height = param.Integer(default=None, bounds=(0, None), doc="""
        Minimal height of the component (in pixels) if height is adjustable.""")

    max_width = param.Integer(default=None, bounds=(0, None), doc="""
        Minimal width of the component (in pixels) if width is adjustable.""")

    max_height = param.Integer(default=None, bounds=(0, None), doc="""
        Minimal height of the component (in pixels) if height is adjustable.""")

    margin = param.Parameter(default=None, doc="""
        Allows to create additional space around the component. May
        be specified as a two-tuple of the form (vertical, horizontal)
        or a four-tuple (top, right, bottom, left).""")

    responsive = param.ObjectSelector(default=False, objects=[False, True, 'width', 'height'])

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

          * factor    : Decimation factor to use when applying
                        decimation.
          * interval  : Interval (in ms) downsampling will be enabled
                        after an interactive event.
          * threshold : Number of samples before downsampling is enabled.
          * timeout   : Timeout (in ms) for checking whether interactive
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
                                            "left", "right", "disable", None],
                                   doc="""
        The toolbar location, must be one of 'above', 'below',
        'left', 'right', None.""")

    xformatter = param.ClassSelector(
        default=None, class_=(util.basestring, TickFormatter, FunctionType), doc="""
        Formatter for ticks along the x-axis.""")

    yformatter = param.ClassSelector(
        default=None, class_=(util.basestring, TickFormatter, FunctionType), doc="""
        Formatter for ticks along the x-axis.""")

    _categorical = False
    _allow_implicit_categories = True

    # Declare which styles cannot be mapped to a non-scalar dimension
    _nonvectorized_styles = []

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
        self.callbacks, self.source_streams = self._construct_callbacks()
        self.static_source = False
        self.streaming = [s for s in self.streams if isinstance(s, Buffer)]
        self.geographic = bool(self.hmap.last.traverse(lambda x: x, Tiles))
        if self.geographic and self.projection is None:
            self.projection = 'mercator'

        # Whether axes are shared between plots
        self._shared = {'x': False, 'y': False}

        # Flag to check whether plot has been updated
        self._updated = False


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
                if handle and handle in TOOLS_MAP:
                    tool_names.append(handle)
                    if handle == 'hover':
                        tool = tools.HoverTool(
                            tooltips=tooltips, tags=['hv_created'],
                            **hover_opts)
                        hover = tool
                    else:
                        tool = TOOLS_MAP[handle]()
                    cb_tools.append(tool)
                    self.handles[handle] = tool

        tool_list = [
            t for t in cb_tools + self.default_tools + self.tools
            if t not in tool_names]

        tool_list = [
            tools.HoverTool(tooltips=tooltips, tags=['hv_created'], mode=tl, **hover_opts)
            if tl in ['vline', 'hline'] else tl for tl in tool_list
        ]

        copied_tools = []
        for tool in tool_list:
            if isinstance(tool, tools.Tool):
                properties = tool.properties_with_values(include_defaults=False)
                tool = type(tool)(**properties)
            copied_tools.append(tool)

        hover_tools = [t for t in copied_tools if isinstance(t, tools.HoverTool)]
        if 'hover' in copied_tools:
            hover = tools.HoverTool(tooltips=tooltips, tags=['hv_created'], **hover_opts)
            copied_tools[copied_tools.index('hover')] = hover
        elif any(hover_tools):
            hover = hover_tools[0]
        if hover:
            self.handles['hover'] = hover

        box_tools = [t for t in copied_tools if isinstance(t, tools.BoxSelectTool)]
        if box_tools:
            self.handles['box_select'] = box_tools[0]
        lasso_tools = [t for t in copied_tools if isinstance(t, tools.LassoSelectTool)]
        if lasso_tools:
            self.handles['lasso_select'] = lasso_tools[0]

        # Link the selection properties between tools
        if box_tools and lasso_tools:
            box_tools[0].js_link('mode', lasso_tools[0], 'mode')
            lasso_tools[0].js_link('mode', box_tools[0], 'mode')

        return copied_tools

    def _update_hover(self, element):
        tool = self.handles['hover']
        if 'hv_created' in tool.tags:
            tooltips, hover_opts = self._hover_opts(element)
            tooltips = [(ttp.pprint_label, '@{%s}' % util.dimension_sanitizer(ttp.name))
                        if isinstance(ttp, Dimension) else ttp for ttp in tooltips]
            tool.tooltips = tooltips
        else:
            plot_opts = element.opts.get('plot', 'bokeh')
            new_hover = [t for t in plot_opts.kwargs.get('tools', [])
                         if isinstance(t, tools.HoverTool)]
            if new_hover:
                tool.tooltips = new_hover[0].tooltips

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

        for k, v in self.overlay_dims.items():
            dim = util.dimension_sanitizer(k.name)
            if dim not in data:
                data[dim] = [v for _ in range(len(list(data.values())[0]))]


    def _merge_ranges(self, plots, xspecs, yspecs, xtype, ytype):
        """
        Given a list of other plots return axes that are shared
        with another plot by matching the dimensions specs stored
        as tags on the dimensions.
        """
        plot_ranges = {}
        for plot in plots:
            if plot is None:
                continue
            if hasattr(plot, 'x_range') and plot.x_range.tags and xspecs is not None:
                if match_dim_specs(plot.x_range.tags[0], xspecs) and match_ax_type(plot.xaxis, xtype):
                    plot_ranges['x_range'] = plot.x_range
                if match_dim_specs(plot.x_range.tags[0], yspecs) and match_ax_type(plot.xaxis, ytype):
                    plot_ranges['y_range'] = plot.x_range
            if hasattr(plot, 'y_range') and plot.y_range.tags and yspecs is not None:
                if match_dim_specs(plot.y_range.tags[0], yspecs) and match_ax_type(plot.yaxis, ytype):
                    plot_ranges['y_range'] = plot.y_range
                if match_dim_specs(plot.y_range.tags[0], xspecs) and match_ax_type(plot.yaxis, xtype):
                    plot_ranges['x_range'] = plot.y_range
        return plot_ranges


    def _get_axis_dims(self, element):
        """Returns the dimensions corresponding to each axis.

        Should return a list of dimensions or list of lists of
        dimensions, which will be formatted to label the axis
        and to link axes.
        """
        dims = element.dimensions()[:2]
        if len(dims) == 1:
            return dims + [None, None]
        else:
            return dims + [None]


    def _axes_props(self, plots, subplots, element, ranges):
        # Get the bottom layer and range element
        el = element.traverse(lambda x: x, [lambda el: isinstance(el, Element) and not isinstance(el, (Annotation, Tiles))])
        el = el[0] if el else element
        if isinstance(el, Graph):
            el = el.nodes

        dims = self._get_axis_dims(el)
        xlabel, ylabel, zlabel = self._get_axis_labels(dims)
        if self.invert_axes:
            xlabel, ylabel = ylabel, xlabel
            dims = dims[:2][::-1]
        xdims, ydims = dims[:2]
        if xdims:
            if not isinstance(xdims, list):
                xdims = [xdims]
            xspecs = tuple((xd.name, xd.label, xd.unit) for xd in xdims)
        else:
            xspecs = None
        if ydims:
            if not isinstance(ydims, list):
                ydims = [ydims]
            yspecs = tuple((yd.name, yd.label, yd.unit) for yd in ydims)
        else:
            yspecs = None

        # Get the Element that determines the range and get_extents
        range_el = el if self.batched and not isinstance(self, OverlayPlot) else element
        l, b, r, t = self.get_extents(range_el, ranges)
        if self.invert_axes:
            l, b, r, t = b, l, t, r

        categorical = any(self.traverse(lambda x: x._categorical))
        if xdims is not None and any(xdim.name in ranges and 'factors' in ranges[xdim.name] for xdim in xdims):
            categorical_x = True
        else:
            categorical_x = any(isinstance(x, (util.basestring, bytes)) for x in (l, r))

        if ydims is not None and any(ydim.name in ranges and 'factors' in ranges[ydim.name] for ydim in ydims):
            categorical_y = True
        else:
            categorical_y = any(isinstance(y, (util.basestring, bytes)) for y in (b, t))

        range_types = (self._x_range_type, self._y_range_type)
        if self.invert_axes: range_types = range_types[::-1]
        x_range_type, y_range_type = range_types
        x_axis_type = 'log' if self.logx else 'auto'
        if xdims:
            if len(xdims) > 1 or x_range_type is FactorRange:
                x_axis_type = 'auto'
                categorical_x = True
            else:
                xtype = el.get_dimension_type(xdims[0])
                if ((xtype is np.object_ and issubclass(type(l), util.datetime_types)) or
                    xtype in util.datetime_types):
                    x_axis_type = 'datetime'

        y_axis_type = 'log' if self.logy else 'auto'
        if ydims:
            if len(ydims) > 1 or y_range_type is FactorRange:
                y_axis_type = 'auto'
                categorical_y = True
            else:
                if isinstance(el, Graph):
                    ytype = el.nodes.get_dimension_type(ydims[0])
                else:
                    ytype = el.get_dimension_type(ydims[0])
                if ((ytype is np.object_ and issubclass(type(b), util.datetime_types))
                    or ytype in util.datetime_types):
                    y_axis_type = 'datetime'

        plot_ranges = {}
        # Try finding shared ranges in other plots in the same Layout
        norm_opts = self.lookup_options(el, 'norm').options
        if plots and self.shared_axes and not norm_opts.get('axiswise', False):
            plot_ranges = self._merge_ranges(plots, xspecs, yspecs, x_axis_type, y_axis_type)

        # Declare shared axes
        x_range, y_range = plot_ranges.get('x_range'), plot_ranges.get('y_range')
        if x_range and not (x_range_type is FactorRange and not isinstance(x_range, FactorRange)):
            self._shared['x'] = True
        if y_range and not (y_range_type is FactorRange and not isinstance(y_range, FactorRange)):
            self._shared['y'] = True

        if self._shared['x']:
            pass
        elif categorical or categorical_x:
            x_axis_type = 'auto'
            plot_ranges['x_range'] = FactorRange()
        else:
            plot_ranges['x_range'] = x_range_type()

        if self._shared['y']:
            pass
        elif categorical or categorical_y:
            y_axis_type = 'auto'
            plot_ranges['y_range'] = FactorRange()
        elif 'y_range' not in plot_ranges:
            plot_ranges['y_range'] = y_range_type()

        x_range, y_range = plot_ranges['x_range'], plot_ranges['y_range']
        if not x_range.tags and xspecs is not None:
            x_range.tags.append(xspecs)
        if not y_range.tags and yspecs is not None:
            y_range.tags.append(yspecs)

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
        properties['x_axis_label'] = xlabel if 'x' in self.labelled or self.xlabel else ' '
        properties['y_axis_label'] = ylabel if 'y' in self.labelled or self.ylabel else ' '

        if not self.show_frame:
            properties['outline_line_alpha'] = 0

        if self.show_title and self.adjoined is None:
            title = self._format_title(key, separator=' ')
        else:
            title = ''

        if self.toolbar != 'disable':
            tools = self._init_tools(element)
            properties['tools'] = tools
            properties['toolbar_location'] = self.toolbar
        else:
            properties['tools'] = []
            properties['toolbar_location'] = None

        if self.renderer.webgl:
            properties['output_backend'] = 'webgl'

        properties.update(**self._plot_properties(key, element))

        with warnings.catch_warnings():
            # Bokeh raises warnings about duplicate tools but these
            # are not really an issue
            warnings.simplefilter('ignore', UserWarning)
            return bokeh.plotting.Figure(x_axis_type=x_axis_type,
                                         y_axis_type=y_axis_type, title=title,
                                         **properties)


    def _plot_properties(self, key, element):
        """
        Returns a dictionary of plot properties.
        """
        init = 'plot' not in self.handles
        size_multiplier = self.renderer.size/100.
        options = self._traverse_options(element, 'plot', ['width', 'height'], defaults=False)

        logger = self.param if init else None
        aspect_props, dimension_props = compute_layout_properties(
            self.width, self.height, self.frame_width, self.frame_height,
            options.get('width'), options.get('height'), self.aspect, self.data_aspect,
            self.responsive, size_multiplier, logger=logger)

        if not init:
            if aspect_props['aspect_ratio'] is None:
                aspect_props['aspect_ratio'] = self.state.aspect_ratio

        if self.dynamic and aspect_props['match_aspect']:
            # Sync the plot size on dynamic plots to support accurate
            # scaling of dimension ranges
            plot_size = [s for s in self.streams if isinstance(s, PlotSize)]
            callbacks = [c for c in self.callbacks if isinstance(c, PlotSizeCallback)]
            if plot_size:
                stream = plot_size[0]
            elif callbacks:
                stream = callbacks[0].streams[0]
            else:
                stream = PlotSize()
                self.callbacks.append(PlotSizeCallback(self, [stream], None))
            stream.add_subscriber(self._update_size)

        plot_props = {
            'align':         self.align,
            'margin':        self.margin,
            'max_width':     self.max_width,
            'max_height':    self.max_height,
            'min_width':     self.min_width,
            'min_height':    self.min_height
        }
        plot_props.update(aspect_props)
        if not self.drawn:
            plot_props.update(dimension_props)

        if self.bgcolor:
            plot_props['background_fill_color'] = self.bgcolor
        if self.border is not None:
            for p in ['left', 'right', 'top', 'bottom']:
                plot_props['min_border_'+p] = self.border
        lod = dict(self.param.defaults().get('lod', {}), **self.lod)
        for lod_prop, v in lod.items():
            plot_props['lod_'+lod_prop] = v
        return plot_props

    def _update_size(self, width, height, scale):
        if self.renderer.mode == 'server':
            return
        self.state.frame_width = width
        self.state.frame_height = height

    def _set_active_tools(self, plot):
        "Activates the list of active tools"
        for tool in self.active_tools:
            if isinstance(tool, util.basestring):
                tool_type = TOOL_TYPES[tool]
                matching = [t for t in plot.toolbar.tools
                            if isinstance(t, tool_type)]
                if not matching:
                    self.param.warning('Tool of type %r could not be found '
                                       'and could not be activated by default.'
                                       % tool)
                    continue
                tool = matching[0]
            if isinstance(tool, tools.Drag):
                plot.toolbar.active_drag = tool
            if isinstance(tool, tools.Scroll):
                plot.toolbar.active_scroll = tool
            if isinstance(tool, tools.Tap):
                plot.toolbar.active_tap = tool
            if isinstance(tool, tools.Inspection):
                plot.toolbar.active_inspect.append(tool)


    def _title_properties(self, key, plot, element):
        if self.show_title and self.adjoined is None:
            title = self._format_title(key, separator=' ')
        else:
            title = ''
        opts = dict(text=title)

        # this will override theme if not set to the default 12pt
        title_font = self._fontsize('title').get('fontsize')
        if title_font != '12pt':
            title_font = title_font if bokeh_version > LooseVersion('2.2.3') else value(title_font)
            opts['text_font_size'] = title_font
        return opts


    def _init_axes(self, plot):
        if self.xaxis is None:
            plot.xaxis.visible = False
        elif isinstance(self.xaxis, util.basestring) and 'top' in self.xaxis:
            plot.above = plot.below
            plot.below = []
            plot.xaxis[:] = plot.above
        self.handles['xaxis'] = plot.xaxis[0]
        self.handles['x_range'] = plot.x_range

        if self.yaxis is None:
            plot.yaxis.visible = False
        elif isinstance(self.yaxis, util.basestring) and'right' in self.yaxis:
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
        # need to copy dictionary by calling dict() on it
        axis_props = dict(theme_attr_json(self.renderer.theme, 'Axis'))

        if ((axis == 'x' and self.xaxis in ['bottom-bare', 'top-bare', 'bare']) or
            (axis == 'y' and self.yaxis in ['left-bare', 'right-bare', 'bare'])):
            zero_pt = '0pt' if bokeh_version > LooseVersion('2.2.3') else value('0pt')
            axis_props['axis_label_text_font_size'] = zero_pt
            axis_props['major_label_text_font_size'] = zero_pt
            axis_props['major_tick_line_color'] = None
            axis_props['minor_tick_line_color'] = None
        else:
            labelsize = self._fontsize('%slabel' % axis).get('fontsize')
            if labelsize:
                axis_props['axis_label_text_font_size'] = labelsize
            ticksize = self._fontsize('%sticks' % axis, common=False).get('fontsize')
            if ticksize:
                ticksize = ticksize if bokeh_version > LooseVersion('2.2.3') else value(ticksize)
                axis_props['major_label_text_font_size'] = ticksize
            rotation = self.xrotation if axis == 'x' else self.yrotation
            if rotation:
                axis_props['major_label_orientation'] = np.radians(rotation)
            ticker = self.xticks if axis == 'x' else self.yticks
            if isinstance(ticker, np.ndarray):
                ticker = list(ticker)
            if isinstance(ticker, Ticker):
                axis_props['ticker'] = ticker
            elif isinstance(ticker, int):
                axis_props['ticker'] = BasicTicker(desired_num_ticks=ticker)
            elif isinstance(ticker, (tuple, list)):
                if all(isinstance(t, tuple) for t in ticker):
                    ticks, labels = zip(*ticker)
                    # Ensure floats which are integers are serialized as ints
                    # because in JS the lookup fails otherwise
                    ticks = [int(t) if isinstance(t, float) and t.is_integer() else t
                             for t in ticks]
                    labels = [l if isinstance(l, util.basestring) else str(l)
                              for l in labels]
                else:
                    ticks, labels = ticker, None
                if ticks and util.isdatetime(ticks[0]):
                    ticks = [util.dt_to_int(tick, 'ms') for tick in ticks]
                axis_props['ticker'] = FixedTicker(ticks=ticks)
                if labels is not None:
                    axis_props['major_label_overrides'] = dict(zip(ticks, labels))
        formatter = self.xformatter if axis == 'x' else self.yformatter
        if formatter:
            formatter = wrap_formatter(formatter, axis)
            if formatter is not None:
                axis_props['formatter'] = formatter
        elif FuncTickFormatter is not None and ax_mapping and isinstance(dimension, Dimension):
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

        if axis == 'x':
            axis_obj = plot.xaxis[0]
        elif axis == 'y':
            axis_obj = plot.yaxis[0]

        if (self.geographic and isinstance(self.projection, str)
            and self.projection == 'mercator'):
            dimension = 'lon' if axis == 'x' else 'lat'
            axis_props['ticker'] = MercatorTicker(dimension=dimension)
            axis_props['formatter'] = MercatorTickFormatter(dimension=dimension)
            box_zoom = self.state.select(type=tools.BoxZoomTool)
            if box_zoom:
                box_zoom[0].match_aspect = True
            wheel_zoom = self.state.select(type=tools.WheelZoomTool)
            if wheel_zoom:
                wheel_zoom[0].zoom_on_axis = False
        elif isinstance(axis_obj, CategoricalAxis):
            for key in list(axis_props):
                if key.startswith('major_label'):
                    # set the group labels equal to major (actually minor)
                    new_key = key.replace('major_label', 'group')
                    axis_props[new_key] = axis_props[key]

            # major ticks are actually minor ticks in a categorical
            # so if user inputs minor ticks sizes, then use that;
            # else keep major (group) == minor (subgroup)
            msize = self._fontsize('minor_{0}ticks'.format(axis),
                common=False).get('fontsize')
            if msize is not None:
                axis_props['major_label_text_font_size'] = msize

        return axis_props


    def _update_plot(self, key, plot, element=None):
        """
        Updates plot parameters on every frame
        """
        plot.update(**self._plot_properties(key, element))
        self._update_labels(key, plot, element)
        self._update_title(key, plot, element)
        self._update_grid(plot)


    def _update_labels(self, key, plot, element):
        el = element.traverse(lambda x: x, [Element])
        el = el[0] if el else element
        dimensions = self._get_axis_dims(el)
        props = {axis: self._axis_properties(axis, key, plot, dim)
                 for axis, dim in zip(['x', 'y'], dimensions)}
        xlabel, ylabel, zlabel = self._get_axis_labels(dimensions)
        if self.invert_axes:
            xlabel, ylabel = ylabel, xlabel
        props['x']['axis_label'] = xlabel if 'x' in self.labelled or self.xlabel else ''
        props['y']['axis_label'] = ylabel if 'y' in self.labelled or self.ylabel else ''
        recursive_model_update(plot.xaxis[0], props.get('x', {}))
        recursive_model_update(plot.yaxis[0], props.get('y', {}))


    def _update_title(self, key, plot, element):
        if plot.title:
            plot.title.update(**self._title_properties(key, plot, element))
        else:
            plot.title = Title(**self._title_properties(key, plot, element))


    def _update_grid(self, plot):
        if not self.show_grid:
            plot.xgrid.grid_line_color = None
            plot.ygrid.grid_line_color = None
            return
        replace = ['bounds', 'bands', 'visible', 'level', 'ticker', 'visible']
        style_items = list(self.gridstyle.items())
        both = {k: v for k, v in style_items if k.startswith('grid_') or k.startswith('minor_grid')}
        xgrid = {k.replace('xgrid', 'grid'): v for k, v in style_items if 'xgrid' in k}
        ygrid = {k.replace('ygrid', 'grid'): v for k, v in style_items if 'ygrid' in k}
        xopts = {k.replace('grid_', '') if any(r in k for r in replace) else k: v
                 for k, v in dict(both, **xgrid).items()}
        yopts = {k.replace('grid_', '') if any(r in k for r in replace) else k: v
                 for k, v in dict(both, **ygrid).items()}
        if plot.xaxis and 'ticker' not in xopts:
            xopts['ticker'] = plot.xaxis[0].ticker
        if plot.yaxis and 'ticker' not in yopts:
            yopts['ticker'] = plot.yaxis[0].ticker
        plot.xgrid[0].update(**xopts)
        plot.ygrid[0].update(**yopts)


    def _update_ranges(self, element, ranges):
        plot = self.handles['plot']
        x_range = self.handles['x_range']
        y_range = self.handles['y_range']

        l, b, r, t = None, None, None, None
        if any(isinstance(r, (Range1d, DataRange1d)) for r in [x_range, y_range]):
            l, b, r, t = self.get_extents(element, ranges)
            if self.invert_axes:
                l, b, r, t = b, l, t, r

        xfactors, yfactors = None, None
        if any(isinstance(ax_range, FactorRange) for ax_range in [x_range, y_range]):
            xfactors, yfactors = self._get_factors(element, ranges)
        framewise = self.framewise
        streaming = (self.streaming and any(stream._triggering and stream.following
                                            for stream in self.streaming))
        xupdate = ((not (self.model_changed(x_range) or self.model_changed(plot))
                    and (framewise or streaming))
                   or xfactors is not None)
        yupdate = ((not (self.model_changed(x_range) or self.model_changed(plot))
                    and (framewise or streaming))
                   or yfactors is not None)

        options = self._traverse_options(element, 'plot', ['width', 'height'], defaults=False)
        fixed_width = (self.frame_width or options.get('width'))
        fixed_height = (self.frame_height or options.get('height'))
        constrained_width = options.get('min_width') or options.get('max_width')
        constrained_height = options.get('min_height') or options.get('max_height')

        data_aspect = (self.aspect == 'equal' or self.data_aspect)
        xaxis, yaxis = self.handles['xaxis'], self.handles['yaxis']
        categorical = isinstance(xaxis, CategoricalAxis) or isinstance(yaxis, CategoricalAxis)
        datetime = isinstance(xaxis, DatetimeAxis) or isinstance(yaxis, CategoricalAxis)

        if data_aspect and (categorical or datetime):
            ax_type = 'categorical' if categorical else 'datetime axes'
            self.param.warning('Cannot set data_aspect if one or both '
                               'axes are %s, the option will '
                               'be ignored.' % ax_type)
        elif data_aspect:
            plot = self.handles['plot']
            xspan = r-l if util.is_number(l) and util.is_number(r) else None
            yspan = t-b if util.is_number(b) and util.is_number(t) else None

            if self.drawn or (fixed_width and fixed_height) or (constrained_width or constrained_height):
                # After initial draw or if aspect is explicit
                # adjust range to match the plot dimension aspect
                ratio = self.data_aspect or 1
                if self.aspect == 'square':
                    frame_aspect = 1
                elif self.aspect and self.aspect != 'equal':
                    frame_aspect = self.aspect
                elif plot.frame_height and plot.frame_width:
                    frame_aspect = plot.frame_height/plot.frame_width
                else:
                    # Cannot force an aspect until we know the frame size
                    return

                range_streams = [s for s in self.streams if isinstance(s, RangeXY)]
                if self.drawn:
                    current_l, current_r = plot.x_range.start, plot.x_range.end
                    current_b, current_t = plot.y_range.start, plot.y_range.end
                    current_xspan, current_yspan = (current_r-current_l), (current_t-current_b)
                else:
                    current_l, current_r, current_b, current_t = l, r, b, t
                    current_xspan, current_yspan = xspan, yspan

                if any(rs._triggering for rs in range_streams):
                    # If the event was triggered by a RangeXY stream
                    # event we want to get the latest range span
                    # values so we do not accidentally trigger a
                    # loop of events
                    l, r, b, t = current_l, current_r, current_b, current_t
                    xspan, yspan = current_xspan, current_yspan

                size_streams = [s for s in self.streams if isinstance(s, PlotSize)]
                if any(ss._triggering for ss in size_streams) and self._updated:
                    # Do not trigger on frame size changes, except for
                    # the initial one which can be important if width
                    # and/or height constraints have forced different
                    # aspect. After initial event we skip because size
                    # changes can trigger event loops if the tick
                    # labels change the canvas size
                    return

                desired_xspan = yspan*(ratio/frame_aspect)
                desired_yspan = xspan/(ratio/frame_aspect)
                if ((np.allclose(desired_xspan, xspan, rtol=0.05) and
                     np.allclose(desired_yspan, yspan, rtol=0.05)) or
                    not (util.isfinite(xspan) and util.isfinite(yspan))):
                    pass
                elif desired_yspan >= yspan:
                    desired_yspan = current_xspan/(ratio/frame_aspect)
                    ypad = (desired_yspan-yspan)/2.
                    b, t = b-ypad, t+ypad
                    yupdate = True
                else:
                    desired_xspan = current_yspan*(ratio/frame_aspect)
                    xpad = (desired_xspan-xspan)/2.
                    l, r = l-xpad, r+xpad
                    xupdate = True
            elif not (fixed_height and fixed_width):
                # Set initial aspect
                aspect = self.get_aspect(xspan, yspan)
                width = plot.frame_width or plot.plot_width or 300
                height = plot.frame_height or plot.plot_height or 300

                if not (fixed_width or fixed_height) and not self.responsive:
                    fixed_height = True

                if fixed_height:
                    plot.frame_height = height
                    plot.frame_width = int(height/aspect)
                    plot.plot_width, plot.plot_height = None, None
                elif fixed_width:
                    plot.frame_width = width
                    plot.frame_height = int(width*aspect)
                    plot.plot_width, plot.plot_height = None, None
                else:
                    plot.aspect_ratio = 1./aspect

            box_zoom = plot.select(type=tools.BoxZoomTool)
            scroll_zoom = plot.select(type=tools.WheelZoomTool)
            if box_zoom:
                box_zoom.match_aspect = True
            if scroll_zoom:
                scroll_zoom.zoom_on_axis = False

        if not self.drawn or xupdate:
            self._update_range(x_range, l, r, xfactors, self.invert_xaxis,
                               self._shared['x'], self.logx, streaming)
        if not self.drawn or yupdate:
            self._update_range(y_range, b, t, yfactors, self.invert_yaxis,
                               self._shared['y'], self.logy, streaming)


    def _update_range(self, axis_range, low, high, factors, invert, shared, log, streaming=False):
        if isinstance(axis_range, (Range1d, DataRange1d)) and self.apply_ranges:
            if isinstance(low, util.cftime_types):
                pass
            elif (low == high and low is not None):
                if isinstance(low, util.datetime_types):
                    offset = np.timedelta64(500, 'ms')
                    low, high = np.datetime64(low), np.datetime64(high)
                    low -= offset
                    high += offset
                else:
                    offset = abs(low*0.1 if low else 0.5)
                    low -= offset
                    high += offset
            if shared:
                shared = (axis_range.start, axis_range.end)
                low, high = util.max_range([(low, high), shared])
            if invert: low, high = high, low
            if not isinstance(low, util.datetime_types) and log and (low is None or low <= 0):
                low = 0.01 if high < 0.01 else 10**(np.log10(high)-2)
                self.param.warning(
                    "Logarithmic axis range encountered value less "
                    "than or equal to zero, please supply explicit "
                    "lower-bound to override default of %.3f." % low)
            updates = {}
            if util.isfinite(low):
                updates['start'] = (axis_range.start, low)
                updates['reset_start'] = updates['start']
            if util.isfinite(high):
                updates['end'] = (axis_range.end, high)
                updates['reset_end'] = updates['end']
            for k, (old, new) in updates.items():
                if isinstance(new, util.cftime_types):
                    new = date_to_integer(new)
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


    def get_aspect(self, xspan, yspan):
        """
        Computes the aspect ratio of the plot
        """
        if 'plot' in self.handles and self.state.frame_width and self.state.frame_height:
            return self.state.frame_width/self.state.frame_height
        elif self.data_aspect:
            return (yspan/xspan)*self.data_aspect
        elif self.aspect == 'equal':
            return yspan/xspan
        elif self.aspect == 'square':
            return 1
        elif self.aspect is not None:
            return self.aspect
        elif self.width is not None and self.height is not None:
            return self.width/self.height
        else:
            return 1


    def _get_factors(self, element, ranges):
        """
        Get factors for categorical axes.
        """
        xdim, ydim = element.dimensions()[:2]
        if xdim.values:
            xvals = xdim.values
        elif 'factors' in ranges.get(xdim.name, {}):
            xvals = ranges[xdim.name]['factors']
        else:
            xvals = element.dimension_values(0, False)

        if ydim.values:
            yvals = ydim.values
        elif 'factors' in ranges.get(ydim.name, {}):
            yvals = ranges[ydim.name]['factors']
        else:
            yvals = element.dimension_values(1, False)
        xvals, yvals = np.asarray(xvals), np.asarray(yvals)
        if not self._allow_implicit_categories:
            xvals = xvals if xvals.dtype.kind in 'SU' else []
            yvals = yvals if yvals.dtype.kind in 'SU' else []
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
        if 'legend_field' in properties and 'legend_label' in properties:
            del properties['legend_label']
        renderer = getattr(plot, plot_method)(**dict(properties, **mapping))
        return renderer, renderer.glyph


    def _element_transform(self, transform, element, ranges):
        return transform.apply(element, ranges=ranges, flat=True)


    def _apply_transforms(self, element, data, ranges, style, group=None):
        new_style = dict(style)
        prefix = group+'_' if group else ''
        for k, v in dict(style).items():
            if isinstance(v, util.basestring):
                if validate(k, v) == True:
                    continue
                elif v in element or (isinstance(element, Graph) and v in element.nodes):
                    v = dim(v)
                elif any(d==v for d in self.overlay_dims):
                    v = dim([d for d in self.overlay_dims if d==v][0])

            if (not isinstance(v, dim) or (group is not None and not k.startswith(group))):
                continue
            elif (not v.applies(element) and v.dimension not in self.overlay_dims):
                new_style.pop(k)
                self.param.warning(
                    'Specified %s dim transform %r could not be applied, '
                    'as not all dimensions could be resolved.' % (k, v))
                continue

            if v.dimension in self.overlay_dims:
                ds = Dataset({d.name: v for d, v in self.overlay_dims.items()},
                             list(self.overlay_dims))
                val = v.apply(ds, ranges=ranges, flat=True)[0]
            else:
                val = self._element_transform(v, element, ranges)

            if (not util.isscalar(val) and len(util.unique_array(val)) == 1 and
                ((not 'color' in k or validate('color', val)) or k in self._nonvectorized_styles)):
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
                elif data and len(val) != len(list(data.values())[0]):
                    if isinstance(element, VectorField):
                        val = np.tile(val, 3)
                    elif isinstance(element, Path) and not isinstance(element, Contours):
                        val = val[:-1]
                    else:
                        continue

            if k == 'angle':
                val = np.deg2rad(val)
            elif k.endswith('font_size'):
                if util.isscalar(val) and isinstance(val, int):
                    val = str(v)+'pt'
                elif isinstance(val, np.ndarray) and val.dtype.kind in 'ifu':
                    val = [str(int(s))+'pt' for s in val]
            if util.isscalar(val):
                key = val
            else:
                # Node marker does not handle {'field': ...}
                key = k if k == 'node_marker' else {'field': k}
                data[k] = val

            # If color is not valid colorspec add colormapper
            numeric = isinstance(val, util.arraylike_types) and val.dtype.kind in 'uifMmb'
            colormap = style.get(prefix+'cmap')
            if ('color' in k and isinstance(val, util.arraylike_types) and
                (numeric or not validate('color', val) or isinstance(colormap, dict))):
                kwargs = {}
                if val.dtype.kind not in 'ifMu':
                    range_key = dim_range_key(v)
                    if range_key in ranges and 'factors' in ranges[range_key]:
                        factors = ranges[range_key]['factors']
                    else:
                        factors = util.unique_array(val)
                    if isinstance(val, util.arraylike_types) and val.dtype.kind == 'b':
                        factors = factors.astype(str)
                    kwargs['factors'] = factors
                cmapper = self._get_colormapper(v, element, ranges,
                                                dict(style), name=k+'_color_mapper',
                                                group=group, **kwargs)
                categorical = isinstance(cmapper, CategoricalColorMapper)
                if categorical and val.dtype.kind in 'ifMub':
                    if v.dimension in element:
                        formatter = element.get_dimension(v.dimension).pprint_value
                    else:
                        formatter = str
                    field = k + '_str__'
                    data[k+'_str__'] = [formatter(d) for d in val]
                else:
                    field = k
                if categorical and getattr(self, 'show_legend', False):
                    legend_prop = 'legend_field' if bokeh_version >= LooseVersion('1.3.5') else 'legend'
                    new_style[legend_prop] = field
                key = {'field': field, 'transform': cmapper}
            new_style[k] = key

        # Process color/alpha styles and expand to fill/line style
        for style, val in list(new_style.items()):
            for s in ('alpha', 'color'):
                if prefix+s != style or style not in data or validate(s, val, True):
                    continue
                supports_fill = any(
                    o.startswith(prefix+'fill') and (prefix != 'edge_' or getattr(self, 'filled', True))
                    for o in self.style_opts)
                for pprefix in [p+'_' for p in property_prefixes]+['']:
                    fill_key = prefix+pprefix+'fill_'+s
                    fill_style = new_style.get(fill_key)

                    # Do not override custom nonselection/muted alpha
                    if ((pprefix in ('nonselection_', 'muted_') and s == 'alpha')
                        or fill_key not in self.style_opts):
                        continue

                    # Override empty and non-vectorized fill_style if not hover style
                    hover = pprefix == 'hover_'
                    if ((fill_style is None or (validate(s, fill_style, True) and not hover))
                        and supports_fill):
                        new_style[fill_key] = val

                    line_key = prefix+pprefix+'line_'+s
                    line_style = new_style.get(line_key)

                    # If glyph has fill and line style is set overriding line color
                    if supports_fill and line_style is not None:
                        continue

                    # If glyph does not support fill override non-vectorized line_color
                    if ((line_style is not None and (validate(s, line_style) and not hover)) or
                        (line_style is None and not supports_fill)):
                        new_style[line_key] = val

        return new_style


    def _glyph_properties(self, plot, element, source, ranges, style, group=None):
        properties = dict(style, source=source)
        if self.show_legend:
            if self.overlay_dims:
                legend = ', '.join([d.pprint_value(v, print_unit=True) for d, v in
                                    self.overlay_dims.items()])
            else:
                legend = element.label
            if legend and self.overlaid:
                legend_prop = 'legend_label' if bokeh_version >= LooseVersion('1.3.5') else 'legend'
                properties[legend_prop] = legend
        return properties


    def _filter_properties(self, properties, glyph_type, allowed):
        glyph_props = dict(properties)
        for gtype in ((glyph_type, '') if glyph_type else ('',)):
            for prop in ('color', 'alpha'):
                glyph_prop = properties.get(gtype+prop)
                if glyph_prop is not None and ('line_'+prop not in glyph_props or gtype):
                    glyph_props['line_'+prop] = glyph_prop
                if glyph_prop is not None and ('fill_'+prop not in glyph_props or gtype):
                    glyph_props['fill_'+prop] = glyph_prop

            props = {k[len(gtype):]: v for k, v in glyph_props.items()
                     if k.startswith(gtype)}
            if self.batched:
                glyph_props = dict(props, **glyph_props)
            else:
                glyph_props.update(props)
        return {k: v for k, v in glyph_props.items() if k in allowed}


    def _update_glyph(self, renderer, properties, mapping, glyph, source, data):
        allowed_properties = glyph.properties()
        properties = mpl_to_bokeh(properties)
        merged = dict(properties, **mapping)
        legend_props = ('legend_field', 'legend_label') if bokeh_version >= LooseVersion('1.3.5') else ('legend',)
        for lp in legend_props:
            legend = merged.pop(lp, None)
            if legend is not None:
                break
        columns = list(source.data.keys())
        glyph_updates = []
        for glyph_type in ('', 'selection_', 'nonselection_', 'hover_', 'muted_'):
            if renderer:
                glyph = getattr(renderer, glyph_type+'glyph', None)
                if glyph == 'auto':
                    base_glyph = renderer.glyph
                    props = base_glyph.properties_with_values()
                    glyph = type(base_glyph)(**{k: v for k, v in props.items()
                                                if not prop_is_none(v)})
                    setattr(renderer, glyph_type+'glyph', glyph)
            if not glyph or (not renderer and glyph_type):
                continue
            filtered = self._filter_properties(merged, glyph_type, allowed_properties)

            # Ensure that data is populated before updating glyph
            dataspecs = glyph.dataspecs()
            for spec in dataspecs:
                new_spec = filtered.get(spec)
                old_spec = getattr(glyph, spec)
                new_field = new_spec.get('field') if isinstance(new_spec, dict) else new_spec
                old_field = old_spec.get('field') if isinstance(old_spec, dict) else old_spec
                if (data is None) or (new_field not in data or new_field in source.data or new_field == old_field):
                    continue
                columns.append(new_field)
            glyph_updates.append((glyph, filtered))

        # If a dataspec has changed and the CDS.data will be replaced
        # the GlyphRenderer will not find the column, therefore we
        # craft an event which will make the column available.
        cds_replace = True if data is None else cds_column_replace(source, data)
        if not cds_replace:
            if not self.static_source:
                self._update_datasource(source, data)
            if hasattr(self, 'selected') and self.selected is not None:
                self._update_selected(source)
        elif self.document:
            server = self.renderer.mode == 'server'
            with hold_policy(self.document, 'collect', server=server):
                empty_data = {c: [] for c in columns}
                event = ModelChangedEvent(self.document, source, 'data',
                                          source.data, empty_data, empty_data,
                                          setter='empty')
                if bokeh_version >= LooseVersion('2.4.0'):
                    self.document.callbacks._held_events.append(event)
                else:
                    self.document._held_events.append(event)

        if legend is not None:
            for leg in self.state.legend:
                for item in leg.items:
                    if renderer in item.renderers:
                        if isinstance(legend, dict):
                            label = legend
                        elif lp != 'legend':
                            prop = 'value' if 'label' in lp else 'field'
                            label = {prop: legend}
                        elif isinstance(item.label, dict):
                            label = {list(item.label)[0]: legend}
                        else:
                            label = {'value': legend}
                        item.label = label

        for glyph, update in glyph_updates:
            glyph.update(**update)

        if data is not None and cds_replace and not self.static_source:
            self._update_datasource(source, data)


    def _postprocess_hover(self, renderer, source):
        """
        Attaches renderer to hover tool and processes tooltips to
        ensure datetime data is displayed correctly.
        """
        hover = self.handles.get('hover')
        if hover is None:
            return
        if not isinstance(hover.tooltips, util.basestring) and 'hv_created' in hover.tags:
            for k, values in source.data.items():
                key = '@{%s}' % k
                if ((isinstance(value, np.ndarray) and value.dtype.kind == 'M') or
                    (len(values) and isinstance(values[0], util.datetime_types))):
                    hover.tooltips = [(l, f+'{%F %T}' if f == key else f) for l, f in hover.tooltips]
                    hover.formatters[key] = "datetime"

        if hover.renderers == 'auto':
            hover.renderers = []
        if renderer not in hover.renderers:
            hover.renderers.append(renderer)

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

        with abbreviated_exception():
            style = self._apply_transforms(element, data, ranges, style)

        if source is None:
            source = self._init_datasource(data)
        self.handles['previous_id'] = current_id
        self.handles['source'] = self.handles['cds'] = source
        self.handles['selected'] = source.selected

        properties = self._glyph_properties(plot, style_element, source, ranges, style)
        if 'legend_label' in properties and 'legend_field' in mapping:
            mapping.pop('legend_field')

        with abbreviated_exception():
            renderer, glyph = self._init_glyph(plot, mapping, properties)
        self.handles['glyph'] = glyph
        if isinstance(renderer, Renderer):
            self.handles['glyph_renderer'] = renderer

        self._postprocess_hover(renderer, source)

        # Update plot, source and glyph
        with abbreviated_exception():
            self._update_glyph(renderer, properties, mapping, glyph, source, source.data)


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

        if self.top_level:
            self.init_links()

        if not self.overlaid:
            self._set_active_tools(plot)
            self._process_legend()
        self._execute_hooks(element)

        self.drawn = True

        return plot


    def _update_glyphs(self, element, ranges, style):
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
        if self.batched:
            data, mapping, style = self.get_batched_data(element, ranges)
        else:
            data, mapping, style = self.get_data(element, ranges, style)

        # Include old data if source static
        if self.static_source:
            for k, v in source.data.items():
                if k not in data:
                    data[k] = v
                elif not len(data[k]) and len(source.data):
                    data[k] = source.data[k]

        with abbreviated_exception():
            style = self._apply_transforms(element, data, ranges, style)

        if glyph:
            properties = self._glyph_properties(plot, element, source, ranges, style)
            renderer = self.handles.get('glyph_renderer')
            if 'visible' in style and hasattr(renderer, 'visible'):
                renderer.visible = style['visible']
            with abbreviated_exception():
                self._update_glyph(renderer, properties, mapping, glyph, source, data)
        elif not self.static_source:
            self._update_datasource(source, data)


    def _reset_ranges(self):
        """
        Resets RangeXY streams if norm option is set to framewise
        """
        # Skipping conditional to temporarily revert fix (see https://github.com/holoviz/holoviews/issues/4396)
        # This fix caused PlotSize change events to rerender
        # rasterized/datashaded with the full extents which was wrong
        if self.overlaid or True:
            return
        for el, callbacks in self.traverse(lambda x: (x.current_frame, x.callbacks)):
            if el is None:
                continue
            for callback in callbacks:
                norm = self.lookup_options(el, 'norm').options
                if norm.get('framewise'):
                    for s in callback.streams:
                        if isinstance(s, RangeXY) and not s._triggering:
                            s.reset()

    def update_frame(self, key, ranges=None, plot=None, element=None):
        """
        Updates an existing plot with data corresponding
        to the key.
        """
        self._reset_ranges()
        reused = isinstance(self.hmap, DynamicMap) and (self.overlaid or self.batched)
        self.prev_frame =  self.current_frame
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

        if not self.overlaid:
            ranges = self.compute_ranges(self.hmap, key, ranges)
        else:
            self.ranges.update(ranges)
        self.param.set_param(**self.lookup_options(style_element, 'plot').options)
        ranges = util.match_spec(style_element, ranges)
        self.current_ranges = ranges
        plot = self.handles['plot']
        if not self.overlaid:
            self._update_ranges(style_element, ranges)
            self._update_plot(key, plot, style_element)
            self._set_active_tools(plot)
            self._updated = True

        if 'hover' in self.handles:
            self._update_hover(element)
            if 'cds' in self.handles:
                cds = self.handles['cds']
                self._postprocess_hover(renderer, cds)

        self._update_glyphs(element, ranges, self.style[self.cyclic_index])
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
            style_group = self._style_groups.get('_'.join(key.split('_')[:-1]))
            group_style = dict(style)
            ds_data = data.get(key, {})
            with abbreviated_exception():
                group_style = self._apply_transforms(element, ds_data, ranges, group_style, style_group)
            if id(ds_data) in source_cache:
                source = source_cache[id(ds_data)]
            else:
                source = self._init_datasource(ds_data)
                source_cache[id(ds_data)] = source
            self.handles[key+'_source'] = source
            properties = self._glyph_properties(plot, element, source, ranges, group_style, style_group)
            properties = self._process_properties(key, properties, mapping.get(key, {}))

            with abbreviated_exception():
                renderer, glyph = self._init_glyph(plot, mapping.get(key, {}), properties, key)
            self.handles[key+'_glyph'] = glyph
            if isinstance(renderer, Renderer):
                self.handles[key+'_glyph_renderer'] = renderer

            self._postprocess_hover(renderer, source)

            # Update plot, source and glyph
            with abbreviated_exception():
                self._update_glyph(renderer, properties, mapping.get(key, {}), glyph,
                                   source, source.data)
        if getattr(self, 'colorbar', False):
            for k, v in list(self.handles.items()):
                if not k.endswith('color_mapper'):
                    continue
                self._draw_colorbar(plot, v, k[:-12])


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


    def _update_glyphs(self, element, ranges, style):
        plot = self.handles['plot']

        # Cache frame object id to skip updating data if unchanged
        previous_id = self.handles.get('previous_id', None)
        if self.batched:
            current_id = tuple(element.traverse(lambda x: x._plot_id, [Element]))
        else:
            current_id = element._plot_id
        self.handles['previous_id'] = current_id
        self.static_source = (self.dynamic and (current_id == previous_id))
        data, mapping, style = self.get_data(element, ranges, style)

        keys = glyph_order(dict(data, **mapping), self._draw_order)
        for key in keys:
            gdata = data.get(key)
            source = self.handles[key+'_source']
            glyph = self.handles.get(key+'_glyph')
            if glyph:
                group_style = dict(style)
                style_group = self._style_groups.get('_'.join(key.split('_')[:-1]))
                with abbreviated_exception():
                    group_style = self._apply_transforms(element, gdata, ranges, group_style, style_group)
                properties = self._glyph_properties(plot, element, source, ranges, group_style, style_group)
                properties = self._process_properties(key, properties, mapping[key])
                renderer = self.handles.get(key+'_glyph_renderer')
                with abbreviated_exception():
                    self._update_glyph(renderer, properties, mapping[key],
                                       glyph, source, gdata)
            elif not self.static_source and gdata is not None:
                self._update_datasource(source, gdata)


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

    color_levels = param.ClassSelector(default=None, class_=(
        (int, list) + ((range,) if sys.version_info.major > 2 else ())), doc="""
        Number of discrete colors to use when colormapping or a set of color
        intervals defining the range of values to map each color to.""")

    cformatter = param.ClassSelector(
        default=None, class_=(util.basestring, TickFormatter, FunctionType), doc="""
        Formatter for ticks along the colorbar axis.""")

    clabel = param.String(default=None, doc="""
        An explicit override of the color bar label. If set, takes precedence
        over the title key in colorbar_opts.""")

    clim = param.Tuple(default=(np.nan, np.nan), length=2, doc="""
        User-specified colorbar axis range limits for the plot, as a tuple (low,high).
        If specified, takes precedence over data and dimension ranges.""")

    clim_percentile = param.ClassSelector(default=False, class_=(int, float, bool), doc="""
        Percentile value to compute colorscale robust to outliers. If
        True, uses 2nd and 98th percentile; otherwise uses the specified
        numerical percentile value.""")

    cformatter = param.ClassSelector(
        default=None, class_=(util.basestring, TickFormatter, FunctionType), doc="""
        Formatter for ticks along the colorbar axis.""")

    cnorm = param.ObjectSelector(default='linear', objects=['linear', 'log', 'eq_hist'], doc="""
        Color normalization to be applied during colormapping.""")

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

    logz = param.Boolean(default=False, doc="""
         Whether to apply log scaling to the z-axis.""")

    symmetric = param.Boolean(default=False, doc="""
        Whether to make the colormap symmetric around zero.""")

    _colorbar_defaults = dict(bar_line_color='black', label_standoff=8,
                              major_tick_line_color='black')

    _default_nan = '#8b8b8b'

    _nonvectorized_styles = base_properties + ['cmap', 'palette']

    def _draw_colorbar(self, plot, color_mapper, prefix=''):
        if CategoricalColorMapper and isinstance(color_mapper, CategoricalColorMapper):
            return
        if EqHistColorMapper and isinstance(color_mapper, EqHistColorMapper) and BinnedTicker:
            ticker = BinnedTicker(mapper=color_mapper)
        elif isinstance(color_mapper, LogColorMapper) and color_mapper.low > 0:
            ticker = LogTicker()
        else:
            ticker = BasicTicker()
        cbar_opts = dict(self.colorbar_specs[self.colorbar_position])

        # Check if there is a colorbar in the same position
        pos = cbar_opts['pos']
        if any(isinstance(model, ColorBar) for model in getattr(plot, pos, [])):
            return

        if self.clabel:
            self.colorbar_opts.update({'title': self.clabel})

        if self.cformatter is not None:
            self.colorbar_opts.update({'formatter': wrap_formatter(self.cformatter, 'c')})

        for tk in ['cticks', 'ticks']:
            ticksize = self._fontsize(tk, common=False).get('fontsize')
            if ticksize is not None:
                self.colorbar_opts.update({'major_label_text_font_size': ticksize})
                break

        for lb in ['clabel', 'labels']:
            labelsize = self._fontsize(lb, common=False).get('fontsize')
            if labelsize is not None:
                self.colorbar_opts.update({'title_text_font_size': labelsize})
                break

        opts = dict(cbar_opts['opts'], color_mapper=color_mapper, ticker=ticker,
                    **self._colorbar_defaults)
        color_bar = ColorBar(**dict(opts, **self.colorbar_opts))

        plot.add_layout(color_bar, pos)
        self.handles[prefix+'colorbar'] = color_bar


    def _get_colormapper(self, eldim, element, ranges, style, factors=None, colors=None,
                         group=None, name='color_mapper'):
        # The initial colormapper instance is cached the first time
        # and then only updated
        if eldim is None and colors is None:
            return None
        dim_name = dim_range_key(eldim)

        # Attempt to find matching colormapper on the adjoined plot
        if self.adjoined:
            cmappers = self.adjoined.traverse(
                lambda x: (x.handles.get('color_dim'),
                           x.handles.get(name),
                           [v for v in x.handles.values()
                            if isinstance(v, ColorMapper)])
                )
            cmappers = [(cmap, mappers) for cdim, cmap, mappers in cmappers
                        if cdim == eldim]
            if cmappers:
                cmapper, mappers  = cmappers[0]
                if not cmapper:
                    if mappers and mappers[0]:
                        cmapper = mappers[0]
                    else:
                        return None
                self.handles['color_mapper'] = cmapper
                return cmapper
            else:
                return None

        ncolors = None if factors is None else len(factors)
        if eldim:
            # check if there's an actual value (not np.nan)
            if all(util.isfinite(cl) for cl in self.clim):
                low, high = self.clim
            elif dim_name in ranges:
                if self.clim_percentile and 'robust' in ranges[dim_name]:
                    low, high = ranges[dim_name]['robust']
                else:
                    low, high = ranges[dim_name]['combined']
                dlow, dhigh = ranges[dim_name]['data']
                if (util.is_int(low, int_like=True) and
                    util.is_int(high, int_like=True) and
                    util.is_int(dlow) and
                    util.is_int(dhigh)):
                    low, high = int(low), int(high)
            elif isinstance(eldim, dim):
                low, high = np.nan, np.nan
            else:
                low, high = element.range(eldim.name)
            if self.symmetric:
                sym_max = max(abs(low), high)
                low, high = -sym_max, sym_max
            low = self.clim[0] if util.isfinite(self.clim[0]) else low
            high = self.clim[1] if util.isfinite(self.clim[1]) else high
        else:
            low, high = None, None

        prefix = '' if group is None else group+'_'
        cmap = colors or style.get(prefix+'cmap', style.get('cmap', 'viridis'))
        nan_colors = {k: rgba_tuple(v) for k, v in self.clipping_colors.items()}
        if isinstance(cmap, dict):
            factors = list(cmap)
            palette = [cmap.get(f, nan_colors.get('NaN', self._default_nan)) for f in factors]
            if isinstance(eldim, dim):
                if eldim.dimension in element:
                    formatter = element.get_dimension(eldim.dimension).pprint_value
                else:
                    formatter = str
            else:
                formatter = eldim.pprint_value
            factors = [formatter(f) for f in factors]
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
                palette, (low, high) = color_intervals(palette, self.color_levels, clip=(low, high))
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
            self.handles['color_dim'] = eldim
        return cmapper


    def _get_color_data(self, element, ranges, style, name='color', factors=None, colors=None,
                        int_categories=False):
        data, mapping = {}, {}
        cdim = element.get_dimension(self.color_index)
        color = style.get(name, None)
        if cdim and ((isinstance(color, util.basestring) and color in element) or isinstance(color, dim)):
            self.param.warning(
                "Cannot declare style mapping for '%s' option and "
                "declare a color_index; ignoring the color_index."
                % name)
            cdim = None
        if not cdim:
            return data, mapping

        cdata = element.dimension_values(cdim)
        field = util.dimension_sanitizer(cdim.name)
        dtypes = 'iOSU' if int_categories else 'OSU'

        if factors is None and (isinstance(cdata, list) or cdata.dtype.kind in dtypes):
            range_key = dim_range_key(cdim)
            if range_key in ranges and 'factors' in ranges[range_key]:
                factors = ranges[range_key]['factors']
            else:
                factors = util.unique_array(cdata)
        if factors is not None and int_categories and cdata.dtype.kind == 'i':
            field += '_str__'
            cdata = [str(f) for f in cdata]
            factors = [str(f) for f in factors]

        mapper = self._get_colormapper(cdim, element, ranges, style,
                                       factors, colors)
        if factors is None and isinstance(mapper, CategoricalColorMapper):
            field += '_str__'
            cdata = [cdim.pprint_value(c) for c in cdata]
            factors = True

        data[field] = cdata
        if factors is not None and self.show_legend:
            legend_prop = 'legend_field' if bokeh_version >= LooseVersion('1.3.5') else 'legend'
            mapping[legend_prop] = field
        mapping[name] = {'field': field, 'transform': mapper}

        return data, mapping


    def _get_cmapper_opts(self, low, high, factors, colors):
        if factors is None:
            if self.cnorm == 'linear':
                colormapper = LinearColorMapper
            if self.cnorm == 'log' or self.logz:
                colormapper = LogColorMapper
                if util.is_int(low) and util.is_int(high) and low == 0:
                    low = 1
                    if 'min' not in colors:
                        # Make integer 0 be transparent
                        colors['min'] = 'rgba(0, 0, 0, 0)'
                elif util.is_number(low) and low <= 0:
                    self.param.warning(
                        "Log color mapper lower bound <= 0 and will not "
                        "render correctly. Ensure you set a positive "
                        "lower bound on the color dimension or using "
                        "the `clim` option."
                    )
            elif self.cnorm == 'eq_hist':
                if EqHistColorMapper is None:
                    raise ImportError("Could not import bokeh.models.EqHistColorMapper. "
                                      "Note that the option cnorm='eq_hist' requires "
                                      "bokeh 2.2.3 or higher.")
                colormapper = EqHistColorMapper
            if isinstance(low, (bool, np.bool_)): low = int(low)
            if isinstance(high, (bool, np.bool_)): high = int(high)
            # Pad zero-range to avoid breaking colorbar (as of bokeh 1.0.4)
            if low == high:
                offset = self.default_span / 2
                low -= offset
                high += offset
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
            opts = dict(factors=list(factors))
            if 'NaN' in colors:
                opts['nan_color'] = colors['NaN']
        return colormapper, opts


    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object and optionally creates a colorbar.
        """
        ret = super(ColorbarPlot, self)._init_glyph(plot, mapping, properties)
        if self.colorbar:
            for k, v in list(self.handles.items()):
                if not k.endswith('color_mapper'):
                    continue
                self._draw_colorbar(plot, v, k[:-12])
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

    legend_opts = param.Dict(default={}, doc="""
        Allows setting specific styling options for the colorbar.""")

    def _process_legend(self, plot=None):
        plot = plot or self.handles['plot']
        if not plot.legend:
            return
        legend = plot.legend[0]
        cmappers = [cmapper for cmapper in self.handles.values()
                   if isinstance(cmapper, CategoricalColorMapper)]
        categorical = bool(cmappers)
        if ((not categorical and not self.overlaid and len(legend.items) == 1)
            or not self.show_legend):
            legend.items[:] = []
        else:
            plot.legend.orientation = 'horizontal' if self.legend_cols else 'vertical'
            pos = self.legend_position
            if pos in self.legend_specs:
                plot.legend[:] = []
                legend.location = self.legend_offset
                if pos in ['top', 'bottom']:
                    plot.legend.orientation = 'horizontal'
                plot.add_layout(legend, self.legend_specs[pos])
            else:
                legend.location = pos

            # Apply muting and misc legend opts
            for leg in plot.legend:
                leg.update(**self.legend_opts)
                for item in leg.items:
                    for r in item.renderers:
                        r.muted = self.legend_muted



class AnnotationPlot(object):
    """
    Mix-in plotting subclass for AnnotationPlots which do not have a legend.
    """


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
                          'title', 'title_format', 'legend_position', 'legend_offset',
                          'legend_cols', 'gridstyle', 'legend_muted', 'padding',
                          'xlabel', 'ylabel', 'xlim', 'ylim', 'zlim',
                          'xformatter', 'yformatter', 'active_tools',
                          'min_height', 'max_height', 'min_width', 'min_height',
                          'margin', 'aspect', 'data_aspect', 'frame_width',
                          'frame_height', 'responsive', 'fontscale']

    @property
    def _x_range_type(self):
        for v in self.subplots.values():
            if not isinstance(v._x_range_type, Range1d):
                return v._x_range_type
        return self._x_range_type

    @property
    def _y_range_type(self):
        for v in self.subplots.values():
            if not isinstance(v._y_range_type, Range1d):
                return v._y_range_type
        return self._y_range_type

    def _process_legend(self, overlay):
        plot = self.handles['plot']
        subplots = self.traverse(lambda x: x, [lambda x: x is not self])
        legend_plots = any(p is not None for p in subplots
                           if isinstance(p, LegendPlot) and
                           not isinstance(p, OverlayPlot))
        non_annotation = [p for p in subplots if not
                          (isinstance(p, OverlayPlot) or isinstance(p, AnnotationPlot))]
        if (not self.show_legend or len(plot.legend) == 0 or
            (len(non_annotation) <= 1 and not (self.dynamic or legend_plots))):
            return super(OverlayPlot, self)._process_legend()
        elif not plot.legend:
            return

        legend = plot.legend[0]

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

        pos = self.legend_position
        orientation = 'horizontal' if self.legend_cols else 'vertical'
        if pos in ['top', 'bottom']:
            orientation = 'horizontal'
        options['orientation'] = orientation

        if overlay is not None and overlay.kdims:
            title = ', '.join([d.label for d in overlay.kdims])
            options['title'] = title

        options.update(self._fontsize('legend', 'label_text_font_size'))
        options.update(self._fontsize('legend_title', 'title_text_font_size'))
        legend.update(**options)

        if pos in self.legend_specs:
            pos = self.legend_specs[pos]
        else:
            legend.location = pos

        if 'legend_items' not in self.handles:
            self.handles['legend_items'] = []
        legend_items = self.handles['legend_items']
        legend_labels = {tuple(sorted(i.label.items())) if isinstance(i.label, dict) else i.label: i
                         for i in legend_items}
        for item in legend.items:
            label = tuple(sorted(item.label.items())) if isinstance(item.label, dict) else item.label
            if not label or (isinstance(item.label, dict) and not item.label.get('value', True)):
                continue
            if label in legend_labels:
                prev_item = legend_labels[label]
                prev_item.renderers[:] = list(util.unique_iterator(prev_item.renderers+item.renderers))
            else:
                legend_labels[label] = item
                legend_items.append(item)
                if item not in self.handles['legend_items']:
                    self.handles['legend_items'].append(item)

        # Ensure that each renderer is only singly referenced by a legend item
        filtered = []
        renderers = []
        for item in legend_items:
            item.renderers[:] = [r for r in item.renderers if r not in renderers]
            if item in filtered or not item.renderers or not any(r.visible for r in item.renderers):
                continue
            renderers += item.renderers
            filtered.append(item)
        legend.items[:] = list(util.unique_iterator(filtered))

        if self.multiple_legends:
            remove_legend(plot, legend)
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
            remove_legend(plot, legend)
            legend.location = self.legend_offset
            plot.add_layout(legend, pos)

        # Apply muting and misc legend opts
        for leg in plot.legend:
            leg.update(**self.legend_opts)
            for item in leg.items:
                for r in item.renderers:
                    r.muted = self.legend_muted or r.muted


    def _init_tools(self, element, callbacks=[]):
        """
        Processes the list of tools to be supplied to the plot.
        """
        hover_tools = {}
        init_tools, tool_types = [], []
        for key, subplot in self.subplots.items():
            el = element.get(key)
            if el is not None:
                el_tools = subplot._init_tools(el, self.callbacks)
                for tool in el_tools:
                    if isinstance(tool, util.basestring):
                        tool_type = TOOL_TYPES.get(tool)
                    else:
                        tool_type = type(tool)
                    if isinstance(tool, tools.HoverTool):
                        tooltips = tuple(tool.tooltips) if tool.tooltips else ()
                        if tooltips in hover_tools:
                            continue
                        else:
                            hover_tools[tooltips] = tool
                    elif tool_type in tool_types:
                        continue
                    else:
                        tool_types.append(tool_type)
                    init_tools.append(tool)
        self.handles['hover_tools'] = hover_tools
        return init_tools


    def _merge_tools(self, subplot):
        """
        Merges tools on the overlay with those on the subplots.
        """
        if self.batched and 'hover' in subplot.handles:
            self.handles['hover'] = subplot.handles['hover']
        elif 'hover' in subplot.handles and 'hover_tools' in self.handles:
            hover = subplot.handles['hover']
            if hover.tooltips and not isinstance(hover.tooltips, util.basestring):
                tooltips = tuple((name, spec.replace('{%F %T}', '')) for name, spec in hover.tooltips)
            else:
                tooltips = ()
            tool = self.handles['hover_tools'].get(tooltips)
            if tool:
                tool_renderers = [] if tool.renderers == 'auto' else tool.renderers
                hover_renderers = [] if hover.renderers == 'auto' else hover.renderers
                renderers = [r for r in tool_renderers + hover_renderers if r is not None]
                tool.renderers = list(util.unique_iterator(renderers))
                if 'hover' not in self.handles:
                    self.handles['hover'] = tool


    def _get_factors(self, overlay, ranges):
        xfactors, yfactors = [], []
        for k, sp in self.subplots.items():
            el = overlay.data.get(k)
            if el is not None:
                elranges = util.match_spec(el, ranges)
                xfs, yfs = sp._get_factors(el, elranges)
                if len(xfs):
                    xfactors.append(xfs)
                if len(yfs):
                    yfactors.append(yfs)
        xfactors = list(util.unique_iterator(chain(*xfactors)))
        yfactors = list(util.unique_iterator(chain(*yfactors)))
        return xfactors, yfactors


    def _get_axis_dims(self, element):
        subplots = list(self.subplots.values())
        if subplots:
            return subplots[0]._get_axis_dims(element)
        return super(OverlayPlot, self)._get_axis_dims(element)


    def initialize_plot(self, ranges=None, plot=None, plots=None):
        key = util.wrap_tuple(self.hmap.last_key)
        nonempty = [(k, el) for k, el in self.hmap.data.items() if el]
        if not nonempty:
            raise SkipRendering('All Overlays empty, cannot initialize plot.')
        dkey, element = nonempty[-1]
        ranges = self.compute_ranges(self.hmap, key, ranges)

        self.tabs = self.tabs or any(isinstance(sp, TablePlot) for sp in self.subplots.values())
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
                # Ensure that all subplots are in the same state
                frame = element.get(key, None)
                subplot.current_frame = frame
                subplot.current_key = dkey
            if self.batched:
                self.handles['plot'] = child
            if self.tabs:
                title = subplot._format_title(key, dimensions=False)
                if not title:
                    title = get_tab_title(key, frame, self.hmap.last)
                panels.append(Panel(child=child, title=title))
            self._merge_tools(subplot)

        if self.tabs:
            self.handles['plot'] = Tabs(
                tabs=panels, width=self.width, height=self.height,
                min_width=self.min_width, min_height=self.min_height,
                max_width=self.max_width, max_height=self.max_height,
                sizing_mode='fixed'
            )
        elif not self.overlaid:
            self._process_legend(element)
            self._set_active_tools(plot)
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
        self._reset_ranges()
        reused = isinstance(self.hmap, DynamicMap) and self.overlaid
        self.prev_frame =  self.current_frame
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

            # Update plot options
            plot_opts = self.lookup_options(element, 'plot').options
            inherited = self._traverse_options(element, 'plot',
                                               self._propagate_options,
                                               defaults=False)
            plot_opts.update(**{k: v[0] for k, v in inherited.items() if k not in plot_opts})
            self.param.set_param(**plot_opts)

            if not self.overlaid and not self.tabs and not self.batched:
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
                    idx, spec, exact = self._match_subplot(k, subplot, items, element)
                    if idx is not None:
                        _, el = items.pop(idx)
                        if not exact:
                            self._update_subplot(subplot, spec)

                # Skip updates to subplots when its streams is not one of
                # the streams that initiated the update
                if (triggering and all(s not in triggering for s in subplot.streams) and
                    not subplot in self.dynamic_subplots):
                    continue
            subplot.update_frame(key, ranges, element=el)

        if not self.batched and isinstance(self.hmap, DynamicMap) and items:
            init_kwargs = {'plots': self.handles['plots']}
            if not self.tabs:
                init_kwargs['plot'] = self.handles['plot']
            self._create_dynamic_subplots(key, items, ranges, **init_kwargs)
            if not self.overlaid and not self.tabs:
                self._process_legend(element)

        if element and not self.overlaid and not self.tabs and not self.batched:
            plot = self.handles['plot']
            self._update_plot(key, plot, element)
            self._set_active_tools(plot)

        self._updated = True
        self._process_legend(element)
        self._execute_hooks(element)
