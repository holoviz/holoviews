import calendar
import datetime as dt
import inspect
import re
import time

from collections import defaultdict
from contextlib import contextmanager
from types import FunctionType

import param
import bokeh
import numpy as np

from bokeh.core.json_encoder import serialize_json # noqa (API import)
from bokeh.core.validation import silence
from bokeh.layouts import WidgetBox, Row, Column
from bokeh.models import tools
from bokeh.models import (
    Model, ToolbarBox, FactorRange, Range1d, Plot, Spacer, CustomJS,
    GridBox, DatetimeAxis, CategoricalAxis
)
from bokeh.models.formatters import (
    FuncTickFormatter, TickFormatter, PrintfTickFormatter
)
from bokeh.models.widgets import DataTable, Tabs, Div
from bokeh.plotting import Figure
from bokeh.themes.theme import Theme

try:
    from bokeh.themes import built_in_themes
except:
    built_in_themes = {}

from ...core.ndmapping import NdMapping
from ...core.overlay import Overlay
from ...core.util import (
    LooseVersion, arraylike_types, callable_name, cftime_types,
    cftime_to_timestamp, isnumeric, pd, unique_array
)
from ...core.spaces import get_nested_dmaps, DynamicMap
from ..util import dim_axis_label

bokeh_version = LooseVersion(bokeh.__version__)  # noqa


TOOL_TYPES = {
    'pan': tools.PanTool,
    'xpan': tools.PanTool,
    'ypan': tools.PanTool,
    'xwheel_pan': tools.WheelPanTool,
    'ywheel_pan': tools.WheelPanTool,
    'wheel_zoom': tools.WheelZoomTool,
    'xwheel_zoom': tools.WheelZoomTool,
    'ywheel_zoom': tools.WheelZoomTool,
    'zoom_in': tools.ZoomInTool,
    'xzoom_in': tools.ZoomInTool,
    'yzoom_in': tools.ZoomInTool,
    'zoom_out': tools.ZoomOutTool,
    'xzoom_out': tools.ZoomOutTool,
    'yzoom_out': tools.ZoomOutTool,
    'click': tools.TapTool,
    'tap': tools.TapTool,
    'crosshair': tools.CrosshairTool,
    'box_select': tools.BoxSelectTool,
    'xbox_select': tools.BoxSelectTool,
    'ybox_select': tools.BoxSelectTool,
    'poly_select': tools.PolySelectTool,
    'lasso_select': tools.LassoSelectTool,
    'box_zoom': tools.BoxZoomTool,
    'xbox_zoom': tools.BoxZoomTool,
    'ybox_zoom': tools.BoxZoomTool,
    'hover': tools.HoverTool,
    'save': tools.SaveTool,
    'undo': tools.UndoTool,
    'redo': tools.RedoTool,
    'reset': tools.ResetTool,
    'help': tools.HelpTool,
    'box_edit': tools.BoxEditTool,
    'point_draw': tools.PointDrawTool,
    'poly_draw': tools.PolyDrawTool,
    'poly_edit': tools.PolyEditTool,
    'freehand_draw': tools.FreehandDrawTool
}


def convert_timestamp(timestamp):
    """
    Converts bokehJS timestamp to datetime64.
    """
    datetime = dt.datetime.utcfromtimestamp(timestamp/1000.)
    return np.datetime64(datetime.replace(tzinfo=None))


def prop_is_none(value):
    """
    Checks if property value is None.
    """
    return (value is None or
            (isinstance(value, dict) and 'value' in value
             and value['value'] is None))


def decode_bytes(array):
    """
    Decodes an array, list or tuple of bytestrings to avoid python 3
    bokeh serialization errors
    """
    if (not len(array) or (isinstance(array, arraylike_types) and array.dtype.kind != 'O')):
        return array
    decoded = [v.decode('utf-8') if isinstance(v, bytes) else v for v in array]
    if isinstance(array, np.ndarray):
        return np.asarray(decoded)
    elif isinstance(array, tuple):
        return tuple(decoded)
    return decoded


def layout_padding(plots, renderer):
    """
    Pads Nones in a list of lists of plots with empty plots.
    """
    widths, heights = defaultdict(int), defaultdict(int)
    for r, row in enumerate(plots):
        for c, p in enumerate(row):
            if p is not None:
                width, height = renderer.get_size(p)
                widths[c] = max(widths[c], width)
                heights[r] = max(heights[r], height)

    expanded_plots = []
    for r, row in enumerate(plots):
        expanded_plots.append([])
        for c, p in enumerate(row):
            if p is None:
                p = empty_plot(widths[c], heights[r])
            elif hasattr(p, 'plot_width') and p.plot_width == 0 and p.plot_height == 0:
                p.plot_width = widths[c]
                p.plot_height = heights[r]
            expanded_plots[r].append(p)
    return expanded_plots


def compute_plot_size(plot):
    """
    Computes the size of bokeh models that make up a layout such as
    figures, rows, columns, widgetboxes and Plot.
    """
    if isinstance(plot, GridBox):
        ndmapping = NdMapping({(x, y): fig for fig, y, x in plot.children}, kdims=['x', 'y'])
        cols = ndmapping.groupby('x')
        rows = ndmapping.groupby('y')
        width = sum([max([compute_plot_size(f)[0] for f in col]) for col in cols])
        height = sum([max([compute_plot_size(f)[1] for f in row]) for row in rows])
        return width, height
    elif isinstance(plot, (Div, ToolbarBox)):
        # Cannot compute size for Div or ToolbarBox
        return 0, 0
    elif isinstance(plot, (Row, Column, WidgetBox, Tabs)):
        if not plot.children: return 0, 0
        if isinstance(plot, Row) or (isinstance(plot, ToolbarBox) and plot.toolbar_location not in ['right', 'left']):
            w_agg, h_agg = (np.sum, np.max)
        elif isinstance(plot, Tabs):
            w_agg, h_agg = (np.max, np.max)
        else:
            w_agg, h_agg = (np.max, np.sum)
        widths, heights = zip(*[compute_plot_size(child) for child in plot.children])
        return w_agg(widths), h_agg(heights)
    elif isinstance(plot, Figure):
        if plot.plot_width:
            width = plot.plot_width
        else:
            width = plot.frame_width + plot.min_border_right + plot.min_border_left
        if plot.plot_height:
            height = plot.plot_height
        else:
            height = plot.frame_height + plot.min_border_bottom + plot.min_border_top
        return width, height
    elif isinstance(plot, (Plot, DataTable, Spacer)):
        return plot.width, plot.height
    else:
        return 0, 0


def compute_layout_properties(
        width, height, frame_width, frame_height, explicit_width,
        explicit_height, aspect, data_aspect, responsive, size_multiplier,
        logger=None):
    """
    Utility to compute the aspect, plot width/height and sizing_mode
    behavior.

    Args:
      width (int): Plot width
      height (int): Plot height
      frame_width (int): Plot frame width
      frame_height (int): Plot frame height
      explicit_width (list): List of user supplied widths
      explicit_height (list): List of user supplied heights
      aspect (float): Plot aspect
      data_aspect (float): Scaling between x-axis and y-axis ranges
      responsive (boolean): Whether the plot should resize responsively
      size_multiplier (float): Multiplier for supplied plot dimensions
      logger (param.Parameters): Parameters object to issue warnings on

    Returns:
      Returns two dictionaries one for the aspect and sizing modes,
      and another for the plot dimensions.
    """
    fixed_width = (explicit_width or frame_width)
    fixed_height = (explicit_height or frame_height)
    fixed_aspect = aspect or data_aspect
    if aspect == 'square':
        aspect = 1
    elif aspect == 'equal':
        data_aspect = 1

    # Plot dimensions
    height = None if height is None else int(height*size_multiplier)
    width = None if width is None else int(width*size_multiplier)
    frame_height = None if frame_height is None else int(frame_height*size_multiplier)
    frame_width = None if frame_width is None else int(frame_width*size_multiplier)
    actual_width = frame_width or width
    actual_height = frame_height or height

    if frame_width is not None:
        width = None
    if frame_height is not None:
        height = None

    sizing_mode = 'fixed'
    if responsive:
        if fixed_height and fixed_width:
            responsive = False
            if logger:
                logger.warning("responsive mode could not be enabled "
                               "because fixed width and height were "
                               "specified.")
        elif fixed_width:
            height = None
            sizing_mode = 'fixed' if fixed_aspect else 'stretch_height'
        elif fixed_height:
            width = None
            sizing_mode = 'fixed' if fixed_aspect else 'stretch_width'
        else:
            width, height = None, None
            if fixed_aspect:
                if responsive == 'width':
                    sizing_mode = 'scale_width'
                elif responsive == 'height':
                    sizing_mode = 'scale_height'
                else:
                    sizing_mode = 'scale_both'
            else:
                if responsive == 'width':
                    sizing_mode = 'stretch_both'
                elif responsive == 'height':
                    sizing_mode = 'stretch_height'
                else:
                    sizing_mode = 'stretch_both'


    if fixed_aspect:
        if ((explicit_width and not frame_width) != (explicit_height and not frame_height)) and logger:
            logger.warning('Due to internal constraints, when aspect and '
                           'width/height is set, the bokeh backend uses '
                           'those values as frame_width/frame_height instead. '
                           'This ensures the aspect is respected, but means '
                           'that the plot might be slightly larger than '
                           'anticipated. Set the frame_width/frame_height '
                           'explicitly to suppress this warning.')

        aspect_type = 'data_aspect' if data_aspect else 'aspect'
        if fixed_width and fixed_height and aspect:
            if aspect == 'equal':
                data_aspect = 1
            elif not data_aspect:
                aspect = None
                if logger:
                    logger.warning(
                        "%s value was ignored because absolute width and "
                        "height values were provided. Either supply "
                        "explicit frame_width and frame_height to achieve "
                        "desired aspect OR supply a combination of width "
                        "or height and an aspect value." % aspect_type)
        elif fixed_width and responsive:
            height = None
            responsive = False
            if logger:
                logger.warning("responsive mode could not be enabled "
                               "because fixed width and aspect were "
                               "specified.")
        elif fixed_height and responsive:
            width = None
            responsive = False
            if logger:
                logger.warning("responsive mode could not be enabled "
                               "because fixed height and aspect were "
                               "specified.")
        elif responsive == 'width':
            sizing_mode = 'scale_width'
        elif responsive == 'height':
            sizing_mode = 'scale_height'

    if responsive == 'width' and fixed_width:
        responsive = False
        if logger:
            logger.warning("responsive width mode could not be enabled "
                           "because a fixed width was defined.")
    if responsive == 'height' and fixed_height:
        responsive = False
        if logger:
            logger.warning("responsive height mode could not be enabled "
                           "because a fixed height was defined.")

    match_aspect = False
    aspect_scale = 1
    aspect_ratio = None
    if data_aspect:
        match_aspect = True
        if (fixed_width and fixed_height):
            frame_width, frame_height = frame_width or width, frame_height or height
        elif fixed_width or not fixed_height:
            height = None
        elif fixed_height or not fixed_width:
            width = None

        aspect_scale = data_aspect
        if aspect == 'equal':
            aspect_scale = 1
        elif responsive:
            aspect_ratio = aspect
    elif (fixed_width and fixed_height):
        pass
    elif isnumeric(aspect):
        if responsive:
            aspect_ratio = aspect
        elif fixed_width:
            frame_width = actual_width
            frame_height = int(actual_width/aspect)
            width, height = None, None
        else:
            frame_width = int(actual_height*aspect)
            frame_height = actual_height
            width, height = None, None
    elif aspect is not None and logger:
        logger.warning('aspect value of type %s not recognized, '
                       'provide a numeric value, \'equal\' or '
                       '\'square\'.')

    return ({'aspect_ratio': aspect_ratio,
             'aspect_scale': aspect_scale,
             'match_aspect': match_aspect,
             'sizing_mode' : sizing_mode},
            {'frame_width' : frame_width,
             'frame_height': frame_height,
             'plot_height' : height,
             'plot_width'  : width})


@contextmanager
def silence_warnings(*warnings):
    """
    Context manager for silencing bokeh validation warnings.
    """
    for warning in warnings:
        silence(warning)
    try:
        yield
    finally:
        for warning in warnings:
            silence(warning, False)


def empty_plot(width, height):
    """
    Creates an empty and invisible plot of the specified size.
    """
    return Spacer(width=width, height=height)


def remove_legend(plot, legend):
    """
    Removes a legend from a bokeh plot.
    """
    valid_places = ['left', 'right', 'above', 'below', 'center']
    plot.legend[:] = [l for l in plot.legend if l is not legend]
    for place in valid_places:
        place = getattr(plot, place)
        if legend in place:
            place.remove(legend)


def font_size_to_pixels(size):
    """
    Convert a fontsize to a pixel value
    """
    if size is None or not isinstance(size, str):
        return
    conversions = {'em': 16, 'pt': 16/12.}
    val = re.findall(r'\d+', size)
    unit = re.findall('[a-z]+', size)
    if (val and not unit) or (val and unit[0] == 'px'):
        return int(val[0])
    elif val and unit[0] in conversions:
        return (int(int(val[0]) * conversions[unit[0]]))


def make_axis(axis, size, factors, dim, flip=False, rotation=0,
              label_size=None, tick_size=None, axis_height=35):
    factors = list(map(dim.pprint_value, factors))
    nchars = np.max([len(f) for f in factors])
    ranges = FactorRange(factors=factors)
    ranges2 = Range1d(start=0, end=1)
    axis_label = dim_axis_label(dim)
    reset = "range.setv({start: 0, end: range.factors.length})"
    customjs = CustomJS(args=dict(range=ranges), code=reset)
    ranges.js_on_change('start', customjs)

    axis_props = {}
    if label_size:
        axis_props['axis_label_text_font_size'] = label_size
    if tick_size:
        axis_props['major_label_text_font_size'] = tick_size

    tick_px = font_size_to_pixels(tick_size)
    if tick_px is None:
        tick_px = 8
    label_px = font_size_to_pixels(label_size)
    if label_px is None:
        label_px = 10

    rotation = np.radians(rotation)
    if axis == 'x':
        align = 'center'
        # Adjust height to compensate for label rotation
        height = int(axis_height + np.abs(np.sin(rotation)) *
                     ((nchars*tick_px)*0.82)) + tick_px + label_px
        opts = dict(x_axis_type='auto', x_axis_label=axis_label,
                    x_range=ranges, y_range=ranges2, plot_height=height,
                    plot_width=size)
    else:
        # Adjust width to compensate for label rotation
        align = 'left' if flip else 'right'
        width = int(axis_height + np.abs(np.cos(rotation)) *
                    ((nchars*tick_px)*0.82)) + tick_px + label_px
        opts = dict(y_axis_label=axis_label, x_range=ranges2,
                    y_range=ranges, plot_width=width, plot_height=size)

    p = Figure(toolbar_location=None, tools=[], **opts)
    p.outline_line_alpha = 0
    p.grid.grid_line_alpha = 0

    if axis == 'x':
        p.align = 'end'
        p.yaxis.visible = False
        axis = p.xaxis[0]
        if flip:
            p.above = p.below
            p.below = []
            p.xaxis[:] = p.above
    else:
        p.xaxis.visible = False
        axis = p.yaxis[0]
        if flip:
            p.right = p.left
            p.left = []
            p.yaxis[:] = p.right
    axis.major_label_orientation = rotation
    axis.major_label_text_align = align
    axis.major_label_text_baseline = 'middle'
    axis.update(**axis_props)
    return p


def hsv_to_rgb(hsv):
    """
    Vectorized HSV to RGB conversion, adapted from:
    http://stackoverflow.com/questions/24852345/hsv-to-rgb-color-conversion
    """
    h, s, v = (hsv[..., i] for i in range(3))
    shape = h.shape
    i = np.int_(h*6.)
    f = h*6.-i

    q = f
    t = 1.-f
    i = np.ravel(i)
    f = np.ravel(f)
    i%=6

    t = np.ravel(t)
    q = np.ravel(q)
    s = np.ravel(s)
    v = np.ravel(v)

    clist = (1-s*np.vstack([np.zeros_like(f),np.ones_like(f),q,t]))*v

    #0:v 1:p 2:q 3:t
    order = np.array([[0,3,1],[2,0,1],[1,0,3],[1,2,0],[3,1,0],[0,1,2]])
    rgb = clist[order[i], np.arange(np.prod(shape))[:,None]]

    return rgb.reshape(shape+(3,))


def pad_width(model, table_padding=0.85, tabs_padding=1.2):
    """
    Computes the width of a model and sets up appropriate padding
    for Tabs and DataTable types.
    """
    if isinstance(model, Row):
        vals = [pad_width(child) for child in model.children]
        width = np.max([v for v in vals if v is not None])
    elif isinstance(model, Column):
        vals = [pad_width(child) for child in model.children]
        width = np.sum([v for v in vals if v is not None])
    elif isinstance(model, Tabs):
        vals = [pad_width(t) for t in model.tabs]
        width = np.max([v for v in vals if v is not None])
        for model in model.tabs:
            model.width = width
            width = int(tabs_padding*width)
    elif isinstance(model, DataTable):
        width = model.width
        model.width = int(table_padding*width)
    elif isinstance(model, (WidgetBox, Div)):
        width = model.width
    elif model:
        width = model.plot_width
    else:
        width = 0
    return width


def pad_plots(plots):
    """
    Accepts a grid of bokeh plots in form of a list of lists and
    wraps any DataTable or Tabs in a WidgetBox with appropriate
    padding. Required to avoid overlap in gridplot.
    """
    widths = []
    for row in plots:
        row_widths = []
        for p in row:
            width = pad_width(p)
            row_widths.append(width)
        widths.append(row_widths)
    plots = [[WidgetBox(p, width=w) if isinstance(p, (DataTable, Tabs)) else p
              for p, w in zip(row, ws)] for row, ws in zip(plots, widths)]
    return plots


def filter_toolboxes(plots):
    """
    Filters out toolboxes out of a list of plots to be able to compose
    them into a larger plot.
    """
    if isinstance(plots, list):
        plots = [filter_toolboxes(plot) for plot in plots]
    elif hasattr(plots, 'children'):
        plots.children = [filter_toolboxes(child) for child in plots.children
                          if not isinstance(child, ToolbarBox)]
    return plots


def py2js_tickformatter(formatter, msg=''):
    """
    Uses py2js to compile a python tick formatter to JS code
    """
    try:
        from pscript import py2js
    except ImportError:
        param.main.param.warning(
            msg+'Ensure pscript is installed ("conda install pscript" '
            'or "pip install pscript")')
        return
    try:
        jscode = py2js(formatter, 'formatter')
    except Exception as e:
        error = 'Pyscript raised an error: {0}'.format(e)
        error = error.replace('%', '%%')
        param.main.param.warning(msg+error)
        return

    args = inspect.getfullargspec(formatter).args
    arg_define = 'var %s = tick;' % args[0] if args else ''
    return_js = 'return formatter();\n'
    jsfunc = '\n'.join([arg_define, jscode, return_js])
    match = re.search(r'(formatter \= function flx_formatter \(.*\))', jsfunc)
    return jsfunc[:match.start()] + 'formatter = function ()' + jsfunc[match.end():]


def get_tab_title(key, frame, overlay):
    """
    Computes a title for bokeh tabs from the key in the overlay, the
    element and the containing (Nd)Overlay.
    """
    if isinstance(overlay, Overlay):
        if frame is not None:
            title = []
            if frame.label:
                title.append(frame.label)
                if frame.group != frame.param.objects('existing')['group'].default:
                    title.append(frame.group)
            else:
                title.append(frame.group)
        else:
            title = key
        title = ' '.join(title)
    else:
        title = ' | '.join([d.pprint_value_string(k) for d, k in
                            zip(overlay.kdims, key)])
    return title


def get_default(model, name, theme=None):
    """
    Looks up the default value for a bokeh model property.
    """
    overrides = None
    if theme is not None:
        if isinstance(theme, str):
            theme = built_in_themes[theme]
        overrides = theme._for_class(model)
    descriptor = model.lookup(name)
    return descriptor.property.themed_default(model, name, overrides)


def filter_batched_data(data, mapping):
    """
    Iterates over the data and mapping for a ColumnDataSource and
    replaces columns with repeating values with a scalar. This is
    purely and optimization for scalar types.
    """
    for k, v in list(mapping.items()):
        if isinstance(v, dict) and 'field' in v:
            if 'transform' in v:
                continue
            v = v['field']
        elif not isinstance(v, str):
            continue
        values = data[v]
        try:
            if len(unique_array(values)) == 1:
                mapping[k] = values[0]
                del data[v]
        except:
            pass

def cds_column_replace(source, data):
    """
    Determine if the CDS.data requires a full replacement or simply
    needs to be updated. A replacement is required if untouched
    columns are not the same length as the columns being updated.
    """
    current_length = [len(v) for v in source.data.values()
                      if isinstance(v, (list,)+arraylike_types)]
    new_length = [len(v) for v in data.values() if isinstance(v, (list, np.ndarray))]
    untouched = [k for k in source.data if k not in data]
    return bool(untouched and current_length and new_length and current_length[0] != new_length[0])


@contextmanager
def hold_policy(document, policy, server=False):
    """
    Context manager to temporary override the hold policy.
    """
    if bokeh_version >= LooseVersion('2.4'):
        old_policy = document.callbacks.hold_value
        document.callbacks._hold = policy
    else:
        old_policy = document._hold
        document._hold = policy
    try:
        yield
    finally:
        if server and not old_policy:
            document.unhold()
        elif bokeh_version >= LooseVersion('2.4'):
            document.callbacks._hold = old_policy
        else:
            document._hold = old_policy


def recursive_model_update(model, props):
    """
    Recursively updates attributes on a model including other
    models. If the type of the new model matches the old model
    properties are simply updated, otherwise the model is replaced.
    """
    updates = {}
    valid_properties = model.properties_with_values()
    for k, v in props.items():
        if isinstance(v, Model):
            nested_model = getattr(model, k)
            if type(v) is type(nested_model):
                nested_props = v.properties_with_values(include_defaults=False)
                recursive_model_update(nested_model, nested_props)
            else:
                try:
                    setattr(model, k, v)
                except Exception as e:
                    if isinstance(v, dict) and 'value' in v:
                        setattr(model, k, v['value'])
                    else:
                        raise e
        elif k in valid_properties and v != valid_properties[k]:
            if isinstance(v, dict) and 'value' in v:
                v = v['value']
            updates[k] = v
    model.update(**updates)


def update_shared_sources(f):
    """
    Context manager to ensures data sources shared between multiple
    plots are cleared and updated appropriately avoiding warnings and
    allowing empty frames on subplots. Expects a list of
    shared_sources and a mapping of the columns expected columns for
    each source in the plots handles.
    """
    def wrapper(self, *args, **kwargs):
        source_cols = self.handles.get('source_cols', {})
        shared_sources = self.handles.get('shared_sources', [])
        doc = self.document
        for source in shared_sources:
            source.data.clear()
            if doc:
                event_obj = doc.callbacks if bokeh_version >= LooseVersion('2.4') else doc
                event_obj._held_events = event_obj._held_events[:-1]

        ret = f(self, *args, **kwargs)

        for source in shared_sources:
            expected = source_cols[id(source)]
            found = [c for c in expected if c in source.data]
            empty = np.full_like(source.data[found[0]], np.NaN) if found else []
            patch = {c: empty for c in expected if c not in source.data}
            source.data.update(patch)
        return ret
    return wrapper


def categorize_array(array, dim):
    """
    Uses a Dimension instance to convert an array of values to categorical
    (i.e. string) values and applies escaping for colons, which bokeh
    treats as a categorical suffix.
    """
    return np.array([dim.pprint_value(x) for x in array])


class periodic(object):
    """
    Mocks the API of periodic Thread in hv.core.util, allowing a smooth
    API transition on bokeh server.
    """

    def __init__(self, document):
        self.document = document
        self.callback = None
        self.period = None
        self.count = None
        self.counter = None
        self._start_time = None
        self.timeout = None
        self._pcb = None

    @property
    def completed(self):
        return self.counter is None

    def start(self):
        self._start_time = time.time()
        if self.document is None:
            raise RuntimeError('periodic was registered to be run on bokeh'
                               'server but no document was found.')
        self._pcb = self.document.add_periodic_callback(self._periodic_callback, self.period)

    def __call__(self, period, count, callback, timeout=None, block=False):
        if isinstance(count, int):
            if count < 0: raise ValueError('Count value must be positive')
        elif not type(count) is type(None):
            raise ValueError('Count value must be a positive integer or None')

        self.callback = callback
        self.period = period*1000.
        self.timeout = timeout
        self.count = count
        self.counter = 0
        return self

    def _periodic_callback(self):
        self.callback(self.counter)
        self.counter += 1

        if self.timeout is not None:
            dt = (time.time() - self._start_time)
            if dt > self.timeout:
                self.stop()
        if self.counter == self.count:
            self.stop()

    def stop(self):
        self.counter = None
        self.timeout = None
        try:
            self.document.remove_periodic_callback(self._pcb)
        except ValueError: # Already stopped
            pass
        self._pcb = None

    def __repr__(self):
        return 'periodic(%s, %s, %s)' % (self.period,
                                         self.count,
                                         callable_name(self.callback))
    def __str__(self):
        return repr(self)


def attach_periodic(plot):
    """
    Attaches plot refresh to all streams on the object.
    """
    def append_refresh(dmap):
        for dmap in get_nested_dmaps(dmap):
            dmap.periodic._periodic_util = periodic(plot.document)
    return plot.hmap.traverse(append_refresh, [DynamicMap])


def date_to_integer(date):
    """Converts support date types to milliseconds since epoch

    Attempts highest precision conversion of different datetime
    formats to milliseconds since the epoch (1970-01-01 00:00:00).
    If datetime is a cftime with a non-standard calendar the
    caveats described in hv.core.util.cftime_to_timestamp apply.

    Args:
        date: Date- or datetime-like object

    Returns:
        Milliseconds since 1970-01-01 00:00:00
    """
    if pd and isinstance(date, pd.Timestamp):
        try:
            date = date.to_datetime64()
        except:
            date = date.to_datetime()

    if isinstance(date, np.datetime64):
        return date.astype('datetime64[ms]').astype(float)
    elif isinstance(date, cftime_types):
        return cftime_to_timestamp(date, 'ms')

    if hasattr(date, 'timetuple'):
        dt_int = calendar.timegm(date.timetuple())*1000
    else:
        raise ValueError('Datetime type not recognized')
    return dt_int


def glyph_order(keys, draw_order=[]):
    """
    Orders a set of glyph handles using regular sort and an explicit
    sort order. The explicit draw order must take the form of a list
    of glyph names while the keys should be glyph names with a custom
    suffix. The draw order may only match subset of the keys and any
    matched items will take precedence over other entries.
    """
    keys = sorted(keys)
    def order_fn(glyph):
        matches = [item for item in draw_order if glyph.startswith(item)]
        return ((draw_order.index(matches[0]), glyph) if matches else
                (1e9+keys.index(glyph), glyph))
    return sorted(keys, key=order_fn)


def colormesh(X, Y):
    """
    Generates line paths for a quadmesh given 2D arrays of X and Y
    coordinates.
    """
    X1 = X[0:-1, 0:-1].ravel()
    Y1 = Y[0:-1, 0:-1].ravel()
    X2 = X[1:, 0:-1].ravel()
    Y2 = Y[1:, 0:-1].ravel()
    X3 = X[1:, 1:].ravel()
    Y3 = Y[1:, 1:].ravel()
    X4 = X[0:-1, 1:].ravel()
    Y4 = Y[0:-1, 1:].ravel()

    X = np.column_stack([X1, X2, X3, X4, X1])
    Y = np.column_stack([Y1, Y2, Y3, Y4, Y1])
    return X, Y


def theme_attr_json(theme, attr):
    if isinstance(theme, str) and theme in built_in_themes:
        return built_in_themes[theme]._json['attrs'].get(attr, {})
    elif isinstance(theme, Theme):
        return theme._json['attrs'].get(attr, {})
    else:
        return {}


def multi_polygons_data(element):
    """
    Expands polygon data which contains holes to a bokeh multi_polygons
    representation. Multi-polygons split by nans are expanded and the
    correct list of holes is assigned to each sub-polygon.
    """
    xs, ys = (element.dimension_values(kd, expanded=False) for kd in element.kdims)
    holes = element.holes()
    xsh, ysh = [], []
    for x, y, multi_hole in zip(xs, ys, holes):
        xhs = [[h[:, 0] for h in hole] for hole in multi_hole]
        yhs = [[h[:, 1] for h in hole] for hole in multi_hole]
        array = np.column_stack([x, y])
        splits = np.where(np.isnan(array[:, :2].astype('float')).sum(axis=1))[0]
        arrays = np.split(array, splits+1) if len(splits) else [array]
        multi_xs, multi_ys = [], []
        for i, (path, hx, hy) in enumerate(zip(arrays, xhs, yhs)):
            if i != (len(arrays)-1):
                path = path[:-1]
            multi_xs.append([path[:, 0]]+hx)
            multi_ys.append([path[:, 1]]+hy)
        xsh.append(multi_xs)
        ysh.append(multi_ys)
    return xsh, ysh


def match_dim_specs(specs1, specs2):
    """Matches dimension specs used to link axes.

    Axis dimension specs consists of a list of tuples corresponding
    to each dimension, each tuple spec has the form (name, label, unit).
    The name and label must match exactly while the unit only has to
    match if both specs define one.
    """
    if (specs1 is None or specs2 is None) or (len(specs1) != len(specs2)):
        return False
    for spec1, spec2 in zip(specs1, specs2):
        for s1, s2 in zip(spec1, spec2):
            if s1 is None or s2 is None:
                continue
            if s1 != s2:
                return False
    return True


def match_ax_type(ax, range_type):
    """
    Ensure the range_type matches the axis model being matched.
    """
    if isinstance(ax[0], CategoricalAxis):
        return range_type == 'categorical'
    elif isinstance(ax[0], DatetimeAxis):
        return range_type == 'datetime'
    else:
        return range_type in ('auto', 'log')


def wrap_formatter(formatter, axis):
    """
    Wraps formatting function or string in
    appropriate bokeh formatter type.
    """
    if isinstance(formatter, TickFormatter):
        pass
    elif isinstance(formatter, FunctionType):
        msg = ('%sformatter could not be '
               'converted to tick formatter. ' % axis)
        jsfunc = py2js_tickformatter(formatter, msg)
        if jsfunc:
            formatter = FuncTickFormatter(code=jsfunc)
        else:
            formatter = None
    else:
        formatter = PrintfTickFormatter(format=formatter)
    return formatter
