import re, time, sys
from distutils.version import LooseVersion
from collections import defaultdict
import datetime as dt

import numpy as np

try:
    from matplotlib import colors
    import matplotlib.cm as cm
except ImportError:
    cm, colors = None, None

import param
import bokeh

bokeh_version = LooseVersion(bokeh.__version__)  # noqa

from bokeh.core.enums import Palette
from bokeh.core.json_encoder import serialize_json # noqa (API import)
from bokeh.core.properties import value
from bokeh.layouts import WidgetBox, Row, Column
from bokeh.models import Model, ToolbarBox, FactorRange, Range1d, Plot, Spacer, CustomJS
from bokeh.models.widgets import DataTable, Tabs, Div
from bokeh.plotting import Figure
from bokeh.themes.theme import Theme

try:
    from bokeh.themes import built_in_themes
except:
    built_in_themes = {}

try:
    from bkcharts import Chart
except:
    Chart = type(None) # Create stub for isinstance check

from ...core.options import abbreviated_exception
from ...core.overlay import Overlay
from ...core.util import (basestring, unique_array, callable_name, pd,
                          dt64_to_dt, _getargspec)
from ...core.spaces import get_nested_dmaps, DynamicMap

from ..util import dim_axis_label, rgb2hex, COLOR_ALIASES

# Conversion between matplotlib and bokeh markers
markers = {'s': {'marker': 'square'},
           'd': {'marker': 'diamond'},
           '+': {'marker': 'cross'},
           '^': {'marker': 'triangle', 'angle': 0},
           '>': {'marker': 'triangle', 'angle': -np.pi/2},
           'v': {'marker': 'triangle', 'angle': np.pi},
           '<': {'marker': 'triangle', 'angle': np.pi/2},
           '1': {'marker': 'triangle', 'angle': 0},
           '2': {'marker': 'triangle', 'angle': -np.pi/2},
           '3': {'marker': 'triangle', 'angle': np.pi},
           '4': {'marker': 'triangle', 'angle': np.pi/2}}


def convert_timestamp(timestamp):
    """
    Converts bokehJS timestamp to datetime64.
    """
    datetime = dt.datetime.fromtimestamp(timestamp/1000., dt.timezone.utc)
    return np.datetime64(datetime.replace(tzinfo=None))


def rgba_tuple(rgba):
    """
    Ensures RGB(A) tuples in the range 0-1 are scaled to 0-255.
    """
    if isinstance(rgba, tuple):
        return tuple(int(c*255) if i<3 else c for i, c in enumerate(rgba))
    else:
        return COLOR_ALIASES.get(rgba, rgba)


def decode_bytes(array):
    """
    Decodes an array, list or tuple of bytestrings to avoid python 3
    bokeh serialization errors
    """
    if (sys.version_info.major == 2 or not len(array) or
        (isinstance(array, np.ndarray) and array.dtype.kind != 'O')):
        return array
    decoded = [v.decode('utf-8') if isinstance(v, bytes) else v for v in array]
    if isinstance(array, np.ndarray):
        return np.asarray(decoded)
    elif isinstance(array, tuple):
        return tuple(decoded)
    return decoded


def get_cmap(cmap):
    """
    Returns matplotlib cmap generated from bokeh palette or
    directly accessed from matplotlib.
    """
    with abbreviated_exception():
        rgb_vals = getattr(Palette, cmap, None)
        if rgb_vals:
            return colors.ListedColormap(rgb_vals, name=cmap)
        return cm.get_cmap(cmap)


def mpl_to_bokeh(properties):
    """
    Utility to process style properties converting any
    matplotlib specific options to their nearest bokeh
    equivalent.
    """
    new_properties = {}
    for k, v in properties.items():
        if k == 's':
            new_properties['size'] = v
        elif k == 'marker':
            new_properties.update(markers.get(v, {'marker': v}))
        elif (k == 'color' or k.endswith('_color')) and not isinstance(v, dict):
            with abbreviated_exception():
                v = COLOR_ALIASES.get(v, v)
            if isinstance(v, tuple):
                with abbreviated_exception():
                    v = rgb2hex(v)
            new_properties[k] = v
        else:
            new_properties[k] = v
    new_properties.pop('cmap', None)
    return new_properties


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
    if isinstance(plot, (Div, ToolbarBox)):
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
        width, height = w_agg(widths), h_agg(heights)
    elif isinstance(plot, (Figure, Chart)):
        width, height = plot.plot_width, plot.plot_height
    elif isinstance(plot, (Plot, DataTable, Spacer)):
        width, height = plot.width, plot.height
    return width, height


def empty_plot(width, height):
    """
    Creates an empty and invisible plot of the specified size.
    """
    x_range = Range1d(start=0, end=1)
    y_range = Range1d(start=0, end=1)
    p = Figure(plot_width=width, plot_height=height,
               x_range=x_range, y_range=y_range)
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.outline_line_alpha = 0
    p.grid.grid_line_alpha = 0
    return p


def font_size_to_pixels(size):
    """
    Convert a fontsize to a pixel value
    """
    if size is None or not isinstance(size, basestring):
        return
    conversions = {'em': 16, 'pt': 16/12.}
    val = re.findall('\d+', size)
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
    ranges.callback = CustomJS(args=dict(range=ranges), code=reset)

    axis_props = {}
    if label_size:
        axis_props['axis_label_text_font_size'] = value(label_size)
    if tick_size:
        axis_props['major_label_text_font_size'] = value(tick_size)

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


def convert_datetime(time):
    return time.astype('datetime64[s]').astype(float)*1000


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
    Uses flexx.pyscript to compile a python tick formatter to JS code
    """
    try:
        from flexx.pyscript import py2js
    except ImportError:
        param.main.warning(msg+'Ensure Flexx is installed '
                           '("conda install -c bokeh flexx" or '
                           '"pip install flexx")')
        return
    try:
        jscode = py2js(formatter, 'formatter')
    except Exception as e:
        error = 'Pyscript raised an error: {0}'.format(e)
        error = error.replace('%', '%%')
        param.main.warning(msg+error)
        return

    args = _getargspec(formatter).args
    arg_define = 'var %s = tick;' % args[0] if args else ''
    return_js = 'return formatter();\n'
    jsfunc = '\n'.join([arg_define, jscode, return_js])
    match = re.search('(formatter \= function \(.*\))', jsfunc )
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
                if frame.group != frame.params('group').default:
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


def expand_batched_style(style, opts, mapping, nvals):
    """
    Computes styles applied to a batched plot by iterating over the
    supplied list of style options and expanding any options found in
    the supplied style dictionary returning a data and mapping defining
    the data that should be added to the ColumnDataSource.
    """
    opts = sorted(opts, key=lambda x: x in ['color', 'alpha'])
    applied_styles = set(mapping)
    style_data, style_mapping = {}, {}
    for opt in opts:
        if 'color' in opt:
            alias = 'color'
        elif 'alpha' in opt:
            alias = 'alpha'
        else:
            alias = None
        if opt not in style or opt in mapping:
            continue
        elif opt == alias:
            if alias in applied_styles:
                continue
            elif 'line_'+alias in applied_styles:
                if 'fill_'+alias not in opts:
                    continue
                opt = 'fill_'+alias
                val = style[alias]
            elif 'fill_'+alias in applied_styles:
                opt = 'line_'+alias
                val = style[alias]
            else:
                val = style[alias]
        else:
            val = style[opt]
        style_mapping[opt] = {'field': opt}
        applied_styles.add(opt)
        if 'color' in opt and isinstance(val, tuple):
            val = rgb2hex(val)
        style_data[opt] = [val]*nvals
    return style_data, style_mapping


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
        elif not isinstance(v, basestring):
            continue
        values = data[v]
        try:
            if len(unique_array(values)) == 1:
                mapping[k] = values[0]
                del data[v]
        except:
            pass


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
                setattr(model, k, v)
        elif k in valid_properties and v != valid_properties[k]:
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
        for source in shared_sources:
            source.data.clear()
            if self.document and self.document._held_events:
                self.document._held_events = self.document._held_events[:-1]

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

    @property
    def completed(self):
        return self.counter is None

    def start(self):
        self._start_time = time.time()
        if self.document is None:
            raise RuntimeError('periodic was registered to be run on bokeh'
                               'server but no document was found.')
        self.document.add_periodic_callback(self._periodic_callback, self.period)

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
            self.document.remove_periodic_callback(self._periodic_callback)
        except ValueError: # Already stopped
            pass

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
    """
    Converts datetime types to bokeh's integer format.
    """
    if isinstance(date, np.datetime64):
        date = dt64_to_dt(date)
    elif pd and isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()
    if isinstance(date, (dt.datetime, dt.date)):
        dt_int = time.mktime(date.timetuple())*1000
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

    >>> glyph_order(['scatter_1', 'patch_1', 'rect_1'], \
                    ['scatter', 'patch'])
    ['scatter_1', 'patch_1', 'rect_1']
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
