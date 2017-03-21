import itertools, inspect, re
from distutils.version import LooseVersion
from collections import defaultdict

import numpy as np

try:
    from matplotlib import colors
    import matplotlib.cm as cm
except ImportError:
    cm, colors = None, None

import param
import bokeh
bokeh_version = LooseVersion(bokeh.__version__)
from bokeh.core.enums import Palette
from bokeh.core.json_encoder import serialize_json # noqa (API import)
from bokeh.core.properties import value
from bokeh.document import Document
from bokeh.models.plots import Plot
from bokeh.models import (GlyphRenderer, Model, HasProps, Column, Row,
                          ToolbarBox, FactorRange, Range1d)
from bokeh.models.widgets import DataTable, Tabs
from bokeh.plotting import Figure
if bokeh_version >= '0.12':
    from bokeh.layouts import WidgetBox

from ...core.options import abbreviated_exception
from ...core.overlay import Overlay

from ..util import dim_axis_label

# Conversion between matplotlib and bokeh markers
markers = {'s': {'marker': 'square'},
           'd': {'marker': 'diamond'},
           '^': {'marker': 'triangle', 'orientation': 0},
           '>': {'marker': 'triangle', 'orientation': np.pi/2},
           'v': {'marker': 'triangle', 'orientation': np.pi},
           '<': {'marker': 'triangle', 'orientation': -np.pi/2},
           '1': {'marker': 'triangle', 'orientation': 0},
           '2': {'marker': 'triangle', 'orientation': np.pi/2},
           '3': {'marker': 'triangle', 'orientation': np.pi},
           '4': {'marker': 'triangle', 'orientation': -np.pi/2}}

# List of models that do not update correctly and must be ignored
# Should only include models that have no direct effect on the display
# and can therefore be safely ignored. Axes currently fail saying
# LinearAxis.computed_bounds cannot be updated
IGNORED_MODELS = ['LinearAxis', 'LogAxis', 'DatetimeAxis', 'DatetimeTickFormatter',
                  'BasicTicker', 'BasicTickFormatter', 'FixedTicker',
                  'FuncTickFormatter', 'LogTickFormatter',
                  'CategoricalTickFormatter']

# List of attributes that can safely be dropped from the references
IGNORED_ATTRIBUTES = ['data', 'palette', 'image', 'x', 'y', 'factors']

# Model priority order to ensure some types are updated before others
MODEL_PRIORITY = ['Range1d', 'Title', 'Image', 'LinearColorMapper',
                  'Plot', 'Range1d', 'FactorRange', 'CategoricalAxis',
                  'LinearAxis', 'ColumnDataSource']


def rgb2hex(rgb):
    """
    Convert RGB(A) tuple to hex.
    """
    if len(rgb) > 3:
        rgb = rgb[:-1]
    return "#{0:02x}{1:02x}{2:02x}".format(*(int(v*255) for v in rgb))


def rgba_tuple(rgba):
    """
    Ensures RGB(A) tuples in the range 0-1 are scaled to 0-255.
    """
    if isinstance(rgba, tuple):
        return tuple(int(c*255) if i<3 else c for i, c in enumerate(rgba))
    else:
        return rgba


def mplcmap_to_palette(cmap, ncolors=None):
    """
    Converts a matplotlib colormap to palette of RGB hex strings."
    """
    if colors is None:
        raise ValueError("Using cmaps on objects requires matplotlib.")
    with abbreviated_exception():
        colormap = cm.get_cmap(cmap) #choose any matplotlib colormap here
        if ncolors:
            return [rgb2hex(colormap(i)) for i in np.linspace(0, 1, ncolors)]
        return [rgb2hex(m) for m in colormap(np.arange(colormap.N))]


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
        elif k == 'color' or k.endswith('_color'):
            with abbreviated_exception():
                v = colors.ColorConverter.colors.get(v, v)
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
                x_range = Range1d(start=0, end=1)
                y_range = Range1d(start=0, end=1)
                p = Figure(plot_width=widths[c], plot_height=heights[r],
                           x_range=x_range, y_range=y_range)
                p.xaxis.visible = False
                p.yaxis.visible = False
                p.outline_line_alpha = 0
                p.grid.grid_line_alpha = 0
            expanded_plots[r].append(p)
    return expanded_plots


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
              label_size=None, tick_size=None, axis_height=40):
    factors = list(map(dim.pprint_value, factors))
    nchars = np.max([len(f) for f in factors])
    ranges = FactorRange(factors=factors)
    ranges2 = Range1d(start=0, end=1)
    axis_label = dim_axis_label(dim)

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
                     ((nchars*tick_px)*0.5)) + tick_px + label_px
        opts = dict(x_axis_type='auto', x_axis_label=axis_label,
                    x_range=ranges, y_range=ranges2, plot_height=height,
                    plot_width=size)
    else:
        # Adjust width to compensate for label rotation
        align = 'left' if flip else 'right'
        width = int(axis_height + np.abs(np.cos(rotation)) *
                    ((nchars*tick_px)*0.5)) + tick_px + label_px
        opts = dict(y_axis_label=axis_label, x_range=ranges2,
                    y_range=ranges, plot_width=width, plot_height=size)

    p = Figure(toolbar_location=None, **opts)
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


def get_ids(obj):
    """
    Returns a list of all ids in the supplied object.  Useful for
    determining if a json representation contains references to other
    objects. Since only the references between objects are required
    this allows determining whether a particular branch of the json
    representation is required.
    """
    ids = []
    if isinstance(obj, list):
        ids = [get_ids(o) for o in obj]
    elif isinstance(obj, dict):
        ids = [(v,) if k == 'id' else get_ids(v)
               for k, v in obj.items() if not k in IGNORED_ATTRIBUTES]
    return list(itertools.chain(*ids))


def replace_models(obj):
    """
    Recursively processes references, replacing Models with there .ref
    values and HasProps objects with their property values.
    """
    if isinstance(obj, Model):
        return obj.ref
    elif isinstance(obj, HasProps):
        return obj.properties_with_values(include_defaults=False)
    elif isinstance(obj, dict):
        return {k: v if k in IGNORED_ATTRIBUTES else replace_models(v)
                   for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_models(v) for v in obj]
    else:
        return obj


def to_references(doc):
    """
    Convert the document to a dictionary of references. Avoids
    unnecessary JSON serialization/deserialization within Python and
    the corresponding performance penalty.
    """
    root_ids = []
    for r in doc._roots:
        root_ids.append(r._id)

    references = {}
    for obj in doc._references_json(doc._all_models.values()):
        obj = replace_models(obj)
        references[obj['id']] = obj
    return references


def compute_static_patch(document, models):
    """
    Computes a patch to update an existing document without
    diffing the json first, making it suitable for static updates
    between arbitrary frames. Note that this only supports changed
    attributes and will break if new models have been added since
    the plot was first created.

    A static patch consists of two components:

    1) The events: Contain references to particular model attributes
       along with the updated value.
    2) The references: Contain a list of all references required to
       resolve the update events.

    This function cleans up the events and references that are sent
    to ensure that only the data that is required is sent. It does so
    by a) filtering the events and references for the models that have
    been requested to be updated and b) cleaning up the references to
    ensure that only the references between objects are sent without
    duplicating any of the data.
    """
    references = to_references(document)
    model_ids = [m.ref['id'] for m in models]

    requested_updates = []
    value_refs = {}
    events = []
    update_types = defaultdict(list)
    for ref_id, obj in references.items():
        if ref_id in model_ids:
            requested_updates += get_ids(obj)
        else:
            continue
        if obj['type'] in MODEL_PRIORITY:
            priority = MODEL_PRIORITY.index(obj['type'])
        else:
            priority = float('inf')
        for key, val in obj['attributes'].items():
            event = Document._event_for_attribute_change(references,
                                                         obj, key, val,
                                                         value_refs)
            events.append((priority, event))
            update_types[obj['type']].append(key)
    events = [delete_refs(e, IGNORED_MODELS, ignored_attributes=IGNORED_ATTRIBUTES)
              for _, e in sorted(events, key=lambda x: x[0])]
    events = [e for e in events if all(i in requested_updates for i in get_ids(e))
              if 'new' in e]
    value_refs = {ref_id: delete_refs(val, IGNORED_MODELS, IGNORED_ATTRIBUTES)
                  for ref_id, val in value_refs.items()}
    references = [val for val in value_refs.values()
                  if val not in [None, {}]]
    return dict(events=events, references=references)


def delete_refs(obj, models=[], dropped_attributes=[], ignored_attributes=[]):
    """
    Recursively traverses the object and looks for models and model
    attributes to be deleted.
    """
    if isinstance(obj, dict):
        if 'type' in obj and obj['type'] in models:
            return None
        new_obj = {}
        for k, v in list(obj.items()):
            # Drop unneccessary attributes, i.e. those that do not
            # contain references to other objects.
            if k in dropped_attributes or (k == 'attributes' and not get_ids(v)):
                continue
            if k in ignored_attributes:
                ref = v
            else:
                ref = delete_refs(v, models, dropped_attributes, ignored_attributes)
            if ref is not None:
                new_obj[k] = ref
        return new_obj
    elif isinstance(obj, list):
        objs = [delete_refs(v, models, dropped_attributes, ignored_attributes)
                for v in obj]
        return [o for o in objs if o is not None]
    else:
        return obj


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


def update_plot(old, new):
    """
    Updates an existing plot or figure with a new plot,
    useful for bokeh charts and mpl conversions, which do
    not allow updating an existing plot easily.

    ALERT: Should be replaced once bokeh supports it directly
    """
    old_renderers = old.select(type=GlyphRenderer)
    new_renderers = new.select(type=GlyphRenderer)

    old.x_range.update(**new.x_range.properties_with_values())
    old.y_range.update(**new.y_range.properties_with_values())
    updated = []
    for new_r in new_renderers:
        for old_r in old_renderers:
            if type(old_r.glyph) == type(new_r.glyph):
                old_renderers.pop(old_renderers.index(old_r))
                new_props = new_r.properties_with_values()
                source = new_props.pop('data_source')
                old_r.glyph.update(**new_r.glyph.properties_with_values())
                old_r.update(**new_props)
                old_r.data_source.data.update(source.data)
                updated.append(old_r)
                break

    for old_r in old_renderers:
        if old_r not in updated:
            emptied = {k: [] for k in old_r.data_source.data}
            old_r.data_source.data.update(emptied)


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
    elif isinstance(model, WidgetBox):
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
    total_width = np.max([np.sum(row) for row in widths])
    return plots, total_width


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

    args = inspect.getargspec(formatter).args
    arg_define = 'var %s = tick;' % args[0] if args else ''
    return_js = 'return formatter();\n'
    jsfunc = '\n'.join([arg_define, jscode, return_js])
    match = re.search('(function \(.*\))', jsfunc )
    return jsfunc[:match.start()] + 'function ()' + jsfunc[match.end():]


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
