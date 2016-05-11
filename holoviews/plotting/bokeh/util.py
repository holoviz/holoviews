from collections import defaultdict
import numpy as np
from ...core.options import abbreviated_exception

try:
    from matplotlib import colors
    import matplotlib.cm as cm
except ImportError:
    cm, colors = None, None

try:
    from bokeh.enums import Palette
    from bokeh.plotting import Plot
    bokeh_lt_011 = True
except:
    from bokeh.core.enums import Palette
    from bokeh.models.plots import Plot
    bokeh_lt_011 = False

from bokeh.models import GlyphRenderer
from bokeh.plotting import Figure

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


def mplcmap_to_palette(cmap):
    """
    Converts a matplotlib colormap to palette of RGB hex strings."
    """
    if colors is None:
        raise ValueError("Using cmaps on objects requires matplotlib.")
    with abbreviated_exception():
        colormap = cm.get_cmap(cmap) #choose any matplotlib colormap here
        return [colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]


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
                    v = colors.rgb2hex(v)
            new_properties[k] = v
        else:
            new_properties[k] = v
    new_properties.pop('cmap', None)
    return new_properties


def layout_padding(plots):
    """
    Temporary workaround to allow empty plots in a
    row of a bokeh GridPlot type. Should be removed
    when https://github.com/bokeh/bokeh/issues/2891
    is resolved.
    """
    widths, heights = defaultdict(int), defaultdict(int)
    for r, row in enumerate(plots):
        for c, p in enumerate(row):
            if p is not None:
                width = p.plot_width if isinstance(p, Plot) else p.width
                height = p.plot_height if isinstance(p, Plot) else p.height
                widths[c] = max(widths[c], width)
                heights[r] = max(heights[r], height)

    expanded_plots = []
    for r, row in enumerate(plots):
        expanded_plots.append([])
        for c, p in enumerate(row):
            if p is None:
                p = Figure(plot_width=widths[c],
                           plot_height=heights[r])
                p.text(x=0, y=0, text=[' '])
                p.xaxis.visible = False
                p.yaxis.visible = False
                p.outline_line_color = None
                p.xgrid.grid_line_color = None
                p.ygrid.grid_line_color = None
            expanded_plots[r].append(p)
    return expanded_plots


def convert_datetime(time):
    return time.astype('datetime64[s]').astype(float)*1000


def models_to_json(models):
    """
    Convert list of bokeh models into json to update plot(s).
    """
    json_data, ids = [], []
    for plotobj in models:
        if plotobj.ref['id'] in ids:
            continue
        else:
            ids.append(plotobj.ref['id'])
        if bokeh_lt_011:
            json = plotobj.vm_serialize(changed_only=True)
        else:
            json = plotobj.to_json(False)
            json.pop('tool_events', None)
            json.pop('renderers', None)
            json_data.append({'id': plotobj.ref['id'],
                              'type': plotobj.ref['type'],
                              'data': json})
    return json_data


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
