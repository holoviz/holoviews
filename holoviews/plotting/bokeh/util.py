from collections import defaultdict
import numpy as np

try:
    from matplotlib import colors
    import matplotlib.cm as cm
except ImportError:
    cm, colors = None, None

from bokeh.enums import Palette
from bokeh.plotting import figure

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
        raise ValueException("Using cmaps on objects requires matplotlib.")
    colormap = cm.get_cmap(cmap) #choose any matplotlib colormap here
    return [colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]


def get_cmap(cmap):
    """
    Returns matplotlib cmap generated from bokeh palette or
    directly accessed from matplotlib.
    """
    rgb_vals = getattr(Palette, cmap, None)
    if rgb_vals:
        return colors.ListedColormap(rgb_vals, name=cmap)
    return cm.get_cmap(cmap)


def map_colors(arr, crange, cmap):
    """
    Maps an array of values to RGB hex strings.
    """
    nanmin = np.nanmin(arr)
    arr = (arr - nanmin) / (np.nanmax(arr)-nanmin)
    return [colors.rgb2hex(cmap(c)) if np.isfinite(c) else '#FFFFFF' for c in arr]


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
            v = colors.ColorConverter.colors.get(v, v)
            if isinstance(v, tuple):
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
            if p:
                widths[c] = max(widths[c], p.plot_width)
                heights[r] = max(heights[r], p.plot_height)

    expanded_plots = []
    for r, row in enumerate(plots):
        expanded_plots.append([])
        for c, p in enumerate(row):
            if p is None:
                p = figure(plot_width=widths[c], 
                           plot_height=heights[r])
                p.text(x=0, y=0, text=[' '])
                p.xaxis.visible = False
                p.yaxis.visible = False
                p.outline_line_color = None
                p.xgrid.grid_line_color = None
                p.ygrid.grid_line_color = None
            expanded_plots[r].append(p)
    return expanded_plots
