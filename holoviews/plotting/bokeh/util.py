import numpy as np

try:
    from matplotlib import colors
    import matplotlib.cm as cm
except ImportError:
    cm, colors = None, None

from bokeh.enums import Palette

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
