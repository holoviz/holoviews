import numpy as np

try:
    from matplotlib import colors
    import matplotlib.cm as cm
except ImportError:
    cm, colors = None, None

from bokeh.enums import Palette

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
    arr = (arr - arr.min()) / (arr.max()-arr.min())
    return [colors.rgb2hex(cmap(c)) for c in arr]
