import numpy as np

try:
    from matplotlib import colors
    import matplotlib.cm as cm
except ImportError:
    cm, colors = None, None

from bokeh.enums import Palette

def mplcmap_to_palette(cmap):
    if colors is None: 
        raise ValueException("Using cmaps on objects requires matplotlib.")
    colormap = cm.get_cmap(cmap) #choose any matplotlib colormap here
    return [colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
