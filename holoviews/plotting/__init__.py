import os

import param

from ..core.options import Cycle, Options, Store
from .annotation import * # pyflakes:ignore (API import)
from .chart import * # pyflakes:ignore (API import)
from .chart3d import * # pyflakes:ignore (API import)
from .plot import * # pyflakes:ignore (API import)
from .raster import * # pyflakes:ignore (API import)
from .tabular import * # pyflakes:ignore (API import)
from . import pandas # pyflakes:ignore (API import)
from . import seaborn # pyflakes:ignore (API import)


GIF_TAG = "<center><img src='data:image/gif;base64,{b64}' style='max-width:100%'/><center/>"
VIDEO_TAG = """
<center><video controls style='max-width:100%'>
<source src="data:video/{mime_type};base64,{b64}" type="video/{mime_type}">
Your browser does not support the video tag.
</video><center/>"""


# <format name> : (animation writer, mime_type,  anim_kwargs, extra_args, tag)
ANIMATION_OPTS = {
    'webm': ('ffmpeg', 'webm', {},
             ['-vcodec', 'libvpx', '-b', '1000k'],
             VIDEO_TAG),
    'h264': ('ffmpeg', 'mp4', {'codec': 'libx264'},
             ['-pix_fmt', 'yuv420p'],
             VIDEO_TAG),
    'gif': ('imagemagick', 'gif', {'fps': 10}, [],
            GIF_TAG),
    'scrubber': ('html', None, {'fps': 5}, None, None)
}

Store.register_plots()

# Charts
Store.options.Curve = Options('style', color=Cycle(), linewidth=2)
Store.options.Scatter = Options('style', color=Cycle(), marker='o')
Store.options.Bars = Options('style', ec='k', fc=Cycle())
Store.options.Histogram = Options('style', ec='k', fc=Cycle())
Store.options.Points = Options('style', color=Cycle(), marker='o')
Store.options.Scatter3D = Options('style', color=Cycle(), marker='o')
# Rasters
Store.options.Image = Options('style', cmap='hot', interpolation='nearest')
Store.options.Raster = Options('style', cmap='hot', interpolation='nearest')
Store.options.HeatMap = Options('style', cmap='RdYlBu_r', interpolation='nearest')
Store.options.HeatMap = Options('plot', show_values=True, xticks=20, yticks=20)
Store.options.RGBA = Options('style', interpolation='nearest')
Store.options.RGB = Options('style', interpolation='nearest')
# Composites
Store.options.GridSpace = Options('style', **{'font.size': 10, 'axes.labelsize': 'small',
                                              'axes.titlesize': 'small'})
# Annotations
Store.options.VLine = Options('style', color=Cycle())
Store.options.HLine = Options('style', color=Cycle())
Store.options.Spline = Options('style', lw=2)
Store.options.Text = Options('style', fontsize=13)
Store.options.Arrow = Options('style', color='k', lw=2, fontsize=13)
# Paths
Store.options.Contours = Options('style', color=Cycle())
Store.options.Path = Options('style', color=Cycle())
Store.options.Box = Options('style', color=Cycle())
Store.options.Bounds = Options('style', color=Cycle())
Store.options.Ellipse = Options('style', color=Cycle())
# Interface
Store.options.TimeSeries = Options('style', color=Cycle())


from ..core import Dimension
from matplotlib.ticker import FormatStrFormatter
Dimension.type_formatters[int] = FormatStrFormatter("%d")
Dimension.type_formatters[float] = FormatStrFormatter("%.3g")
Dimension.type_formatters[np.float32] = FormatStrFormatter("%.3g")
Dimension.type_formatters[np.float64] = FormatStrFormatter("%.3g")

# Defining the most common style options for HoloViews
GrayNearest = Options(key='style', cmap='gray', interpolation='nearest')

def public(obj):
    if not isinstance(obj, type): return False
    baseclasses = [Plot]
    return any([issubclass(obj, bc) for bc in baseclasses])


_public = ["PlotSaver", "GrayNearest"] + list(set([_k for _k, _v in locals().items() if public(_v)]))
__all__ = _public
