import os

from matplotlib import ticker
from matplotlib import rc_params_from_file

from ...core.options import Cycle, Palette, Options, StoreOptions
from ...core import Dimension, Layout, NdLayout, Overlay, HoloMap
from .annotation import * # pyflakes:ignore (API import)
from .chart import * # pyflakes:ignore (API import)
from .chart3d import * # pyflakes:ignore (API import)
from .path import * # pyflakes:ignore (API import)
from .plot import * # pyflakes:ignore (API import)
from .raster import * # pyflakes:ignore (API import)
from .tabular import * # pyflakes:ignore (API import)
from . import pandas # pyflakes:ignore (API import)
from . import seaborn # pyflakes:ignore (API import)

from renderer import MPLPlotRenderer, ANIMATION_OPTS


def opts(el, percent_size):
    "Returns the plot options with supplied size (if not overridden)"
    obj = el.last if isinstance(el, HoloMap) else el
    options = MPLPlotRenderer.get_plot_size(obj, percent_size) #  Store.registry[type(el)].renderer
    options.update(Store.lookup_options(obj, 'plot').options)
    return options


def set_style(key):
    """
    Select a style by name, e.g. set_style('default'). To revert to the
    previous style use the key 'unset' or False.
    """
    if key is None:
        return
    elif not key or key in ['unset', 'backup']:
        if 'backup' in styles:
            plt.rcParams.update(styles['backup'])
        else:
            raise Exception('No style backed up to restore')
    elif key not in styles:
        raise KeyError('%r not in available styles.')
    else:
        path = os.path.join(os.path.dirname(__file__), styles[key])
        new_style = rc_params_from_file(path)
        styles['backup'] = dict(plt.rcParams)

        plt.rcParams.update(new_style)

styles = {'default': './default.mplstyle'}
set_style('default')


# Upgrade Dimension formatters to matplotlib
wrapped_formatters = {k: fn if isinstance(fn, ticker.Formatter) else ticker.FuncFormatter(fn)
                      for k, fn in Dimension.type_formatters.items()}
Dimension.type_formatters.update(wrapped_formatters)

# Define matplotlib based style cycles and Palettes
Cycle.default_cycles.update({'default_colors': plt.rcParams['axes.color_cycle']})
Palette.colormaps.update({cm: plt.get_cmap(cm) for cm in plt.cm.datad})

style_aliases = {'edgecolor': ['ec', 'ecolor'], 'facecolor': ['fc'],
                 'linewidth': ['lw'], 'edgecolors': ['ec', 'edgecolor'],
                 'linestyle': ['ls'], 'size': ['s'], 'color': ['c'],
                 'markeredgecolor': ['mec'], 'markeredgewidth': ['mew'],
                 'markerfacecolor': ['mfc'], 'markersize': ['ms']}

def default_options(options):
    # Charts
    options.Curve = Options('style', color=Cycle(), linewidth=2)
    options.Scatter = Options('style', color=Cycle(), marker='o')
    options.ErrorBars = Options('style', ecolor='k')
    options.Bars = Options('style', ec='k', color=Cycle())
    options.Histogram = Options('style', ec='k', fc=Cycle())
    options.Points = Options('style', color=Cycle(), marker='o')
    options.Scatter3D = Options('style', color=Cycle(), marker='o')
    # Rasters
    options.Image = Options('style', cmap='hot', interpolation='nearest')
    options.Raster = Options('style', cmap='hot', interpolation='nearest')
    options.HeatMap = Options('style', cmap='RdYlBu_r', interpolation='nearest')
    options.HeatMap = Options('plot', show_values=True, xticks=20, yticks=20)
    options.RGB = Options('style', interpolation='nearest')
    # Composites
    options.Layout = Options('plot', sublabel_format='{Alpha}')
    options.GridSpace = Options('style', **{'font.size': 10, 'axes.labelsize': 'small',
                                                  'axes.titlesize': 'small'})
    # Annotations
    options.VLine = Options('style', color=Cycle())
    options.HLine = Options('style', color=Cycle())
    options.Spline = Options('style', linewidth=2, ec='r')
    options.Text = Options('style', fontsize=13)
    options.Arrow = Options('style', color='k', linewidth=2, fontsize=13)
    # Paths
    options.Contours = Options('style', color=Cycle())
    options.Path = Options('style', color=Cycle())
    options.Box = Options('style', color=Cycle())
    options.Bounds = Options('style', color=Cycle())
    options.Ellipse = Options('style', color=Cycle())
    # Interface
    options.TimeSeries = Options('style', color=Cycle())

# Register the default options
Store.option_setters.append(default_options)

# Register default Element options
Store.register_plots(style_aliases=style_aliases)


# Defining the most common style options for HoloViews
GrayNearest = Options(key='style', cmap='gray', interpolation='nearest')
