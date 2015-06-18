
try:
    # Switching to 'agg' backend (may be overridden in holoviews.rc)
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
except:
    pass


import os
from matplotlib import ticker
from matplotlib import rc_params_from_file

from ...core.options import Cycle, Palette, Options, StoreOptions
from ...core import Dimension, Layout, NdLayout, Overlay, HoloMap
from ...core.element import * # pyflakes:ignore (API import)
from ..plot import PlotWrapper
from .annotation import * # pyflakes:ignore (API import)
from .chart import * # pyflakes:ignore (API import)
from .chart3d import * # pyflakes:ignore (API import)
from .path import * # pyflakes:ignore (API import)
from .plot import * # pyflakes:ignore (API import)
from .raster import * # pyflakes:ignore (API import)
from .tabular import * # pyflakes:ignore (API import)
from . import pandas # pyflakes:ignore (API import)
from . import seaborn # pyflakes:ignore (API import)

from .renderer import MPLRenderer



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

Store.renderers['matplotlib'] = MPLRenderer

# Defines a wrapper around GridPlot and RasterGridPlot
# switching to RasterGridPlot if the plot only contains
# Raster Elements
BasicGridPlot = GridPlot
def grid_wrapper(grid, **kwargs):
    raster_fn = lambda x: True if isinstance(x, Raster) else False
    all_raster = all(grid.traverse(raster_fn, [Element]))
    if all_raster:
        plot_class = RasterGridPlot
    else:
        plot_class = BasicGridPlot
    return plot_class(grid, **kwargs)

GridPlot = PlotWrapper(grid_wrapper, plots=[GridPlot, RasterGridPlot])

# Register default Elements
Store.register({Curve: CurvePlot,
                Scatter: PointPlot,
                Bars: BarPlot,
                Histogram: HistogramPlot,
                Points: PointPlot,
                VectorField: VectorFieldPlot,
                ErrorBars: ErrorPlot,

                # General plots
                GridSpace: GridPlot,
                NdLayout: LayoutPlot,
                Layout: LayoutPlot,
                AdjointLayout: AdjointLayoutPlot,

                # Element plots
                NdOverlay: OverlayPlot,
                Overlay: OverlayPlot,

                # Chart 3D
                Surface: SurfacePlot,
                Scatter3D: Scatter3DPlot,

                # Tabular plots
                ItemTable: TablePlot,
                Table: TablePlot,

                # Raster plots
                Raster: RasterPlot,
                HeatMap: RasterPlot,
                Image: RasterPlot,
                RGB: RasterPlot,
                HSV: RasterPlot,

                # Annotation plots
                VLine: VLinePlot,
                HLine: HLinePlot,
                Arrow: ArrowPlot,
                Spline: SplinePlot,
                Text: TextPlot,

                # Path plots
                Contours: PathPlot,
                Path:     PathPlot,
                Box:      PathPlot,
                Bounds:   PathPlot,
                Ellipse:  PathPlot,
                Polygons: PolygonPlot}, 'matplotlib', style_aliases=style_aliases)


MPLPlot.sideplots.update({Histogram: SideHistogramPlot,
                          GridSpace: GridPlot})

options = Store.options(backend='matplotlib')
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

