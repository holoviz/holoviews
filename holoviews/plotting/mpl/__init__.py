import os
from distutils.version import LooseVersion

from matplotlib import rc_params_from_file
from matplotlib.colors import ListedColormap

from ...core import Layout, Collator, GridMatrix, config
from ...core.options import Cycle, Palette, Options
from ...core.overlay import NdOverlay, Overlay
from ...element import * # noqa (API import)
from ..plot import PlotSelector
from .annotation import * # noqa (API import)
from .chart import * # noqa (API import)
from .chart3d import * # noqa (API import)
from .graphs import * # noqa (API import)
from .path import * # noqa (API import)
from .plot import * # noqa (API import)
from .raster import * # noqa (API import)
from .stats import * # noqa (API import)
from .tabular import * # noqa (API import)

from .renderer import MPLRenderer


mpl_ge_150 = LooseVersion(mpl.__version__) >= '1.5.0'


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
        new_style = rc_params_from_file(path, use_default_template=False)
        styles['backup'] = dict(plt.rcParams)

        plt.rcParams.update(new_style)


# Define matplotlib based style cycles and Palettes
def get_color_cycle():
    if mpl_ge_150:
        cyl = mpl.rcParams['axes.prop_cycle']
        # matplotlib 1.5 verifies that axes.prop_cycle *is* a cycler
        # but no guarantee that there's a `color` key.
        # so users could have a custom rcParams w/ no color...
        try:
            return [x['color'] for x in cyl]
        except KeyError:
            pass  # just return axes.color style below
    return mpl.rcParams['axes.color_cycle']


styles = {'default': './default.mplstyle',
          'default>1.5': './default1.5.mplstyle'}

if config.style_17:
    if mpl_ge_150:
        set_style('default>1.5')
    else:
        set_style('default')
    Cycle.default_cycles.update({'default_colors': get_color_cycle()})
else:
    Cycle.default_cycles['default_colors'] =  ['#30a2da', '#fc4f30', '#e5ae38',
                                               '#6d904f', '#8b8b8b']

# Define Palettes and cycles from matplotlib colormaps
Palette.colormaps.update({cm: plt.get_cmap(cm) for cm in plt.cm.datad
                          if 'spectral' not in cm and 'Vega' not in cm})
listed_cmaps = [cm for cm in Palette.colormaps.values() if isinstance(cm, ListedColormap)]
Cycle.default_cycles.update({cm.name: list(cm.colors) for cm in listed_cmaps})

style_aliases = {'edgecolor': ['ec', 'ecolor'], 'facecolor': ['fc'],
                 'linewidth': ['lw'], 'edgecolors': ['ec', 'edgecolor'],
                 'size': ['s'], 'color': ['c'], 'markeredgecolor': ['mec'],
                 'markeredgewidth': ['mew'], 'markerfacecolor': ['mfc'],
                 'markersize': ['ms']}

Store.renderers['matplotlib'] = MPLRenderer.instance()

if len(Store.renderers) == 1:
    Store.current_backend = 'matplotlib'

# Defines a wrapper around GridPlot and RasterGridPlot
# switching to RasterGridPlot if the plot only contains
# Raster Elements
BasicGridPlot = GridPlot
def grid_selector(grid):
    raster_fn = lambda x: True if isinstance(x, Raster) else False
    all_raster = all(grid.traverse(raster_fn, [Element]))
    return 'RasterGridPlot' if all_raster else 'GridPlot'

GridPlot = PlotSelector(grid_selector,
                        plot_classes=[('GridPlot', BasicGridPlot),
                                      ('RasterGridPlot', RasterGridPlot)])

# Register default Elements
Store.register({Curve: CurvePlot,
                Scatter: PointPlot,
                Bars: BarPlot,
                Histogram: HistogramPlot,
                Points: PointPlot,
                VectorField: VectorFieldPlot,
                ErrorBars: ErrorPlot,
                Spread: SpreadPlot,
                Spikes: SpikesPlot,
                BoxWhisker: BoxPlot,
                Area: AreaPlot,

                # General plots
                GridSpace: GridPlot,
                GridMatrix: GridPlot,
                NdLayout: LayoutPlot,
                Layout: LayoutPlot,
                AdjointLayout: AdjointLayoutPlot,

                # Element plots
                NdOverlay: OverlayPlot,
                Overlay: OverlayPlot,

                # Chart 3D
                Surface: SurfacePlot,
                Trisurface: TrisurfacePlot,
                Scatter3D: Scatter3DPlot,

                # Tabular plots
                ItemTable: TablePlot,
                Table: TablePlot,
                Collator: TablePlot,

                # Raster plots
                QuadMesh: QuadMeshPlot,
                Raster: RasterPlot,
                HeatMap: HeatMapPlot,
                Image: RasterPlot,
                RGB: RasterPlot,
                HSV: RasterPlot,

                # Graph Elements
                Graph: GraphPlot,
                Nodes: PointPlot,
                EdgePaths: PathPlot,

                # Annotation plots
                VLine: VLinePlot,
                HLine: HLinePlot,
                Arrow: ArrowPlot,
                Spline: SplinePlot,
                Text: TextPlot,

                # Path plots
                Contours: ContourPlot,
                Path:     PathPlot,
                Box:      PathPlot,
                Bounds:   PathPlot,
                Ellipse:  PathPlot,
                Polygons: PolygonPlot,

                # Statistics elements
                Distribution: DistributionPlot,
                Bivariate: BivariatePlot}, 'matplotlib', style_aliases=style_aliases)


MPLPlot.sideplots.update({Histogram: SideHistogramPlot,
                          GridSpace: GridPlot,
                          Spikes: SideSpikesPlot,
                          BoxWhisker: SideBoxPlot})

if config.style_17:
    CurvePlot.show_grid = True
    SideHistogramPlot.show_grid = True
    PointPlot.show_grid = True

    MPLPlot.show_frame = True
    for framelesscls in [RasterGridPlot, GridPlot,
                         AdjoinedPlot, Plot3D, CurvePlot, HistogramPlot]:
        framelesscls.show_frame = False
else:
    # Raster types, Path types and VectorField should have frames
    for framedcls in [VectorFieldPlot, ContourPlot, PathPlot, RasterPlot,
                      QuadMeshPlot, HeatMapPlot, PolygonPlot]:
        framedcls.show_frame = True


options = Store.options(backend='matplotlib')
dflt_cmap = 'hot' if config.style_17 else 'fire'
# Default option definitions
# Note: *No*short aliases here! e.g use 'facecolor' instead of 'fc'

# Charts
options.Curve = Options('style', color=Cycle(), linewidth=2)
options.Scatter = Options('style', color=Cycle(), marker='o', cmap=dflt_cmap)

if not config.style_17:
    options.Points = Options('plot', show_frame=True)

options.ErrorBars = Options('style', ecolor='k')
options.Spread = Options('style', facecolor=Cycle(), alpha=0.6, edgecolor='k', linewidth=0.5)
options.Bars = Options('style', ec='k', color=Cycle())
options.Histogram = Options('style', ec='k', facecolor=Cycle())
options.Points = Options('style', color=Cycle(), marker='o', cmap=dflt_cmap)
options.Scatter3D = Options('style', c=Cycle(), marker='o')
options.Scatter3D = Options('plot', fig_size=150)
options.Surface = Options('plot', fig_size=150)
options.Surface = Options('style', cmap='fire')
options.Spikes = Options('style', color='black', cmap='fire')
options.Area = Options('style', facecolor=Cycle(), edgecolor='black')
options.BoxWhisker = Options('style', boxprops=dict(color='k', linewidth=1.5),
                             whiskerprops=dict(color='k', linewidth=1.5))

# Rasters
options.Image = Options('style', cmap=dflt_cmap, interpolation='nearest')
options.GridImage = Options('style', cmap=dflt_cmap, interpolation='nearest')
options.Raster = Options('style', cmap=dflt_cmap, interpolation='nearest')
options.QuadMesh = Options('style', cmap=dflt_cmap)
options.HeatMap = Options('style', cmap='RdYlBu_r', interpolation='nearest')
options.HeatMap = Options('plot', show_values=True, xticks=20, yticks=20)
options.RGB = Options('style', interpolation='nearest')
# Composites
options.Layout = Options('plot', sublabel_format='{Alpha}')
options.GridMatrix = Options('plot', fig_size=160, shared_xaxis=True,
                             shared_yaxis=True, xaxis=None, yaxis=None)

# Annotations
options.VLine = Options('style', color=Cycle())
options.HLine = Options('style', color=Cycle())
if config.style_17:
    options.Spline = Options('style', linewidth=2, edgecolor='r')
else:
    options.Spline = Options('style', edgecolor=Cycle())

options.Text = Options('style', fontsize=13)
options.Arrow = Options('style', color='k', linewidth=2, fontsize=13)
# Paths
options.Contours = Options('style', color=Cycle(), cmap='viridis')
options.Contours = Options('plot', show_legend=True)
options.Path = Options('style', color=Cycle(), cmap='viridis')

if config.style_17:
    options.Box = Options('style', color=Cycle())
    options.Bounds = Options('style', color=Cycle())
    options.Ellipse = Options('style', color=Cycle())
else:
    options.Box = Options('style', color='black')
    options.Bounds = Options('style', color='black')
    options.Ellipse = Options('style', color='black')
    options.Polygons = Options('style', facecolor=Cycle(), edgecolor='black',
                               cmap='viridis')

# Interface
options.TimeSeries = Options('style', color=Cycle())

# Graphs
options.Graph = Options('style', node_edgecolors='black', node_facecolors=Cycle(),
                        edge_color='black', node_size=20)
options.Nodes = Options('style', edgecolors='black', facecolors=Cycle(),
                        marker='o', s=20**2)
options.EdgePaths = Options('style', color='black')

# Statistics
options.Distribution = Options('style', facecolor=Cycle(), edgecolor='black',
                               alpha=0.5)
