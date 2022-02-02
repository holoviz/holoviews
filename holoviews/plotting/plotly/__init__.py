import plotly

from param import concrete_descendents

from ...core import (
    Overlay, NdOverlay, Layout, NdLayout, GridSpace, GridMatrix, config
)
from ...core.options import Store, Cycle, Options
from ...core.util import LooseVersion, VersionError
from ...element import *              # noqa (Element import for registration)

from .element import ElementPlot
from .renderer import PlotlyRenderer

from .annotation import *            # noqa (API import)
from .tiles import *                 # noqa (API import)
from .element import *               # noqa (API import)
from .chart import *                 # noqa (API import)
from .chart3d import *               # noqa (API import)
from .raster import *                # noqa (API import)
from .plot import *                  # noqa (API import)
from .stats import *                 # noqa (API import)
from .tabular import *               # noqa (API import)
from .callbacks import *             # noqa (API import)
from .shapes import *                # noqa (API import)
from .images import *                # noqa (API import)

if LooseVersion(plotly.__version__) < LooseVersion('4.0.0'):
    raise VersionError(
        "The plotly extension requires a plotly version >=4.0.0, "
        "please upgrade from plotly %s to a more recent version."
        % plotly.__version__, plotly.__version__, '4.0.0')

Store.renderers['plotly'] = PlotlyRenderer.instance()

if len(Store.renderers) == 1:
    Store.set_current_backend('plotly')

Store.register({Points: ScatterPlot,
                Scatter: ScatterPlot,
                Curve: CurvePlot,
                Area: AreaPlot,
                Spread: SpreadPlot,
                ErrorBars: ErrorBarsPlot,

                # Statistics elements
                Bivariate: BivariatePlot,
                Distribution: DistributionPlot,
                Bars: BarPlot,
                Histogram: HistogramPlot,
                BoxWhisker: BoxWhiskerPlot,
                Violin: ViolinPlot,

                # Raster plots
                Raster: RasterPlot,
                Image: RasterPlot,
                HeatMap: HeatMapPlot,
                QuadMesh: QuadMeshPlot,
                RGB: RGBPlot,

                # 3D Plot
                Scatter3D: Scatter3DPlot,
                Surface: SurfacePlot,
                Path3D: Path3DPlot,
                TriSurface: TriSurfacePlot,

                # Tabular
                Table: TablePlot,
                ItemTable: TablePlot,

                # Annotations
                Labels: LabelPlot,
                Tiles: TilePlot,

                # Shapes
                Box: PathShapePlot,
                Bounds: PathShapePlot,
                Ellipse: PathShapePlot,
                Rectangles: BoxShapePlot,
                Segments: SegmentShapePlot,
                Path: PathsPlot,
                HLine: HVLinePlot,
                VLine: HVLinePlot,
                HSpan: HVSpanPlot,
                VSpan: HVSpanPlot,

                # Container Plots
                Overlay: OverlayPlot,
                NdOverlay: OverlayPlot,
                Layout: LayoutPlot,
                AdjointLayout: AdjointLayoutPlot,
                NdLayout: LayoutPlot,
                GridSpace: GridPlot,
                GridMatrix: GridPlot}, backend='plotly')


options = Store.options(backend='plotly')

if config.no_padding:
    for plot in concrete_descendents(ElementPlot).values():
        plot.padding = 0

dflt_cmap = config.default_cmap
dflt_shape_line_color = '#2a3f5f'  # Line color of default plotly template

point_size = np.sqrt(6) # Matches matplotlib default
Cycle.default_cycles['default_colors'] =  ['#30a2da', '#fc4f30', '#e5ae38',
                                           '#6d904f', '#8b8b8b']

# Charts
options.Curve = Options('style', color=Cycle(), line_width=2)
options.ErrorBars = Options('style', color='black')
options.Scatter = Options('style', color=Cycle(), cmap=dflt_cmap)
options.Points = Options('style', color=Cycle(), cmap=dflt_cmap)
options.Area = Options('style', color=Cycle(), line_width=2)
options.Spread = Options('style', color=Cycle(), line_width=2)
options.TriSurface = Options('style', cmap=dflt_cmap)
options.Histogram = Options('style', color=Cycle(), line_width=1, line_color='black')

# Rasters
options.Image = Options('style', cmap=config.default_gridded_cmap)
options.Raster = Options('style', cmap=config.default_gridded_cmap)
options.QuadMesh = Options('style', cmap=config.default_gridded_cmap)
options.HeatMap = Options('style', cmap=config.default_heatmap_cmap)

# Disable padding for image-like elements
options.Image = Options("plot", padding=0)
options.Raster = Options("plot", padding=0)
options.RGB = Options("plot", padding=0)

# 3D
options.Scatter3D = Options('style', color=Cycle(), size=6)

# Annotations
options.VSpan = Options('style', fillcolor=Cycle(), opacity=0.5)
options.HSpan = Options('style', fillcolor=Cycle(), opacity=0.5)

# Shapes
options.Rectangles = Options('style', line_color=dflt_shape_line_color)
options.Bounds = Options('style', line_color=dflt_shape_line_color)
options.Path = Options('style', line_color=dflt_shape_line_color)
options.Segments = Options('style', line_color=dflt_shape_line_color)
options.Box = Options('style', line_color=dflt_shape_line_color)
