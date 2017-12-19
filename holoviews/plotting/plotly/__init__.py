from ...core.options import Store, Cycle, Options
from ...core import (Overlay, NdOverlay, Layout, NdLayout, GridSpace,
                     GridMatrix, config)
from ...element import *              # noqa (Element import for registration)
from .renderer import PlotlyRenderer

from .element import *                # noqa (API import)
from .chart import *                 # noqa (API import)
from .chart3d import *               # noqa (API import)
from .raster import *                # noqa (API import)
from .plot import *                  # noqa (API import)
from .tabular import *               # noqa (API import)

Store.renderers['plotly'] = PlotlyRenderer.instance()

if len(Store.renderers) == 1:
    Store.current_backend = 'plotly'

Store.register({Points: PointPlot,
                Scatter: PointPlot,
                Curve: CurvePlot,
                ErrorBars: ErrorBarsPlot,
                Bivariate: BivariatePlot,
                Distribution: DistributionPlot,
                Bars: BarPlot,
                BoxWhisker: BoxWhiskerPlot,

                # Raster plots
                Raster: RasterPlot,
                Image: RasterPlot,
                HeatMap: HeatMapPlot,
                QuadMesh: QuadMeshPlot,

                # 3D Plot
                Scatter3D: Scatter3dPlot,
                Surface: SurfacePlot,
                TriSurface: TriSurfacePlot,
                Trisurface: TriSurfacePlot, # Alias, remove in 2.0

                # Tabular
                Table: TablePlot,
                ItemTable: TablePlot,

                # Container Plots
                Overlay: OverlayPlot,
                NdOverlay: OverlayPlot,
                Layout: LayoutPlot,
                NdLayout: LayoutPlot,
                GridSpace: GridPlot,
                GridMatrix: GridPlot}, backend='plotly')


options = Store.options(backend='plotly')

dflt_cmap = 'hot' if config.style_17 else 'fire'

point_size = np.sqrt(6) # Matches matplotlib default
Cycle.default_cycles['default_colors'] =  ['#30a2da', '#fc4f30', '#e5ae38',
                                           '#6d904f', '#8b8b8b']

# Charts
options.Curve = Options('style', color=Cycle(), width=2)
options.ErrorBars = Options('style', color='black')
options.Scatter = Options('style', color=Cycle())
options.Points = Options('style', color=Cycle())
options.TriSurface = Options('style', cmap='viridis')

# Rasters
options.Image = Options('style', cmap=dflt_cmap)
options.Raster = Options('style', cmap=dflt_cmap)
options.QuadMesh = Options('style', cmap=dflt_cmap)
options.HeatMap = Options('style', cmap='RdBu_r')
