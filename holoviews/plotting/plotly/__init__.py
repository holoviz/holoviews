from ...core.options import Store, Cycle, Options
from ...core import (Overlay, NdOverlay, Layout, NdLayout, GridSpace,
                     GridMatrix)
from ...interface.seaborn import *
from ...element import * 
from .renderer import PlotlyRenderer
from .element import *
from .chart import *
from .chart3d import *
from .raster import *
from .plot import *
from .tabular import *

Store.renderers['plotly'] = PlotlyRenderer.instance()

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

                # 3D Plot
                Scatter3D: Scatter3dPlot,
                Surface: SurfacePlot,
                Trisurface: TrisurfacePlot,

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


point_size = np.sqrt(6) # Matches matplotlib default
Cycle.default_cycles['default_colors'] =  ['#30a2da', '#fc4f30', '#e5ae38',
                                           '#6d904f', '#8b8b8b']

# Charts
options.Curve = Options('style', color=Cycle(), width=2)
options.ErrorBars = Options('style', color='black')
options.Scatter = Options('style', color=Cycle())
options.Points = Options('style', color=Cycle())
