from ...core.options import Store
from ...core import Overlay, NdOverlay
from ...interface.seaborn import *
from ...element import * 
from .renderer import PlotlyRenderer
from .element import *
from .chart import *
from .raster import *

Store.renderers['plotly'] = PlotlyRenderer

Store.register({Points: PointPlot,
                Scatter: PointPlot,
                Curve: CurvePlot,
                Raster: RasterPlot,
                Image: RasterPlot,
                Surface: SurfacePlot,
                Bivariate: BivariatePlot,
                Distribution: DistributionPlot,
                HeatMap: HeatMapPlot,
                VectorField: VectorPlot,
                ErrorBars: ErrorBarsPlot,
                Scatter3D: Scatter3dPlot,
                Overlay: OverlayPlot,
                NdOverlay: OverlayPlot}, backend='plotly')
