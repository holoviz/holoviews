from __future__ import absolute_import, division, unicode_literals

from ...core.options import Store, Cycle, Options
from ...core import (Overlay, NdOverlay, Layout, NdLayout, GridSpace,
                     GridMatrix, config)
from ...element import *              # noqa (Element import for registration)
from .renderer import PlotlyRenderer

from .annotation import *            # noqa (API import)
from .element import *               # noqa (API import)
from .chart import *                 # noqa (API import)
from .chart3d import *               # noqa (API import)
from .raster import *                # noqa (API import)
from .plot import *                  # noqa (API import)
from .stats import *                 # noqa (API import)
from .tabular import *               # noqa (API import)
from ...core.util import LooseVersion, VersionError
import plotly

if LooseVersion(plotly.__version__) < '3.4.0':
    raise VersionError(
        "The plotly extension requires a plotly version >=3.4.0, "
        "please upgrade from plotly %s to a more recent version."
        % plotly.__version__, plotly.__version__, '3.4.0')

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
                BoxWhisker: BoxWhiskerPlot,
                Violin: ViolinPlot,

                # Raster plots
                Raster: RasterPlot,
                Image: RasterPlot,
                HeatMap: HeatMapPlot,
                QuadMesh: QuadMeshPlot,

                # 3D Plot
                Scatter3D: Scatter3DPlot,
                Surface: SurfacePlot,
                Path3D: Path3DPlot,
                TriSurface: TriSurfacePlot,
                Trisurface: TriSurfacePlot, # Alias, remove in 2.0

                # Tabular
                Table: TablePlot,
                ItemTable: TablePlot,

                # Annotations
                Labels: LabelPlot,

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
options.Curve = Options('style', color=Cycle(), line_width=2)
options.ErrorBars = Options('style', color='black')
options.Scatter = Options('style', color=Cycle())
options.Points = Options('style', color=Cycle())
options.Area = Options('style', color=Cycle(), line_width=2)
options.Spread = Options('style', color=Cycle(), line_width=2)
options.TriSurface = Options('style', cmap='viridis')

# Rasters
options.Image = Options('style', cmap=dflt_cmap)
options.Raster = Options('style', cmap=dflt_cmap)
options.QuadMesh = Options('style', cmap=dflt_cmap)
options.HeatMap = Options('style', cmap='RdBu_r')
