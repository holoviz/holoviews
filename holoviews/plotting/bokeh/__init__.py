import numpy as np

from ...core import (Store, Overlay, NdOverlay, Layout, AdjointLayout,
                     GridSpace, GridMatrix, NdLayout)
from ...element import (Curve, Points, Scatter, Image, Raster, Path,
                        RGB, Histogram, Spread, HeatMap, Contours, Bars,
                        Box, Bounds, Ellipse, Polygons, BoxWhisker,
                        ErrorBars, Text, HLine, VLine, Spline, Spikes,
                        Table, ItemTable, Area, HSV, QuadMesh, VectorField)
from ...core.options import Options, Cycle

try:
    from ...interface import DFrame
except:
    DFrame = None

from .annotation import TextPlot, LineAnnotationPlot, SplinePlot
from .callbacks import Callback # noqa (API import)
from .element import OverlayPlot
from .chart import (PointPlot, CurvePlot, SpreadPlot, ErrorPlot, HistogramPlot,
                    SideHistogramPlot, BoxPlot, BarPlot, SpikesPlot,
                    SideSpikesPlot, AreaPlot, VectorFieldPlot)
from .path import PathPlot, PolygonPlot
from .plot import GridPlot, LayoutPlot, AdjointLayoutPlot
from .raster import (RasterPlot, RGBPlot, HeatmapPlot,
                     HSVPlot, QuadMeshPlot)
from .renderer import BokehRenderer
from .tabular import TablePlot
from .util import bokeh_version


Store.renderers['bokeh'] = BokehRenderer.instance()

if len(Store.renderers) == 1:
    Store.current_backend = 'bokeh'

associations = {Overlay: OverlayPlot,
                NdOverlay: OverlayPlot,
                GridSpace: GridPlot,
                GridMatrix: GridPlot,
                AdjointLayout: AdjointLayoutPlot,
                Layout: LayoutPlot,
                NdLayout: LayoutPlot,

                # Charts
                Curve: CurvePlot,
                Points: PointPlot,
                Scatter: PointPlot,
                ErrorBars: ErrorPlot,
                Spread: SpreadPlot,
                Spikes: SpikesPlot,
                Area: AreaPlot,
                VectorField: VectorFieldPlot,

                # Rasters
                Image: RasterPlot,
                RGB: RGBPlot,
                HSV: HSVPlot,
                Raster: RasterPlot,
                HeatMap: HeatmapPlot,
                Histogram: HistogramPlot,
                QuadMesh: QuadMeshPlot,

                # Paths
                Path: PathPlot,
                Contours: PathPlot,
                Path:     PathPlot,
                Box:      PathPlot,
                Bounds:   PathPlot,
                Ellipse:  PathPlot,
                Polygons: PolygonPlot,

                # Annotations
                HLine: LineAnnotationPlot,
                VLine: LineAnnotationPlot,
                Text: TextPlot,
                Spline: SplinePlot,

                # Tabular
                Table: TablePlot,
                ItemTable: TablePlot}

if DFrame is not None:
    associations[DFrame] = TablePlot

Store.register(associations,
               'bokeh')


AdjointLayoutPlot.registry[Histogram] = SideHistogramPlot
AdjointLayoutPlot.registry[Spikes] = SideSpikesPlot

try:
    import pandas # noqa (Conditional import)
    Store.register({BoxWhisker: BoxPlot,
                    Bars: BarPlot}, 'bokeh')
except ImportError:
    pass

point_size = np.sqrt(6) # Matches matplotlib default
Cycle.default_cycles['default_colors'] =  ['#30a2da', '#fc4f30', '#e5ae38',
                                           '#6d904f', '#8b8b8b']

options = Store.options(backend='bokeh')

# Charts
options.Curve = Options('style', color=Cycle(), line_width=2)
options.Scatter = Options('style', color=Cycle(), size=point_size, cmap='hot')
options.Points = Options('style', color=Cycle(), size=point_size, cmap='hot')
options.Histogram = Options('style', line_color='black', fill_color=Cycle())
options.ErrorBars = Options('style', color='black')
options.Spread = Options('style', color=Cycle(), alpha=0.6, line_color='black')

options.Spikes = Options('style', color='black')
options.Area = Options('style', color=Cycle(), line_color='black')
options.VectorField = Options('style', color='black')

# Paths
options.Contours = Options('style', color=Cycle())
options.Path = Options('style', color=Cycle())
options.Box = Options('style', color='black')
options.Bounds = Options('style', color='black')
options.Ellipse = Options('style', color='black')
options.Polygons = Options('style', color=Cycle(), line_color='black')

# Rasters
options.Image = Options('style', cmap='hot')
options.GridImage = Options('style', cmap='hot')
options.Raster = Options('style', cmap='hot')
options.QuadMesh = Options('style', cmap='hot', line_alpha=0)
options.HeatMap = Options('style', cmap='RdYlBu_r', line_alpha=0)

# Annotations
options.HLine = Options('style', color=Cycle(), line_width=3, alpha=1)
options.VLine = Options('style', color=Cycle(), line_width=3, alpha=1)

# Define composite defaults
options.GridMatrix = Options('plot', shared_xaxis=True, shared_yaxis=True,
                             xaxis=None, yaxis=None)

if bokeh_version >= '0.12.5':
    options.Overlay = Options('style', click_policy='mute')
    options.NdOverlay = Options('style', click_policy='mute')
    options.Curve = Options('style', muted_alpha=0.2)
    options.Path = Options('style', muted_alpha=0.2)
    options.Scatter = Options('style', muted_alpha=0.2)
    options.Points = Options('style', muted_alpha=0.2)
    options.Polygons = Options('style', muted_alpha=0.2)

