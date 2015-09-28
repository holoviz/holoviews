from ...core import Store, Overlay, NdOverlay, Layout, AdjointLayout, GridSpace
from ...element import (Curve, Points, Scatter, Image, Raster, Path,
                        RGB, Histogram, Spread, HeatMap, Contours,
                        Path, Box, Bounds, Ellipse, Polygons,
                        ErrorBars, Text, HLine, VLine, Spline,
                        Table, ItemTable, Surface, Scatter3D)
from ...core.options import Options, Cycle, OptionTree
from ...interface import DFrame
from ..plot import PlotSelector
from ..mpl import SurfacePlot, Scatter3DPlot

from .annotation import TextPlot, LineAnnotationPlot, SplinePlot
from .element import OverlayPlot, BokehMPLWrapper, BokehMPLRawWrapper
from .chart import (PointPlot, CurvePlot, SpreadPlot, ErrorPlot, HistogramPlot,
                    AdjointHistogramPlot)
from .path import PathPlot, PolygonPlot
from .plot import GridPlot, LayoutPlot, AdjointLayoutPlot
from .raster import RasterPlot, RGBPlot, HeatmapPlot
from .renderer import BokehRenderer
from .tabular import TablePlot

Store.renderers['bokeh'] = BokehRenderer

Store.register({Overlay: OverlayPlot,
                NdOverlay: OverlayPlot,
                Curve: CurvePlot,
                Points: PointPlot,
                Scatter: PointPlot,
                Spread: SpreadPlot,
                HLine: LineAnnotationPlot,
                VLine: LineAnnotationPlot,
                GridSpace: GridPlot,
                Image: RasterPlot,
                RGB: RGBPlot,
                Raster: RasterPlot,
                HeatMap: HeatmapPlot,
                Histogram: HistogramPlot,
                AdjointLayout: AdjointLayoutPlot,
                Layout: LayoutPlot,
                Path: PathPlot,
                Contours: PathPlot,
                Path:     PathPlot,
                Box:      PathPlot,
                Bounds:   PathPlot,
                Ellipse:  PathPlot,
                Polygons: PolygonPlot,
                ErrorBars: ErrorPlot,
                Text: TextPlot,
                Spline: SplinePlot,
                Table: TablePlot,
                ItemTable: TablePlot,
                DFrame: TablePlot,
                Surface: PlotSelector(lambda x: 'bokeh',
                                      [('mpl', SurfacePlot),
                                       ('bokeh', BokehMPLRawWrapper)], True),
                Scatter3D: PlotSelector(lambda x: 'bokeh',
                                        [('mpl', Scatter3DPlot),
                                         ('bokeh', BokehMPLRawWrapper)], True)},
               'bokeh')


AdjointLayoutPlot.registry[Histogram] = AdjointHistogramPlot


try:
    from ..mpl.seaborn import TimeSeriesPlot, BivariatePlot, DistributionPlot
    from ...interface.seaborn import Bivariate, TimeSeries, Distribution
    Store.register({Distribution: PlotSelector(lambda x: 'bokeh',
                                               [('mpl', DistributionPlot),
                                                ('bokeh', BokehMPLWrapper)],
                                               True),
                    TimeSeries: PlotSelector(lambda x: 'bokeh',
                                             [('mpl', TimeSeriesPlot),
                                              ('bokeh', BokehMPLWrapper)],
                                             True),
                    Bivariate: PlotSelector(lambda x: 'bokeh',
                                        [('mpl', BivariatePlot),
                                         ('bokeh', BokehMPLWrapper)], True)},
                   'bokeh')
except ImportError:
    pass


Cycle.default_cycles['default_colors'] =  ['#30a2da', '#fc4f30', '#e5ae38',
                                           '#6d904f', '#8b8b8b']

options = Store.options(backend='bokeh')

# Charts
options.Curve = Options('style', color=Cycle(), line_width=2)
options.Scatter = Options('style', color=Cycle())
options.ErrorBars = Options('style', color='black')
options.Spread = Options('style', fill_color=Cycle(), fill_alpha=0.6, line_color='black')
options.Histogram = Options('style', fill_color="#036564", line_color="#033649")
options.Points = Options('style', color=Cycle())

# Paths
options.Contours = Options('style', color=Cycle())
options.Path = Options('style', color=Cycle())
options.Box = Options('style', color='black')
options.Bounds = Options('style', color='black')
options.Ellipse = Options('style', color='black')
options.Polygons = Options('style', color=Cycle())

# Rasters
options.Image = Options('style', cmap='hot')
options.Raster = Options('style', cmap='hot')
options.QuadMesh = Options('style', cmap='hot')
options.HeatMap = Options('style', cmap='RdYlBu_r', line_alpha=0)
