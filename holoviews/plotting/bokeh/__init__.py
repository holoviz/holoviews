from ...core import Store, Overlay, NdOverlay, Layout, AdjointLayout
from ...element import Curve, Points, Scatter, Image, Raster, Path, RGB, Histogram
from ...element import Contours, Path, Box, Bounds, Ellipse, Polygons, ErrorBars, Text
from ...interface.seaborn import Bivariate, TimeSeries
from ...core.options import Options, Cycle, OptionTree
from ..plot import PlotSelector
from ..mpl.seaborn import TimeSeriesPlot, BivariatePlot
from .plot import *
from .element import *
from .renderer import BokehRenderer

Store.renderers['bokeh'] = BokehRenderer

def wrapper(obj):
    return 'bokeh'

Store.register({Overlay: OverlayPlot,
                NdOverlay: OverlayPlot,
                Curve: CurvePlot,
                Points: PointPlot,
                Scatter: PointPlot,
                LinkedScatter: LinkedScatterPlot,
                Image: RasterPlot,
                RGB: RasterPlot,
                Raster: RasterPlot,
                Histogram: HistogramPlot,
                AdjointLayout: AdjointLayoutPlot,
                Layout: LayoutPlot,
                Path: PathPlot,
                TimeSeries: PlotSelector(wrapper, [('mpl', TimeSeriesPlot), ('bokeh', BokehMPLWrapper)], True),
                Bivariate: PlotSelector(wrapper, [('mpl', BivariatePlot), ('bokeh', BokehMPLWrapper)], True),
                Contours: PathPlot,
                Path:     PathPlot,
                Box:      PathPlot,
                Bounds:   PathPlot,
                Ellipse:  PathPlot,
                Polygons: PolygonPlot,
                ErrorBars: ErrorPlot,
                Text: TextPlot}, 'bokeh')

Cycle.default_cycles['default_colors'] =  ['#30a2da', '#fc4f30', '#e5ae38',
                                           '#6d904f', '#8b8b8b']

options = Store.options(backend='bokeh')
options.Scatter = Options('style', color=Cycle())
options.Curve = Options('style', color=Cycle(), line_width=2)
options.Polygons = Options('style', color=Cycle())
options.LinkedScatter = Options('style', size=12, color=Cycle(),
                                marker=Cycle(values=['circle', 'square', 'triangle', 'diamond', 'inverted_triangle']))
options.Histogram = Options('style', fill_color="#036564", line_color="#033649")
