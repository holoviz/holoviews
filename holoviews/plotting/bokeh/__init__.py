from ...core import Store, Overlay, NdOverlay, Layout, AdjointLayout
from ...element import Curve, Points, Scatter, Image, Raster, Path, RGB
from ...element import Contours, Path, Box, Bounds, Ellipse, Polygons, ErrorBars, Text
from ...core.options import Options, Cycle, OptionTree
from .plot import *
from .element import *
from .renderer import BokehRenderer

Store.renderers['bokeh'] = BokehRenderer

Store.register({Overlay: OverlayPlot,
                NdOverlay: OverlayPlot,
                Curve: CurvePlot,
                Points: PointPlot,
                Scatter: PointPlot,
                LinkedScatter: LinkedScatterPlot,
                Image: RasterPlot,
                RGB: RasterPlot,
                Raster: RasterPlot,
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
                Text: TextPlot}, 'bokeh')

Cycle.default_cycles['default_colors'] =  ['#30a2da', '#fc4f30', '#e5ae38',
                                           '#6d904f', '#8b8b8b']

options = Store.options(backend='bokeh')
options.Scatter = Options('style', color=Cycle())
options.Curve = Options('style', color=Cycle(), line_width=2)
options.Polygons = Options('style', color=Cycle())
options.LinkedScatter = Options('style', size=12, color=Cycle(),
                                marker=Cycle(values=['circle', 'square', 'triangle', 'diamond', 'inverted_triangle']))
