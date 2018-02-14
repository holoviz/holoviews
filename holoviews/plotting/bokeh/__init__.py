from __future__ import absolute_import

from distutils.version import LooseVersion

import numpy as np
import bokeh
from bokeh.palettes import all_palettes

from ...core import (Store, Overlay, NdOverlay, Layout, AdjointLayout,
                     GridSpace, GridMatrix, NdLayout, config)
from ...element import (Curve, Points, Scatter, Image, Raster, Path,
                        RGB, Histogram, Spread, HeatMap, Contours, Bars,
                        Box, Bounds, Ellipse, Polygons, BoxWhisker, Arrow,
                        ErrorBars, Text, HLine, VLine, Spline, Spikes,
                        Table, ItemTable, Area, HSV, QuadMesh, VectorField,
                        Graph, Nodes, EdgePaths, Distribution, Bivariate,
                        TriMesh, Violin, Chord)
from ...core.options import Options, Cycle, Palette
from ...core.util import VersionError

if LooseVersion(bokeh.__version__) < '0.12.10':
    raise VersionError("The bokeh extension requires a bokeh version >=0.12.10, "
                       "please upgrade from bokeh %s to a more recent version."
                       % bokeh.__version__, bokeh.__version__, '0.12.10')

try:
    from ...interface import DFrame
except:
    DFrame = None

from .annotation import TextPlot, LineAnnotationPlot, SplinePlot, ArrowPlot
from .callbacks import Callback # noqa (API import)
from .element import OverlayPlot, ElementPlot
from .chart import (PointPlot, CurvePlot, SpreadPlot, ErrorPlot, HistogramPlot,
                    SideHistogramPlot, BarPlot, SpikesPlot, SideSpikesPlot,
                    AreaPlot, VectorFieldPlot)
from .graphs import GraphPlot, NodePlot, TriMeshPlot, ChordPlot
from .path import PathPlot, PolygonPlot, ContourPlot
from .plot import GridPlot, LayoutPlot, AdjointLayoutPlot
from .raster import RasterPlot, RGBPlot, HeatMapPlot, HSVPlot, QuadMeshPlot
from .renderer import BokehRenderer
from .stats import DistributionPlot, BivariatePlot, BoxWhiskerPlot, ViolinPlot
from .tabular import TablePlot
from .util import bokeh_version # noqa (API import)


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
                Bars: BarPlot,
                Points: PointPlot,
                Scatter: PointPlot,
                ErrorBars: ErrorPlot,
                Spread: SpreadPlot,
                Spikes: SpikesPlot,
                Area: AreaPlot,
                VectorField: VectorFieldPlot,
                Histogram: HistogramPlot,

                # Rasters
                Image: RasterPlot,
                RGB: RGBPlot,
                HSV: HSVPlot,
                Raster: RasterPlot,
                HeatMap: HeatMapPlot,
                QuadMesh: QuadMeshPlot,

                # Paths
                Path: PathPlot,
                Contours: ContourPlot,
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
                Arrow: ArrowPlot,

                # Graph Elements
                Graph: GraphPlot,
                Chord: ChordPlot,
                Nodes: NodePlot,
                EdgePaths: PathPlot,
                TriMesh: TriMeshPlot,

                # Tabular
                Table: TablePlot,
                ItemTable: TablePlot,

                # Statistics
                Distribution: DistributionPlot,
                Bivariate: BivariatePlot,
                BoxWhisker: BoxWhiskerPlot,
                Violin: ViolinPlot}


if DFrame is not None:
    associations[DFrame] = TablePlot

Store.register(associations, 'bokeh')

if config.style_17:
    ElementPlot.show_grid = True
    RasterPlot.show_grid = True

    ElementPlot.show_frame = True
else:
    # Raster types, Path types and VectorField should have frames
    for framedcls in [VectorFieldPlot, ContourPlot, PathPlot, PolygonPlot,
                      RasterPlot, RGBPlot, HSVPlot, QuadMeshPlot, HeatMapPlot]:
        framedcls.show_frame = True


AdjointLayoutPlot.registry[Histogram] = SideHistogramPlot
AdjointLayoutPlot.registry[Spikes] = SideSpikesPlot


point_size = np.sqrt(6) # Matches matplotlib default
Cycle.default_cycles['default_colors'] =  ['#30a2da', '#fc4f30', '#e5ae38',
                                           '#6d904f', '#8b8b8b']

# Register bokeh.palettes with Palette and Cycle
def colormap_generator(palette):
    return lambda value: palette[int(value*(len(palette)-1))]

Palette.colormaps.update({name: colormap_generator(p[max(p.keys())])
                          for name, p in all_palettes.items()})

Cycle.default_cycles.update({name: p[max(p.keys())] for name, p in all_palettes.items()
                             if max(p.keys()) < 256})

dflt_cmap = 'hot' if config.style_17 else 'fire'
options = Store.options(backend='bokeh')

# Charts
options.Curve = Options('style', color=Cycle(), line_width=2)
options.BoxWhisker = Options('style', box_fill_color=Cycle(), whisker_color='black',
                             box_line_color='black', outlier_color='black')
options.Scatter = Options('style', color=Cycle(), size=point_size, cmap=dflt_cmap)
options.Points = Options('style', color=Cycle(), size=point_size, cmap=dflt_cmap)
if not config.style_17:
    options.Points = Options('plot', show_frame=True)

options.Histogram = Options('style', line_color='black', fill_color=Cycle(), muted_alpha=0.2)
options.ErrorBars = Options('style', color='black')
options.Spread = Options('style', color=Cycle(), alpha=0.6, line_color='black')
options.Bars = Options('style', color=Cycle(), line_color='black', width=0.8)

options.Spikes = Options('style', color='black', cmap='fire')
options.Area = Options('style', color=Cycle(), alpha=1, line_color='black')
options.VectorField = Options('style', color='black')

# Paths
if not config.style_17:
    options.Contours = Options('plot', show_legend=True)
options.Contours = Options('style', color=Cycle(), cmap='viridis')
options.Path = Options('style', color=Cycle(), cmap='viridis')
options.Box = Options('style', color='black')
options.Bounds = Options('style', color='black')
options.Ellipse = Options('style', color='black')
options.Polygons = Options('style', color=Cycle(), line_color='black',
                           cmap='viridis')

# Rasters
options.Image = Options('style', cmap=dflt_cmap)
options.GridImage = Options('style', cmap=dflt_cmap)
options.Raster = Options('style', cmap=dflt_cmap)
options.QuadMesh = Options('style', cmap=dflt_cmap, line_alpha=0)
options.HeatMap = Options('style', cmap='RdYlBu_r', line_alpha=0)

# Annotations
options.HLine = Options('style', color=Cycle(), line_width=3, alpha=1)
options.VLine = Options('style', color=Cycle(), line_width=3, alpha=1)
options.Arrow = Options('style', arrow_size=10)

# Graphs
options.Graph = Options('style', node_size=15, node_fill_color=Cycle(),
                        node_line_color='black',
                        node_nonselection_fill_color=Cycle(),
                        node_hover_line_color='black',
                        node_hover_fill_color='limegreen',
                        node_nonselection_alpha=0.2,
                        edge_nonselection_alpha=0.2,
                        node_nonselection_line_color='black',
                        edge_line_color='black', edge_line_width=2,
                        edge_nonselection_line_color='black',
                        edge_hover_line_color='limegreen')
options.TriMesh = Options('style', node_size=5, node_line_color='black',
                          node_fill_color='white', edge_line_color='black',
                          node_hover_fill_color='limegreen',
                          edge_hover_line_color='limegreen',
                          edge_nonselection_alpha=0.2,
                          edge_nonselection_line_color='black',
                          node_nonselection_alpha=0.2,
                          edge_line_width=1)
options.TriMesh = Options('plot', tools=[])
options.Chord = Options('style', node_size=15, node_fill_color=Cycle(),
                        node_line_color='black',
                        node_selection_fill_color='limegreen',
                        node_nonselection_fill_color=Cycle(),
                        node_hover_line_color='black',
                        node_nonselection_line_color='black',
                        node_selection_line_color='black',
                        node_hover_fill_color='limegreen',
                        node_nonselection_alpha=0.2,
                        edge_nonselection_alpha=0.1,
                        edge_line_color='black', edge_line_width=1,
                        edge_nonselection_line_color='black',
                        edge_hover_line_color='limegreen',
                        edge_selection_line_color='limegreen',
                        label_text_font_size='8pt')
options.Chord = Options('plot', xaxis=None, yaxis=None)
options.Nodes = Options('style', line_color='black', color=Cycle(),
                        size=20, nonselection_fill_color=Cycle(),
                        selection_fill_color='limegreen',
                        hover_fill_color='indianred')
options.Nodes = Options('plot', tools=['hover', 'tap'])
options.EdgePaths = Options('style', color='black', nonselection_alpha=0.2,
                            line_width=2, selection_color='limegreen',
                            hover_line_color='indianred')
options.EdgePaths = Options('plot', tools=['hover', 'tap'])

# Define composite defaults
options.GridMatrix = Options('plot', shared_xaxis=True, shared_yaxis=True,
                             xaxis=None, yaxis=None)

options.Overlay = Options('style', click_policy='mute')
options.NdOverlay = Options('style', click_policy='mute')
options.Curve = Options('style', muted_alpha=0.2)
options.Path = Options('style', muted_alpha=0.2)
options.Scatter = Options('style', muted_alpha=0.2)
options.Points = Options('style', muted_alpha=0.2)
options.Polygons = Options('style', muted_alpha=0.2)

# Statistics
options.Distribution = Options('style', fill_color=Cycle(), line_color='black',
                               fill_alpha=0.5, muted_alpha=0.2)
options.Violin = Options('style', violin_fill_color=Cycle(),
                         violin_line_color='black', violin_fill_alpha=0.5,
                         stats_color='black', box_color='black',
                         median_color='white')
