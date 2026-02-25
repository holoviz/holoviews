from ..core import HoloMap
from ..core.data import DataConversion, Dataset
from ..core.element import Element
from .annotation import (
    Annotation,
    Arrow,
    Div,
    HLine,
    HLines,
    HSpan,
    HSpans,
    Labels,
    Slope,
    Spline,
    Text,
    VectorizedAnnotation,
    VLine,
    VLines,
    VSpan,
    VSpans,
)
from .chart import (  # noqa: F401
    Area,
    Bars,
    Chart,
    Curve,
    ErrorBars,
    Histogram,
    Scatter,
    Spikes,
    Spread,
)
from .chart3d import Path3D, Scatter3D, Surface, TriSurface
from .geom import Points, Rectangles, Segments, VectorField
from .graphs import (  # noqa: F401
    Chord,
    EdgePaths,
    Graph,
    Nodes,
    TriMesh,
    circular_layout,
)
from .path import Bounds, Box, Contours, Dendrogram, Ellipse, Path, Polygons
from .raster import HSV, RGB, HeatMap, Image, ImageStack, QuadMesh, Raster
from .sankey import Sankey
from .stats import Bivariate, BoxWhisker, Distribution, HexTiles, Violin
from .tabular import ItemTable, Table
from .tiles import Tiles


class ElementConversion(DataConversion):
    """ElementConversion is a subclass of DataConversion providing
    concrete methods to convert a Dataset to specific Element
    types.

    """

    def bars(self, kdims=None, vdims=None, groupby=None, **kwargs):
        return self(Bars, kdims, vdims, groupby, **kwargs)

    def box(self, kdims=None, vdims=None, groupby=None, **kwargs):
        return self(BoxWhisker, kdims, vdims, groupby, **kwargs)

    def bivariate(self, kdims=None, vdims=None, groupby=None, **kwargs):
        return self(Bivariate, kdims, vdims, groupby, **kwargs)

    def curve(self, kdims=None, vdims=None, groupby=None, **kwargs):
        return self(Curve, kdims, vdims, groupby, **kwargs)

    def errorbars(self, kdims=None, vdims=None, groupby=None, **kwargs):
        return self(ErrorBars, kdims, vdims, groupby, **kwargs)

    def distribution(self, dim=None, groupby=None, **kwargs):
        if groupby is None:
            groupby = []
        if dim is None:
            if self._element.vdims:
                dim = self._element.vdims[0]
            else:
                raise Exception('Must supply an explicit value dimension '
                                'if no value dimensions are defined ')
        if groupby:
            reindexed = self._element.reindex(groupby, [dim])
            kwargs['kdims'] = dim
            kwargs['vdims'] = None
            return reindexed.groupby(groupby, HoloMap, Distribution, **kwargs)
        else:
            element = self._element
            params = dict(kdims=[element.get_dimension(dim)],
                          label=element.label)
            if element.group != element.param['group'].default:
                params['group'] = element.group
            return Distribution((element.dimension_values(dim),),
                                **dict(params, **kwargs))

    def heatmap(self, kdims=None, vdims=None, groupby=None, **kwargs):
        return self(HeatMap, kdims, vdims, groupby, **kwargs)

    def image(self, kdims=None, vdims=None, groupby=None, **kwargs):
        return self(Image, kdims, vdims, groupby, **kwargs)

    def points(self, kdims=None, vdims=None, groupby=None, **kwargs):
        return self(Points, kdims, vdims, groupby, **kwargs)

    def raster(self, kdims=None, vdims=None, groupby=None, **kwargs):
        heatmap = self.heatmap(kdims, vdims, **kwargs)
        return Raster(heatmap.data, **self._element.param.values(onlychanged=True))

    def scatter(self, kdims=None, vdims=None, groupby=None, **kwargs):
        return self(Scatter, kdims, vdims, groupby, **kwargs)

    def scatter3d(self, kdims=None, vdims=None, groupby=None, **kwargs):
        return self(Scatter3D, kdims, vdims, groupby, **kwargs)

    def spikes(self, kdims=None, vdims=None, groupby=None, **kwargs):
        return self(Spikes, kdims, vdims, groupby, **kwargs)

    def spread(self, kdims=None, vdims=None, groupby=None, **kwargs):
        return self(Spread, kdims, vdims, groupby, **kwargs)

    def surface(self, kdims=None, vdims=None, groupby=None, **kwargs):
        heatmap = self.heatmap(kdims, vdims, **kwargs)
        return Surface(heatmap.data, **self._table.param.values(onlychanged=True))

    def trisurface(self, kdims=None, vdims=None, groupby=None, **kwargs):
        return self(TriSurface, kdims, vdims, groupby, **kwargs)

    def vectorfield(self, kdims=None, vdims=None, groupby=None, **kwargs):
        return self(VectorField, kdims, vdims, groupby, **kwargs)

    def violin(self, kdims=None, vdims=None, groupby=None, **kwargs):
        return self(Violin, kdims, vdims, groupby, **kwargs)

    def labels(self, kdims=None, vdims=None, groupby=None, **kwargs):
        return self(Labels, kdims, vdims, groupby, **kwargs)

    def chord(self, kdims=None, vdims=None, groupby=None, **kwargs):
        return self(Chord, kdims, vdims, groupby, **kwargs)

    def hextiles(self, kdims=None, vdims=None, groupby=None, **kwargs):
        return self(HexTiles, kdims, vdims, groupby, **kwargs)

    def area(self, kdims=None, vdims=None, groupby=None, **kwargs):
        return self(Area, kdims, vdims, groupby, **kwargs)

    def table(self, kdims=None, vdims=None, groupby=None, **kwargs):
        return self(Table, kdims, vdims, groupby, **kwargs)


Dataset._conversion_interface = ElementConversion


__all__ = [
    "HSV",
    "RGB",
    "Annotation",
    "Area",
    "Arrow",
    "Bars",
    "Bivariate",
    "Bounds",
    "Box",
    "BoxWhisker",
    "Chord",
    "Contours",
    "Curve",
    "Dataset",
    "Dendrogram",
    "Distribution",
    "Div",
    "EdgePaths",
    "Element",
    "Ellipse",
    "ErrorBars",
    "Graph",
    "HLine",
    "HLines",
    "HSpan",
    "HSpans",
    "HeatMap",
    "HexTiles",
    "Histogram",
    "Image",
    "ImageStack",
    "ItemTable",
    "Labels",
    "Nodes",
    "Path",
    "Path3D",
    "Points",
    "Polygons",
    "QuadMesh",
    "Raster",
    "Rectangles",
    "Sankey",
    "Scatter",
    "Scatter3D",
    "Segments",
    "Slope",
    "Spikes",
    "Spline",
    "Spread",
    "Surface",
    "Table",
    "Text",
    "Tiles",
    "TriMesh",
    "TriSurface",
    "VLine",
    "VLines",
    "VSpan",
    "VSpans",
    "VectorField",
    "VectorizedAnnotation",
    "Violin",
]
