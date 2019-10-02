from ..core import HoloMap
from ..core.data import Dataset, DataConversion
from .annotation import * # noqa (API import)
from .chart import * # noqa (API import)
from .geom import * # noqa (API import)
from .chart3d import * # noqa (API import)
from .graphs import * # noqa (API import)
from .path import * # noqa (API import)
from .raster import * # noqa (API import)
from .sankey import * # noqa (API import)
from .stats import * # noqa (API import)
from .tabular import * # noqa (API import)
from .tiles import * # noqa (API import)


class ElementConversion(DataConversion):
    """
    ElementConversion is a subclass of DataConversion providing
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

    def distribution(self, dim=None, groupby=[], **kwargs):
        if dim is None:
            if self._element.vdims:
                dim = self._element.vdims[0]
            else:
                raise Exception('Must supply an explicit value dimension '
                                'if no value dimensions are defined ')
        if groupby:
            reindexed = self._element.reindex(groupby, [dim])
            return reindexed.groupby(groupby, HoloMap, Distribution, **kwargs)
        else:
            element = self._element
            params = dict(kdims=[element.get_dimension(dim)],
                          label=element.label)
            if element.group != element.params()['group'].default:
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
        return Raster(heatmap.data, **dict(self._element.get_param_values(onlychanged=True)))

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
        return Surface(heatmap.data, **dict(self._table.get_param_values(onlychanged=True)))

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


def public(obj):
    if not isinstance(obj, type) or getattr(obj, 'abstract', False) and not obj is Element:
        return False
    return issubclass(obj, Element)

__all__ = list(set([_k for _k, _v in locals().items() if public(_v)]))
