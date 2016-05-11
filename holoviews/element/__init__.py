from ..core import ViewableElement
from ..core.data import Dataset, DataConversion
from .annotation import * # noqa (API import)
from .chart import * # noqa (API import)
from .chart3d import * # noqa (API import)
from .path import * # noqa (API import)
from .raster import * # noqa (API import)
from .tabular import * # noqa (API import)


class ElementConversion(DataConversion):
    """
    ElementConversion is a subclass of DataConversion providing
    concrete methods to convert a Dataset to specific Element
    types.
    """

    def bars(self, kdims=None, vdims=None, mdims=None, **kwargs):
        return self(Bars, kdims, vdims, mdims, **kwargs)

    def box(self, kdims=None, vdims=None, mdims=None, **kwargs):
        return self(BoxWhisker, kdims, vdims, mdims, **kwargs)

    def bivariate(self, kdims=None, vdims=None, mdims=None, **kwargs):
        from ..interface.seaborn import Bivariate
        return self(Bivariate, kdims, vdims, mdims, **kwargs)

    def curve(self, kdims=None, vdims=None, mdims=None, **kwargs):
        return self(Curve, kdims, vdims, mdims, sort=True, **kwargs)

    def errorbars(self, kdims=None, vdims=None, mdims=None, **kwargs):
        return self(ErrorBars, kdims, vdims, mdims, sort=True, **kwargs)

    def distribution(self, dim=None, mdims=[], **kwargs):
        from ..interface.seaborn import Distribution
        if dim is None:
            if self._element.vdims:
                dim = self._element.vdims[0]
            else:
                raise Exception('Must supply an explicit value dimension '
                                'if no value dimensions are defined ')
        if mdims:
            reindexed = self._element.reindex(mdims, [dim])
            return reindexed.groupby(mdims, HoloMap, Distribution, **kwargs)
        else:
            element = self._element
            params = dict(vdims=[element.get_dimension(dim)],
                          label=element.label)
            if element.group != element.params()['group'].default:
                params['group'] = element.group
            return Distribution((element.dimension_values(dim),),
                                **dict(params, **kwargs))

    def heatmap(self, kdims=None, vdims=None, mdims=None, **kwargs):
        return self(HeatMap, kdims, vdims, mdims, **kwargs)

    def points(self, kdims=None, vdims=None, mdims=None, **kwargs):
        return self(Points, kdims, vdims, mdims, **kwargs)

    def raster(self, kdims=None, vdims=None, mdims=None, **kwargs):
        heatmap = self.heatmap(kdims, vdims, **kwargs)
        return Raster(heatmap.data, **dict(self._element.get_param_values(onlychanged=True)))

    def regression(self, kdims=None, vdims=None, mdims=None, **kwargs):
        from ..interface.seaborn import Regression
        return self(Regression, kdims, vdims, mdims, **kwargs)

    def scatter(self, kdims=None, vdims=None, mdims=None, **kwargs):
        return self(Scatter, kdims, vdims, mdims, **kwargs)

    def scatter3d(self, kdims=None, vdims=None, mdims=None, **kwargs):
        return self(Scatter3D, kdims, vdims, mdims, **kwargs)

    def spikes(self, kdims=None, vdims=None, mdims=None, **kwargs):
        return self(Spikes, kdims, vdims, mdims, **kwargs)

    def spread(self, kdims=None, vdims=None, mdims=None, **kwargs):
        return self(Spread, kdims, vdims, mdims, sort=True, **kwargs)

    def surface(self, kdims=None, vdims=None, mdims=None, **kwargs):
        heatmap = self.heatmap(kdims, vdims, **kwargs)
        return Surface(heatmap.data, **dict(self._table.get_param_values(onlychanged=True)))

    def trisurface(self, kdims=None, vdims=None, mdims=None, **kwargs):
        return self(Trisurface, kdims, vdims, mdims, **kwargs)

    def vectorfield(self, kdims=None, vdims=None, mdims=None, **kwargs):
        return self(VectorField, kdims, vdims, mdims, **kwargs)


Dataset._conversion_interface = ElementConversion


def public(obj):
    if not isinstance(obj, type): return False
    return issubclass(obj, ViewableElement)

__all__ = list(set([_k for _k, _v in locals().items() if public(_v)]))
