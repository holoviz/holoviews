import param

from ..core import Dimension, Element3D
from .chart import Chart, Scatter
from .raster import Image


class Surface(Image, Element3D):
    """
    Surface Element represents a 3D surface in space.
    The data should be supplied as a dense NxM matrix.
    """

    extents = param.Tuple(default=(None, None, None,
                                   None, None, None),
        doc="""Allows overriding the extents of the Element
               in 3D space defined as (xmin, ymin, zmin,
               xmax, ymax, zmax).""")

    kdims = param.List(default=[Dimension('x'), Dimension('y')],
                                bounds=(2,2), doc="""
        The Surface x and y dimensions of the space defined
        by the supplied extent.""")

    vdims = param.List(default=[Dimension('z')], bounds=(1,1), doc="""
        The Surface height dimension.""")

    group = param.String(default='Surface', constant=True)

    def __init__(self, data, kdims=None, vdims=None, extents=None, **params):
        extents = extents if extents else (None, None, None, None, None, None)
        Image.__init__(self, data, kdims=kdims, vdims=vdims, extents=extents, **params)



class Trisurface(Element3D, Scatter):
    """
    Trisurface object represents a number of coordinates in 3D-space,
    represented as a Surface of triangular polygons.
    """

    kdims = param.List(default=[Dimension('x'),
                                Dimension('y'),
                                Dimension('z')])

    vdims = param.List(default=[], doc="""
        Trisurface can have optional value dimensions,
        which may be mapped onto color and size.""")

    group = param.String(default='Trisurface', constant=True)

    def __getitem__(self, slc):
        return Chart.__getitem__(self, slc)



class Scatter3D(Element3D, Scatter):
    """
    Scatter3D object represents a number of coordinates in
    3D-space. Additionally Scatter3D points may have any number
    of value dimensions. The data may therefore be supplied
    as NxD matrix where N represents the number of samples,
    and D the number of key and value dimensions.
    """

    kdims = param.List(default=[Dimension('x'),
                                Dimension('y'),
                                Dimension('z')])

    vdims = param.List(default=[], doc="""
        Scatter3D can have optional value dimensions,
        which may be mapped onto color and size.""")

    group = param.String(default='Scatter3D', constant=True)

    def __getitem__(self, slc):
        return Chart.__getitem__(self, slc)
