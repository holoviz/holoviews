import param

from ..core import Dimension, Element3D
from .chart import Chart
from .raster import Raster


class Surface(Element3D, Raster):
    """
    Surface Element represents a 3D surface in space.
    The data should be supplied as a dense NxM matrix.
    """

    key_dimensions = param.List(default=[Dimension('x'),
                                         Dimension('y')],
                                bounds=(2,2), doc="""
        The Surface x and y dimensions of the space defined
        by the supplied extent.""")

    value_dimensions = param.List(default=[Dimension('z')],
                                  bounds=(1,1), doc="""
        The Surface height dimension.""")

    group = param.String(default='Surface')

    def __init__(self, data, extents=(0, 0, 0, 1, 1, None), **params):
        super(Surface, self).__init__(data, extents=extents, **params)



class Scatter3D(Element3D, Chart):
    """
    Scatter3D object represents a number of coordinates in
    3D-space. Additionally Scatter3D points may have any number
    of value dimensions. The data may therefore be supplied
    as NxD matrix where N represents the number of samples,
    and D the number of key and value dimensions.
    """

    key_dimensions = param.List(default=[Dimension('x'),
                                         Dimension('y'),
                                         Dimension('z')])

    value_dimensions = param.List(default=[], doc="""
        Scatter3D can have optional value dimensions,
        which may be mapped onto color and size.""")

    group = param.String(default='Scatter3D')

    def __getitem__(self, slc):
        return Chart.__getitem__(self, slc)
