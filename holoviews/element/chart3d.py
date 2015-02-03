import param

from ..core import Dimension, Element3D
from .chart import Chart
from .raster import Raster


class Surface(Element3D, Raster):
    """
    Surface Element represents a 3D surface in a 3D space.
    """

    key_dimensions = param.Boolean(default=[Dimension('x'),
                                            Dimension('y'),
                                            Dimension('z')])

    value = param.String(default='Surface')

    def __init__(self, data, extents=(0, 0, 0, 1, 1, None), **params):
        super(Surface, self).__init__(data, extents=extents, **params)



class Scatter3D(Element3D, Chart):
    """
    Scatter3D object represents a number of coordinates in
    3D-space. Additionally a value dimension may be supplied.
    """

    key_dimensions = param.List(default=[Dimension('x'),
                                         Dimension('y'),
                                         Dimension('z')])

    value = param.String(default='Scatter3D')
