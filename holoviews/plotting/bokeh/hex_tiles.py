import param
import numpy as np

from ...core import Dimension, Operation
from ...core.options import Compositor
from ...core.util import basestring
from ...element import HexTiles
from .element import ColorbarPlot


def round_hex(q, r):
    """
    Rounds fractional hex coordinates
    """
    x = q
    z = r
    y = -x-z

    rx = np.round(x)
    ry = np.round(y)
    rz = np.round(z)

    dx = np.abs(rx - x)
    dy = np.abs(ry - y)
    dz = np.abs(rz - z)

    cond1 = (dx > dy) & (dx > dz)
    q = np.where(cond1              , -(ry + rz), rx)
    r = np.where(~cond1 & ~(dy > dz), -(rx + ry), rz)

    return q.astype(int), r.astype(int)

HEX_FLAT = [2.0/3.0, 0.0, -1.0/3.0, np.sqrt(3.0)/3.0]
HEX_POINTY = [np.sqrt(3.0)/3.0, -1.0/3.0, 0.0, 2.0/3.0]


def coords_to_hex(x, y, orientation, xsize, ysize):
    """
    Converts array x, y coordinates to hexagonal grid coordinates
    """
    orientation = HEX_FLAT if orientation == 'flat' else HEX_POINTY
    x =  x / xsize
    y = -y / ysize
    q = orientation[0] * x + orientation[1] * y
    r = orientation[2] * x + orientation[3] * y
    return round_hex(q, r)


class hex_binning(Operation):
    """
    Applies hex binning by computing aggregates on a hexagonal grid.

    Should not be user facing as the returned element is not directly
    useable.
    """

    aggregator = param.Callable(default=np.size)

    gridsize = param.ClassSelector(default=50, class_=(int, tuple))

    min_count = param.Number(default=None)

    orientation = param.ObjectSelector(default='pointy', objects=['flat', 'pointy'])

    def _process(self, element, key=None):

        gridsize, aggregator, orientation = self.p.gridsize, self.p.aggregator, self.p.orientation

        # Determine sampling
        (x0, x1), (y0, y1) = (element.range(i) for i in range(2))
        if isinstance(gridsize, tuple):
            sx, sy = gridsize
        else:
            sx, sy = gridsize, gridsize
        xsize = ((x1-x0)/sx)*(2.0/3.0)
        ysize = ((y1-y0)/sy)*(2.0/3.0)

        # Compute hexagonal coordinates
        x, y = (element.dimension_values(i) for i in range(2))
        q, r = coords_to_hex(x, y, orientation, xsize, ysize)
        coords = q, r

        # Get aggregation values
        if aggregator is np.size:
            aggregator = np.sum
            values = (np.full_like(q, 1),)
            vdims = ['Count']
        elif not element.vdims:
            raise ValueError('HexTiles aggregated by value must '
                             'define a value dimensions.')
        else:
            vdims = element.vdims
            values = tuple(element.dimension_values(vdim) for vdim in vdims)

        # Add empty bins
        if self.p.min_count == 0:
            rs = np.arange(int(x0/xsize), int(x1/xsize)+1)
            qs = np.arange(int(y0/ysize), int(y1/ysize)+1)
            qs, rs = np.meshgrid(qs, rs)
            coords = (np.concatenate([coords[0], qs.flat]),
                      np.concatenate([coords[1], rs.flat]))
            zeros = np.full_like(qs, 0).flat
            values = tuple(np.concatenate([vals, zeros]) for vals in values)

        # Construct aggregate
        data = coords + values
        xd, yd = element.kdims
        xd, yd = xd(range=(x0, x1)), yd(range=(y0, y1))
        agg = element.clone(data, kdims=[xd, yd], vdims=vdims).aggregate(function=aggregator)
        if self.p.min_count is not None and self.p.min_count > 1:
            agg = agg[:, :, self.p.min_count:]
        return agg


compositor = Compositor(
    "HexTiles", hex_binning, None, 'data', output_type=HexTiles,
    transfer_options=True, transfer_parameters=True, backends=['bokeh']
)
Compositor.register(compositor)


class HexTilesPlot(ColorbarPlot):

    aggregator = param.Callable(default=np.size)

    color_index = param.ClassSelector(default=2, class_=(basestring, int),
                                     allow_None=True, doc="""
      Index of the dimension from which the colors will the drawn.""")

    gridsize = param.ClassSelector(default=50, class_=(int, tuple))

    max_scale = param.Number(default=0.9, bounds=(0, 1))

    min_count = param.Number(default=None)

    orientation = param.ObjectSelector(default='pointy', objects=['flat', 'pointy'])

    size_index = param.ClassSelector(default=None, class_=(basestring, int),
                                     allow_None=True, doc="""
      Index of the dimension from which the sizes will the drawn.""")

    _plot_methods = dict(single='hex_tile')

    def _hover_opts(self, element):
        if self.aggregator is np.size:
            dims = [Dimension('Count')]
        else:
            dims = element.vdims
        return dims, {}

    def get_data(self, element, ranges, style):
        q, r = (element.dimension_values(i) for i in range(2))
        x, y = element.kdims

        (x0, x1), (y0, y1) = ranges[x.name], ranges[y.name]
        if isinstance(self.gridsize, tuple):
            sx, sy = self.gridsize
        else:
            sx, sy = self.gridsize, self.gridsize
        xsize = ((x1-x0)/sx)*(2.0/3.0)
        ysize = ((y1-y0)/sy)*(2.0/3.0)
        size = xsize if self.orientation == 'flat' else ysize
        scale = ysize/xsize

        mapping = {'q': 'q', 'r': 'r'}
        data = {'q': q, 'r': r}
        cdata, cmapping = self._get_color_data(element, ranges, style)
        data.update(cdata)
        mapping.update(cmapping)
        self._get_hover_data(data, element)
        style['orientation'] = self.orientation+'top'
        style['size'] = size
        style['aspect_scale'] = scale
        scale_dim = element.get_dimension(self.size_index)
        if scale_dim is not None:
            sizes = element.dimension_values(scale_dim)
            mapping['scale'] = 'scale'
            data['scale'] = ((sizes - sizes.min()) / sizes.ptp()) * self.max_scale

        return data, mapping, style
