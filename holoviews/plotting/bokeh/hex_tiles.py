from __future__ import absolute_import, division, unicode_literals
import types

import param
import numpy as np

try:
    from bokeh.util.hex import cartesian_to_axial
except:
    cartesian_to_axial = None

from ...core import Dimension, Operation
from ...core.options import Compositor
from ...core.util import isfinite, max_range
from ...element import HexTiles
from .element import ColorbarPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import base_properties, line_properties, fill_properties


class hex_binning(Operation):
    """
    Applies hex binning by computing aggregates on a hexagonal grid.

    Should not be user facing as the returned element is not directly
    useable.
    """

    aggregator = param.ClassSelector(
        default=np.size, class_=(types.FunctionType, tuple), doc="""
      Aggregation function or dimension transform used to compute bin
      values. Defaults to np.size to count the number of values
      in each bin.""")

    gridsize = param.ClassSelector(default=50, class_=(int, tuple))

    invert_axes = param.Boolean(default=False)

    min_count = param.Number(default=None)

    orientation = param.ObjectSelector(default='pointy', objects=['flat', 'pointy'])

    def _process(self, element, key=None):
        gridsize, aggregator, orientation = self.p.gridsize, self.p.aggregator, self.p.orientation

        # Determine sampling
        indexes = [1, 0] if self.p.invert_axes else [0, 1]
        (x0, x1), (y0, y1) = (element.range(i) for i in indexes)
        if isinstance(gridsize, tuple):
            sx, sy = gridsize
        else:
            sx, sy = gridsize, gridsize
        xsize = ((x1-x0)/sx)*(2.0/3.0)
        ysize = ((y1-y0)/sy)*(2.0/3.0)
        size = xsize if self.orientation == 'flat' else ysize
        if isfinite(ysize) and isfinite(xsize) and not xsize == 0:
            scale = ysize/xsize
        else:
            scale = 1

        # Compute hexagonal coordinates
        x, y = (element.dimension_values(i) for i in indexes)
        if not len(x):
            return element.clone([])
        finite = isfinite(x) & isfinite(y)
        x, y = x[finite], y[finite]
        q, r = cartesian_to_axial(x, y, size, orientation+'top', scale)
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

        # Construct aggregate
        data = coords + values
        xd, yd = (element.get_dimension(i) for i in indexes)
        xdn, ydn = xd.clone(range=(x0, x1)), yd.clone(range=(y0, y1))
        kdims = [ydn, xdn] if self.p.invert_axes else [xdn, ydn]
        agg = (
            element.clone(data, kdims=kdims, vdims=vdims)
            .aggregate(function=aggregator)
        )
        if self.p.min_count is not None and self.p.min_count > 1:
            agg = agg[:, :, self.p.min_count:]
        agg.cdims = {xd.name: xdn, yd.name: ydn}
        return agg


compositor = Compositor(
    "HexTiles", hex_binning, None, 'data', output_type=HexTiles,
    transfer_options=True, transfer_parameters=True, backends=['bokeh']
)
Compositor.register(compositor)


class HexTilesPlot(ColorbarPlot):

    aggregator = param.ClassSelector(
        default=np.size, class_=(types.FunctionType, tuple), doc="""
      Aggregation function or dimension transform used to compute
      bin values.  Defaults to np.size to count the number of values
      in each bin.""")

    gridsize = param.ClassSelector(default=50, class_=(int, tuple), doc="""
      Number of hexagonal bins along x- and y-axes. Defaults to uniform
      sampling along both axes when setting and integer but independent
      bin sampling can be specified a tuple of integers corresponding to
      the number of bins along each axis.""")

    min_count = param.Number(default=None, doc="""
      The display threshold before a bin is shown, by default bins with
      a count of less than 1 are hidden.""")

    orientation = param.ObjectSelector(default='pointy', objects=['flat', 'pointy'],
                                       doc="""
      The orientation of hexagon bins. By default the pointy side is on top.""")

    selection_display = BokehOverlaySelectionDisplay()

    style_opts = base_properties + line_properties + fill_properties + ['cmap', 'scale']

    _nonvectorized_styles = base_properties + ['cmap', 'line_dash']
    _plot_methods = dict(single='hex_tile')

    def get_extents(self, element, ranges, range_type='combined'):
        xdim, ydim = element.kdims[:2]
        ranges[xdim.name]['data'] = xdim.range
        ranges[ydim.name]['data'] = ydim.range
        xd = element.cdims.get(xdim.name)
        if xd and xdim.name in ranges:
            ranges[xdim.name]['hard'] = xd.range
            ranges[xdim.name]['soft'] = max_range([xd.soft_range, ranges[xdim.name]['soft']])
        yd = element.cdims.get(ydim.name)
        if yd and ydim.name in ranges:
            ranges[ydim.name]['hard'] = yd.range
            ranges[ydim.name]['hard'] = max_range([yd.soft_range, ranges[ydim.name]['soft']])
        return super(HexTilesPlot, self).get_extents(element, ranges, range_type)

    def _hover_opts(self, element):
        if self.aggregator is np.size:
            dims = [Dimension('Count')]
        else:
            dims = element.vdims
        return dims, {}

    def get_data(self, element, ranges, style):
        mapping = {'q': 'q', 'r': 'r'}
        if not len(element):
            data = {'q': [], 'r': []}
            return data, mapping, style

        q, r = (element.dimension_values(i) for i in range(2))
        x, y = element.kdims[::-1] if self.invert_axes else element.kdims
        (x0, x1), (y0, y1) = x.range, y.range
        if isinstance(self.gridsize, tuple):
            sx, sy = self.gridsize
        else:
            sx, sy = self.gridsize, self.gridsize
        xsize = ((x1-x0)/sx)*(2.0/3.0)
        ysize = ((y1-y0)/sy)*(2.0/3.0)
        size = xsize if self.orientation == 'flat' else ysize
        scale = ysize/xsize

        data = {'q': q, 'r': r}
        cdata, cmapping = self._get_color_data(element, ranges, style, element.vdims[0])
        data.update(cdata)
        mapping.update(cmapping)
        if self.min_count is not None and self.min_count <= 0:
            cmapper = cmapping['color']['transform']
            cmapper.low = self.min_count
            self.state.background_fill_color = cmapper.palette[0]

        self._get_hover_data(data, element, element.vdims)
        style['orientation'] = self.orientation+'top'
        style['size'] = size
        style['aspect_scale'] = scale

        return data, mapping, style
