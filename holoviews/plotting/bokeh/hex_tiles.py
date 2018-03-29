import param
import numpy as np

try:
    from bokeh.util.hex import cartesian_to_axial
except:
    cartesian_to_axial = None

from ...core import Dimension, Operation
from ...core.options import Compositor, SkipRendering
from ...core.util import basestring
from ...element import HexTiles
from .element import ColorbarPlot, line_properties, fill_properties
from .util import bokeh_version


class hex_binning(Operation):
    """
    Applies hex binning by computing aggregates on a hexagonal grid.

    Should not be user facing as the returned element is not directly
    useable.
    """

    aggregator = param.Callable(default=np.size)

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
        scale = ysize/xsize

        # Compute hexagonal coordinates
        x, y = (element.dimension_values(i) for i in indexes)
        if not len(x):
            return element.clone([])
        finite = np.isfinite(x) & np.isfinite(y)
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
        xd, yd = xd(range=(x0, x1)), yd(range=(y0, y1))
        kdims = [yd, xd] if self.p.invert_axes else [xd, yd]
        agg = element.clone(data, kdims=kdims, vdims=vdims).aggregate(function=aggregator)
        if self.p.min_count is not None and self.p.min_count > 1:
            agg = agg[:, :, self.p.min_count:]
        return agg


compositor = Compositor(
    "HexTiles", hex_binning, None, 'data', output_type=HexTiles,
    transfer_options=True, transfer_parameters=True, backends=['bokeh']
)
Compositor.register(compositor)


class HexTilesPlot(ColorbarPlot):

    aggregator = param.Callable(default=np.size, doc="""
      Aggregation function used to compute bin values. Any NumPy
      reduction is allowed, defaulting to np.size to count the number
      of values in each bin.""")

    color_index = param.ClassSelector(default=2, class_=(basestring, int),
                                     allow_None=True, doc="""
      Index of the dimension from which the colors will the drawn.""")

    gridsize = param.ClassSelector(default=50, class_=(int, tuple), doc="""
      Number of hexagonal bins along x- and y-axes. Defaults to uniform
      sampling along both axes when setting and integer but independent
      bin sampling can be specified a tuple of integers corresponding to
      the number of bins along each axis.""")

    max_scale = param.Number(default=0.9, bounds=(0, None), doc="""
      When size_index is enabled this defines the maximum size of each
      bin relative to uniform tile size, i.e. for a value of 1, the
      largest bin will match the size of bins when scaling is disabled.
      Setting value larger than 1 will result in overlapping bins.""")

    min_scale = param.Number(default=0, bounds=(0, None), doc="""
      When size_index is enabled this defines the minimum size of each
      bin relative to uniform tile size, i.e. for a value of 1, the
      smallest bin will match the size of bins when scaling is disabled.
      Setting value larger than 1 will result in overlapping bins.""")

    min_count = param.Number(default=None, doc="""
      The display threshold before a bin is shown, by default bins with
      a count of less than 1 are hidden.""")

    orientation = param.ObjectSelector(default='pointy', objects=['flat', 'pointy'],
                                       doc="""
      The orientation of hexagon bins. By default the pointy side is on top.""")

    size_index = param.ClassSelector(default=None, class_=(basestring, int),
                                     allow_None=True, doc="""
      Index of the dimension from which the sizes will the drawn.""")

    _plot_methods = dict(single='hex_tile')

    style_opts = ['cmap', 'color'] + line_properties + fill_properties

    def _hover_opts(self, element):
        if self.aggregator is np.size:
            dims = [Dimension('Count')]
        else:
            dims = element.vdims
        return dims, {}

    def get_data(self, element, ranges, style):
        if bokeh_version < '0.12.15':
            raise SkipRendering('Plotting HexTiles with bokeh requires bokeh '
                                'version >=0.12.15, skipping plot.')

        mapping = {'q': 'q', 'r': 'r'}
        if not len(element):
            data = {'q': [], 'r': []}
            return data, mapping, style
        q, r = (element.dimension_values(i) for i in range(2))
        x, y = element.kdims[::-1] if self.invert_axes else element.kdims
        (x0, x1), (y0, y1) = ranges[x.name], ranges[y.name]
        if isinstance(self.gridsize, tuple):
            sx, sy = self.gridsize
        else:
            sx, sy = self.gridsize, self.gridsize
        xsize = ((x1-x0)/sx)*(2.0/3.0)
        ysize = ((y1-y0)/sy)*(2.0/3.0)
        size = xsize if self.orientation == 'flat' else ysize
        scale = ysize/xsize

        data = {'q': q, 'r': r}
        cdata, cmapping = self._get_color_data(element, ranges, style)
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
        scale_dim = element.get_dimension(self.size_index)
        if scale_dim is not None:
            sizes = element.dimension_values(scale_dim)
            if self.aggregator is np.size:
                ptp = sizes.max()
                baseline = 0
            else:
                ptp = sizes.ptp()
                baseline = sizes.min()
            if self.min_scale > self.max_scale:
                raise ValueError('min_scale parameter must be smaller '
                                 'than max_scale parameter.')
            scale = self.max_scale - self.min_scale
            mapping['scale'] = 'scale'
            data['scale'] = (((sizes - baseline) / ptp) * scale) + self.min_scale

        return data, mapping, style
