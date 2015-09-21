import numpy as np

import param

from ...core import Dimension
from ...core.util import max_range
from ...element import Chart, Raster, Points, Polygons
from ..util import compute_sizes, get_sideplot_ranges
from .element import ElementPlot, line_properties, fill_properties
from .path import PathPlot, PolygonPlot
from .util import map_colors, get_cmap


class PointPlot(ElementPlot):

    color_index = param.Integer(default=3, doc="""
      Index of the dimension from which the color will the drawn""")

    size_index = param.Integer(default=2, doc="""
      Index of the dimension from which the sizes will the drawn.""")

    radius_index = param.Integer(default=None, doc="""
      Index of the dimension from which the sizes will the drawn.""")

    scaling_factor = param.Number(default=1, bounds=(1, None), doc="""
      If values are supplied the area of the points is computed relative
      to the marker size. It is then multiplied by scaling_factor to the power
      of the ratio between the smallest point and all other points.
      For values of 1 scaling by the values is disabled, a factor of 2
      allows for linear scaling of the area and a factor of 4 linear
      scaling of the point width.""")

    size_fn = param.Callable(default=np.abs, doc="""
      Function applied to size values before applying scaling,
      to remove values lower than zero.""")

    style_opts = (['cmap', 'palette', 'marker', 'size', 's', 'alpha', 'color'] +
                  line_properties + fill_properties)

    _plot_method = 'scatter'


    def get_data(self, element, ranges=None):
        style = self.style[self.cyclic_index]
        dims = element.dimensions(label=True)

        mapping = dict(x=dims[0], y=dims[1])
        data = {}

        cmap = style.get('palette', style.get('cmap', None))
        if self.color_index < len(dims) and cmap:
            mapping['color'] = 'color'
            cmap = get_cmap(cmap)
            colors = element.data[:, self.color_index]
            crange = ranges.get(dims[self.color_index], None)
            data['color'] = map_colors(colors, crange, cmap)
        if self.size_index < len(dims):
            mapping['size'] = 'size'
            ms = style.get('size', 1)
            sizes = element.data[:, self.size_index]
            data['size'] = compute_sizes(sizes, self.size_fn,
                                         self.scaling_factor, ms)
        data[dims[0]] = element.data[:, 0]
        data[dims[1]] = element.data[:, 1]
        if 'hover' in self.tools:
            for d in dims[2:]:
                data[d] = element.dimension_values(d)
        return data, mapping



class CurvePlot(ElementPlot):

    style_opts = ['color'] + line_properties
    _plot_method = 'line'

    def get_data(self, element, ranges=None):
        return (dict(x=element.data[:, 0], y=element.data[:, 1]),
                dict(x='x', y='y'))


class SpreadPlot(PolygonPlot):

    style_opts = ['color'] + line_properties + fill_properties

    def __init__(self, *args, **kwargs):
        super(SpreadPlot, self).__init__(*args, **kwargs)

    def get_data(self, element, ranges=None):
        lower = element.data[:, 1] - element.data[:, 2]
        upper = element.data[:, 1] + element.data[:, 3]
        band_x = np.append(element.data[:, 0], element.data[::-1, 0])
        band_y = np.append(lower, upper[::-1])
        return dict(xs=[band_x], ys=[band_y]), self._mapping


class HistogramPlot(ElementPlot):

    style_opts = ['color'] + line_properties + fill_properties
    _plot_method = 'quad'

    def get_data(self, element, ranges=None):
        mapping = dict(top='top', bottom=0, left='left', right='right')
        data = dict(top=element.values, left=element.edges[:-1],
                    right=element.edges[1:])

        if 'hover' in self.default_tools + self.tools:
            data.update({d: element.dimension_values(d)
                         for d in element.dimensions(label=True)})
        return (data, mapping)


class AdjointHistogramPlot(HistogramPlot):

    style_opts = HistogramPlot.style_opts + ['cmap']

    width = param.Integer(default=125)

    def get_data(self, element, ranges=None):
        if self.invert_axes:
            mapping = dict(top='left', bottom='right', left=0, right='top')
        else:
            mapping = dict(top='top', bottom=0, left='left', right='right')
        data = dict(top=element.values, left=element.edges[:-1],
                    right=element.edges[1:])

        dim = element.get_dimension(0).name
        main = self.adjoined.main
        range_item, main_range, dim = get_sideplot_ranges(self, element, main, ranges)
        vals = element.dimension_values(dim)
        if isinstance(range_item, (Raster, Points, Polygons)):
            style = self.lookup_options(range_item, 'style')[self.cyclic_index]
        else:
            style = {}

        if 'cmap' in style or 'palette' in style:
            cmap = get_cmap(style.get('cmap', style.get('palette', None)))
            colors = map_colors(vals, main_range, cmap)
            data['color'] = colors
            mapping['fill_color'] = 'color'

        if 'hover' in self.default_tools + self.tools:
            data.update({d: element.dimension_values(d)
                         for d in element.dimensions(label=True)})
        return (data, mapping)



class ErrorPlot(PathPlot):

    horizontal = param.Boolean(default=False)

    style_opts = ['color'] + line_properties

    def get_data(self, element, ranges=None):
        data = element.data
        err_xs = []
        err_ys = []
        for x, y, neg, pos in data:
            if self.horizontal:
                err_xs.append((x - neg, x + pos))
                err_ys.append((y, y))
            else:
                err_xs.append((x, x))
                err_ys.append((y - neg, y + pos))
        return (dict(xs=err_xs, ys=err_ys), self._mapping)
