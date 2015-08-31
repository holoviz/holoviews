import numpy as np

import param

from ...core import Dimension
from ...core.util import max_range
from ...element import Chart
from ..util import compute_sizes
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

        mapping = dict(x= dims[0], y=dims[1])
        data = {}

        cmap = style.get('palette', style.get('cmap', None))
        if self.color_index < len(dims) and cmap:
            mapping['fill_color'] = dims[self.color_index]
            cmap = get_cmap(cmap)
            colors = element.dimension_values(self.color_index)
            data[dims[self.color_index]] = map_colors(colors, ranges, cmap)
        if self.size_index < len(dims):
            mapping['size'] = dims[self.size_index]
            val_dim = dims[self.size_index]
            ms = style.get('size', 1)
            sizes = element.dimension_values(self.size_index)
            data[dims[self.size_index]] = compute_sizes(sizes, self.size_fn,
                                                        self.scaling_factor, ms)
        data[dims[0]] = element.dimension_values(0)
        data[dims[1]] = element.dimension_values(1)
        return data, mapping



class CurvePlot(ElementPlot):

    style_opts = ['color'] + line_properties
    _plot_method = 'line'

    def get_data(self, element, ranges=None):
        return (dict(x=element.dimension_values(0),
                     y=element.dimension_values(1)),
                dict(x='x', y='y'))


class SpreadPlot(PolygonPlot):

    style_opts = ['color'] + line_properties + fill_properties

    def __init__(self, *args, **kwargs):
        super(SpreadPlot, self).__init__(*args, **kwargs)

    def get_data(self, element, ranges=None):
        lower = element.dimension_values(1) - element.dimension_values(2)
        upper = element.dimension_values(1) + element.dimension_values(3)
        band_x = np.append(element.dimension_values(0), element.dimension_values(0)[::-1])
        band_y = np.append(lower, upper[::-1])
        return dict(xs=[band_x], ys=[band_y]), self._mapping


class HistogramPlot(ElementPlot):

    style_opts = ['color'] + line_properties + fill_properties
    _plot_method = 'quad'

    def get_data(self, element, ranges=None):
        mapping = dict(top='top', bottom=0, left='left', right='right')
        return (dict(top=element.values, left=element.edges[:-1],
                     right=element.edges[1:]), mapping)



class ErrorPlot(PathPlot):

    horizontal = param.Boolean(default=False)

    style_opts = ['color'] + line_properties

    def get_data(self, element, ranges=None):
        data = [element.dimension_values(i)
                for i in range(element.dimensions())]
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
