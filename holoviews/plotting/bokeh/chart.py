import numpy as np

import param

from ...core import Dimension
from ...core.util import max_range
from ...element import Chart
from ..util import compute_sizes
from .element import ElementPlot, line_properties, fill_properties
from .path import PathPlot
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

    style_opts = (['cmap', 'palette', 'marker', 'size', 'alpha', 'color'] +
                  line_properties + fill_properties)

    def get_data(self, element, ranges=None):
        dims = element.dimensions(label=True)
        data = {}
        cmap = self.style.get('palette', self.style.get('cmap', None))
        if self.color_index < len(dims) and cmap:
            cmap = get_cmap(cmap)
            colors = element.data[:, self.color_index]
            data[dims[self.color_index]] = map_colors(colors, ranges, cmap)
        if self.size_index < len(dims):
            val_dim = dims[self.size_index]
            ms = self.style.get('size', 1)
            sizes = element.data[:, self.size_index]
            data[dims[self.size_index]] = compute_sizes(sizes, self.size_fn,
                                                        self.scaling_factor, ms)
        data[dims[0]] = element.data[:, 0]
        data[dims[1]] = element.data[:, 1]
        return data

    def _glyph_kwargs(self, element):
        dims = element.dimensions(label=True)
        kwargs = dict(self.style, x=dims[0], y=dims[1])
        if self.color_index < len(dims):
            kwargs['fill_color'] = dims[self.color_index]
            kwargs.pop('cmap', None)
        if self.size_index < len(dims):
            kwargs.pop('size', None)
            kwargs['size'] = dims[self.size_index]
        return kwargs


    def init_glyph(self, element, plot, source, ranges):
        plot.scatter(source=source, legend=element.label,
                     **self._glyph_kwargs(element))



class CurvePlot(ElementPlot):

    style_opts = ['color'] + line_properties
    
    def get_data(self, element, ranges=None):
        return dict(x=element.data[:, 0], y=element.data[:, 1])
    
    def init_glyph(self, element, plot, source, ranges):
        plot.line(x='x', y='y', source=source, legend=element.label, **self.style)


class SpreadPlot(ElementPlot):

    style_opts = ['color'] + line_properties + fill_properties

    def __init__(self, *args, **kwargs):
        super(SpreadPlot, self).__init__(*args, **kwargs)
        self._extent = None

    def get_data(self, element, ranges=None):
        lower = element.data[:, 1] - element.data[:, 2]
        upper = element.data[:, 1] + element.data[:, 3]
        band_x = np.append(element.data[:, 0], element.data[::-1, 0])
        band_y = np.append(lower, upper[::-1])
        return dict(xs=[band_x], ys=[band_y])
        
    def init_glyph(self, element, plot, source, ranges):
        self.handles['patches'] = plot.patches(xs='xs', ys='ys', source=source, 
                                               legend=element.label, **self.style)

    def get_extents(self, view, ranges):
        x0, y0, x1, y1 = super(SpreadPlot, self).get_extents(view, ranges)
        normopts = self.lookup_options(view, 'norm')
        if normopts.options.get('framewise', False):
            y0 = view.data[:, 1] - view.data[:, 2]
            y1 = view.data[:, 1] + view.data[:, 3]
        else:
            if not self._extent:
                max_spread = lambda x: (np.min(x.data[:, 1] - x.data[:, 2]),
                                        np.max(x.data[:,1] + x.data[:, 3]))
                y0, y1 = max_range(self.map.traverse(max_spread, (type(view),)))
                self._extent = (y0, y1)
            else:
                y0, y1 = self._extent
        return x0, y0, x1, y1


class HistogramPlot(ElementPlot):

    style_opts = ['color'] + line_properties + fill_properties

    def get_data(self, element, ranges=None):
        return dict(top=element.values, left=element.edges[:-1],
                    right=element.edges[1:])

    def init_glyph(self, element, plot, source, ranges):
        self.handles['lines'] = plot.quad(top='top', bottom=0, left='left',
                                          right='right', source=source,
                                          legend=element.label, **self.style)


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
        return dict(xs=err_xs, ys=err_ys)



