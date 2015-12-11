import numpy as np
from bokeh.models import Circle
import param

from ...core import Dimension
from ...core.util import max_range
from ...element import Chart, Raster, Points, Polygons, Spikes
from ..util import compute_sizes, get_sideplot_ranges
from .element import ElementPlot, line_properties, fill_properties
from .path import PathPlot, PolygonPlot
from .util import map_colors, get_cmap, mpl_to_bokeh


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

    style_opts = (['cmap', 'palette', 'marker', 'size', 's', 'alpha', 'color',
                   'unselected_color'] +
                  line_properties + fill_properties)

    _plot_method = 'scatter'


    def get_data(self, element, ranges=None):
        style = self.style[self.cyclic_index]
        dims = element.dimensions(label=True)

        mapping = dict(x=dims[0], y=dims[1])
        data = {}

        cmap = style.get('palette', style.get('cmap', None))
        if self.color_index < len(dims) and cmap:
            map_key = 'color_' + dims[self.color_index]
            mapping['color'] = map_key
            cmap = get_cmap(cmap)
            colors = element.dimension_values(self.color_index)
            crange = ranges.get(dims[self.color_index], None)
            data[map_key] = map_colors(colors, crange, cmap)
        if self.size_index < len(dims):
            map_key = 'size_' + dims[self.size_index]
            mapping['size'] = map_key
            ms = style.get('size', 1)
            sizes = element.dimension_values(self.size_index)
            data[map_key] = compute_sizes(sizes, self.size_fn,
                                          self.scaling_factor, ms)
        data[dims[0]] = element.dimension_values(0)
        data[dims[1]] = element.dimension_values(1)
        if 'hover' in self.tools:
            for d in dims[2:]:
                data[d] = element.dimension_values(d)
        return data, mapping


    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object.
        """
        properties = mpl_to_bokeh(properties)
        unselect_color = properties.pop('unselected_color', None)
        if (any(t in self.tools for t in ['box_select', 'lasso_select'])
            and unselect_color is not None):
            source = properties.pop('source')
            color = properties.pop('color', None)
            color = mapping.pop('color', color)
            properties.pop('legend', None)
            unselected = Circle(**dict(properties, fill_color=unselect_color, **mapping))
            selected = Circle(**dict(properties, fill_color=color, **mapping))
            plot.add_glyph(source, selected, selection_glyph=selected,
                           nonselection_glyph=unselected)
        else:
            getattr(plot, self._plot_method)(**dict(properties, **mapping))



class CurvePlot(ElementPlot):

    style_opts = ['color'] + line_properties
    _plot_method = 'line'

    def get_data(self, element, ranges=None):
        x = element.get_dimension(0).name
        y = element.get_dimension(1).name
        return ({x: element.dimension_values(0),
                 y: element.dimension_values(1)},
                dict(x=x, y=y))


class SpreadPlot(PolygonPlot):

    style_opts = ['color'] + line_properties + fill_properties

    def __init__(self, *args, **kwargs):
        super(SpreadPlot, self).__init__(*args, **kwargs)

    def get_data(self, element, ranges=None):

        xvals = element.dimension_values(0)
        mean = element.dimension_values(1)
        neg_error = element.dimension_values(2)
        pos_idx = 3 if len(element.dimensions()) > 3 else 2
        pos_error = element.dimension_values(pos_idx)

        lower = mean - neg_error
        upper = mean + pos_error
        band_x = np.append(xvals, xvals[::-1])
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


class SideHistogramPlot(HistogramPlot):

    style_opts = HistogramPlot.style_opts + ['cmap']

    height = param.Integer(default=125, doc="The height of the plot")

    width = param.Integer(default=125, doc="The width of the plot")

    show_title = param.Boolean(default=False, doc="""
        Whether to display the plot title.""")

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
        if isinstance(range_item, (Raster, Points, Polygons, Spikes)):
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
        data = element.array(dimensions=element.dimensions()[0:4])
        err_xs = []
        err_ys = []
        for row in data:
            x, y = row[0:2]
            if len(row) > 3:
                neg, pos = row[2:]
            else:
                neg, pos = row[2], row[2]

            if self.horizontal:
                err_xs.append((x - neg, x + pos))
                err_ys.append((y, y))
            else:
                err_xs.append((x, x))
                err_ys.append((y - neg, y + pos))
        return (dict(xs=err_xs, ys=err_ys), self._mapping)


class SpikesPlot(PathPlot):

    color_index = param.Integer(default=1, doc="""
      Index of the dimension from which the color will the drawn""")

    spike_length = param.Number(default=0.5, doc="""
      The length of each spike if Spikes object is one dimensional.""")

    position = param.Number(default=0., doc="""
      The position of the lower end of each spike.""")

    style_opts = (['color', 'cmap', 'palette'] + line_properties)

    def get_extents(self, element, ranges):
        l, b, r, t = super(SpikesPlot, self).get_extents(element, ranges)
        if len(element.dimensions()) == 1:
            b, t = self.position, self.position+self.spike_length
        return l, b, r, t


    def get_data(self, element, ranges=None):
        style = self.style[self.cyclic_index]
        dims = element.dimensions(label=True)

        pos = self.position
        if len(dims) > 1:
            xs, ys = zip(*(((x, x), (pos, pos+y))
                           for x, y in element.array()))
            mapping = dict(xs=dims[0], ys=dims[1])
            keys = (dims[0], dims[1])
        else:
            height = self.spike_length
            xs, ys = zip(*(((x[0], x[0]), (pos, pos+height))
                           for x in element.array()))
            mapping = dict(xs=dims[0], ys='heights')
            keys = (dims[0], 'heights')

        if self.invert_axes: keys = keys[::-1]
        data = dict(zip(keys, (xs, ys)))

        cmap = style.get('palette', style.get('cmap', None))        
        if self.color_index < len(dims) and cmap:
            cdim = dims[self.color_index]
            map_key = 'color_' + cdim
            mapping['color'] = map_key
            cmap = get_cmap(cmap)
            colors = element.dimension_values(cdim)
            crange = ranges.get(cdim, None)
            data[map_key] = map_colors(colors, crange, cmap)

        return data, mapping



class SideSpikesPlot(SpikesPlot):
    """
    SpikesPlot with useful defaults for plotting adjoined rug plot.
    """

    xaxis = param.ObjectSelector(default='top-bare',
                                 objects=['top', 'bottom', 'bare', 'top-bare',
                                          'bottom-bare', None], doc="""
        Whether and where to display the xaxis, bare options allow suppressing
        all axis labels including ticks and xlabel. Valid options are 'top',
        'bottom', 'bare', 'top-bare' and 'bottom-bare'.""")

    yaxis = param.ObjectSelector(default='right-bare',
                                      objects=['left', 'right', 'bare', 'left-bare',
                                               'right-bare', None], doc="""
        Whether and where to display the yaxis, bare options allow suppressing
        all axis labels including ticks and ylabel. Valid options are 'left',
        'right', 'bare' 'left-bare' and 'right-bare'.""")

    height = param.Integer(default=80, doc="Height of plot")

    width = param.Integer(default=80, doc="Width of plot")
