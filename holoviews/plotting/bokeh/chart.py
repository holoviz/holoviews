import numpy as np
from bokeh.charts import Bar, BoxPlot as BokehBoxPlot
from bokeh.models import Circle, GlyphRenderer, ColumnDataSource, Range1d
import param

from ...core import Dimension
from ...core.util import max_range
from ...element import Chart, Raster, Points, Polygons
from ..util import compute_sizes, get_sideplot_ranges, match_spec
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



class ChartPlot(ElementPlot):


    def initialize_plot(self, ranges=None, plot=None, plots=None, source=None):
        """
        Initializes a new plot object with the last available frame.
        """
        # Get element key and ranges for frame
        element = self.hmap.last
        key = self.keys[-1]
        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = match_spec(element, ranges)
        self.current_ranges = ranges
        self.current_frame = element
        self.current_key = key

        # Initialize plot, source and glyph
        if plot is not None:
            raise Exception("Can't overlay Bokeh Charts based plot properties")

        init_element = element.clone(element.interface.concat(self.hmap.values()))
        plot = self._init_chart(init_element, ranges)
        self.handles['plot'] = plot
        self.handles['glyph_renderers'] = [r for r in plot.renderers
                                           if isinstance(r, GlyphRenderer)]
        self._update_chart(element, ranges)

        # Update plot, source and glyph
        self.drawn = True

        return plot

    def update_frame(self, key, ranges=None, plot=None, element=None):
        """
        Updates an existing plot with data corresponding
        to the key.
        """
        element = self._get_frame(key)
        if not element:
            if self.dynamic and self.overlaid:
                self.current_key = key
                element = self.current_frame
            else:
                element = self._get_frame(key)
        else:
            self.current_key = key
            self.current_frame = element

        self.set_param(**self.lookup_options(element, 'plot').options)
        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = match_spec(element, ranges)
        self.current_ranges = ranges

        self._update_chart(element, ranges)


    def _update_chart(self, element, ranges):
        new_chart = self._init_chart(element, ranges)
        old_chart = self.handles['plot']
        old_renderers = old_chart.select(type=GlyphRenderer)
        new_renderers = new_chart.select(type=GlyphRenderer)

        old_chart.y_range.update(**new_chart.y_range.properties_with_values())
        updated = []
        for new_r in new_renderers:
            for old_r in old_renderers:
                if type(old_r.glyph) == type(new_r.glyph):
                    old_renderers.pop(old_renderers.index(old_r))
                    new_props = new_r.properties_with_values()
                    source = new_props.pop('data_source')
                    old_r.glyph.update(**new_r.glyph.properties_with_values())
                    old_r.update(**new_props)
                    old_r.data_source.data.update(source.data)
                    updated.append(old_r)
                    break

        for old_r in old_renderers:
            if old_r not in updated:
                emptied = {k: [] for k in old_r.data_source.data}
                old_r.data_source.data.update(emptied)


    @property
    def current_handles(self):
        plot = self.handles['plot']
        sources = plot.select(type=ColumnDataSource)
        return sources


class BoxPlot(ChartPlot):
    """
    BoxPlot generates a box and whisker plot from a BoxWhisker
    Element. This allows plotting the median, mean and various
    percentiles. Displaying outliers is currently not supported
    as they cannot be consistently updated.
    """

    def _init_chart(self, element, ranges):
        plot = BokehBoxPlot(element.dframe(),
                            label=element.dimensions('key', True),
                            values=element.dimensions('value', True)[0])

        # Disable outliers for now as they cannot be consistently updated.
        plot.renderers = [r for r in plot.renderers
                          if not (isinstance(r, GlyphRenderer) and
                                  isinstance(r.glyph, Circle))]
        return plot


class BarPlot(ChartPlot):
    """
    BoxPlot generates a box and whisker plot from a BoxWhisker
    Element. This allows plotting the median, mean and various
    percentiles. Displaying outliers is currently not supported
    as they cannot be consistently updated.
    """

    def _init_chart(self, element):
        plot = Bar(element.dframe(),
                   label=element.dimensions('key', True),
                   values=element.dimensions('value', True)[0])
        return plot


