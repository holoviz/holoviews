import numpy as np
from bokeh.charts import Bar, BoxPlot as BokehBoxPlot
from bokeh.models import Circle, GlyphRenderer, ColumnDataSource, Range1d
import param

from ...element import Raster, Points, Polygons, Spikes
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

    scaling_method = param.ObjectSelector(default="area",
                                          objects=["width", "area"],
                                          doc="""
      Determines whether the `scaling_factor` should be applied to
      the width or area of each point (default: "area").""")

    scaling_factor = param.Number(default=1, bounds=(1, None), doc="""
      Scaling factor which is applied to either the width or area
      of each point, depending on the value of `scaling_method`.""")

    size_fn = param.Callable(default=np.abs, doc="""
      Function applied to size values before applying scaling,
      to remove values lower than zero.""")

    style_opts = (['cmap', 'palette', 'marker', 'size', 's', 'alpha', 'color',
                   'unselected_color'] +
                  line_properties + fill_properties)

    _plot_method = 'scatter'


    def get_data(self, element, ranges=None, empty=False):
        style = self.style[self.cyclic_index]
        dims = element.dimensions(label=True)

        mapping = dict(x=dims[0], y=dims[1])
        data = {}

        cmap = style.get('palette', style.get('cmap', None))
        if self.color_index < len(dims) and cmap:
            map_key = 'color_' + dims[self.color_index]
            mapping['color'] = map_key
            if empty:
                data[map_key] = []
            else:
                cmap = get_cmap(cmap)
                colors = element.dimension_values(self.color_index)
                crange = ranges.get(dims[self.color_index], None)
                data[map_key] = map_colors(colors, crange, cmap)
        if self.size_index < len(dims) and self.scaling_factor != 1:
            map_key = 'size_' + dims[self.size_index]
            mapping['size'] = map_key
            if empty:
                data[map_key] = []
            else:
                ms = style.get('size', 1)
                sizes = element.dimension_values(self.size_index)
                data[map_key] = compute_sizes(sizes, self.size_fn,
                                    self.scaling_factor, self.scaling_method, ms)

        data[dims[0]] = [] if empty else element.dimension_values(0)
        data[dims[1]] = [] if empty else element.dimension_values(1)
        if 'hover' in self.tools+self.default_tools:
            for d in dims:
                data[d] = [] if empty else element.dimension_values(d)
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
            renderer = plot.add_glyph(source, selected, selection_glyph=selected,
                                      nonselection_glyph=unselected)
        else:
            renderer = getattr(plot, self._plot_method)(**dict(properties, **mapping))
        return renderer, renderer.glyph


class CurvePlot(ElementPlot):

    style_opts = ['color'] + line_properties
    _plot_method = 'line'

    def get_data(self, element, ranges=None, empty=False):
        x = element.get_dimension(0).name
        y = element.get_dimension(1).name
        return ({x: [] if empty else element.dimension_values(0),
                 y: [] if empty else element.dimension_values(1)},
                dict(x=x, y=y))


class SpreadPlot(PolygonPlot):

    style_opts = ['color'] + line_properties + fill_properties

    def __init__(self, *args, **kwargs):
        super(SpreadPlot, self).__init__(*args, **kwargs)

    def get_data(self, element, ranges=None, empty=None):
        if empty:
            return dict(xs=[], ys=[]), self._mapping

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

    def get_data(self, element, ranges=None, empty=None):
        mapping = dict(top='top', bottom=0, left='left', right='right')
        if empty:
            data = dict(top=[], left=[], right=[])
        else:
            data = dict(top=element.values, left=element.edges[:-1],
                        right=element.edges[1:])

        if 'hover' in self.default_tools + self.tools:
            data.update({d: [] if empty else element.dimension_values(d)
                         for d in element.dimensions(label=True)})
        return (data, mapping)


class SideHistogramPlot(HistogramPlot):

    style_opts = HistogramPlot.style_opts + ['cmap']

    height = param.Integer(default=125, doc="The height of the plot")

    width = param.Integer(default=125, doc="The width of the plot")

    show_title = param.Boolean(default=False, doc="""
        Whether to display the plot title.""")

    def get_data(self, element, ranges=None, empty=None):
        if self.invert_axes:
            mapping = dict(top='left', bottom='right', left=0, right='top')
        else:
            mapping = dict(top='top', bottom=0, left='left', right='right')

        if empty:
            data = dict(top=[], left=[], right=[])
        else:
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
            data['color'] = [] if empty else map_colors(vals, main_range, cmap)
            mapping['fill_color'] = 'color'

        if 'hover' in self.default_tools + self.tools:
            data.update({d: [] if empty else element.dimension_values(d)
                         for d in element.dimensions(label=True)})
        return (data, mapping)



class ErrorPlot(PathPlot):

    horizontal = param.Boolean(default=False)

    style_opts = ['color'] + line_properties

    def get_data(self, element, ranges=None, empty=False):
        if empty:
            return dict(xs=[], ys=[]), self._mapping

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


    def get_data(self, element, ranges=None, empty=False):
        style = self.style[self.cyclic_index]
        dims = element.dimensions(label=True)

        pos = self.position
        if empty:
            xs, ys, keys = [], [], []
            mapping = dict(xs=dims[0], ys=dims[1] if len(dims) > 1 else 'heights')
        elif len(dims) > 1:
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

        if not empty and self.invert_axes: keys = keys[::-1]
        data = dict(zip(keys, (xs, ys)))

        cmap = style.get('palette', style.get('cmap', None))        
        if self.color_index < len(dims) and cmap:
            cdim = dims[self.color_index]
            map_key = 'color_' + cdim
            mapping['color'] = map_key
            if empty:
                colors = []
            else:
                cmap = get_cmap(cmap)
                cvals = element.dimension_values(cdim)
                crange = ranges.get(cdim, None)
                colors = map_colors(cvals, crange, cmap)
            data[map_key] = colors

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



class ChartPlot(ElementPlot):
    """
    ChartPlot creates and updates Bokeh high-level Chart instances.
    The current implementation requires creating a new Chart for each
    frame and updating the existing Chart. Once Bokeh supports updating
    Charts directly this workaround will no longer be required.
    """

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
        self._update_chart(key, element, ranges)

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

        self.style = self.lookup_options(element, 'style')
        self.set_param(**self.lookup_options(element, 'plot').options)
        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = match_spec(element, ranges)
        self.current_ranges = ranges

        self._update_chart(key, element, ranges)


    def _update_chart(self, key, element, ranges):
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

        properties = self._plot_properties(key, old_chart, element)
        old_chart.update(**properties)


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

    style_opts = ['color', 'whisker_color'] + line_properties

    def _init_chart(self, element, ranges):
        properties = self.style[self.cyclic_index]
        dframe = element.dframe()
        label = element.dimensions('key', True)
        if len(element.dimensions()) == 1:
            dframe[''] = ''
            label = ['']
        plot = BokehBoxPlot(dframe, label=label,
                            values=element.dimensions('value', True)[0],
                            **properties)

        # Disable outliers for now as they cannot be consistently updated.
        plot.renderers = [r for r in plot.renderers
                          if not (isinstance(r, GlyphRenderer) and
                                  isinstance(r.glyph, Circle))]
        return plot


class BarPlot(ChartPlot):
    """
    BarPlot allows generating single- or multi-category
    bar Charts, by selecting which key dimensions are
    mapped onto separate groups, categories and stacks.
    """

    group_index = param.Integer(default=0, doc="""
       Index of the dimension in the supplied Bars
       Element, which will be laid out into groups.""")

    category_index = param.Integer(default=1, doc="""
       Index of the dimension in the supplied Bars
       Element, which will be laid out into categories.""")

    stack_index = param.Integer(default=2, doc="""
       Index of the dimension in the supplied Bars
       Element, which will stacked.""")

    style_opts = ['bar_width', 'max_height', 'color', 'fill_alpha']

    def _init_chart(self, element, ranges):
        kdims = element.dimensions('key', True)
        vdim = element.dimensions('value', True)[0]

        kwargs = self.style[self.cyclic_index]
        if self.group_index < element.ndims:
            kwargs['label'] = kdims[self.group_index]
        if self.category_index < element.ndims:
            kwargs['group'] = kdims[self.category_index]
        if self.stack_index < element.ndims:
            kwargs['stack'] = kdims[self.stack_index]
        crange = Range1d(*ranges.get(vdim))
        plot = Bar(element.dframe(), values=vdim,
                   continuous_range=crange, **kwargs)
        return plot

