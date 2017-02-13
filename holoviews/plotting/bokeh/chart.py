from collections import defaultdict

import numpy as np
import param
try:
    from bokeh.charts import Bar, BoxPlot as BokehBoxPlot
except:
    Bar, BokehBoxPlot = None, None
from bokeh.models import (Circle, GlyphRenderer, ColumnDataSource,
                          Range1d, CustomJS, FactorRange, HoverTool)
from bokeh.models.tools import BoxSelectTool

from ...element import Raster, Points, Polygons, Spikes
from ...core.util import max_range, basestring, dimension_sanitizer
from ...core.options import abbreviated_exception
from ...operation import interpolate_curve
from ..util import compute_sizes, get_sideplot_ranges, match_spec, map_colors
from .element import ElementPlot, ColorbarPlot, LegendPlot, line_properties, fill_properties
from .path import PathPlot, PolygonPlot
from .util import get_cmap, mpl_to_bokeh, update_plot, rgb2hex, bokeh_version


class PointPlot(ColorbarPlot):

    color_index = param.ClassSelector(default=3, class_=(basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    size_index = param.ClassSelector(default=2, class_=(basestring, int),
                                     allow_None=True, doc="""
      Index of the dimension from which the sizes will the drawn.""")

    scaling_method = param.ObjectSelector(default="area",
                                          objects=["width", "area"],
                                          doc="""
      Determines whether the `scaling_factor` should be applied to
      the width or area of each point (default: "area").""")

    scaling_factor = param.Number(default=1, bounds=(0, None), doc="""
      Scaling factor which is applied to either the width or area
      of each point, depending on the value of `scaling_method`.""")

    size_fn = param.Callable(default=np.abs, doc="""
      Function applied to size values before applying scaling,
      to remove values lower than zero.""")

    style_opts = (['cmap', 'palette', 'marker', 'size', 's', 'alpha', 'color',
                   'unselected_color'] +
                  line_properties + fill_properties)

    _plot_methods = dict(single='scatter', batched='scatter')

    def get_data(self, element, ranges=None, empty=False):
        style = self.style[self.cyclic_index]
        dims = element.dimensions(label=True)

        xidx, yidx = (1, 0) if self.invert_axes else (0, 1)
        mapping = dict(x=dims[xidx], y=dims[yidx])
        data = {}

        cdim = element.get_dimension(self.color_index)
        if cdim:
            mapper = self._get_colormapper(cdim, element, ranges, style)
            data[cdim.name] = [] if empty else element.dimension_values(cdim)
            mapping['color'] = {'field': cdim.name,
                                'transform': mapper}

        sdim = element.get_dimension(self.size_index)
        if sdim:
            map_key = 'size_' + sdim.name
            if empty:
                data[map_key] = []
                mapping['size'] = map_key
            else:
                ms = style.get('size', np.sqrt(6))**2
                sizes = element.dimension_values(self.size_index)
                sizes = compute_sizes(sizes, self.size_fn,
                                      self.scaling_factor,
                                      self.scaling_method, ms)
                if sizes is None:
                    eltype = type(element).__name__
                    self.warning('%s dimension is not numeric, cannot '
                                 'use to scale %s size.' % (sdim, eltype))
                else:
                    data[map_key] = np.sqrt(sizes)
                    mapping['size'] = map_key

        xdim, ydim = dims[xidx], dims[yidx]
        data[xdim] = [] if empty else element.dimension_values(xidx)
        data[ydim] = [] if empty else element.dimension_values(yidx)
        self._categorize_data(data, (xdim, ydim), element.dimensions())
        self._get_hover_data(data, element, empty)
        return data, mapping


    def get_batched_data(self, element, ranges=None, empty=False):
        data = defaultdict(list)
        for key, el in element.items():
            style = self.lookup_options(el, 'style')
            style = style.max_cycles(len(self.ordering))
            self.set_param(**self.lookup_options(el, 'plot').options)
            eldata, elmapping = self.get_data(el, ranges, empty)
            for k, eld in eldata.items():
                data[k].append(eld)
            if 'color' not in elmapping:
                zorder = self.get_zorder(element, key, el)
                val = style[zorder].get('color')
                elmapping['color'] = 'color'
                if isinstance(val, tuple):
                    val = rgb2hex(val)
                data['color'].append([val]*len(data[k][-1]))
        data = {k: np.concatenate(v) for k, v in data.items()}
        return data, elmapping


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
            plot_method = self._plot_methods.get('batched' if self.batched else 'single')
            renderer = getattr(plot, plot_method)(**dict(properties, **mapping))
        if self.colorbar and 'color_mapper' in self.handles:
            self._draw_colorbar(plot, self.handles['color_mapper'])
        return renderer, renderer.glyph


class CurvePlot(ElementPlot):

    interpolation = param.ObjectSelector(objects=['linear', 'steps-mid',
                                                  'steps-pre', 'steps-post'],
                                         default='linear', doc="""
        Defines how the samples of the Curve are interpolated,
        default is 'linear', other options include 'steps-mid',
        'steps-pre' and 'steps-post'.""")

    style_opts = ['color'] + line_properties
    _plot_methods = dict(single='line', batched='multi_line')
    _mapping = {p: p for p in ['xs', 'ys', 'color', 'line_alpha']}

    def get_data(self, element, ranges=None, empty=False):
        if 'steps' in self.interpolation:
            element = interpolate_curve(element, interpolation=self.interpolation)
        xidx, yidx = (1, 0) if self.invert_axes else (0, 1)
        x = element.get_dimension(xidx).name
        y = element.get_dimension(yidx).name
        data = {x: [] if empty else element.dimension_values(xidx),
                y: [] if empty else element.dimension_values(yidx)}
        self._get_hover_data(data, element, empty)
        self._categorize_data(data, (x, y), element.dimensions())
        return (data, dict(x=x, y=y))

    def _hover_opts(self, element):
        if self.batched:
            dims = list(self.hmap.last.kdims)
            line_policy = 'prev'
        else:
            dims = list(self.overlay_dims.keys())+element.dimensions()
            line_policy = 'nearest'
        return dims, dict(line_policy=line_policy)

    def get_batched_data(self, overlay, ranges=None, empty=False):
        data = defaultdict(list)
        opts = ['color', 'line_alpha', 'line_color']
        for key, el in overlay.data.items():
            eldata, elmapping = self.get_data(el, ranges, empty)
            for k, eld in eldata.items():
                data[k].append(eld)

            # Add options
            style = self.lookup_options(el, 'style')
            style = style.max_cycles(len(self.ordering))
            zorder = self.get_zorder(overlay, key, el)
            style = style[zorder]
            for opt in opts:
                if opt not in style:
                    continue
                val = style[opt]
                if opt == 'color' and isinstance(val, tuple):
                    val = rgb2hex(val)
                data[opt].append([val])

            for d, k in zip(overlay.kdims, key):
                sanitized = dimension_sanitizer(d.name)
                data[sanitized].append([k])
        data = {opt: vals for opt, vals in data.items()
                if not any(v is None for v in vals)}
        return data, dict(xs=elmapping['x'], ys=elmapping['y'],
                          **{o: o for o in opts if o in data})


class AreaPlot(PolygonPlot):

    def get_extents(self, element, ranges):
        vdims = element.vdims
        vdim = vdims[0].name
        if len(vdims) > 1:
            ranges[vdim] = max_range([ranges[vd.name] for vd in vdims])
        else:
            vdim = vdims[0].name
            ranges[vdim] = (np.nanmin([0, ranges[vdim][0]]), ranges[vdim][1])
        return super(AreaPlot, self).get_extents(element, ranges)

    def get_data(self, element, ranges=None, empty=False):
        mapping = dict(self._mapping)
        if empty: return {'xs': [], 'ys': []}
        xs = element.dimension_values(0)
        x2 = np.hstack((xs[::-1], xs))

        if len(element.vdims) > 1:
            bottom = element.dimension_values(2)
        else:
            bottom = np.zeros(len(element))
        ys = np.hstack((bottom[::-1], element.dimension_values(1)))

        if self.invert_axes:
            data = dict(xs=[ys], ys=[x2])
        else:
            data = dict(xs=[x2], ys=[ys])
        return data, mapping


class SpreadPlot(PolygonPlot):

    style_opts = ['color'] + line_properties + fill_properties

    def get_data(self, element, ranges=None, empty=None):
        if empty:
            return dict(xs=[], ys=[]), dict(self._mapping)

        xvals = element.dimension_values(0)
        mean = element.dimension_values(1)
        neg_error = element.dimension_values(2)
        pos_idx = 3 if len(element.dimensions()) > 3 else 2
        pos_error = element.dimension_values(pos_idx)

        lower = mean - neg_error
        upper = mean + pos_error
        band_x = np.append(xvals, xvals[::-1])
        band_y = np.append(lower, upper[::-1])
        if self.invert_axes:
            data = dict(xs=[band_y], ys=[band_x])
        else:
            data = dict(xs=[band_x], ys=[band_y])
        return data, dict(self._mapping)


class HistogramPlot(ElementPlot):

    style_opts = ['color'] + line_properties + fill_properties
    _plot_methods = dict(single='quad')

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
        self._get_hover_data(data, element, empty)
        return (data, mapping)

    def get_extents(self, element, ranges):
        x0, y0, x1, y1 = super(HistogramPlot, self).get_extents(element, ranges)
        y0 = np.nanmin([0, y0])
        y1 = np.nanmax([0, y1])
        return (x0, y0, x1, y1)



class SideHistogramPlot(ColorbarPlot, HistogramPlot):

    style_opts = HistogramPlot.style_opts + ['cmap']

    height = param.Integer(default=125, doc="The height of the plot")

    width = param.Integer(default=125, doc="The width of the plot")

    show_title = param.Boolean(default=False, doc="""
        Whether to display the plot title.""")

    default_tools = param.List(default=['save', 'pan', 'wheel_zoom',
                                        'box_zoom', 'reset', 'box_select'],
        doc="A list of plugin tools to use on the plot.")

    _callback = """
    color_mapper.low = cb_data['geometry']['y0'];
    color_mapper.high = cb_data['geometry']['y1'];
    source.trigger('change')
    main_source.trigger('change')
    """

    def get_data(self, element, ranges=None, empty=None):
        if self.invert_axes:
            mapping = dict(top='right', bottom='left', left=0, right='top')
        else:
            mapping = dict(top='top', bottom=0, left='left', right='right')

        if empty:
            data = dict(top=[], left=[], right=[])
        else:
            data = dict(top=element.values, left=element.edges[:-1],
                        right=element.edges[1:])

        color_dims = self.adjoined.traverse(lambda x: x.handles.get('color_dim'))
        dim = color_dims[0] if color_dims else None
        cmapper = self._get_colormapper(dim, element, {}, {})
        if cmapper:
            data[dim.name] = [] if empty else element.dimension_values(dim)
            mapping['fill_color'] = {'field': dim.name,
                                     'transform': cmapper}
        self._get_hover_data(data, element, empty)
        return (data, mapping)


    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object.
        """
        ret = super(SideHistogramPlot, self)._init_glyph(plot, mapping, properties)
        if not 'field' in mapping.get('fill_color', {}):
            return ret
        dim = mapping['fill_color']['field']
        sources = self.adjoined.traverse(lambda x: (x.handles.get('color_dim'),
                                                     x.handles.get('source')))
        sources = [src for cdim, src in sources if cdim == dim]
        tools = [t for t in self.handles['plot'].tools
                 if isinstance(t, BoxSelectTool)]
        if not tools or not sources:
            return
        box_select, main_source = tools[0], sources[0]
        handles = {'color_mapper': self.handles['color_mapper'],
                   'source': self.handles['source'],
                   'main_source': main_source}
        if box_select.callback:
            box_select.callback.code += self._callback
            box_select.callback.args.update(handles)
        else:
            box_select.callback = CustomJS(args=handles, code=self._callback)
        return ret


class ErrorPlot(PathPlot):

    horizontal = param.Boolean(default=False)

    style_opts = ['color'] + line_properties

    def get_data(self, element, ranges=None, empty=False):
        if empty:
            return dict(xs=[], ys=[]), dict(self._mapping)

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

        if self.invert_axes:
            data = dict(xs=err_ys, ys=err_xs)
        else:
            data = dict(xs=err_xs, ys=err_ys)
        self._categorize_data(data, ('xs', 'ys'), element.dimensions())
        return (data, dict(self._mapping))


class SpikesPlot(PathPlot, ColorbarPlot):

    color_index = param.ClassSelector(default=1, allow_None=True,
                                      class_=(basestring, int), doc="""
      Index of the dimension from which the color will the drawn""")

    spike_length = param.Number(default=0.5, doc="""
      The length of each spike if Spikes object is one dimensional.""")

    position = param.Number(default=0., doc="""
      The position of the lower end of each spike.""")

    show_legend = param.Boolean(default=True, doc="""
        Whether to show legend for the plot.""")

    style_opts = (['color', 'cmap', 'palette'] + line_properties)

    def get_extents(self, element, ranges):
        l, b, r, t = super(SpikesPlot, self).get_extents(element, ranges)
        if len(element.dimensions()) == 1:
            b, t = self.position, self.position+self.spike_length
        else:
            b = np.nanmin([0, b])
            t = np.nanmax([0, t])
        return l, b, r, t

    def get_data(self, element, ranges=None, empty=False):
        style = self.style[self.cyclic_index]
        dims = element.dimensions(label=True)

        pos = self.position
        mapping = dict(xs='xs', ys='ys')
        if empty:
            xs, ys = [], []
        elif len(dims) > 1:
            xs, ys = zip(*(((x, x), (pos+y, pos))
                           for x, y in element.array(dims[:2])))
        else:
            height = self.spike_length
            xs, ys = zip(*(((x[0], x[0]), (pos+height, pos))
                           for x in element.array(dims[:1])))

        if not empty and self.invert_axes: xs, ys = ys, xs
        data = dict(zip(('xs', 'ys'), (xs, ys)))
        cdim = element.get_dimension(self.color_index)
        if cdim:
            cmapper = self._get_colormapper(cdim, element, ranges, style)
            data[cdim.name] = [] if empty else element.dimension_values(cdim)
            mapping['color'] = {'field': cdim.name,
                                'transform': cmapper}

        if any(isinstance(t, HoverTool) for t in self.state.tools):
            for d in dims:
                data[dimension_sanitizer(d)] = element.dimension_values(d)

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

    border = param.Integer(default=30 if bokeh_version < '0.12' else 5,
                           doc="Default borders on plot")

    height = param.Integer(default=100 if bokeh_version < '0.12' else 50,
                           doc="Height of plot")

    width = param.Integer(default=100 if bokeh_version < '0.12' else 50,
                          doc="Width of plot")



class ChartPlot(LegendPlot):
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
        with abbreviated_exception():
            plot = self._init_chart(init_element, ranges)

        if plot.legend:
            self._process_legend(plot)

        self.handles['plot'] = plot
        self.handles['glyph_renderers'] = [r for r in plot.renderers
                                           if isinstance(r, GlyphRenderer)]
        self._update_chart(key, element, ranges)

        # Update plot, source and glyph
        self.drawn = True

        return plot

    def _process_legend(self, plot):
        legend = plot.legend[0]
        if not self.show_legend:
            legend.items[:] = []
        else:
            plot.legend.orientation = 'horizontal' if self.legend_cols else 'vertical'
            pos = self.legend_position
            if pos in self.legend_specs:
                opts = self.legend_specs[pos]
                plot.legend[:] = []
                legend.plot = None
                legend.location = opts['loc']
                plot.add_layout(legend, opts['pos'])
            else:
                legend.location = pos

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
        with abbreviated_exception():
            new_chart = self._init_chart(element, ranges)
        old_chart = self.handles['plot']
        update_plot(old_chart, new_chart)
        properties = self._plot_properties(key, old_chart, element)
        old_chart.update(**properties)


    @property
    def current_handles(self):
        return self.state.select(type=(ColumnDataSource, Range1d))


class BoxPlot(ChartPlot):
    """
    BoxPlot generates a box and whisker plot from a BoxWhisker
    Element. This allows plotting the median, mean and various
    percentiles.
    """

    style_opts = ['color', 'whisker_color', 'marker'] + line_properties

    def _init_chart(self, element, ranges):
        properties = self.style[self.cyclic_index]
        label = element.dimensions('key', True)
        dframe = element.dframe()

        # Fix for displaying datetimes which are not handled by bokeh
        for kd in element.kdims:
            col = dframe[kd.name]
            if col.dtype.kind in ('M',):
                dframe[kd.name] = [kd.pprint_value(v).replace(':', ';')
                                   for v in col]

        if not element.kdims:
            dframe[''] = ''
            label = ['']

        return BokehBoxPlot(dframe, label=label, values=element.vdims[0].name,
                            **properties)


    def _update_chart(self, key, element, ranges):
        super(BoxPlot, self)._update_chart(key, element, ranges)
        vdim = element.vdims[0].name
        start, end = ranges[vdim]
        self.state.y_range.start = start
        self.state.y_range.end = end



class BarPlot(ChartPlot):
    """
    BarPlot allows generating single- or multi-category
    bar Charts, by selecting which key dimensions are
    mapped onto separate groups, categories and stacks.
    """

    group_index = param.Integer(default=0, doc="""
       Index of the dimension in the supplied Bars
       Element, which will be laid out into groups.""")

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
        if self.stack_index < element.ndims:
            kwargs['stack'] = kdims[self.stack_index]
        crange = Range1d(*ranges.get(vdim))
        plot = Bar(element.dframe(), values=vdim,
                   continuous_range=crange, **kwargs)
        return plot
