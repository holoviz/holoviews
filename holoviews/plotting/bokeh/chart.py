from collections import defaultdict

import numpy as np
import param
try:
    from bokeh.charts import Bar, BoxPlot as BokehBoxPlot
except:
    Bar, BokehBoxPlot = None, None
from bokeh.models import ( GlyphRenderer, ColumnDataSource,
                          Range1d, CustomJS, HoverTool)
from bokeh.models.tools import BoxSelectTool

from ...core.util import max_range, basestring, dimension_sanitizer
from ...core.options import abbreviated_exception
from ...core.spaces import DynamicMap
from ...operation import interpolate_curve
from ..util import compute_sizes,  match_spec, get_min_distance
from .element import (ElementPlot, ColorbarPlot, LegendPlot, line_properties,
                      fill_properties)
from .path import PathPlot, PolygonPlot
from .util import update_plot, bokeh_version, expand_batched_style


class PointPlot(LegendPlot, ColorbarPlot):

    color_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    size_index = param.ClassSelector(default=None, class_=(basestring, int),
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

    style_opts = (['cmap', 'palette', 'marker', 'size'] +
                  line_properties + fill_properties)

    _plot_methods = dict(single='scatter', batched='scatter')
    _batched_style_opts = line_properties + fill_properties + ['size']

    def _get_size_data(self, element, ranges, style):
        data, mapping = {}, {}
        sdim = element.get_dimension(self.size_index)
        if sdim:
            map_key = 'size_' + sdim.name
            ms = style.get('size', np.sqrt(6))**2
            sizes = element.dimension_values(self.size_index)
            sizes = compute_sizes(sizes, self.size_fn,
                                  self.scaling_factor,
                                  self.scaling_method, ms)
            if sizes is None:
                eltype = type(element).__name__
                self.warning('%s dimension is not numeric, cannot '
                             'use to scale %s size.' % (sdim.pprint_label, eltype))
            else:
                data[map_key] = np.sqrt(sizes)
                mapping['size'] = map_key
        return data, mapping


    def get_data(self, element, ranges=None, empty=False):
        style = self.style[self.cyclic_index]
        dims = element.dimensions(label=True)

        xidx, yidx = (1, 0) if self.invert_axes else (0, 1)
        mapping = dict(x=dims[xidx], y=dims[yidx])
        data = {}

        xdim, ydim = dims[xidx], dims[yidx]
        data[xdim] = [] if empty else element.dimension_values(xidx)
        data[ydim] = [] if empty else element.dimension_values(yidx)
        self._categorize_data(data, (xdim, ydim), element.dimensions())

        cdata, cmapping = self._get_color_data(element, ranges, style)
        data.update(cdata)
        mapping.update(cmapping)

        sdata, smapping = self._get_size_data(element, ranges, style)
        data.update(sdata)
        mapping.update(smapping)

        self._get_hover_data(data, element, empty)
        return data, mapping


    def get_batched_data(self, element, ranges=None, empty=False):
        data = defaultdict(list)
        zorders = self._updated_zorders(element)
        styles = self.lookup_options(element.last, 'style')
        styles = styles.max_cycles(len(self.ordering))
        for (key, el), zorder in zip(element.data.items(), zorders):
            self.set_param(**self.lookup_options(el, 'plot').options)
            eldata, elmapping = self.get_data(el, ranges, empty)
            for k, eld in eldata.items():
                data[k].append(eld)

            # Apply static styles
            nvals = len(list(eldata.values())[0])
            style = styles[zorder]
            sdata, smapping = expand_batched_style(style, self._batched_style_opts,
                                                   elmapping, nvals)
            elmapping.update(smapping)
            for k, v in sdata.items():
                data[k].append(v)

            if any(isinstance(t, HoverTool) for t in self.state.tools):
                for dim, k in zip(element.dimensions(), key):
                    sanitized = dimension_sanitizer(dim.name)
                    data[sanitized].append([k]*nvals)

        data = {k: np.concatenate(v) for k, v in data.items()}
        return data, elmapping



class VectorFieldPlot(ColorbarPlot):

    arrow_heads = param.Boolean(default=True, doc="""
       Whether or not to draw arrow heads.""")

    color_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    size_index = param.ClassSelector(default=None, class_=(basestring, int),
                                     allow_None=True, doc="""
      Index of the dimension from which the sizes will the drawn.""")

    normalize_lengths = param.Boolean(default=True, doc="""
       Whether to normalize vector magnitudes automatically. If False,
       it will be assumed that the lengths have already been correctly
       normalized.""")

    rescale_lengths = param.Boolean(default=True, doc="""
       Whether the lengths will be rescaled to take into account the
       smallest non-zero distance between two vectors.""")

    style_opts = line_properties
    _plot_methods = dict(single='segment')

    def _get_lengths(self, element, ranges):
        mag_dim = element.get_dimension(self.size_index)
        (x0, x1), (y0, y1) = (element.range(i) for i in range(2))
        base_dist = get_min_distance(element)
        if mag_dim:
            magnitudes = element.dimension_values(mag_dim)
            _, max_magnitude = ranges[mag_dim.name]
            if self.normalize_lengths and max_magnitude != 0:
                magnitudes = magnitudes / max_magnitude
            if self.rescale_lengths:
                magnitudes *= base_dist
        else:
            magnitudes = np.ones(len(element))*base_dist
        return magnitudes


    def get_data(self, element, ranges=None, empty=False):
        style = self.style[self.cyclic_index]

        # Get x, y, angle, magnitude and color data
        xidx, yidx = (1, 0) if self.invert_axes else (0, 1)
        rads = element.dimension_values(2)
        lens = self._get_lengths(element, ranges)
        cdim = element.get_dimension(self.color_index)
        cdata, cmapping = self._get_color_data(element, ranges, style,
                                               name='line_color')

        # Compute segments and arrowheads
        xs = element.dimension_values(xidx)
        ys = element.dimension_values(yidx)
        xoffsets = np.cos(rads)*lens/2.
        yoffsets = np.sin(rads)*lens/2.
        x0s, x1s = (xs + xoffsets, xs - xoffsets)
        y0s, y1s = (ys + yoffsets, ys - yoffsets)

        if self.arrow_heads:
            arrow_len = (lens/4.)
            xa1s = x0s - np.cos(rads+np.pi/4)*arrow_len
            ya1s = y0s - np.sin(rads+np.pi/4)*arrow_len
            xa2s = x0s - np.cos(rads-np.pi/4)*arrow_len
            ya2s = y0s - np.sin(rads-np.pi/4)*arrow_len
            x0s = np.concatenate([x0s, x0s, x0s])
            x1s = np.concatenate([x1s, xa1s, xa2s])
            y0s = np.concatenate([y0s, y0s, y0s])
            y1s = np.concatenate([y1s, ya1s, ya2s])
            if cdim:
                color = cdata.get(cdim.name)
                color = np.concatenate([color, color, color])
        elif cdim:
            color = cdata.get(cdim.name)

        data = {'x0': x0s, 'x1': x1s, 'y0': y0s, 'y1': y1s}
        mapping = dict(x0='x0', x1='x1', y0='y0', y1='y1')
        if cdim:
            data[cdim.name] = color
            mapping.update(cmapping)

        return (data, mapping)



class CurvePlot(ElementPlot):

    interpolation = param.ObjectSelector(objects=['linear', 'steps-mid',
                                                  'steps-pre', 'steps-post'],
                                         default='linear', doc="""
        Defines how the samples of the Curve are interpolated,
        default is 'linear', other options include 'steps-mid',
        'steps-pre' and 'steps-post'.""")

    style_opts = line_properties
    _plot_methods = dict(single='line', batched='multi_line')
    _batched_style_opts = line_properties

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

        zorders = self._updated_zorders(overlay)
        styles = self.lookup_options(overlay.last, 'style')
        styles = styles.max_cycles(len(self.ordering))

        for (key, el), zorder in zip(overlay.data.items(), zorders):
            self.set_param(**self.lookup_options(el, 'plot').options)
            eldata, elmapping = self.get_data(el, ranges, empty)
            for k, eld in eldata.items():
                data[k].append(eld)

            # Apply static styles
            style = styles[zorder]
            sdata, smapping = expand_batched_style(style, self._batched_style_opts,
                                                   elmapping, nvals=1)
            elmapping.update(smapping)
            for k, v in sdata.items():
                data[k].append(v[0])

            for d, k in zip(overlay.kdims, key):
                sanitized = dimension_sanitizer(d.name)
                data[sanitized].append(k)
        data = {opt: vals for opt, vals in data.items()
                if not any(v is None for v in vals)}
        mapping = {{'x': 'xs', 'y': 'ys'}.get(k, k): v
                   for k, v in elmapping.items()}
        return data, mapping


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

    style_opts = line_properties + fill_properties

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

    style_opts = line_properties + fill_properties
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

    style_opts = line_properties

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
            if self.batched:
                bs, ts = [], []
                # Iterate over current NdOverlay and compute extents
                # from position and length plot options
                for el in self.current_frame.values():
                    opts = self.lookup_options(el, 'plot').options
                    pos = opts.get('position', self.position)
                    length = opts.get('spike_length', self.spike_length)
                    bs.append(pos)
                    ts.append(pos+length)
                b = np.nanmin(bs)
                t = np.nanmax(ts)
            else:
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
            xs, ys = zip(*((np.array([x, x]), np.array([pos+y, pos]))
                           for x, y in element.array(dims[:2])))
        else:
            height = self.spike_length
            xs, ys = zip(*((np.array([x[0], x[0]]), np.array([pos+height, pos]))
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

    def update_frame(self, key, ranges=None, plot=None, element=None):
        """
        Updates an existing plot with data corresponding
        to the key.
        """
        reused = isinstance(self.hmap, DynamicMap) and (self.overlaid or self.batched)
        if not reused and element is None:
            element = self._get_frame(key)
        elif element is not None:
            self.current_key = key
            self.current_frame = element

        if element is None or (not self.dynamic and self.static):
            return

        max_cycles = len(self.style._options)
        self.style = self.lookup_options(element, 'style').max_cycles(max_cycles)

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

    style_opts = ['whisker_color', 'marker'] + line_properties

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
