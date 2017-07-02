from collections import defaultdict

import numpy as np
import param
from bokeh.models import (DataRange1d, CategoricalColorMapper, CustomJS,
                          HoverTool, FactorRange)
from bokeh.models.tools import BoxSelectTool

from ...core import Dataset, OrderedDict
from ...core.dimension import Dimension
from ...core.util import (max_range, basestring, dimension_sanitizer,
                          wrap_tuple, unique_iterator)
from ...element import Bars
from ...operation import interpolate_curve
from ..util import compute_sizes, get_min_distance, dim_axis_label
from .element import (ElementPlot, ColorbarPlot, LegendPlot, CompositeElementPlot,
                      line_properties, fill_properties)
from .path import PathPlot, PolygonPlot
from .util import bokeh_version, expand_batched_style, categorize_array, rgb2hex


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

    pivot = param.ObjectSelector(default='mid', objects=['mid', 'tip', 'tail'],
                                 doc="""
       The point around which the arrows should pivot valid options
       include 'mid', 'tip' and 'tail'.""")

    rescale_lengths = param.Boolean(default=True, doc="""
       Whether the lengths will be rescaled to take into account the
       smallest non-zero distance between two vectors.""")

    style_opts = line_properties + ['scale', 'cmap']
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
            magnitudes = np.ones(len(element))
            if self.rescale_lengths:
                magnitudes *= base_dist

        return magnitudes

    def _glyph_properties(self, *args):
        properties = super(VectorFieldPlot, self)._glyph_properties(*args)
        properties.pop('scale', None)
        return properties


    def get_data(self, element, ranges=None, empty=False):
        style = self.style[self.cyclic_index]
        input_scale = style.pop('scale', 1.0)

        # Get x, y, angle, magnitude and color data
        xidx, yidx = (1, 0) if self.invert_axes else (0, 1)
        rads = element.dimension_values(2)
        lens = self._get_lengths(element, ranges)/input_scale
        cdim = element.get_dimension(self.color_index)
        cdata, cmapping = self._get_color_data(element, ranges, style,
                                               name='line_color')

        # Compute segments and arrowheads
        xs = element.dimension_values(xidx)
        ys = element.dimension_values(yidx)

        # Compute offset depending on pivot option
        xoffsets = np.cos(rads)*lens/2.
        yoffsets = np.sin(rads)*lens/2.
        if self.pivot == 'mid':
            nxoff, pxoff = xoffsets, xoffsets
            nyoff, pyoff = yoffsets, yoffsets
        elif self.pivot == 'tip':
            nxoff, pxoff = 0, xoffsets*2
            nyoff, pyoff = 0, yoffsets*2
        elif self.pivot == 'tail':
            nxoff, pxoff = xoffsets*2, 0
            nyoff, pyoff = yoffsets*2, 0
        x0s, x1s = (xs + nxoff, xs - pxoff)
        y0s, y1s = (ys + nyoff, ys - pyoff)

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
        ylow, yhigh = element.get_dimension(1).range
        y0 = np.nanmin([0, y0]) if ylow is None or not np.isfinite(ylow) else ylow
        y1 = np.nanmax([0, y1]) if yhigh is None or not np.isfinite(yhigh) else yhigh
        return (x0, y0, x1, y1)



class SideHistogramPlot(ColorbarPlot, HistogramPlot):

    style_opts = HistogramPlot.style_opts + ['cmap']

    height = param.Integer(default=125, doc="The height of the plot")

    width = param.Integer(default=125, doc="The width of the plot")

    show_title = param.Boolean(default=False, doc="""
        Whether to display the plot title.""")

    default_tools = param.List(default=['save', 'pan', 'wheel_zoom',
                                        'box_zoom', 'reset', 'ybox_select'],
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
        if cmapper and dim in element.dimensions():
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

    color_index = param.ClassSelector(default=None, allow_None=True,
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
        if len(element) == 0:
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



class BarPlot(ColorbarPlot, LegendPlot):
    """
    BarPlot allows generating single- or multi-category
    bar Charts, by selecting which key dimensions are
    mapped onto separate groups, categories and stacks.
    """

    color_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
       Index of the dimension from which the color will the drawn""")

    group_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
       Index of the dimension in the supplied Bars
       Element, which will be laid out into groups.""")

    stack_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
       Index of the dimension in the supplied Bars
       Element, which will stacked.""")

    style_opts = line_properties + fill_properties + ['width', 'cmap']

    _plot_methods = dict(single=('vbar', 'hbar'), batched=('vbar', 'hbar'))

    # Declare that y-range should auto-range if not bounded
    _y_range_type = DataRange1d

    def get_extents(self, element, ranges):
        """
        Make adjustments to plot extents by computing
        stacked bar heights, adjusting the bar baseline
        and forcing the x-axis to be categorical.
        """
        if self.batched:
            overlay = self.current_frame
            element = Bars(overlay.table(), kdims=element.kdims+overlay.kdims,
                           vdims=element.vdims)
            for kd in overlay.kdims:
                ranges[kd.name] = overlay.range(kd)

        stacked = element.get_dimension(self.stack_index)
        extents = super(BarPlot, self).get_extents(element, ranges)
        xdim = element.kdims[0]
        ydim = element.vdims[0]

        # Compute stack heights
        if stacked:
            ds = Dataset(element)
            y0, y1 = ds.aggregate(xdim, function=np.sum).range(ydim)
        else:
            y0, y1 = ranges[ydim.name]

        # Set y-baseline
        if y0 < 0:
            y1 = max([y1, 0])
        else:
            y0 = None if self.logy else 0

        # Ensure x-axis is picked up as categorical
        x0 = xdim.pprint_value(extents[0])
        x1 = xdim.pprint_value(extents[2])
        return (x0, y0, x1, y1)

    def _get_axis_labels(self, *args, **kwargs):
        """
        Override axis mapping by setting the first key and value
        dimension as the x-axis and y-axis labels.
        """
        element = self.current_frame
        if self.batched:
            element = element.last
        return (dim_axis_label(element.kdims[0]),
                dim_axis_label(element.vdims[0]), None)

    def get_group(self, xvals, nshift, ngroups, width, xdim):
        """
        Adjust x-value positions on categorical axes to stop
        x-axis overlapping. Currently bokeh uses a suffix
        of the format ':%f' with a floating value to set up
        offsets within a single category.
        """
        adjusted_xvals = []
        gwidth = float(width)/ngroups
        offset = (1.-width)/2. + gwidth/2.
        for x in xvals:
            adjustment = (offset+nshift/float(ngroups)*width)
            xcat = xdim.pprint_value(x).replace(':',';')
            adjusted_xvals.append(xcat+':%.4f' % adjustment)
        return adjusted_xvals

    def get_stack(self, xvals, yvals, baselines):
        """
        Iterates over a x- and y-values in a stack layer
        and appropriately offsets the layer on top of the
        previous layer.
        """
        bottoms, tops = [], []
        for x, y in zip(xvals, yvals):
            bottom = baselines[x]
            top = bottom+y
            baselines[x] = top
            bottoms.append(bottom)
            tops.append(top)
        return bottoms, tops

    def _glyph_properties(self, *args):
        props = super(BarPlot, self)._glyph_properties(*args)
        del props['width']
        return props

    def get_data(self, element, ranges, empty):
        # Get x, y, group, stack and color dimensions
        group_dim = element.get_dimension(self.group_index)
        stack_dim = element.get_dimension(self.stack_index)
        if stack_dim:
            group_dim = stack_dim
            grouping = 'stacked'
        elif group_dim:
            grouping = 'grouped'
            group_dim = group_dim
        else:
            grouping, group_dim = None, None
        xdim = element.get_dimension(0)
        ydim = element.get_dimension(element.vdims[0])
        no_cidx = self.color_index is None
        color_index = group_dim if no_cidx else self.color_index
        color_dim = element.get_dimension(color_index)
        if color_dim:
            self.color_index = color_dim.name

        # Define style information
        style = self.style[self.cyclic_index]
        width = style.get('width', 1)
        cmap = style.get('cmap')
        hover = any(t == 'hover' or isinstance(t, HoverTool)
                    for t in self.tools+self.default_tools)

        # Group by stack or group dim if necessary
        if group_dim is None:
            grouped = {0: element}
        else:
            grouped = element.groupby(group_dim, group_type=Dataset,
                                      container_type=OrderedDict,
                                      datatype=['dataframe', 'dictionary'])

        # Map attributes to data
        if grouping == 'stacked':
            mapping = {'x': xdim.name, 'top': 'top',
                       'bottom': 'bottom', 'width': width}
        elif grouping == 'grouped':
            if len(grouped):
                gwidth = width / float(len(grouped))
            else:
                gwidth = width
            mapping = {'x': 'xoffsets', 'top': ydim.name, 'bottom': 0,
                       'width': gwidth}
        else:
            mapping = {'x': xdim.name, 'top': ydim.name, 'bottom': 0, 'width': width}

        # Get colors
        cdim = color_dim or group_dim
        cvals = element.dimension_values(cdim, expanded=False) if cdim else None
        if cvals is not None:
            if cvals.dtype.kind in 'if' and no_cidx:
                cvals = categorize_array(cvals, group_dim)
            factors = None if cvals.dtype.kind in 'if' else list(cvals)
            if cdim is xdim and factors:
                factors = list(categorize_array(factors, xdim))
            if cmap is None and factors:
                styles = self.style.max_cycles(len(factors))
                colors = [styles[i]['color'] for i in range(len(factors))]
                colors = [rgb2hex(c) if isinstance(c, tuple) else c for c in colors]
            else:
                colors = None
        else:
            factors, colors = None, None

        # Iterate over stacks and groups and accumulate data
        data = defaultdict(list)
        baselines = defaultdict(float)
        for i, (k, ds) in enumerate(grouped.items()):
            xs = ds.dimension_values(xdim)
            ys = ds.dimension_values(ydim)

            # Apply stacking or grouping
            if grouping == 'stacked':
                bs, ts = self.get_stack(xs, ys, baselines)
                data['bottom'].append(bs)
                data['top'].append(ts)
                data[xdim.name].append(xs)
                if hover: data[ydim.name].append(ys)
            elif grouping == 'grouped':
                xoffsets = self.get_group(xs, i, len(grouped), width, xdim)
                data['xoffsets'].append(xoffsets)
                data[ydim.name].append(ys)
                if hover: data[xdim.name].append(xs)
            else:
                data[xdim.name].append(xs)
                data[ydim.name].append(ys)

            # Add group dimension to data
            if group_dim and group_dim not in ds.dimensions():
                k = k[0] if isinstance(k, tuple) else k
                gval = group_dim.pprint_value(k).replace(':', ';')
                ds = ds.add_dimension(group_dim.name, ds.ndims, gval)

            # Get colormapper
            cdata, cmapping = self._get_color_data(ds, ranges, dict(style),
                                                   factors=factors, colors=colors)

            # Skip if no colormapper applied
            if 'color' not in cmapping:
                continue

            # Enable legend if colormapper is categorical
            cmapper = cmapping['color']['transform']
            if ('color' in cmapping and self.show_legend and
                isinstance(cmapper, CategoricalColorMapper)):
                mapping['legend'] = cdim.name

            # Merge data and mappings
            mapping.update(cmapping)
            for k, cd in cdata.items():
                # If values have already been added, skip
                if no_cidx and cd.dtype.kind in 'if':
                    cd = categorize_array(cd, group_dim)
                if not len(data[k]) == i+1:
                    data[k].append(cd)

            # Fill in missing hover data if dimension other than group_dim is colormapped
            if hover and group_dim and cdim != group_dim:
                data[group_dim.name].append(ds.dimension_values(group_dim))

        # Concatenate the stacks or groups
        sanitized_data = {}
        for col, vals in data.items():
            if len(vals) == 1:
                sanitized_data[dimension_sanitizer(col)] = vals[0]
            elif vals:
                sanitized_data[dimension_sanitizer(col)] = np.concatenate(vals)

        for name, val in mapping.items():
            sanitized = None
            if isinstance(val, basestring):
                sanitized = dimension_sanitizer(mapping[name])
                mapping[name] = sanitized
            elif isinstance(val, dict) and 'field' in val:
                sanitized = dimension_sanitizer(val['field'])
                val['field'] = sanitized
            if sanitized is not None and sanitized not in sanitized_data:
                sanitized_data[sanitized] = []

        # Ensure x-values are categorical
        xname = dimension_sanitizer(xdim.name)
        if xname in sanitized_data:
            sanitized_data[xname] = categorize_array(sanitized_data[xname], xdim)

        # If axes inverted change mapping to match hbar signature
        if self.invert_axes:
            mapping.update({'y': mapping.pop('x'), 'left': mapping.pop('bottom'),
                            'right': mapping.pop('top'), 'height': mapping.pop('width')})

        return sanitized_data, mapping

    def get_batched_data(self, element, ranges, empty):
        el = element.last
        collapsed = Bars(element.table(), kdims=el.kdims+element.kdims,
                            vdims=el.vdims)
        return self.get_data(collapsed, ranges, empty)



class BoxWhiskerPlot(CompositeElementPlot, ColorbarPlot, LegendPlot):

    color_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    # X-axis is categorical
    _x_range_type = FactorRange

    # Declare that y-range should auto-range if not bounded
    _y_range_type = DataRange1d

    # Map each glyph to a style group
    _style_groups = {'rect': 'whisker', 'segment': 'whisker',
                     'vbar': 'box', 'hbar': 'box', 'circle': 'outlier'}

    # Define all the glyph handles to update
    _update_handles = ([glyph+'_'+model for model in ['glyph', 'glyph_renderer', 'source']
                        for glyph in ['vbar_1', 'vbar_2', 'segment_1', 'segment_2',
                                      'rect_1', 'rect_2', 'circle', 'hbar_1', 'hbar_2']] +
                       ['color_mapper', 'colorbar'])

    style_opts = (['whisker_'+p for p in line_properties] +\
                  ['box_'+p for p in fill_properties+line_properties] +\
                  ['outlier_'+p for p in fill_properties+line_properties] + ['width', 'cmap'])

    def get_extents(self, element, ranges):
        """
        Extents are set to '' and None because x-axis is categorical and
        y-axis auto-ranges.
        """
        return ('', None, '', None)

    def _get_axis_labels(self, *args, **kwargs):
        """
        Override axis labels to group all key dimensions together.
        """
        element = self.current_frame
        xlabel = ', '.join([kd.pprint_label for kd in element.kdims])
        ylabel = element.vdims[0].pprint_label
        return xlabel, ylabel, None

    def _glyph_properties(self, plot, element, source, ranges):
        properties = dict(self.style[self.cyclic_index], source=source)
        if self.show_legend and not element.kdims:
            properties['legend'] = element.label
        return properties

    def _get_factors(self, element):
        """
        Get factors for categorical axes.
        """
        if not element.kdims:
            return [element.label], []
        else:
            factors = [', '.join([d.pprint_value(v).replace(':', ';')
                                  for d, v in zip(element.kdims, key)])
                       for key in element.groupby(element.kdims).data.keys()]
            if self.invert_axes:
                return [], factors
            else:
                return factors, []

    def get_data(self, element, ranges=None, empty=False):
        if element.kdims:
            groups = element.groupby(element.kdims).data
        else:
            groups = dict([(element.label, element)])
        style = self.style[self.cyclic_index]
        vdim = dimension_sanitizer(element.vdims[0].name)

        # Define CDS data
        r1_data, r2_data = ({'index': [], 'top': [], 'bottom': []} for i in range(2))
        s1_data, s2_data = ({'x0': [], 'y0': [], 'x1': [], 'y1': []} for i in range(2))
        w1_data, w2_data = ({'index': [], vdim: []} for i in range(2))
        out_data = defaultdict(list, {'index': [], vdim: []})

        # Define glyph-data mapping
        width = style.get('width', 0.7)
        if self.invert_axes:
            vbar_map = {'y': 'index', 'left': 'top', 'right': 'bottom', 'height': width}
            seg_map = {'y0': 'x0', 'y1': 'x1', 'x0': 'y0', 'x1': 'y1'}
            whisk_map = {'y': 'index', 'x': vdim, 'height': 0.2, 'width': 0.001}
            out_map = {'y': 'index', 'x': vdim}
        else:
            vbar_map = {'x': 'index', 'top': 'top', 'bottom': 'bottom', 'width': width}
            seg_map = {'x0': 'x0', 'x1': 'x1', 'y0': 'y0', 'y1': 'y1'}
            whisk_map = {'x': 'index', 'y': vdim, 'width': 0.2, 'height': 0.001}
            out_map = {'x': 'index', 'y': vdim}
        vbar2_map = dict(vbar_map)

        # Get color values
        if self.color_index is not None:
            cdim = element.get_dimension(self.color_index)
            cidx = element.get_dimension_index(self.color_index)
        else:
            cdim, cidx = None, None

        factors = []
        for key, g in groups.items():
            # Compute group label
            if element.kdims:
                label = ', '.join([d.pprint_value(v).replace(':', ';')
                                   for d, v in zip(element.kdims, key)])
            else:
                label = key

            # Add color factor
            if cidx is not None and cidx<element.ndims:
                factors.append(wrap_tuple(key)[cidx])
            else:
                factors.append(label)

            # Compute statistics
            vals = g.dimension_values(g.vdims[0])
            qmin, q1, q2, q3, qmax = (np.percentile(vals, q=q) for q in range(0,125,25))
            iqr = q3 - q1
            upper = q3 + 1.5*iqr
            lower = q1 - 1.5*iqr
            outliers = vals[(vals>upper) | (vals<lower)]

            # Add to CDS data
            for data in [r1_data, r2_data, w1_data, w2_data]:
                data['index'].append(label)
            for data in [s1_data, s2_data]:
                data['x0'].append(label)
                data['x1'].append(label)
            r1_data['top'].append(q2)
            r2_data['top'].append(q1)
            r1_data['bottom'].append(q3)
            r2_data['bottom'].append(q2)
            s1_data['y0'].append(upper)
            s2_data['y0'].append(lower)
            s1_data['y1'].append(q3)
            s2_data['y1'].append(q1)
            w1_data[vdim].append(lower)
            w2_data[vdim].append(upper)
            if len(outliers):
                out_data['index'] += [label]*len(outliers)
                out_data[vdim] += list(outliers)
                if any(isinstance(t, HoverTool) for t in self.state.tools):
                    for kd, k in zip(element.kdims, wrap_tuple(key)):
                        out_data[dimension_sanitizer(kd.name)] += [k]*len(outliers)

        # Define combined data and mappings
        bar_glyph = 'hbar' if self.invert_axes else 'vbar'
        data = {
            bar_glyph+'_1': r1_data, bar_glyph+'_2': r2_data, 'segment_1': s1_data,
            'segment_2': s2_data, 'rect_1': w1_data, 'rect_2': w2_data,
            'circle': out_data
        }
        mapping = {
            bar_glyph+'_1': vbar_map, bar_glyph+'_2': vbar2_map, 'segment_1': seg_map,
            'segment_2': seg_map, 'rect_1': whisk_map, 'rect_2': whisk_map,
            'circle': out_map
        }

        # Cast data to arrays to take advantage of base64 encoding
        for gdata in [r1_data, r2_data, s1_data, s2_data, w1_data, w2_data, out_data]:
            for k, values in gdata.items():
                gdata[k] = np.array(values)

        # Return if not grouped
        if not element.kdims:
            return data, mapping

        # Define color dimension and data
        if cidx is None or cidx>=element.ndims:
            cdim = Dimension('index')
        else:
            r1_data[dimension_sanitizer(cdim.name)] = factors
            r2_data[dimension_sanitizer(cdim.name)] = factors
            factors = list(unique_iterator(factors))

        # Get colors and define categorical colormapper
        cname = dimension_sanitizer(cdim.name)
        cmap = style.get('cmap')
        if cmap is None:
            styles = self.style.max_cycles(len(factors))
            colors = [styles[i].get('box_color', styles[i]['box_fill_color'])
                      for i in range(len(factors))]
            colors = [rgb2hex(c) if isinstance(c, tuple) else c for c in colors]
        else:
            colors = None
        mapper = self._get_colormapper(cdim, element, ranges, style, factors, colors)
        vbar_map['fill_color'] = {'field': cname, 'transform': mapper}
        vbar2_map['fill_color'] = {'field': cname, 'transform': mapper}
        vbar_map['legend'] = cdim.name

        return data, mapping

