from collections import defaultdict

import numpy as np
import param
from bokeh.models import (CategoricalColorMapper, CustomJS, HoverTool,
                          FactorRange, Whisker, Band, Range1d)
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
from .util import expand_batched_style, categorize_array, rgb2hex, mpl_to_bokeh


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
        if not sdim or self.static_source:
            return data, mapping
        
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


    def get_data(self, element, ranges, style):
        dims = element.dimensions(label=True)

        xidx, yidx = (1, 0) if self.invert_axes else (0, 1)
        mapping = dict(x=dims[xidx], y=dims[yidx])
        data = {}

        if not self.static_source:
            xdim, ydim = dims[xidx], dims[yidx]
            data[xdim] = element.dimension_values(xidx)
            data[ydim] = element.dimension_values(yidx)
            self._categorize_data(data, (xdim, ydim), element.dimensions())

        cdata, cmapping = self._get_color_data(element, ranges, style)
        data.update(cdata)
        mapping.update(cmapping)

        sdata, smapping = self._get_size_data(element, ranges, style)
        data.update(sdata)
        mapping.update(smapping)

        self._get_hover_data(data, element)
        return data, mapping, style


    def get_batched_data(self, element, ranges):
        data = defaultdict(list)
        zorders = self._updated_zorders(element)
        for (key, el), zorder in zip(element.data.items(), zorders):
            self.set_param(**self.lookup_options(el, 'plot').options)
            style = self.lookup_options(element.last, 'style')
            style = style.max_cycles(len(self.ordering))[zorder]
            eldata, elmapping, style = self.get_data(el, ranges, style)
            for k, eld in eldata.items():
                data[k].append(eld)

            # Skip if data is empty
            if not eldata:
                continue

            # Apply static styles
            nvals = len(list(eldata.values())[0])
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
        return data, elmapping, style



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


    def get_data(self, element, ranges, style):
        input_scale = style.pop('scale', 1.0)

        # Get x, y, angle, magnitude and color data
        rads = element.dimension_values(2)
        if self.invert_axes:
            xidx, yidx = (1, 0)
            rads = rads+1.5*np.pi
        else:
            xidx, yidx = (0, 1)
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

        return (data, mapping, style)



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

    def get_data(self, element, ranges, style):
        xidx, yidx = (1, 0) if self.invert_axes else (0, 1)
        x = element.get_dimension(xidx).name
        y = element.get_dimension(yidx).name
        if self.static_source:
            return {}, dict(x=x, y=y), style

        if 'steps' in self.interpolation:
            element = interpolate_curve(element, interpolation=self.interpolation)
        data = {x: element.dimension_values(xidx),
                y: element.dimension_values(yidx)}
        self._get_hover_data(data, element)
        self._categorize_data(data, (x, y), element.dimensions())
        return (data, dict(x=x, y=y), style)

    def _hover_opts(self, element):
        if self.batched:
            dims = list(self.hmap.last.kdims)
            line_policy = 'prev'
        else:
            dims = list(self.overlay_dims.keys())+element.dimensions()
            line_policy = 'nearest'
        return dims, dict(line_policy=line_policy)

    def get_batched_data(self, overlay, ranges):
        data = defaultdict(list)

        zorders = self._updated_zorders(overlay)
        for (key, el), zorder in zip(overlay.data.items(), zorders):
            self.set_param(**self.lookup_options(el, 'plot').options)
            style = self.lookup_options(el, 'style')
            style = style.max_cycles(len(self.ordering))[zorder]
            eldata, elmapping, style = self.get_data(el, ranges, style)

            # Skip if data empty
            if not eldata:
                continue

            for k, eld in eldata.items():
                data[k].append(eld)

            # Apply static styles
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
        return data, mapping, style



class HistogramPlot(ElementPlot):

    style_opts = line_properties + fill_properties
    _plot_methods = dict(single='quad')

    def get_data(self, element, ranges, style):
        if self.invert_axes:
            mapping = dict(top='left', bottom='right', left=0, right='top')
        else:
            mapping = dict(top='top', bottom=0, left='left', right='right')
        if self.static_source:
            data = dict(top=[], left=[], right=[])
        else:
            data = dict(top=element.values, left=element.edges[:-1],
                        right=element.edges[1:])
        self._get_hover_data(data, element)
        return (data, mapping, style)

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

    def get_data(self, element, ranges, style):
        if self.invert_axes:
            mapping = dict(top='right', bottom='left', left=0, right='top')
        else:
            mapping = dict(top='top', bottom=0, left='left', right='right')

        if self.static_source:
            data = dict(top=[], left=[], right=[])
        else:
            data = dict(top=element.values, left=element.edges[:-1],
                        right=element.edges[1:])

        color_dims = [d for d in self.adjoined.traverse(lambda x: x.handles.get('color_dim'))
                      if d is not None]
        dim = color_dims[0] if color_dims else None
        cmapper = self._get_colormapper(dim, element, {}, {})
        if cmapper and dim in element.dimensions():
            data[dim.name] = [] if self.static_source else element.dimension_values(dim)
            mapping['fill_color'] = {'field': dim.name,
                                     'transform': cmapper}
        self._get_hover_data(data, element)
        return (data, mapping, style)


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



class ErrorPlot(ElementPlot):

    style_opts = line_properties

    _mapping = dict(base="base", upper="upper", lower="lower")

    _plot_methods = dict(single=Whisker)

    def get_data(self, element, ranges, style):
        mapping = dict(self._mapping)
        if self.static_source:
            return {}, mapping, style

        base = element.dimension_values(0)
        ys = element.dimension_values(1)
        if len(element.vdims) > 2:
            neg, pos = (element.dimension_values(vd) for vd in element.vdims[1:3])
            lower, upper = ys-neg, ys+pos
        else:
            err = element.dimension_values(2)
            lower, upper = ys-err, ys+err
        data = dict(base=base, lower=lower, upper=upper)

        if self.invert_axes:
            mapping['dimension'] = 'width'
        else:
            mapping['dimension'] = 'height'
        self._categorize_data(data, ('base',), element.dimensions())
        return (data, mapping, style)


    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object.
        """
        properties.pop('legend', None)
        for prop in ['color', 'alpha']:
            if prop not in properties:
                continue
            pval = properties.pop(prop)
            line_prop = 'line_%s' % prop
            fill_prop = 'fill_%s' % prop
            if line_prop not in properties:
                properties[line_prop] = pval
            if fill_prop not in properties and fill_prop in self.style_opts:
                properties[fill_prop] = pval
        properties = mpl_to_bokeh(properties)
        plot_method = self._plot_methods['single']
        glyph = plot_method(**dict(properties, **mapping))
        plot.add_layout(glyph)
        return None, glyph



class SpreadPlot(ErrorPlot):

    style_opts = line_properties + fill_properties
    _plot_methods = dict(single=Band)



class AreaPlot(SpreadPlot):

    def get_extents(self, element, ranges):
        vdims = element.vdims
        vdim = vdims[0].name
        if len(vdims) > 1:
            ranges[vdim] = max_range([ranges[vd.name] for vd in vdims])
        else:
            vdim = vdims[0].name
            ranges[vdim] = (np.nanmin([0, ranges[vdim][0]]), ranges[vdim][1])
        return super(AreaPlot, self).get_extents(element, ranges)

    def get_data(self, element, ranges, style):
        mapping = dict(self._mapping)
        if self.static_source:
            return {}, mapping, style

        xs = element.dimension_values(0)
        if len(element.vdims) > 1:
            lower = element.dimension_values(2)
        else:
            lower = np.zeros(len(element))
        upper = element.dimension_values(1)
        data = dict(base=xs, upper=upper, lower=lower)

        if self.invert_axes:
            mapping['dimension'] = 'width'
        else:
            mapping['dimension'] = 'height'
        return data, mapping, style



class SpikesPlot(ColorbarPlot):

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

    _plot_methods = dict(single='segment')

    def get_extents(self, element, ranges):
        l, b, r, t = super(SpikesPlot, self).get_extents(element, ranges)
        if len(element.dimensions()) == 1:
            if self.batched:
                bs, ts = [], []
                # Iterate over current NdOverlay and compute extents
                # from position and length plot options
                frame = self.current_frame or self.hmap.last 
                for el in frame.values():
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

    def get_data(self, element, ranges, style):
        dims = element.dimensions(label=True)

        data = {}
        pos = self.position
        if len(element) == 0 or self.static_source:
            data = {'x': [], 'y0': [], 'y1': []}
        else:
            data['x'] = element.dimension_values(0)
            data['y0'] = np.full(len(element), pos)
            if len(dims) > 1:
                data['y1'] = element.dimension_values(1)+pos
            else:
                data['y1'] = data['y0']+self.spike_length

        if self.invert_axes:
            mapping = {'x0': 'y0', 'x1': 'y1', 'y0': 'x', 'y1': 'x'}
        else:
            mapping = {'x0': 'x', 'x1': 'x', 'y0': 'y0', 'y1': 'y1'}
        cdim = element.get_dimension(self.color_index)
        if cdim:
            cmapper = self._get_colormapper(cdim, element, ranges, style)
            data[cdim.name] = [] if self.static_source else element.dimension_values(cdim)
            mapping['color'] = {'field': cdim.name,
                                'transform': cmapper}

        if any(isinstance(t, HoverTool) for t in self.state.tools) and not self.static_source:
            for d in dims:
                data[dimension_sanitizer(d)] = element.dimension_values(d)

        return data, mapping, style


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

    border = param.Integer(default=5, doc="Default borders on plot")

    height = param.Integer(default=50, doc="Height of plot")

    width = param.Integer(default=50, doc="Width of plot")



class BarPlot(ColorbarPlot, LegendPlot):
    """
    BarPlot allows generating single- or multi-category
    bar Charts, by selecting which key dimensions are
    mapped onto separate groups, categories and stacks.
    """

    color_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
       Index of the dimension from which the color will the drawn""")

    group_index = param.ClassSelector(default=1, class_=(basestring, int),
                                      allow_None=True, doc="""
       Index of the dimension in the supplied Bars
       Element, which will be laid out into groups.""")

    stack_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
       Index of the dimension in the supplied Bars
       Element, which will stacked.""")

    style_opts = line_properties + fill_properties + ['width', 'cmap']

    _plot_methods = dict(single=('vbar', 'hbar'))

    # Declare that y-range should auto-range if not bounded
    _y_range_type = Range1d

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
            pos_range = ds.select(**{ydim.name: (0, None)}).aggregate(xdim, function=np.sum).range(ydim)
            neg_range = ds.select(**{ydim.name: (None, 0)}).aggregate(xdim, function=np.sum).range(ydim)
            y0, y1 = max_range([pos_range, neg_range])
        else:
            y0, y1 = ranges[ydim.name]

        # Set y-baseline
        if y0 < 0:
            y1 = max([y1, 0])
        elif self.logy:
            y0 = (ydim.range[0] or (10**(np.log10(y1)-2)) if y1 else 0.01)
        else:
            y0 = 0

        # Ensure x-axis is picked up as categorical
        x0 = xdim.pprint_value(extents[0])
        x1 = xdim.pprint_value(extents[2])
        return (x0, y0, x1, y1)


    def _get_factors(self, element):
        """
        Get factors for categorical axes.
        """
        gdim = element.get_dimension(self.group_index)
        if gdim not in element.kdims:
            gdim = None
        sdim = element.get_dimension(self.stack_index)
        if sdim not in element.kdims:
            sdim = None

        xdim, ydim = element.dimensions()[:2]
        xvals = element.dimension_values(0, False)
        xvals = [x if xvals.dtype.kind in 'SU' else xdim.pprint_value(x)
                 for x in xvals]
        if gdim and not sdim:
            gvals = element.dimension_values(gdim, False)
            gvals = [g if gvals.dtype.kind in 'SU' else gdim.pprint_value(g) for g in gvals]
            coords = ([(x, g) for x in xvals for g in gvals], [])
        else:
            coords = (xvals, [])
        if self.invert_axes: coords = coords[::-1]
        return coords

    def _get_axis_labels(self, *args, **kwargs):
        """
        Override axis mapping by setting the first key and value
        dimension as the x-axis and y-axis labels.
        """
        element = self.current_frame
        if self.batched:
            element = element.last
        xlabel = dim_axis_label(element.kdims[0])
        gdim = element.get_dimension(self.group_index)
        if gdim and gdim in element.kdims:
            xlabel = ', '.join([xlabel, dim_axis_label(gdim)])
        return (xlabel, dim_axis_label(element.vdims[0]), None)

    def get_stack(self, xvals, yvals, baselines, sign='positive'):
        """
        Iterates over a x- and y-values in a stack layer
        and appropriately offsets the layer on top of the
        previous layer.
        """
        bottoms, tops = [], []
        for x, y in zip(xvals, yvals):
            baseline = baselines[x][sign]
            if sign == 'positive':
                bottom = baseline
                top = bottom+y
                baseline = top
            else:
                top = baseline
                bottom = top+y
                baseline = bottom
            baselines[x][sign] = baseline
            bottoms.append(bottom)
            tops.append(top)
        return bottoms, tops

    def _glyph_properties(self, *args):
        props = super(BarPlot, self)._glyph_properties(*args)
        del props['width']
        return props


    def _add_color_data(self, ds, ranges, style, cdim, data, mapping, factors, colors):
        # Get colormapper
        cdata, cmapping = self._get_color_data(ds, ranges, dict(style),
                                               factors=factors, colors=colors)
        if 'color' not in cmapping:
            return

        # Enable legend if colormapper is categorical
        cmapper = cmapping['color']['transform']
        if ('color' in cmapping and self.show_legend and
            isinstance(cmapper, CategoricalColorMapper)):
            mapping['legend'] = cdim.name

        # Merge data and mappings
        mapping.update(cmapping)
        for k, cd in cdata.items():
            if self.color_index is None and cd.dtype.kind in 'if':
                cd = categorize_array(cd, cdim)
            if k not in data or len(data[k]) != [len(data[key]) for key in data if key != k][0]:
                data[k].append(cd)
            else:
                data[k][-1] = cd


    def get_data(self, element, ranges, style):
        # Get x, y, group, stack and color dimensions
        grouping = None
        group_dim = element.get_dimension(self.group_index)
        if group_dim not in element.kdims:
            group_dim = None
        else:
            grouping = 'grouped'
        stack_dim = element.get_dimension(self.stack_index)
        if stack_dim not in element.kdims:
            stack_dim = None
        else:
            grouping = 'stacked'
            group_dim = None

        xdim = element.get_dimension(0)
        ydim = element.vdims[0]
        no_cidx = self.color_index is None
        color_index = (group_dim or stack_dim) if no_cidx else self.color_index
        color_dim = element.get_dimension(color_index)
        if color_dim:
            self.color_index = color_dim.name

        # Define style information
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

        y0, y1 = ranges.get(ydim.name, (None, None))
        if self.logy:
            bottom = (ydim.range[0] or (10**(np.log10(y1)-2)) if y1 else 0.01)
        else:
            bottom = 0
        # Map attributes to data
        if grouping == 'stacked':
            mapping = {'x': xdim.name, 'top': 'top',
                       'bottom': 'bottom', 'width': width}
        elif grouping == 'grouped':
            mapping = {'x': 'xoffsets', 'top': ydim.name, 'bottom': bottom,
                       'width': width}
        else:
            mapping = {'x': xdim.name, 'top': ydim.name, 'bottom': bottom, 'width': width}

        # Get colors
        cdim = color_dim or group_dim
        cvals = element.dimension_values(cdim, expanded=False) if cdim else None
        if cvals is not None:
            if cvals.dtype.kind in 'if' and no_cidx:
                cvals = categorize_array(cvals, color_dim)

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
        baselines = defaultdict(lambda: {'positive': bottom, 'negative': 0})
        for i, (k, ds) in enumerate(grouped.items()):
            k = k[0] if isinstance(k, tuple) else k
            if group_dim:
                gval = k if isinstance(k, basestring) else group_dim.pprint_value(k)
            # Apply stacking or grouping
            if grouping == 'stacked':
                for sign, slc in [('negative', (None, 0)), ('positive', (0, None))]:
                    slc_ds = ds.select(**{ds.vdims[0].name: slc})
                    xs = slc_ds.dimension_values(xdim)
                    ys = slc_ds.dimension_values(ydim)
                    bs, ts = self.get_stack(xs, ys, baselines, sign)
                    data['bottom'].append(bs)
                    data['top'].append(ts)
                    data[xdim.name].append(xs)
                    data[stack_dim.name].append(slc_ds.dimension_values(stack_dim))
                    if hover: data[ydim.name].append(ys)
                    self._add_color_data(slc_ds, ranges, style, cdim, data,
                                         mapping, factors, colors)
            elif grouping == 'grouped':
                xs = ds.dimension_values(xdim)
                ys = ds.dimension_values(ydim)
                xoffsets = [(x if xs.dtype.kind in 'SU' else xdim.pprint_value(x), gval)
                            for x in xs]
                data['xoffsets'].append(xoffsets)
                data[ydim.name].append(ys)
                if hover: data[xdim.name].append(xs)
                if group_dim not in ds.dimensions():
                    ds = ds.add_dimension(group_dim.name, ds.ndims, gval)
                data[group_dim.name].append(ds.dimension_values(group_dim))
            else:
                data[xdim.name].append(ds.dimension_values(xdim))
                data[ydim.name].append(ds.dimension_values(ydim))

            if hover:
                for vd in ds.vdims[1:]:
                    data[vd.name].append(ds.dimension_values(vd))

            if not grouping == 'stacked':
                self._add_color_data(ds, ranges, style, cdim, data,
                                     mapping, factors, colors)

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

        return sanitized_data, mapping, style



class BoxWhiskerPlot(CompositeElementPlot, ColorbarPlot, LegendPlot):

    color_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    # X-axis is categorical
    _x_range_type = FactorRange

    # Declare that y-range should auto-range if not bounded
    _y_range_type = Range1d

    # Map each glyph to a style group
    _style_groups = {'rect': 'whisker', 'segment': 'whisker',
                     'vbar': 'box', 'hbar': 'box', 'circle': 'outlier'}

    style_opts = (['whisker_'+p for p in line_properties] +\
                  ['box_'+p for p in fill_properties+line_properties] +\
                  ['outlier_'+p for p in fill_properties+line_properties] + ['width', 'cmap'])

    _stream_data = False # Plot does not support streaming data

    def get_extents(self, element, ranges):
        """
        Extents are set to '' and None because x-axis is categorical and
        y-axis auto-ranges.
        """
        yrange = ranges.get(element.vdims[0].name, (np.NaN, np.NaN))
        return ('', yrange[0], '', yrange[1])

    def _get_axis_labels(self, *args, **kwargs):
        """
        Override axis labels to group all key dimensions together.
        """
        element = self.current_frame
        xlabel = ', '.join([kd.pprint_label for kd in element.kdims])
        ylabel = element.vdims[0].pprint_label
        return xlabel, ylabel, None

    def _glyph_properties(self, plot, element, source, ranges, style):
        properties = dict(style, source=source)
        if self.show_legend and not element.kdims:
            properties['legend'] = element.label
        return properties

    def _get_factors(self, element):
        """
        Get factors for categorical axes.
        """
        if not element.kdims:
            xfactors, yfactors =  [element.label], []
        else:
            factors = [tuple(d.pprint_value(v) for d, v in zip(element.kdims, key))
                       for key in element.groupby(element.kdims).data.keys()]
            factors = [f[0] if len(f) == 1 else f for f in factors]
            xfactors, yfactors = factors, []
        return (yfactors, xfactors) if self.invert_axes else (xfactors, yfactors)

    def get_data(self, element, ranges, style):
        if element.kdims:
            groups = element.groupby(element.kdims).data
        else:
            groups = dict([(element.label, element)])
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
                label = tuple(d.pprint_value(v) for d, v in zip(element.kdims, key))
                if len(label) == 1:
                    label = label[0]
            else:
                label = key

            # Add color factor
            if cidx is not None and cidx<element.ndims:
                factors.append(cdim.pprint_value(wrap_tuple(key)[cidx]))
            else:
                factors.append(label)

            # Compute statistics
            vals = g.dimension_values(g.vdims[0])
            if len(vals):
                qmin, q1, q2, q3, qmax = (np.percentile(vals, q=q) for q in range(0,125,25))
                iqr = q3 - q1
                upper = min(q3 + 1.5*iqr, vals.max())
                lower = max(q1 - 1.5*iqr, vals.min())
            else:
                q1, q2, q3 = 0, 0, 0
                lower, upper = 0, 0
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
            return data, mapping, style

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
            cycle_style = self.lookup_options(element, 'style')
            styles = cycle_style.max_cycles(len(factors))
            colors = [styles[i].get('box_color', styles[i]['box_fill_color'])
                      for i in range(len(factors))]
            colors = [rgb2hex(c) if isinstance(c, tuple) else c for c in colors]
        else:
            colors = None
        mapper = self._get_colormapper(cdim, element, ranges, style, factors, colors)
        vbar_map['fill_color'] = {'field': cname, 'transform': mapper}
        vbar2_map['fill_color'] = {'field': cname, 'transform': mapper}
        if self.show_legend:
            vbar_map['legend'] = cdim.name

        return data, mapping, style

