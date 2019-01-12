from __future__ import absolute_import, division, unicode_literals

from collections import defaultdict

import numpy as np
import param
from bokeh.transform import jitter

from ...core.dimension import dimension_name
from ...core.util import basestring, dimension_sanitizer
from ...util.transform import dim
from ..util import compute_sizes, get_min_distance
from .element import ColorbarPlot, LegendPlot
from .styles import expand_batched_style, line_properties, fill_properties


class PointPlot(LegendPlot, ColorbarPlot):

    jitter = param.Number(default=None, bounds=(0, None), doc="""
      The amount of jitter to apply to offset the points along the x-axis.""")

    # Deprecated parameters

    color_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
        Deprecated in favor of color style mapping, e.g. `color=dim('color')`""")

    size_index = param.ClassSelector(default=None, class_=(basestring, int),
                                     allow_None=True, doc="""
        Deprecated in favor of size style mapping, e.g. `size=dim('size')`""")

    scaling_method = param.ObjectSelector(default="area",
                                          objects=["width", "area"],
                                          doc="""
        Deprecated in favor of size style mapping, e.g.
        size=dim('size')**2.""")

    scaling_factor = param.Number(default=1, bounds=(0, None), doc="""
      Scaling factor which is applied to either the width or area
      of each point, depending on the value of `scaling_method`.""")

    size_fn = param.Callable(default=np.abs, doc="""
      Function applied to size values before applying scaling,
      to remove values lower than zero.""")

    style_opts = (['cmap', 'palette', 'marker', 'size', 'angle'] +
                  line_properties + fill_properties)

    _plot_methods = dict(single='scatter', batched='scatter')
    _batched_style_opts = line_properties + fill_properties + ['size']

    def _get_size_data(self, element, ranges, style):
        data, mapping = {}, {}
        sdim = element.get_dimension(self.size_index)
        ms = style.get('size', np.sqrt(6))
        if sdim and ((isinstance(ms, basestring) and ms in element) or isinstance(ms, dim)):
            self.param.warning(
                "Cannot declare style mapping for 'size' option and "
                "declare a size_index; ignoring the size_index.")
            sdim = None
        if not sdim or self.static_source:
            return data, mapping

        map_key = 'size_' + sdim.name
        ms = ms**2
        sizes = element.dimension_values(self.size_index)
        sizes = compute_sizes(sizes, self.size_fn,
                              self.scaling_factor,
                              self.scaling_method, ms)
        if sizes is None:
            eltype = type(element).__name__
            self.param.warning(
                '%s dimension is not numeric, cannot use to scale %s size.'
                % (sdim.pprint_label, eltype))
        else:
            data[map_key] = np.sqrt(sizes)
            mapping['size'] = map_key
        return data, mapping


    def get_data(self, element, ranges, style):
        dims = (dimension_sanitizer(d) for d in element.dimensions(label=True))

        xidx, yidx = (1, 0) if self.invert_axes else (0, 1)
        mapping = dict(x=dims[xidx], y=dims[yidx])
        data = {}

        if not self.static_source or self.batched:
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

        if 'angle' in style and isinstance(style['angle'], (int, float)):
            style['angle'] = np.deg2rad(style['angle'])

        if self.jitter:
            axrange = 'y_range' if self.invert_axes else 'x_range'
            mapping['x'] = jitter(dims[xidx], self.jitter,
                                  range=self.handles[axrange])

        self._get_hover_data(data, element)
        return data, mapping, style


    def get_batched_data(self, element, ranges):
        data = defaultdict(list)
        zorders = self._updated_zorders(element)
        for (key, el), zorder in zip(element.data.items(), zorders):
            self.param.set_param(**self.lookup_options(el, 'plot').options)
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

            if 'hover' in self.handles:
                for d, k in zip(element.dimensions(), key):
                    sanitized = dimension_sanitizer(d.name)
                    data[sanitized].append([k]*nvals)

        data = {k: np.concatenate(v) for k, v in data.items()}
        return data, elmapping, style


class VectorFieldPlot(ColorbarPlot):

    arrow_heads = param.Boolean(default=True, doc="""
        Whether or not to draw arrow heads.""")

    magnitude = param.ClassSelector(class_=(basestring, dim), doc="""
        Dimension or dimension value transform that declares the magnitude
        of each vector. Magnitude is expected to be scaled between 0-1,
        by default the magnitudes are rescaled relative to the minimum
        distance between vectors, this can be disabled with the
        rescale_lengths option.""")

    pivot = param.ObjectSelector(default='mid', objects=['mid', 'tip', 'tail'],
                                 doc="""
        The point around which the arrows should pivot valid options
        include 'mid', 'tip' and 'tail'.""")

    rescale_lengths = param.Boolean(default=True, doc="""
        Whether the lengths will be rescaled to take into account the
        smallest non-zero distance between two vectors.""")

    # Deprecated parameters

    color_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
        Deprecated in favor of dimension value transform on color option,
        e.g. `color=dim('Magnitude')`.
        """)

    size_index = param.ClassSelector(default=None, class_=(basestring, int),
                                     allow_None=True, doc="""
        Deprecated in favor of the magnitude option, e.g.
        `magnitude=dim('Magnitude')`.
        """)

    normalize_lengths = param.Boolean(default=True, doc="""
        Deprecated in favor of rescaling length using dimension value
        transforms using the magnitude option, e.g.
        `dim('Magnitude').norm()`.""")

    style_opts = line_properties + ['scale', 'cmap']

    _nonvectorized_styles = ['scale', 'cmap']

    _plot_methods = dict(single='segment')

    def _get_lengths(self, element, ranges):
        size_dim = element.get_dimension(self.size_index)
        mag_dim = self.magnitude
        if size_dim and mag_dim:
            self.param.warning(
                "Cannot declare style mapping for 'magnitude' option "
                "and declare a size_index; ignoring the size_index.")
        elif size_dim:
            mag_dim = size_dim
        elif isinstance(mag_dim, basestring):
            mag_dim = element.get_dimension(mag_dim)

        (x0, x1), (y0, y1) = (element.range(i) for i in range(2))
        if mag_dim:
            if isinstance(mag_dim, dim):
                magnitudes = mag_dim.apply(element, flat=True)
            else:
                magnitudes = element.dimension_values(mag_dim)
                _, max_magnitude = ranges[dimension_name(mag_dim)]['combined']
                if self.normalize_lengths and max_magnitude != 0:
                    magnitudes = magnitudes / max_magnitude
            if self.rescale_lengths:
                base_dist = get_min_distance(element)
                magnitudes *= base_dist
        else:
            magnitudes = np.ones(len(element))
            if self.rescale_lengths:
                base_dist = get_min_distance(element)
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

        color = None
        if self.arrow_heads:
            arrow_len = (lens/4.)
            xa1s = x0s - np.cos(rads+np.pi/4)*arrow_len
            ya1s = y0s - np.sin(rads+np.pi/4)*arrow_len
            xa2s = x0s - np.cos(rads-np.pi/4)*arrow_len
            ya2s = y0s - np.sin(rads-np.pi/4)*arrow_len
            x0s = np.tile(x0s, 3)
            x1s = np.concatenate([x1s, xa1s, xa2s])
            y0s = np.tile(y0s, 3)
            y1s = np.concatenate([y1s, ya1s, ya2s])
            if cdim and cdim.name in cdata:
                color = np.tile(cdata[cdim.name], 3)
        elif cdim:
            color = cdata.get(cdim.name)

        data = {'x0': x0s, 'x1': x1s, 'y0': y0s, 'y1': y1s}
        mapping = dict(x0='x0', x1='x1', y0='y0', y1='y1')
        if cdim and color is not None:
            data[cdim.name] = color
            mapping.update(cmapping)

        return (data, mapping, style)
