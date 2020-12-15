from __future__ import absolute_import, division, unicode_literals

from collections import defaultdict

import param
import numpy as np

from ...core import util
from ...element import Contours, Polygons
from ...util.transform import dim
from .callbacks import PolyDrawCallback, PolyEditCallback
from .element import ColorbarPlot, LegendPlot, OverlayPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import (
    expand_batched_style, base_properties, line_properties, fill_properties,
    mpl_to_bokeh, validate
)
from .util import bokeh_version, multi_polygons_data


class PathPlot(LegendPlot, ColorbarPlot):

    selected = param.List(default=None, doc="""
        The current selection as a list of integers corresponding
        to the selected items.""")

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    # Deprecated options

    color_index = param.ClassSelector(default=None, class_=(util.basestring, int),
                                      allow_None=True, doc="""
        Deprecated in favor of color style mapping, e.g. `color=dim('color')`""")

    style_opts = base_properties + line_properties + ['cmap']

    _plot_methods = dict(single='multi_line', batched='multi_line')
    _mapping = dict(xs='xs', ys='ys')
    _nonvectorized_styles = base_properties + ['cmap']
    _batched_style_opts = line_properties

    def _element_transform(self, transform, element, ranges):
        if isinstance(element, Contours):
            return super(PathPlot, self)._element_transform(transform, element, ranges)
        return np.concatenate([transform.apply(el, ranges=ranges, flat=True)
                               for el in element.split()])

    def _hover_opts(self, element):
        cdim = element.get_dimension(self.color_index)
        if self.batched:
            dims = list(self.hmap.last.kdims)+self.hmap.last.last.vdims
        else:
            dims = list(self.overlay_dims.keys())+self.hmap.last.vdims
        if cdim not in dims and cdim is not None:
            dims.append(cdim)
        return dims, {}


    def _get_hover_data(self, data, element):
        """
        Initializes hover data based on Element dimension values.
        """
        if 'hover' not in self.handles or self.static_source:
            return

        for k, v in self.overlay_dims.items():
            dim = util.dimension_sanitizer(k.name)
            if dim not in data:
                data[dim] = [v for _ in range(len(list(data.values())[0]))]


    def get_data(self, element, ranges, style):
        color = style.get('color', None)
        cdim = None
        if isinstance(color, util.basestring) and not validate('color', color):
            cdim = element.get_dimension(color)
        elif self.color_index is not None:
            cdim = element.get_dimension(self.color_index)

        scalar = element.interface.isunique(element, cdim, per_geom=True) if cdim else False
        style_mapping = {
            (s, v) for s, v in style.items() if (s not in self._nonvectorized_styles) and
            ((isinstance(v, util.basestring) and v in element) or isinstance(v, dim)) and
            not (not isinstance(v, dim) and v == color and s == 'color')}
        mapping = dict(self._mapping)

        if (not cdim or scalar) and not style_mapping and 'hover' not in self.handles:
            if self.static_source:
                data = {}
            else:
                paths = element.split(datatype='columns', dimensions=element.kdims)
                xs, ys = ([path[kd.name] for path in paths] for kd in element.kdims)
                if self.invert_axes:
                    xs, ys = ys, xs
                data = dict(xs=xs, ys=ys)
            return data, mapping, style

        hover = 'hover' in self.handles
        vals = defaultdict(list)
        if hover:
            vals.update({util.dimension_sanitizer(vd.name): [] for vd in element.vdims})
        if cdim and self.color_index is not None:
            dim_name = util.dimension_sanitizer(cdim.name)
            cmapper = self._get_colormapper(cdim, element, ranges, style)
            mapping['line_color'] = {'field': dim_name, 'transform': cmapper}
            vals[dim_name] = []

        xpaths, ypaths = [], []
        for path in element.split():
            if cdim and self.color_index is not None:
                scalar = path.interface.isunique(path, cdim, per_geom=True)
                cvals = path.dimension_values(cdim, not scalar)
                vals[dim_name].append(cvals[:-1])
            cols = path.columns(path.kdims)
            xs, ys = (cols[kd.name] for kd in element.kdims)
            alen = len(xs)
            xpaths += [xs[s1:s2+1] for (s1, s2) in zip(range(alen-1), range(1, alen+1))]
            ypaths += [ys[s1:s2+1] for (s1, s2) in zip(range(alen-1), range(1, alen+1))]
            if not hover:
                continue
            for vd in element.vdims:
                if vd == cdim:
                    continue
                values = path.dimension_values(vd)[:-1]
                vd_name = util.dimension_sanitizer(vd.name)
                vals[vd_name].append(values)

        values = {d: np.concatenate(vs) if len(vs) else [] for d, vs in vals.items()}
        if self.invert_axes:
            xpaths, ypaths = ypaths, xpaths
        data = dict(xs=xpaths, ys=ypaths, **values)
        self._get_hover_data(data, element)
        return data, mapping, style


    def get_batched_data(self, element, ranges=None):
        data = defaultdict(list)

        zorders = self._updated_zorders(element)
        for (key, el), zorder in zip(element.data.items(), zorders):
            el_opts = self.lookup_options(el, 'plot').options
            self.param.set_param(**{k: v for k, v in el_opts.items()
                                    if k not in OverlayPlot._propagate_options})
            style = self.lookup_options(el, 'style')
            style = style.max_cycles(len(self.ordering))[zorder]
            self.overlay_dims = dict(zip(element.kdims, key))
            eldata, elmapping, style = self.get_data(el, ranges, style)
            for k, eld in eldata.items():
                data[k].extend(eld)

            # Skip if data is empty
            if not eldata:
                continue

            # Apply static styles
            nvals = len(list(eldata.values())[0])
            sdata, smapping = expand_batched_style(style, self._batched_style_opts,
                                                   elmapping, nvals)
            elmapping.update({k: v for k, v in smapping.items() if k not in elmapping})
            for k, v in sdata.items():
                data[k].extend(list(v))

        return data, elmapping, style


class ContourPlot(PathPlot):

    selected = param.List(default=None, doc="""
        The current selection as a list of integers corresponding
        to the selected items.""")

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    # Deprecated options

    color_index = param.ClassSelector(default=0, class_=(util.basestring, int),
                                      allow_None=True, doc="""
        Deprecated in favor of color style mapping, e.g. `color=dim('color')`""")

    _color_style = 'line_color'
    _nonvectorized_styles = base_properties + ['cmap']

    def __init__(self, *args, **params):
        super(ContourPlot, self).__init__(*args, **params)
        self._has_holes = None

    def _hover_opts(self, element):
        if self.batched:
            dims = list(self.hmap.last.kdims)+self.hmap.last.last.vdims
        else:
            dims = list(self.overlay_dims.keys())+self.hmap.last.vdims
        return dims, {}

    def _get_hover_data(self, data, element):
        """
        Initializes hover data based on Element dimension values.
        If empty initializes with no data.
        """
        if 'hover' not in self.handles or self.static_source:
            return

        interface = element.interface
        scalar_kwargs = {'per_geom': True} if interface.multi else {}
        npath = len([vs for vs in data.values()][0])
        for d in element.vdims:
            dim = util.dimension_sanitizer(d.name)
            if dim not in data:
                if element.level is not None:
                    data[dim] = np.full(npath, element.level)
                elif interface.isunique(element, d, **scalar_kwargs):
                    data[dim] = element.dimension_values(d, expanded=False)
                else:
                    data[dim] = element.split(datatype='array', dimensions=[d])

        for k, v in self.overlay_dims.items():
            dim = util.dimension_sanitizer(k.name)
            if dim not in data:
                data[dim] = [v for _ in range(len(list(data.values())[0]))]

    def get_data(self, element, ranges, style):
        if self._has_holes is None:
            draw_callbacks = any(isinstance(cb, (PolyDrawCallback, PolyEditCallback))
                                 for cb in self.callbacks)
            has_holes = (isinstance(element, Polygons) and not draw_callbacks)
            self._has_holes = has_holes
        else:
            has_holes = self._has_holes

        if not element.interface.multi:
            element = element.clone([element.data], datatype=type(element).datatype)

        if self.static_source:
            data = dict()
            xs = self.handles['cds'].data['xs']
        else:
            if has_holes:
                xs, ys = multi_polygons_data(element)
            else:
                xs, ys = (list(element.dimension_values(kd, expanded=False))
                          for kd in element.kdims)
            if self.invert_axes:
                xs, ys = ys, xs
            data = dict(xs=xs, ys=ys)
        mapping = dict(self._mapping)
        self._get_hover_data(data, element)

        color, fill_color = style.get('color'), style.get('fill_color')
        if (((isinstance(color, dim) and color.applies(element)) or color in element) or
            (isinstance(fill_color, dim) and fill_color.applies(element)) or fill_color in element):
            cdim = None
        elif None not in [element.level, self.color_index] and element.vdims:
            cdim = element.vdims[0]
        else:
            cidx = self.color_index+2 if isinstance(self.color_index, int) else self.color_index
            cdim = element.get_dimension(cidx)

        if cdim is None:
            return data, mapping, style

        ncontours = len(xs)
        dim_name = util.dimension_sanitizer(cdim.name)
        if element.level is not None:
            values = np.full(ncontours, float(element.level))
        else:
            values = element.dimension_values(cdim, expanded=False)
        data[dim_name] = values

        factors = None
        if cdim.name in ranges and 'factors' in ranges[cdim.name]:
            factors = ranges[cdim.name]['factors']
        elif values.dtype.kind in 'SUO' and len(values):
            if isinstance(values[0], np.ndarray):
                values = np.concatenate(values)
            factors = util.unique_array(values)
        cmapper = self._get_colormapper(cdim, element, ranges, style, factors)
        mapping[self._color_style] = {'field': dim_name, 'transform': cmapper}
        if self.show_legend:
            legend_prop = 'legend_field' if bokeh_version >= '1.3.5' else 'legend'
            mapping[legend_prop] = dim_name
        return data, mapping, style

    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object.
        """
        plot_method = properties.pop('plot_method', None)
        properties = mpl_to_bokeh(properties)
        data = dict(properties, **mapping)
        if self._has_holes:
            plot_method = 'multi_polygons'
        elif plot_method is None:
            plot_method = self._plot_methods.get('single')
        renderer = getattr(plot, plot_method)(**data)
        if self.colorbar:
            for k, v in list(self.handles.items()):
                if not k.endswith('color_mapper'):
                    continue
                self._draw_colorbar(plot, v, k[:-12])
        return renderer, renderer.glyph


class PolygonPlot(ContourPlot):

    style_opts = base_properties + line_properties + fill_properties + ['cmap']
    _plot_methods = dict(single='patches', batched='patches')
    _batched_style_opts = line_properties + fill_properties
    _color_style = 'fill_color'

    selection_display = BokehOverlaySelectionDisplay()
