from collections import defaultdict

import param
import numpy as np

from ...core import util
from .element import ColorbarPlot, LegendPlot, line_properties, fill_properties
from .util import expand_batched_style


class PathPlot(ColorbarPlot):

    color_index = param.ClassSelector(default=None, class_=(util.basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    style_opts = line_properties + ['cmap']
    _plot_methods = dict(single='multi_line', batched='multi_line')
    _mapping = dict(xs='xs', ys='ys')
    _batched_style_opts = line_properties

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
        cdim = element.get_dimension(self.color_index)
        inds = (1, 0) if self.invert_axes else (0, 1)
        mapping = dict(self._mapping)
        if not cdim:
            if self.static_source:
                data = {}
            else:
                paths = element.split(datatype='array', dimensions=element.kdims)
                xs, ys = ([path[:, idx] for path in paths] for idx in inds)
                data = dict(xs=xs, ys=ys)
            return data, mapping, style

        dim_name = util.dimension_sanitizer(cdim.name)
        cidx = element.get_dimension_index(cdim)
        if not self.static_source:
            paths, cvals = [], []
            for path in element.split(datatype='array'):
                splits = [0]+list(np.where(np.diff(path[:, cidx])!=0)[0]+1)
                for (s1, s2) in zip(splits[:-1], splits[1:]):
                    cvals.append(path[s1, cidx])
                    paths.append(path[s1:s2+1, :2])
            xs, ys = ([path[:, idx] for path in paths] for idx in inds)
            data = dict(xs=xs, ys=ys, **{dim_name: np.array(cvals)})
        cmapper = self._get_colormapper(cdim, element, ranges, style)
        mapping['line_color'] = {'field': dim_name, 'transform': cmapper}
        self._get_hover_data(data, element)
        return data, mapping, style


    def get_batched_data(self, element, ranges=None):
        data = defaultdict(list)

        zorders = self._updated_zorders(element)
        for (key, el), zorder in zip(element.data.items(), zorders):
            self.set_param(**self.lookup_options(el, 'plot').options)
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


class ContourPlot(LegendPlot, PathPlot):

    color_index = param.ClassSelector(default=0, class_=(util.basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    _color_style = 'line_color'

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

        for d in element.vdims:
            dim = util.dimension_sanitizer(d.name)
            if dim not in data:
                if element.interface.isscalar(element, d):
                    data[dim] = element.dimension_values(d, expanded=False)
                else:
                    data[dim] = element.split(datatype='array', dimensions=[d])
            elif isinstance(data[dim], np.ndarray) and data[dim].dtype.kind == 'M':
                data[dim+'_dt_strings'] = [d.pprint_value(v) for v in data[dim]]

        for k, v in self.overlay_dims.items():
            dim = util.dimension_sanitizer(k.name)
            if dim not in data:
                data[dim] = [v for _ in range(len(list(data.values())[0]))]

    def get_data(self, element, ranges, style):
        paths = element.split(datatype='array', dimensions=element.kdims)
        if self.static_source:
            data = dict()
        else:
            inds = (1, 0) if self.invert_axes else (0, 1)
            xs, ys = ([path[:, idx] for path in paths] for idx in inds)
            data = dict(xs=xs, ys=ys)
        mapping = dict(self._mapping)

        if None not in [element.level, self.color_index] and element.vdims:
            cdim = element.vdims[0]
        else:
            cidx = self.color_index+2 if isinstance(self.color_index, int) else self.color_index
            cdim = element.get_dimension(cidx)
        if cdim is None:
            return data, mapping, style

        ncontours = len(paths)
        dim_name = util.dimension_sanitizer(cdim.name)
        if element.level is not None:
            values = np.full(ncontours, float(element.level))
        else:
            values = element.dimension_values(cdim, expanded=False)
        data[dim_name] = values
        factors = list(np.unique(values)) if values.dtype.kind in 'SUO' else None
        cmapper = self._get_colormapper(cdim, element, ranges, style, factors)
        mapping[self._color_style] = {'field': dim_name, 'transform': cmapper}
        self._get_hover_data(data, element)
        if self.show_legend:
            mapping['legend'] = dim_name
        return data, mapping, style


class PolygonPlot(ContourPlot):

    style_opts = ['cmap'] + line_properties + fill_properties
    _plot_methods = dict(single='patches', batched='patches')
    _batched_style_opts = line_properties + fill_properties
    _color_style = 'fill_color'
