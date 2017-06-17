from collections import defaultdict

import param
import numpy as np

from bokeh.models import HoverTool, FactorRange

from ...core import util
from .element import ElementPlot, ColorbarPlot, line_properties, fill_properties
from .util import expand_batched_style


class PathPlot(ElementPlot):

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    style_opts = line_properties
    _plot_methods = dict(single='multi_line', batched='multi_line')
    _mapping = dict(xs='xs', ys='ys')
    _batched_style_opts = line_properties

    def _hover_opts(self, element):
        if self.batched:
            dims = list(self.hmap.last.kdims)
        else:
            dims = list(self.overlay_dims.keys())
        return dims, {}

    def get_data(self, element, ranges=None, empty=False):
        xidx, yidx = (1, 0) if self.invert_axes else (0, 1)
        xs = [] if empty else [path[:, xidx] for path in element.data]
        ys = [] if empty else [path[:, yidx] for path in element.data]
        return dict(xs=xs, ys=ys), dict(self._mapping)


    def _categorize_data(self, data, cols, dims):
        """
        Transforms non-string or integer types in datasource if the
        axis to be plotted on is categorical. Accepts the column data
        source data, the columns corresponding to the axes and the
        dimensions for each axis, changing the data inplace.
        """
        if self.invert_axes:
            cols = cols[::-1]
            dims = dims[:2][::-1]
        ranges = [self.handles['%s_range' % ax] for ax in 'xy']
        for i, col in enumerate(cols):
            column = data[col]
            if (isinstance(ranges[i], FactorRange) and
                (isinstance(column, list) or column.dtype.kind not in 'SU')):
                data[col] = [[dims[i].pprint_value(v) for v in vals] for vals in column]


    def get_batched_data(self, element, ranges=None, empty=False):
        data = defaultdict(list)

        zorders = self._updated_zorders(element)
        styles = self.lookup_options(element.last, 'style')
        styles = styles.max_cycles(len(self.ordering))

        for (key, el), zorder in zip(element.data.items(), zorders):
            self.set_param(**self.lookup_options(el, 'plot').options)
            self.overlay_dims = dict(zip(element.kdims, key))
            eldata, elmapping = self.get_data(el, ranges, empty)
            for k, eld in eldata.items():
                data[k].extend(eld)

            # Apply static styles
            nvals = len(list(eldata.values())[0])
            style = styles[zorder]
            sdata, smapping = expand_batched_style(style, self._batched_style_opts,
                                                   elmapping, nvals)
            elmapping.update({k: v for k, v in smapping.items() if k not in elmapping})
            for k, v in sdata.items():
                data[k].extend(list(v))

        return data, elmapping


class ContourPlot(ColorbarPlot, PathPlot):

    style_opts = line_properties + ['cmap']

    def get_data(self, element, ranges=None, empty=False):
        data, mapping = super(ContourPlot, self).get_data(element, ranges, empty)
        ncontours = len(list(data.values())[0])
        style = self.style[self.cyclic_index]
        if element.vdims and element.level is not None and 'cmap' in style:
            cdim = element.vdims[0]
            dim_name = util.dimension_sanitizer(cdim.name)
            cmapper = self._get_colormapper(cdim, element, ranges, style)
            data[dim_name] = [] if empty else np.full(ncontours, float(element.level))
            mapping['line_color'] = {'field': dim_name,
                                     'transform': cmapper}
        return data, mapping


class PolygonPlot(ColorbarPlot, PathPlot):

    style_opts = ['cmap', 'palette'] + line_properties + fill_properties
    _plot_methods = dict(single='patches', batched='patches')
    _style_opts = ['color', 'cmap', 'palette'] + line_properties + fill_properties
    _batched_style_opts = line_properties + fill_properties

    def _hover_opts(self, element):
        if self.batched:
            dims = list(self.hmap.last.kdims)
        else:
            dims = list(self.overlay_dims.keys())
        dims += element.vdims
        return dims, {}

    def get_data(self, element, ranges=None, empty=False):
        xs = [] if empty else [path[:, 0] for path in element.data]
        ys = [] if empty else [path[:, 1] for path in element.data]
        data = dict(xs=ys, ys=xs) if self.invert_axes else dict(xs=xs, ys=ys)

        style = self.style[self.cyclic_index]
        mapping = dict(self._mapping)

        if element.vdims and element.level is not None:
            cdim = element.vdims[0]
            dim_name = util.dimension_sanitizer(cdim.name)
            cmapper = self._get_colormapper(cdim, element, ranges, style)
            data[dim_name] = [] if empty else [element.level for _ in range(len(xs))]
            mapping['fill_color'] = {'field': dim_name,
                                     'transform': cmapper}

        if any(isinstance(t, HoverTool) for t in self.state.tools):
            dim_name = util.dimension_sanitizer(element.vdims[0].name)
            for k, v in self.overlay_dims.items():
                dim = util.dimension_sanitizer(k.name)
                data[dim] = [v for _ in range(len(xs))]
            data[dim_name] = [element.level for _ in range(len(xs))]

        return data, mapping
