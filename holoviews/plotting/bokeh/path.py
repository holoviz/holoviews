from collections import defaultdict

import numpy as np
import param

from bokeh.models import HoverTool

from ...core import util
from ..util import map_colors
from .element import ElementPlot, line_properties, fill_properties
from .util import get_cmap, rgb2hex


class PathPlot(ElementPlot):

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    style_opts = ['color'] + line_properties
    _plot_methods = dict(single='multi_line', batched='multi_line')
    _mapping = dict(xs='xs', ys='ys')

    def get_data(self, element, ranges=None, empty=False):
        xidx, yidx = (1, 0) if self.invert_axes else (0, 1)
        xs = [] if empty else [path[:, xidx] for path in element.data]
        ys = [] if empty else [path[:, yidx] for path in element.data]
        return dict(xs=xs, ys=ys), dict(self._mapping)

    def get_batched_data(self, element, ranges=None, empty=False):
        data = defaultdict(list)
        style = self.style.max_cycles(len(self.ordering))
        for key, el in element.data.items():
            self.overlay_dims = dict(zip(element.kdims, key))
            eldata, elmapping = self.get_data(el, ranges, empty)
            for k, eld in eldata.items():
                data[k].extend(eld)
            zorder = self.get_zorder(element, key, el)
            val = style[zorder].get('color')
            if val:
                elmapping['line_color'] = 'color'
                if isinstance(val, tuple):
                    val = rgb2hex(val)
                data['color'] += [val for _ in range(len(eldata.values()[0]))]
        return data, elmapping


class PolygonPlot(PathPlot):

    style_opts = ['color', 'cmap', 'palette'] + line_properties + fill_properties
    _plot_methods = dict(single='patches', batched='patches')

    def _init_tools(self, element):
        """
        Processes the list of tools to be supplied to the plot.
        """
        tools = self.default_tools + self.tools
        if 'hover' not in tools:
            return tools
        tools.pop(tools.index('hover'))
        if self.batched:
            dims = self.hmap.last.kdims
        else:
            dims = list(self.overlay_dims.keys())
        dims += element.vdims
        tooltips = [(d.pprint_label, '@'+util.dimension_sanitizer(d.name))
                    for d in dims]
        tools.append(HoverTool(tooltips=tooltips))
        return tools


    def get_data(self, element, ranges=None, empty=False):
        xs = [] if empty else [path[:, 0] for path in element.data]
        ys = [] if empty else [path[:, 1] for path in element.data]
        data = dict(xs=ys, ys=xs) if self.invert_axes else dict(xs=xs, ys=ys)

        style = self.style[self.cyclic_index]
        cmap = style.get('palette', style.get('cmap', None))
        mapping = dict(self._mapping)
        if cmap and element.level is not None:
            cmap = get_cmap(cmap)
            colors = map_colors(np.array([element.level]), ranges[element.vdims[0].name], cmap)
            mapping['color'] = 'color'
            data['color'] = [] if empty else list(colors)*len(element.data)
            dim_name = util.dimension_sanitizer(element.vdims[0].name)
        if 'hover' in self.tools+self.default_tools:
            for k, v in self.overlay_dims.items():
                dim = util.dimension_sanitizer(k.name)
                data[dim] = [v for _ in range(len(xs))]
            data[dim_name] = [element.level for _ in range(len(xs))]

        return data, mapping


    def get_batched_data(self, element, ranges=None, empty=False):
        data = defaultdict(list)
        style = self.style.max_cycles(len(self.ordering))
        for key, el in element.data.items():
            self.overlay_dims = dict(zip(element.kdims, key))
            eldata, elmapping = self.get_data(el, ranges, empty)
            for k, eld in eldata.items():
                data[k].extend(eld)
            if 'color' not in eldata:
                zorder = self.get_zorder(element, key, el)
                val = style[zorder].get('color')
                elmapping['color'] = 'color'
                if isinstance(val, tuple):
                    val = rgb2hex(val)
                data['color'] += [val for _ in range(len(eldata['xs']))]

        return data, elmapping
