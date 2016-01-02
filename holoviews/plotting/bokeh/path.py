import numpy as np

from .element import ElementPlot, line_properties, fill_properties
from .util import get_cmap, map_colors


class PathPlot(ElementPlot):

    style_opts = ['color'] + line_properties
    _plot_method = 'multi_line'
    _mapping = dict(xs='xs', ys='ys')

    def get_data(self, element, ranges=None, empty=False):
        xs = [] if empty else [path[:, 0] for path in element.data]
        ys = [] if empty else [path[:, 1] for path in element.data]
        return dict(xs=xs, ys=ys), self._mapping


class PolygonPlot(PathPlot):

    style_opts = ['color', 'cmap', 'palette'] + line_properties + fill_properties
    _plot_method = 'patches'

    def get_data(self, element, ranges=None, empty=False):
        xs = [] if empty else [path[:, 0] for path in element.data]
        ys = [] if empty else [path[:, 1] for path in element.data]
        data = dict(xs=xs, ys=ys)

        style = self.style[self.cyclic_index]
        cmap = style.get('palette', style.get('cmap', None))
        mapping = dict(self._mapping)
        if cmap and element.level is not None:
            cmap = get_cmap(cmap)
            colors = map_colors(np.array([element.level]), ranges[element.vdims[0].name], cmap)
            mapping['color'] = 'color'
            data['color'] = [] if empty else list(colors)*len(element.data)

        return data, mapping


class GeoMapPlot(PolygonPlot):
    """
    GeoMapPlot draws paths representing the region boundaries in the
    supplied GeoMap. If the GeoMap has an associated value the color of
    regions will be drawn from the supplied cmap, otherwise the supplied
    facecolor will apply.
    """

    aspect = param.Parameter(default='equal', doc="""
        GeoMap elements use an 'equal' aspect ratio by default but
        may be set to an explicit aspect ratio or to 'square'.""")
