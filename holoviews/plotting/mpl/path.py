from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection
import numpy as np
import param

from ...core.util import match_spec
from .element import ElementPlot, ColorbarPlot


class PathPlot(ElementPlot):

    aspect = param.Parameter(default='equal', doc="""
        PathPlots axes usually define single space so aspect of Paths
        follows aspect in data coordinates by default.""")

    style_opts = ['alpha', 'color', 'linestyle', 'linewidth', 'visible']

    def __init__(self, *args, **params):
        super(PathPlot, self).__init__(*args, **params)

    def initialize_plot(self, ranges=None):
        lines = self.hmap.last
        key = self.keys[-1]
        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = match_spec(lines, ranges)
        style = self.style[self.cyclic_index]
        label = lines.label if self.show_legend else ''
        line_segments = LineCollection(lines.data, label=label,
                                       zorder=self.zorder, **style)
        self.handles['artist'] = line_segments
        self.handles['axis'].add_collection(line_segments)

        return self._finalize_axis(key, ranges=ranges)


    def update_handles(self, axis, element, key, ranges=None):
        artist = self.handles['artist']
        artist.set_paths(element.data)
        visible = self.style[self.cyclic_index].get('visible', True)
        artist.set_visible(visible)



class PolygonPlot(ColorbarPlot):
    """
    PolygonPlot draws the polygon paths in the supplied Polygons
    object. If the Polygon has an associated value the color of
    Polygons will be drawn from the supplied cmap, otherwise the
    supplied facecolor will apply. Facecolor also determines the color
    for non-finite values.
    """

    style_opts = ['alpha', 'cmap', 'facecolor', 'edgecolor', 'linewidth',
                  'hatch', 'linestyle', 'joinstyle', 'fill', 'capstyle']

    def initialize_plot(self, ranges=None):
        element = self.hmap.last
        key = self.keys[-1]
        axis = self.handles['axis']
        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = match_spec(element, ranges)
        collection, polys = self._create_polygons(element, ranges)
        axis.add_collection(collection)
        self.handles['polys'] = polys

        if self.colorbar:
            self._draw_colorbar(collection, element)

        self.handles['artist'] = collection

        return self._finalize_axis(self.keys[-1], ranges=ranges)


    def _create_polygons(self, element, ranges):
        value = element.level
        vdim = element.vdims[0]

        style = self.style[self.cyclic_index]
        polys = []
        for segments in element.data:
            if segments.shape[0]:
                polys.append(Polygon(segments))
        legend = element.label if self.show_legend else ''
        collection = PatchCollection(polys, clim=ranges[vdim.name],
                                     zorder=self.zorder, label=legend, **style)
        if value is not None and np.isfinite(value):
            collection.set_array(np.array([value]*len(polys)))
        return collection, polys


    def update_handles(self, axis, element, key, ranges=None):
        vdim = element.vdims[0]
        collection = self.handles['artist']
        value = element.level

        if any(not np.array_equal(data, poly.get_xy()) for data, poly in
               zip(element.data, self.handles['polys'])):
            collection.remove()
            collection, polys = self._create_polygons(element, ranges)
            self.handles['polys'] = polys
            self.handles['artist'] = collection
            axis.add_collection(collection)
        elif value is not None and np.isfinite(value):
            collection.set_array(np.array([value]*len(element.data)))
            collection.set_clim(ranges[vdim.name])
