from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection
import numpy as np
import param

from ..core.options import Store
from ..core.util import match_spec
from ..element import Path, Box, Bounds, Ellipse, Polygons, Contours
from .element import ElementPlot


class PathPlot(ElementPlot):

    style_opts = ['alpha', 'color', 'linestyle', 'linewidth', 'visible']

    def __init__(self, *args, **params):
        self.aspect = 'equal'
        super(PathPlot, self).__init__(*args, **params)

    def __call__(self, ranges=None):
        lines = self.map.last
        key = self.keys[-1]
        ranges = self.compute_ranges(self.map, key, ranges)
        ranges = match_spec(lines, ranges)
        style = self.style[self.cyclic_index]
        line_segments = LineCollection(lines.data, zorder=self.zorder, **style)
        self.handles['line_segments'] = line_segments
        self.handles['axis'].add_collection(line_segments)

        return self._finalize_axis(key, ranges=ranges)


    def update_handles(self, axis, view, key, ranges=None):
        self.handles['line_segments'].set_paths(view.data)



class PolygonPlot(ElementPlot):
    """
    PolygonPlot draws the polygon paths in the supplied Polygons
    object. If the Polygon has an associated value the color of
    Polygons will be drawn from the supplied cmap, otherwise the
    supplied facecolor will apply. Facecolor also determines the color
    for non-finite values.
    """

    colorbar = param.Boolean(default=False, doc="""
        Whether to draw a colorbar.""")

    style_opts = ['alpha', 'cmap', 'facecolor', 'edgecolor', 'linewidth',
                  'hatch', 'linestyle', 'joinstyle', 'fill', 'capstyle']

    def __call__(self, ranges=None):
        element = self.map.last
        key = self.keys[-1]
        axis = self.handles['axis']
        ranges = self.compute_ranges(self.map, key, ranges)
        ranges = match_spec(element, ranges)
        collection, polys = self._create_polygons(element, ranges)
        axis.add_collection(collection)
        self.handles['polys'] = polys

        if self.colorbar:
            self._draw_colorbar(collection)

        self.handles['polygons'] = collection

        return self._finalize_axis(self.keys[-1], ranges=ranges)


    def _create_polygons(self, element, ranges):
        value = element.level
        vdim = element.value_dimensions[0]

        style = self.style[self.cyclic_index]
        polys = []
        for segments in element.data:
            if segments.shape[0]:
                polys.append(Polygon(segments))
        collection = PatchCollection(polys, clim=ranges[vdim.name],
                                     zorder=self.zorder, **style)
        if value is not None and np.isfinite(value):
            collection.set_array(np.array([value]*len(polys)))
        return collection, polys


    def update_handles(self, axis, element, key, ranges=None):
        vdim = element.value_dimensions[0]
        collection = self.handles['polygons']
        value = element.level

        if any(not np.array_equal(data, poly.get_xy()) for data, poly in
               zip(element.data, self.handles['polys'])):
            collection.remove()
            collection, polys = self._create_polygons(element, ranges)
            self.handles['polys'] = polys
            self.handles['polygons'] = collection
            axis.add_collection(collection)
        elif value is not None and np.isfinite(value):
            collection.set_array(np.array([value]*len(element.data)))
            collection.set_clim(ranges[vdim.name])
        if self.colorbar:
            self._draw_colorbar(collection)


Store.registry.update({
    Contours: PathPlot,
    Path:     PathPlot,
    Box:      PathPlot,
    Bounds:   PathPlot,
    Ellipse:  PathPlot,
    Polygons: PolygonPlot})
