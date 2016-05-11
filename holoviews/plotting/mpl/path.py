from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection
import numpy as np
import param

from .element import ElementPlot, ColorbarPlot


class PathPlot(ElementPlot):

    aspect = param.Parameter(default='equal', doc="""
        PathPlots axes usually define single space so aspect of Paths
        follows aspect in data coordinates by default.""")

    style_opts = ['alpha', 'color', 'linestyle', 'linewidth', 'visible']

    def get_data(self, element, ranges, style):
        return (element.data,), style, {}

    def init_artists(self, ax, plot_args, plot_kwargs):
        line_segments = LineCollection(*plot_args, **plot_kwargs)
        ax.add_collection(line_segments)
        return {'artist': line_segments}

    def update_handles(self, key, axis, element, ranges, style):
        artist = self.handles['artist']
        data, style, axis_kwargs = self.get_data(element, ranges, style)
        artist.set_paths(data[0])
        artist.set_visible(style.get('visible', True))
        return axis_kwargs


class PolygonPlot(ColorbarPlot):
    """
    PolygonPlot draws the polygon paths in the supplied Polygons
    object. If the Polygon has an associated value the color of
    Polygons will be drawn from the supplied cmap, otherwise the
    supplied facecolor will apply. Facecolor also determines the color
    for non-finite values.
    """

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    style_opts = ['alpha', 'cmap', 'facecolor', 'edgecolor', 'linewidth',
                  'hatch', 'linestyle', 'joinstyle', 'fill', 'capstyle']


    def get_data(self, element, ranges, style):
        value = element.level
        vdim = element.vdims[0]
        polys = []
        for segments in element.data:
            if segments.shape[0]:
                polys.append(Polygon(segments))

        if value is not None and np.isfinite(value):
            self._norm_kwargs(element, ranges, style, vdim)
            style['clim'] = style.pop('vmin'), style.pop('vmax')
            style['array'] = np.array([value]*len(polys))
        return (polys,), style, {}

    def init_artists(self, ax, plot_args, plot_kwargs):
        collection = PatchCollection(*plot_args, **plot_kwargs)
        ax.add_collection(collection)
        if self.colorbar:
            self._draw_colorbar(collection, self.current_frame)
        return {'artist': collection, 'polys': plot_args[0]}


    def update_handles(self, key, axis, element, ranges, style):
        value = element.level
        vdim = element.vdims[0]
        collection = self.handles['artist']
        if any(not np.array_equal(data, poly.get_xy()) for data, poly in
               zip(element.data, self.handles['polys'])):
            return super(PolygonPlot, self).update_handles(key, axis, element, ranges, style)
        elif value is not None and np.isfinite(value):
            self._norm_kwargs(element, ranges, style, vdim)
            collection.set_array(np.array([value]*len(element.data)))
            collection.set_clim((style['vmin'], style['vmax']))
            if 'norm' in style:
                collection.norm = style['norm']
