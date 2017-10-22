from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection, LineCollection
import numpy as np
import param

from ...core import util
from .element import ElementPlot, ColorbarPlot


class PathPlot(ColorbarPlot):

    aspect = param.Parameter(default='square', doc="""
        PathPlots axes usually define single space so aspect of Paths
        follows aspect in data coordinates by default.""")
    
    color_index = param.ClassSelector(default=None, class_=(util.basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    style_opts = ['alpha', 'color', 'linestyle', 'linewidth', 'visible', 'cmap']

    def _finalize_artist(self, element):
        if self.colorbar:
            self._draw_colorbar(element.get_dimension(self.color_index))

    def get_data(self, element, ranges, style):
        cdim = element.get_dimension(self.color_index)
        if cdim: cidx = element.get_dimension_index(cdim)
        if not cdim:
            paths = element.split(datatype='array', dimensions=element.kdims)
            if self.invert_axes:
                paths = [p[:, ::-1] for p in paths]
            return (paths,), style, {}
        paths, cvals = [], []
        for path in element.split(datatype='array'):
            splits = [0]+list(np.where(np.diff(path[:, cidx])!=0)[0]+1)
            for (s1, s2) in zip(splits[:-1], splits[1:]):
                cvals.append(path[s1, cidx])
                paths.append(path[s1:s2+1, :2])
        self._norm_kwargs(element, ranges, style, cdim)
        style['array'] = np.array(cvals)
        style['clim'] = style.pop('vmin', None), style.pop('vmax', None)
        return (paths,), style, {}

    def init_artists(self, ax, plot_args, plot_kwargs):
        line_segments = LineCollection(*plot_args, **plot_kwargs)
        ax.add_collection(line_segments)
        return {'artist': line_segments}

    def update_handles(self, key, axis, element, ranges, style):
        artist = self.handles['artist']
        data, style, axis_kwargs = self.get_data(element, ranges, style)
        artist.set_paths(data[0])
        if 'array' in style:
            artist.set_array(style['array'])
            artist.set_clim(style['clim'])
        if 'norm' in style:
            artist.set_norm(style['norm'])
        artist.set_visible(style.get('visible', True))
        return axis_kwargs


class ContourPlot(PathPlot):

    color_index = param.ClassSelector(default=0, class_=(util.basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    def _finalize_artist(self, element):
        if self.colorbar:
            cidx = self.color_index+2 if isinstance(self.color_index, int) else self.color_index
            cdim = element.get_dimension(cidx)
            self._draw_colorbar(cdim)

    def get_data(self, element, ranges, style):
        if None not in [element.level, self.color_index]:
            cdim = element.vdims[0]
        else:
            cidx = self.color_index+2 if isinstance(self.color_index, int) else self.color_index
            cdim = element.get_dimension(cidx)
        paths = element.split(datatype='array', dimensions=element.kdims)
        if self.invert_axes:
            paths = [p[:, ::-1] for p in paths]

        if cdim is None:
            return (paths,), style, {}

        if element.level is not None:
            style['array'] = np.full(len(paths), element.level)
        else:
            style['array'] = element.dimension_values(cdim, expanded=False)
        self._norm_kwargs(element, ranges, style, cdim)
        style['clim'] = style.pop('vmin'), style.pop('vmax')
        return (paths,), style, {}

    
class PolygonPlot(ContourPlot):
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

    def init_artists(self, ax, plot_args, plot_kwargs):
        polys = PolyCollection(*plot_args, **plot_kwargs)
        ax.add_collection(polys)
        return {'artist': polys}
