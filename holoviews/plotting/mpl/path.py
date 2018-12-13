from __future__ import absolute_import, division, unicode_literals

import param
import numpy as np

from matplotlib.collections import PatchCollection, LineCollection

from ...core import util
from ...core.options import abbreviated_exception
from ...element import Polygons
from .element import ColorbarPlot
from .util import polygons_to_path_patches


class PathPlot(ColorbarPlot):

    aspect = param.Parameter(default='square', doc="""
        PathPlots axes usually define single space so aspect of Paths
        follows aspect in data coordinates by default.""")

    color_index = param.ClassSelector(default=None, class_=(util.basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    style_opts = ['alpha', 'color', 'linestyle', 'linewidth', 'visible', 'cmap']

    def get_data(self, element, ranges, style):
        with abbreviated_exception():
            style = self._apply_transforms(element, ranges, style)

        cdim = element.get_dimension(self.color_index)
        if cdim: cidx = element.get_dimension_index(cdim)
        style_mapping = any(True for v in style.values() if isinstance(v, np.ndarray))
        if not (cdim or style_mapping):
            paths = element.split(datatype='array', dimensions=element.kdims)
            if self.invert_axes:
                paths = [p[:, ::-1] for p in paths]
            return (paths,), style, {}
        paths, cvals = [], []
        for path in element.split(datatype='array'):
            length = len(path)
            for (s1, s2) in zip(range(length-1), range(1, length+1)):
                if cdim:
                    cvals.append(path[s1, cidx])
                paths.append(path[s1:s2+1, :2])
        if cdim:
            self._norm_kwargs(element, ranges, style, cdim)
            style['array'] = np.array(cvals)
        if 'c' in style:
            style['array'] = style.pop('c')
        if 'vmin' in style:
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
        if 'colors' in style:
            artist.set_edgecolors(style['colors'])
        if 'facecolors' in style:
            artist.set_facecolors(style['facecolors'])
        if 'linewidth' in style:
            artist.set_linewidths(style['linewidth'])
        return axis_kwargs


class ContourPlot(PathPlot):

    color_index = param.ClassSelector(default=0, class_=(util.basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    def init_artists(self, ax, plot_args, plot_kwargs):
        line_segments = LineCollection(*plot_args, **plot_kwargs)
        ax.add_collection(line_segments)
        return {'artist': line_segments}

    def get_data(self, element, ranges, style):
        if isinstance(element, Polygons):
            color_prop = 'facecolors'
            subpaths = polygons_to_path_patches(element)
            paths = [path for subpath in subpaths for path in subpath]
            if self.invert_axes:
                for p in paths:
                    p._path.vertices = p._path.vertices[:, ::-1]
        else:
            color_prop = 'colors'
            paths = element.split(datatype='array', dimensions=element.kdims)
            if self.invert_axes:
                paths = [p[:, ::-1] for p in paths]

        # Process style transform
        with abbreviated_exception():
            style = self._apply_transforms(element, ranges, style)

        if 'c' in style:
            style['array'] = style.pop('c')
            style['clim'] = style.pop('vmin'), style.pop('vmax')
        elif isinstance(style.get('color'), np.ndarray):
            style[color_prop] = style.pop('color')

        # Process deprecated color_index
        if None not in [element.level, self.color_index]:
            cdim = element.vdims[0]
        elif 'array' not in style:
            cidx = self.color_index+2 if isinstance(self.color_index, int) else self.color_index
            cdim = element.get_dimension(cidx)
        else:
            cdim = None

        if cdim is None:
            return (paths,), style, {}

        if element.level is not None:
            array = np.full(len(paths), element.level)
        else:
            array = element.dimension_values(cdim, expanded=False)
            if len(paths) != len(array):
                # If there are multi-geometries the list of scalar values
                # will not match the list of paths and has to be expanded
                array = np.array([v for v, sps in zip(array, subpaths)
                                  for _ in range(len(sps))])

        if array.dtype.kind not in 'uif':
            array = util.search_indices(array, util.unique_array(array))
        style['array'] = array
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
                  'hatch', 'linestyle', 'joinstyle', 'fill', 'capstyle',
                  'color']

    def init_artists(self, ax, plot_args, plot_kwargs):
        polys = PatchCollection(*plot_args, **plot_kwargs)
        ax.add_collection(polys)
        return {'artist': polys}
