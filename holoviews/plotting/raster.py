import copy
from itertools import product

import numpy as np
from matplotlib import pyplot as plt

import param

from ..core.options import Store
from ..core import CompositeOverlay
from ..core import traversal
from ..element.raster import HeatMap, Matrix, Raster, RGB
from .element import ElementPlot, OverlayPlot
from .plot import Plot, GridPlot


class MatrixPlot(ElementPlot):

    normalize_individually = param.Boolean(default=False)

    show_values = param.Boolean(default=True, doc="""
        Whether to annotate the values when displaying a HeatMap.""")

    style_opts = ['alpha', 'cmap', 'interpolation', 'visible',
                  'filterrad', 'origin', 'clims']

    def __call__(self, ranges=None):
        view = self.map.last
        axis = self.handles['axis']

        ranges = self.compute_ranges(self.map, self.map.last_key, ranges)
        ranges = self.match_range(view, ranges)

        (l, b, r, t) = (0, 0, 1, 1) if isinstance(view, HeatMap)\
            else self.map.last.extents
        xticks, yticks = self._compute_ticks(view)

        opts = Store.lookup_options(view, 'style')[self.cyclic_index]
        data = view.data
        clims = opts.pop('clims', None)
        if view.depth != 1:
            opts.pop('cmap', None)
        elif isinstance(view, HeatMap):
            data = view.data
            data = np.ma.array(data, mask=np.isnan(data))
            cmap_name = opts.pop('cmap', None)
            cmap = copy.copy(plt.cm.get_cmap('gray' if cmap_name is None else cmap_name))
            cmap.set_bad('w', 1.)
            opts['cmap'] = cmap

        im = axis.imshow(data, extent=[l, r, b, t], zorder=self.zorder, **opts)
        if clims is None:
            val_dim = [d.name for d in view.value_dimensions][0]
            clims = ranges.get(val_dim)
        im.set_clim(clims)
        self.handles['im'] = im

        if isinstance(view, HeatMap):
            self.handles['axis'].set_aspect(float(r - l)/(t-b))
            self.handles['annotations'] = {}
            if self.show_values:
                self._annotate_values(view)

        return self._finalize_axis(self.map.last_key, ranges=ranges,
                                   xticks=xticks, yticks=yticks)


    def _compute_ticks(self, view):
        if isinstance(view, HeatMap):
            dim1_keys, dim2_keys = view.dense_keys()
            num_x, num_y = len(dim1_keys), len(dim2_keys)
            xstep, ystep = 1.0/num_x, 1.0/num_y
            xpos = np.linspace(xstep/2., 1.0-xstep/2., num_x)
            ypos = np.linspace(ystep/2., 1.0-ystep/2., num_y)
            return (xpos, dim1_keys), (ypos, dim2_keys)
        else:
            return None, None


    def _annotate_values(self, view):
        axis = self.handles['axis']
        dim1_keys, dim2_keys = view.dense_keys()
        num_x, num_y = len(dim1_keys), len(dim2_keys)
        xstep, ystep = 1.0/num_x, 1.0/num_y
        xpos = np.linspace(xstep/2., 1.0-xstep/2., num_x)
        ypos = np.linspace(ystep/2., 1.0-ystep/2., num_y)
        coords = product(dim1_keys, dim2_keys)
        plot_coords = product(xpos, ypos)
        for plot_coord, coord in zip(plot_coords, coords):
            text = round(view._data.get(coord, np.NaN), 3)
            if plot_coord not in self.handles['annotations']:
                annotation = axis.annotate(text, xy=plot_coord,
                                           xycoords='axes fraction',
                                           horizontalalignment='center',
                                           verticalalignment='center')
                self.handles['annotations'][plot_coord] = annotation
            else:
                self.handles['annotations'][plot_coord].set_text(text)
        old_coords = set(self.handles['annotations'].keys()) - set(product(xpos, ypos))
        for plot_coord in old_coords:
            annotation = self.handles['annotations'].pop(plot_coord)
            annotation.remove()


    def update_handles(self, axis, view, key, ranges=None):
        im = self.handles.get('im', None)
        im.set_data(view.data)

        if isinstance(view, HeatMap) and self.show_values:
           self._annotate_values(view)

        val_dim = [d.name for d in view.value_dimensions][0]
        im.set_clim(ranges.get(val_dim))



class MatrixGridPlot(GridPlot, OverlayPlot):
    """
    MatrixGridPlot evenly spaces out plots of individual projections on
    a grid, even when they differ in size. Since this class uses a single
    axis to generate all the individual plots it is much faster than the
    equivalent using subplots.
    """

    border = param.Number(default=10, doc="""
        Aggregate border as a fraction of total plot size.""")

    show_frame = param.Boolean(default=False)

    show_title = param.Boolean(default=True)

    style_opts = ['alpha', 'cmap', 'interpolation', 'visible',
                  'filterrad', 'origin']

    def __init__(self, layout, keys=None, dimensions=None, ranges=None, **params):
        if not keys or not dimensions:
            dimensions, keys = traversal.unique_dimkeys(layout)
        Plot.__init__(self, dimensions=dimensions, keys=keys, **params)
        self.cyclic_index = 0
        self.zorder = 0
        self.overlaid = False
        self.map = {}
        xkeys, ykeys = zip(*layout.data.keys())
        self._xkeys = sorted(set(xkeys))
        self._ykeys = sorted(set(ykeys))
        self.rows, self.cols = layout.shape
        _, _, self.layout = self._create_subplots(layout, ranges, create_axis=False)


    def get_extents(self, view, ranges):
        width, height, _, _, _, _ = self._compute_borders()
        return (0, 0, width, height)


    def _get_frame(self, key):
        return GridPlot._get_frame(self, key)


    def __call__(self, ranges=None):
        width, height, b_w, b_h, widths, heights = self._compute_borders()

        key = self.keys[-1]
        ranges = self.compute_ranges(self.layout, key, ranges)
        self.handles['projs'] = []
        x, y = b_w, b_h
        xticks, yticks = [], []
        for xidx, xkey in enumerate(self._xkeys):
            w = widths[xidx]
            for yidx, ykey in enumerate(self._ykeys):
                h = heights[yidx]
                vmap = self.layout.get((xkey, ykey), None)
                pane = vmap.get(key, None) if vmap else None
                if pane:
                    if issubclass(vmap.type, CompositeOverlay): pane = pane.last
                    data = pane.data if pane else None
                else:
                    pane = vmap.last.last if issubclass(vmap.type, CompositeOverlay) else vmap.last
                    data = pane.data
                ranges = self.compute_ranges(vmap, key, ranges)
                opts = Store.lookup_options(pane, 'style')[self.cyclic_index]
                plot = self.handles['axis'].imshow(data, extent=(x,x+w, y, y+h), **opts)
                valrange = self.match_range(pane, ranges)[pane.value_dimensions[0].name]
                plot.set_clim(valrange)
                if key not in vmap:
                    plot.set_visible(False)
                self.handles['projs'].append(plot)
                y += h + b_h
                if xidx == 0:
                    yticks.append(y-b_h-h/2.)
            y = b_h
            x += w + b_w
            xticks.append(x-b_w-w/2.)


        return self._finalize_axis(key, ranges=ranges,
                                   title=self._format_title(key),
                                   xticks=(xticks, self._process_ticklabels(self._xkeys)),
                                   yticks=(yticks, self._process_ticklabels(self._ykeys)),
                                   xlabel=str(self.layout.get_dimension(0)),
                                   ylabel=str(self.layout.get_dimension(1)))


    def update_frame(self, key, ranges=None):
        grid_values = self.layout.values()
        ranges = self.compute_ranges(self.layout, key, ranges)
        for i, plot in enumerate(self.handles['projs']):
            view = grid_values[i].get(key, None)
            if view:
                plot.set_visible(True)
                data = view.values()[0].data if isinstance(view, CompositeOverlay) else view.data
                plot.set_data(data)
            else:
                plot.set_visible(False)

        self._finalize_axis(key, ranges=ranges, title=self._format_title(key))


    def _compute_borders(self):
        width_extents = [self.layout[xkey, :].extents for xkey in self._xkeys]
        height_extents = [self.layout[:, ykey].extents for ykey in self._ykeys]
        widths = [extent[2]-extent[0] for extent in width_extents]
        heights = [extent[3]-extent[1] for extent in height_extents]
        width, height = np.sum(widths), np.sum(heights)
        border_width = (width/10.)/(len(widths)+1)
        border_height = (height/10.)/(len(heights)+1)
        width += width/10.
        height += height/10.

        return width, height, border_width, border_height, widths, heights


    def __len__(self):
        return max([len(self.keys), 1])


Store.defaults.update({Raster: MatrixPlot,
                       HeatMap: MatrixPlot,
                       Matrix: MatrixPlot,
                       RGB: MatrixPlot})

