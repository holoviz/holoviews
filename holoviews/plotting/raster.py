import copy
from itertools import product

import numpy as np
from matplotlib import pyplot as plt

import param

from ..core import CompositeOverlay, Element
from ..element.raster import HeatMap, Matrix, Raster
from .plot import Plot, OverlayPlot, GridPlot


class MatrixPlot(Plot):

    normalize_individually = param.Boolean(default=False)

    show_values = param.Boolean(default=True, doc="""
        Whether to annotate the values when displaying a HeatMap.""")

    style_opts = ['alpha', 'cmap', 'interpolation', 'visible',
                  'filterrad', 'origin', 'clims']

    def __call__(self, axis=None, lbrt=None):

        self.ax = self._init_axis(axis)
        view = self._map.last

        (l, b, r, t) = (0, 0, 1, 1) if isinstance(view, HeatMap)\
            else self._map.last.lbrt
        xticks, yticks = self._compute_ticks(view)

        opts = self.settings.closest(view, 'style')[self.cyclic_index]
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

        im = self.ax.imshow(data, extent=[l, r, b, t], zorder=self.zorder, **opts)
        if clims is None:
            val_dim = [d.name for d in view.value_dimensions][0]
            clims = view.range(val_dim) if self.normalize_individually else self._map.range(val_dim)
        im.set_clim(clims)
        self.handles['im'] = im

        if isinstance(view, HeatMap):
            self.ax.set_aspect(float(r - l)/(t-b))
            self.handles['annotations'] = {}
            if self.show_values:
                self._annotate_values(view)

        return self._finalize_axis(self._keys[-1], lbrt=(l, b, r, t),
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
                annotation = self.ax.annotate(text, xy=plot_coord,
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



    def update_handles(self, view, key, lbrt=None):
        im = self.handles.get('im', None)
        im.set_data(view.data)

        if isinstance(view, HeatMap) and self.show_values:
           self._annotate_values(view)

        if self.normalize_individually:
            im.set_clim(view.range)



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

    def __init__(self, grid, **params):
        self.layout = params.pop('layout', None)
        self.grid = grid.clone()
        for k, vmap in grid.data.items():
            self.grid[k] = self._check_map(self.grid[k])
        Plot.__init__(self, **params)
        self._keys = self.grid.all_keys
        xkeys, ykeys = zip(*self.grid.data.keys())
        self._xkeys = sorted(set(xkeys))
        self._ykeys = sorted(set(ykeys))


    def __call__(self, axis=None):
        width, height, b_w, b_h, widths, heights = self._compute_borders()
        self.ax = self._init_axis(axis)

        self.handles['projs'] = []
        key = self._keys[-1]
        x, y = b_w, b_h
        xticks, yticks = [], []
        for xidx, xkey in enumerate(self._xkeys):
            w = widths[xidx]
            for yidx, ykey in enumerate(self._ykeys):
                h = heights[yidx]
                vmap = self.grid.get((xkey, ykey), None)
                pane = vmap.get(key, None) if vmap else None
                if pane:
                    if issubclass(vmap.type, CompositeOverlay): pane = pane.last
                    data = pane.data if pane else None
                else:
                    pane = vmap.last.last if issubclass(vmap.type, CompositeOverlay) else vmap.last
                    data = pane.data
                opts = self.settings(pane, 'style').settings[self.cyclic_index]
                plot = self.ax.imshow(data, extent=(x,x+w, y, y+h), **opts)
                if key not in vmap:
                    plot.set_visible(False)
                self.handles['projs'].append(plot)
                y += h + b_h
                if xidx == 0:
                    yticks.append(y-b_h-h/2.)
            y = b_h
            x += w + b_w
            xticks.append(x-b_w-w/2.)

        return self._finalize_axis(key, lbrt=(0, 0, width, height),
                                   title=self._format_title(key),
                                   xticks=(xticks, self._process_ticklabels(self._xkeys)),
                                   yticks=(yticks, self._process_ticklabels(self._ykeys)),
                                   xlabel=str(self.grid.get_dimension(0)),
                                   ylabel=str(self.grid.get_dimension(1)))


    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        key = self._keys[n]
        grid_values = self.grid.values()
        for i, plot in enumerate(self.handles['projs']):
            view = grid_values[i].get(key, None)
            if view:
                plot.set_visible(True)
                data = view.values()[0].data if isinstance(view, CompositeOverlay) else view.data
                plot.set_data(data)
            else:
                plot.set_visible(False)

        self._finalize_axis(key, title=self._format_title(key))


    def _compute_borders(self):
        width_lbrts = [self.grid[xkey, :].lbrt for xkey in self._xkeys]
        height_lbrts = [self.grid[:, ykey].lbrt for ykey in self._ykeys]
        widths = [lbrt[2]-lbrt[0] for lbrt in width_lbrts]
        heights = [lbrt[3]-lbrt[1] for lbrt in height_lbrts]
        width, height = np.sum(widths), np.sum(heights)
        border_width = (width/10.)/(len(widths)+1)
        border_height = (height/10.)/(len(heights)+1)
        width += width/10.
        height += height/10.

        return width, height, border_width, border_height, widths, heights


    def __len__(self):
        return max([len(self._keys), 1])


Plot.defaults.update({Raster: MatrixPlot,
                      HeatMap: MatrixPlot,
                      Matrix: MatrixPlot})

