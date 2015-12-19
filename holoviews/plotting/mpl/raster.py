import copy
from itertools import product

import numpy as np
from matplotlib import pyplot as plt

import param

from ...core import CompositeOverlay, Element
from ...core import traversal
from ...core.util import match_spec, max_range
from ...element.raster import HeatMap, Image, Raster, RGB
from .element import ColorbarPlot, OverlayPlot
from .plot import MPLPlot, GridPlot


class RasterPlot(ColorbarPlot):

    aspect = param.Parameter(default='equal', doc="""
        Raster elements respect the aspect ratio of the
        Images by default but may be set to an explicit
        aspect ratio or to 'square'.""")

    colorbar = param.Boolean(default=False, doc="""
        Whether to add a colorbar to the plot.""")

    situate_axes = param.Boolean(default=False, doc="""
        Whether to situate the image relative to other plots. """)

    show_values = param.Boolean(default=False, doc="""
        Whether to annotate each pixel with its value.""")

    symmetric = param.Boolean(default=False, doc="""
        Whether to make the colormap symmetric around zero.""")

    style_opts = ['alpha', 'cmap', 'interpolation', 'visible',
                  'filterrad', 'clims', 'norm']


    def __init__(self, *args, **kwargs):
        super(RasterPlot, self).__init__(*args, **kwargs)
        if self.hmap.type == Raster:
            self.invert_yaxis = not self.invert_yaxis


    def get_extents(self, element, ranges):
        extents = super(RasterPlot, self).get_extents(element, ranges)
        if self.situate_axes:
            return extents
        else:
            if isinstance(element, Image):
                return element.bounds.lbrt()
            else:
                return element.extents


    def initialize_plot(self, ranges=None):
        element = self.hmap.last
        axis = self.handles['axis']

        ranges = self.compute_ranges(self.hmap, self.keys[-1], ranges)
        ranges = match_spec(element, ranges)

        xticks, yticks = self._compute_ticks(element, ranges)

        opts = self.style[self.cyclic_index]
        if element.depth != 1:
            opts.pop('cmap', None)

        data = element.data
        if isinstance(element, Image):
            l, b, r, t = element.bounds.lbrt()
        else:
            l, b, r, t = element.extents
        if self.invert_yaxis and type(element) is Raster:
            b, t = t, b

        if isinstance(element, RGB):
            data = element.rgb.data
        elif isinstance(element, HeatMap):
            data = element.raster
            data = np.ma.array(data, mask=np.logical_not(np.isfinite(data)))
            cmap_name = opts.pop('cmap', None)
            cmap = copy.copy(plt.cm.get_cmap('gray' if cmap_name is None else cmap_name))
            cmap.set_bad('w', 1.)
            opts['cmap'] = cmap

        clim, norm, opts = self._norm_kwargs(element, ranges, opts)
        im = axis.imshow(data, extent=[l, r, b, t], zorder=self.zorder,
                         clim=clim, norm=norm, **opts)
        self.handles['artist'] = im

        if isinstance(element, HeatMap):
            self.handles['axis'].set_aspect(float(r - l)/(t-b))
            self.handles['annotations'] = {}

            if self.show_values:
                self._annotate_values(element)

        return self._finalize_axis(self.keys[-1], ranges=ranges,
                                   xticks=xticks, yticks=yticks)


    def _compute_ticks(self, element, ranges):
        if isinstance(element, HeatMap):
            xdim, ydim = element.kdims
            dim1_keys, dim2_keys = [np.unique(element.dimension_values(i))
                                    for i in range(2)]
            num_x, num_y = len(dim1_keys), len(dim2_keys)
            x0, y0, x1, y1 = element.extents
            xstep, ystep = ((x1-x0)/num_x, (y1-y0)/num_y)
            xpos = np.linspace(x0+xstep/2., x1-xstep/2., num_x)
            ypos = np.linspace(y0+ystep/2., y1-ystep/2., num_y)
            xlabels = [xdim.pprint_value(k) for k in dim1_keys]
            ylabels = [ydim.pprint_value(k) for k in dim2_keys]
            return (xpos, xlabels), (ypos, ylabels)
        else:
            return None, None


    def _annotate_values(self, element):
        axis = self.handles['axis']
        val_dim = element.vdims[0]
        d1keys, d2keys = element.dense_keys()
        vals = np.rot90(element.raster, 3).flatten()
        d1uniq, d2uniq = [np.unique(element.dimension_values(i)) for i in range(2)]
        num_x, num_y = len(d1uniq), len(d2uniq)
        xstep, ystep = 1.0/num_x, 1.0/num_y
        xpos = np.linspace(xstep/2., 1.0-xstep/2., num_x)
        ypos = np.linspace(ystep/2., 1.0-ystep/2., num_y)
        plot_coords = product(xpos, ypos)
        for plot_coord, v in zip(plot_coords, vals):
            text = val_dim.pprint_value(v)
            text = '' if v is np.nan else text
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


    def update_handles(self, axis, element, key, ranges=None):
        im = self.handles.get('artist', None)
        data = np.ma.array(element.data,
                           mask=np.logical_not(np.isfinite(element.data)))
        im.set_data(data)

        if isinstance(element, HeatMap) and self.show_values:
           self._annotate_values(element)

        xdim, ydim = element.kdims
        if isinstance(element, Image):
            l, b, r, t = element.bounds.lbrt()
        else:
            l, b, r, t = element.extents
            if type(element) == Raster:
                b, t = t, b

        opts = self.style[self.cyclic_index]

        clim, norm, opts = self._norm_kwargs(element, ranges, opts)
        im.set_clim(clim)
        if norm:
            im.norm = norm
        im.set_extent((l, r, b, t))
        xticks, yticks = self._compute_ticks(element, ranges)
        return {'xticks': xticks, 'yticks': yticks}


class QuadMeshPlot(ColorbarPlot):

    symmetric = param.Boolean(default=False, doc="""
        Whether to make the colormap symmetric around zero.""")

    style_opts = ['alpha', 'cmap', 'clim', 'edgecolors', 'norm', 'shading',
                  'linestyles', 'linewidths', 'hatch', 'visible']

    def initialize_plot(self, ranges=None):
        key = self.hmap.keys()[-1]
        element = self.hmap.last
        axis = self.handles['axis']

        ranges = self.compute_ranges(self.hmap, self.keys[-1], ranges)
        ranges = match_spec(element, ranges)
        self._init_cmesh(axis, element, ranges)

        return self._finalize_axis(key, ranges)

    def _init_cmesh(self, axis, element, ranges):
        opts = self.style[self.cyclic_index]
        if 'cmesh' in self.handles:
            self.handles['cmesh'].remove()
        clims = opts.get('clim', ranges.get(element.get_dimension(2).name))
        data = np.ma.array(element.data[2],
                           mask=np.logical_not(np.isfinite(element.data[2])))
        cmesh_data = list(element.data[:2]) + [data]
        clim, norm, opts = self._norm_kwargs(element, ranges, opts)
        self.handles['artist'] = axis.pcolormesh(*cmesh_data, zorder=self.zorder,
                                                 vmin=clim[0], vmax=clim[1], norm=norm,
                                                 **opts)
        self.handles['locs'] = np.concatenate(element.data[:2])


    def update_handles(self, axis, element, key, ranges=None):
        cmesh = self.handles['artist']
        opts = self.style[self.cyclic_index]
        locs = np.concatenate(element.data[:2])
        if (locs != self.handles['locs']).any():
            self._init_cmesh(axis, element, ranges)
        else:
            mask_array = np.logical_not(np.isfinite(element.data[2]))
            data = np.ma.array(element.data[2], mask=mask_array)
            cmesh.set_array(data.ravel())
            clim, norm, opts = self._norm_kwargs(element, ranges, opts)
            cmesh.set_clim(clim)
            if norm:
                cmesh.norm = norm


class RasterGridPlot(GridPlot, OverlayPlot):
    """
    RasterGridPlot evenly spaces out plots of individual projections on
    a grid, even when they differ in size. Since this class uses a single
    axis to generate all the individual plots it is much faster than the
    equivalent using subplots.
    """

    # Parameters inherited from OverlayPlot that are not part of the
    # GridPlot interface. Some of these may be enabled in future in
    # conjunction with GridPlot.

    apply_extents = param.Parameter(precedence=-1)
    apply_ranges = param.Parameter(precedence=-1)
    apply_ticks = param.Parameter(precedence=-1)
    bgcolor = param.Parameter(precedence=-1)
    invert_axes = param.Parameter(precedence=-1)
    invert_xaxis = param.Parameter(precedence=-1)
    invert_yaxis = param.Parameter(precedence=-1)
    legend_cols = param.Parameter(precedence=-1)
    legend_position = param.Parameter(precedence=-1)
    logx = param.Parameter(precedence=-1)
    logy = param.Parameter(precedence=-1)
    logz = param.Parameter(precedence=-1)
    show_grid = param.Parameter(precedence=-1)
    style_grouping = param.Parameter(precedence=-1)
    xticks = param.Parameter(precedence=-1)
    yticks = param.Parameter(precedence=-1)
    zticks = param.Parameter(precedence=-1)
    zaxis = param.Parameter(precedence=-1)
    zrotation = param.Parameter(precedence=-1)


    def __init__(self, layout, keys=None, dimensions=None, create_axes=False, ranges=None,
                 layout_num=1, **params):
        if not keys or not dimensions:
            dimensions, keys = traversal.unique_dimkeys(layout)
        MPLPlot.__init__(self, dimensions=dimensions, keys=keys, **params)
        self.layout = layout
        self.cyclic_index = 0
        self.zorder = 0
        self.layout_num = layout_num
        self.overlaid = False
        self.hmap = {}
        if layout.ndims > 1:
            xkeys, ykeys = zip(*layout.data.keys())
        else:
            xkeys = layout.keys()
            ykeys = [None]
        self._xkeys = sorted(set(xkeys))
        self._ykeys = sorted(set(ykeys))
        self._xticks, self._yticks = [], []
        self.rows, self.cols = layout.shape
        self.fig_inches = self._get_size()
        _, _, self.layout = self._create_subplots(layout, None, ranges, create_axes=False)
        self.border_extents = self._compute_borders()
        width, height, _, _, _, _ = self.border_extents
        if self.aspect == 'equal':
            self.aspect = float(width/height)

    def _finalize_artist(self, key):
        pass

    def get_extents(self, view, ranges):
        width, height, _, _, _, _ = self.border_extents
        return (0, 0, width, height)


    def _get_frame(self, key):
        return GridPlot._get_frame(self, key)


    def initialize_plot(self, ranges=None):
        width, height, b_w, b_h, widths, heights = self.border_extents

        key = self.keys[-1]
        ranges = self.compute_ranges(self.layout, key, ranges)
        self.handles['projs'] = {}
        x, y = b_w, b_h
        for xidx, xkey in enumerate(self._xkeys):
            w = widths[xidx]
            for yidx, ykey in enumerate(self._ykeys):
                h = heights[yidx]
                if self.layout.ndims > 1:
                    vmap = self.layout.get((xkey, ykey), None)
                else:
                    vmap = self.layout.get(xkey, None)
                pane = vmap.select(**{d.name: val for d, val in zip(self.dimensions, key)
                                    if d in vmap.kdims})
                if pane:
                    if issubclass(vmap.type, CompositeOverlay): pane = pane.values()[-1]
                    data = pane.data if pane else None
                else:
                    pane = vmap.last.values()[-1] if issubclass(vmap.type, CompositeOverlay) else vmap.last
                    data = pane.data
                ranges = self.compute_ranges(vmap, key, ranges)
                opts = self.lookup_options(pane, 'style')[self.cyclic_index]
                plot = self.handles['axis'].imshow(data, extent=(x,x+w, y, y+h), **opts)
                valrange = match_spec(pane, ranges)[pane.vdims[0].name]
                plot.set_clim(valrange)
                if data is None:
                    plot.set_visible(False)
                self.handles['projs'][(xkey, ykey)] = plot
                y += h + b_h
                if xidx == 0:
                    self._yticks.append(y-b_h-h/2.)
            y = b_h
            x += w + b_w
            self._xticks.append(x-b_w-w/2.)

        grid_dims = self.layout.kdims
        ydim = grid_dims[1] if self.layout.ndims > 1 else None
        xticks = (self._xticks, self._process_ticklabels(self._xkeys, grid_dims[0]))
        yticks = (self._yticks, self._process_ticklabels(self._ykeys, ydim))
        ylabel = str(self.layout.kdims[1]) if self.layout.ndims > 1 else ''

        return self._finalize_axis(key, ranges=ranges,
                                   title=self._format_title(key),
                                   xticks=xticks, yticks=yticks,
                                   xlabel=str(self.layout.get_dimension(0)),
                                   ylabel=ylabel)


    def update_frame(self, key, ranges=None):
        grid = self._get_frame(key)
        ranges = self.compute_ranges(self.layout, key, ranges)
        for xkey in self._xkeys:
            for ykey in self._ykeys:
                plot = self.handles['projs'][(xkey, ykey)]
                grid_key = (xkey, ykey) if self.layout.ndims > 1 else (xkey,)
                element = grid.data.get(grid_key, None)
                if element:
                    plot.set_visible(True)
                    data = element.values()[0].data if isinstance(element, CompositeOverlay) else element.data
                    plot.set_data(data)
                else:
                    plot.set_visible(False)

        xdim = self.layout.kdims[0]
        ydim = self.layout.kdims[1] if self.layout.ndims > 1 else None

        self._finalize_axis(key, ranges=ranges, title=self._format_title(key),
                            xticks=(self._xticks, self._process_ticklabels(self._xkeys, xdim)),
                            yticks=(self._yticks, self._process_ticklabels(self._ykeys, ydim)))


    def _axis_labels(self, view, subplots, xlabel=None, ylabel=None, zlabel=None):
        xdim = self.layout.kdims[0]
        ydim = self.layout.kdims[1] if self.layout.ndims > 1 else None
        return xlabel if xlabel else str(xdim), ylabel if ylabel or not ydim else str(ydim), zlabel


    def _compute_borders(self):
        ndims = self.layout.ndims
        xkey, ykey = self._xkeys[0], self._ykeys[0]
        width_fn = lambda x: x.range(0)
        height_fn = lambda x: x.range(1)
        if ndims > 1:
            vert_section = self.layout[xkey, slice(None)]
        else:
            vert_section = [self.layout[xkey]]
        horz_section = self.layout[(slice(None), ykey) if ndims > 1 else slice(None)]
        height_extents = [max_range(hm.traverse(height_fn, [Element])) for hm in vert_section]
        width_extents = [max_range(hm.traverse(width_fn, [Element])) for hm in horz_section]
        widths = [extent[0]-extent[1] for extent in width_extents]
        heights = [extent[0]-extent[1] for extent in height_extents]
        width, height = np.sum(widths), np.sum(heights)
        border_width = (width*self.padding)/(len(widths)+1)
        border_height = (height*self.padding)/(len(heights)+1)
        width += width*self.padding
        height += height*self.padding

        return width, height, border_width, border_height, widths, heights


    def __len__(self):
        return max([len(self.keys), 1])
