import copy
from itertools import product

import numpy as np
from matplotlib import pyplot as plt

import param

from ...core import CompositeOverlay, Element
from ...core import traversal
from ...core.util import match_spec, max_range
from ...element.raster import Image, Raster, RGB
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


    def _compute_ticks(self, element, ranges):
        return None, None


    def get_data(self, element, ranges, style):
        xticks, yticks = self._compute_ticks(element, ranges)

        if element.depth != 1:
            style.pop('cmap', None)

        data = element.data
        if isinstance(element, Image):
            l, b, r, t = element.bounds.lbrt()
        else:
            l, b, r, t = element.extents
        if self.invert_yaxis and type(element) is Raster:
            b, t = t, b

        if isinstance(element, RGB):
            data = element.rgb.data
        vdim = element.vdims[0]
        self._norm_kwargs(element, ranges, style, vdim)
        style['extent'] = [l, r, b, t]

        return [data], style, {'xticks': xticks, 'yticks': yticks}


    def init_artists(self, ax, plot_args, plot_kwargs):
        im = ax.imshow(*plot_args, **plot_kwargs)
        return {'artist': im}


    def update_handles(self, key, axis, element, ranges, style):
        im = self.handles['artist']
        data, style, axis_kwargs = self.get_data(element, ranges, style)
        l, r, b, t = style['extent']
        im.set_data(data[0])
        im.set_extent((l, r, b, t))
        im.set_clim((style['vmin'], style['vmax']))
        if 'norm' in style:
            im.norm = style['norm']

        return axis_kwargs


class HeatMapPlot(RasterPlot):

    show_values = param.Boolean(default=False, doc="""
        Whether to annotate each pixel with its value.""")

    def _annotate_plot(self, ax, annotations):
        handles = {}
        for plot_coord, text in annotations.items():
            handles[plot_coord] = ax.annotate(text, xy=plot_coord,
                                              xycoords='axes fraction',
                                              horizontalalignment='center',
                                              verticalalignment='center')
        return handles


    def _annotate_values(self, element):
        val_dim = element.vdims[0]
        vals = np.rot90(element.raster, 3).flatten()
        d1uniq, d2uniq = [element.dimension_values(i, False) for i in range(2)]
        num_x, num_y = len(d1uniq), len(d2uniq)
        xstep, ystep = 1.0/num_x, 1.0/num_y
        xpos = np.linspace(xstep/2., 1.0-xstep/2., num_x)
        ypos = np.linspace(ystep/2., 1.0-ystep/2., num_y)
        plot_coords = product(xpos, ypos)
        annotations = {}
        for plot_coord, v in zip(plot_coords, vals):
            text = val_dim.pprint_value(v)
            text = '' if v is np.nan else text
            annotations[plot_coord] = text
        return annotations


    def _compute_ticks(self, element, ranges):
        xdim, ydim = element.kdims
        dim1_keys, dim2_keys = [element.dimension_values(i, False)
                                for i in range(2)]
        num_x, num_y = len(dim1_keys), len(dim2_keys)
        x0, y0, x1, y1 = element.extents
        xstep, ystep = ((x1-x0)/num_x, (y1-y0)/num_y)
        xpos = np.linspace(x0+xstep/2., x1-xstep/2., num_x)
        ypos = np.linspace(y0+ystep/2., y1-ystep/2., num_y)
        xlabels = [xdim.pprint_value(k) for k in dim1_keys]
        ylabels = [ydim.pprint_value(k) for k in dim2_keys]
        return list(zip(xpos, xlabels)), list(zip(ypos, ylabels))


    def init_artists(self, ax, plot_args, plot_kwargs):
        l, r, b, t = plot_kwargs['extent']
        ax.set_aspect(float(r - l)/(t-b))

        handles = {}
        annotations = plot_kwargs.pop('annotations', None)
        handles['artist'] = ax.imshow(*plot_args, **plot_kwargs)
        if self.show_values and annotations:
            handles['annotations'] = self._annotate_plot(ax, annotations)
        return handles


    def get_data(self, element, ranges, style):
        _, style, axis_kwargs = super(HeatMapPlot, self).get_data(element, ranges, style)
        data = element.raster
        data = np.ma.array(data, mask=np.logical_not(np.isfinite(data)))
        cmap_name = style.pop('cmap', None)
        cmap = copy.copy(plt.cm.get_cmap('gray' if cmap_name is None else cmap_name))
        cmap.set_bad('w', 1.)
        style['cmap'] = cmap
        style['annotations'] = self._annotate_values(element)
        return [data], style, axis_kwargs


    def update_handles(self, key, axis, element, ranges, style):
        im = self.handles['artist']
        data, style, axis_kwargs = self.get_data(element, ranges, style)
        l, r, b, t = style['extent']
        im.set_data(data[0])
        im.set_extent((l, r, b, t))
        im.set_clim((style['vmin'], style['vmax']))
        if 'norm' in style:
            im.norm = style['norm']

        if self.show_values:
            annotations = self.handles['annotations']
            for annotation in annotations.values():
                try:
                    annotation.remove()
                except:
                    pass
            annotations = self._annotate_plot(axis, style['annotations'])
            self.handles['annotations'] = annotations
        return axis_kwargs




class QuadMeshPlot(ColorbarPlot):

    style_opts = ['alpha', 'cmap', 'clim', 'edgecolors', 'norm', 'shading',
                  'linestyles', 'linewidths', 'hatch', 'visible']

    def get_data(self, element, ranges, style):
        data = np.ma.array(element.data[2],
                           mask=np.logical_not(np.isfinite(element.data[2])))
        cmesh_data = list(element.data[:2]) + [data]
        style['locs'] = np.concatenate(element.data[:2])
        vdim = element.vdims[0]
        self._norm_kwargs(element, ranges, style, vdim)
        return tuple(cmesh_data), style, {}


    def init_artists(self, ax, plot_args, plot_kwargs):
        locs = plot_kwargs.pop('locs')
        artist = ax.pcolormesh(*plot_args, **plot_kwargs)
        return {'artist': artist, 'locs': locs}


    def update_handles(self, key, axis, element, ranges, style):
        cmesh = self.handles['artist']
        locs = np.concatenate(element.data[:2])

        if (locs != self.handles['locs']).any():
            return super(QuadMeshPlot, self).update_handles(key, axis, element,
                                                            ranges, style)
        else:
            data, style, axis_kwargs = self.get_data(element, ranges, style)
            cmesh.set_array(data[-1])
            cmesh.set_clim((style['vmin'], style['vmax']))
            if 'norm' in style:
                cmesh.norm = style['norm']
            return axis_kwargs


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
    invert_zaxis = param.Parameter(precedence=-1)
    labelled = param.Parameter(precedence=-1)
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
        _, _, b_w, b_h, widths, heights = self.border_extents

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
                cdim = pane.vdims[0].name
                valrange = match_spec(pane, ranges).get(cdim, pane.range(cdim))
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

        kwargs = self._get_axis_kwargs()
        return self._finalize_axis(key, ranges=ranges, **kwargs)


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

        kwargs = self._get_axis_kwargs()
        return self._finalize_axis(key, ranges=ranges, **kwargs)


    def _get_axis_kwargs(self):
        xdim = self.layout.kdims[0]
        ydim = self.layout.kdims[1] if self.layout.ndims > 1 else None
        xticks = (self._xticks, self._process_ticklabels(self._xkeys, xdim))
        yticks = (self._yticks, self._process_ticklabels(self._ykeys, ydim))
        return dict(xlabel=xdim.pprint_label, ylabel=ydim.pprint_label if ydim else '',
                    xticks=xticks, yticks=yticks)


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
