from itertools import product

import numpy as np
import param

from ...core import CompositeOverlay, Element
from ...core import traversal
from ...core.util import match_spec, max_range, unique_iterator, unique_array, is_nan
from ...element.raster import Image, Raster, RGB
from .element import ColorbarPlot, OverlayPlot
from .plot import MPLPlot, GridPlot, mpl_rc_context
from .util import get_raster_array


class RasterPlot(ColorbarPlot):

    aspect = param.Parameter(default='equal', doc="""
        Raster elements respect the aspect ratio of the
        Images by default but may be set to an explicit
        aspect ratio or to 'square'.""")

    colorbar = param.Boolean(default=False, doc="""
        Whether to add a colorbar to the plot.""")

    situate_axes = param.Boolean(default=True, doc="""
        Whether to situate the image relative to other plots. """)

    style_opts = ['alpha', 'cmap', 'interpolation', 'visible',
                  'filterrad', 'clims', 'norm']

    _plot_methods = dict(single='imshow')

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

        if isinstance(element, RGB):
            style.pop('cmap', None)

        data = get_raster_array(element)
        if type(element) is Raster:
            l, b, r, t = element.extents
            if self.invert_axes:
                data = data[:, ::-1]
            else:
                data = data[::-1]
        else:
            l, b, r, t = element.bounds.lbrt()
            if self.invert_axes:
                data = data[::-1, ::-1]

        if self.invert_axes:
            data = data.transpose([1, 0, 2]) if isinstance(element, RGB) else data.T
            l, b, r, t = b, l, t, r

        vdim = element.vdims[0]
        self._norm_kwargs(element, ranges, style, vdim)
        style['extent'] = [l, r, b, t]
        style['origin'] = 'upper'

        return [data], style, {'xticks': xticks, 'yticks': yticks}

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

    clipping_colors = param.Dict(default={'NaN': 'white'}, doc="""
        Dictionary to specify colors for clipped values, allows
        setting color for NaN values and for values above and below
        the min and max value. The min, max or NaN color may specify
        an RGB(A) color as a color hex string of the form #FFFFFF or
        #FFFFFFFF or a length 3 or length 4 tuple specifying values in
        the range 0-1 or a named HTML color.""")

    show_values = param.Boolean(default=False, doc="""
        Whether to annotate each pixel with its value.""")

    def _annotate_plot(self, ax, annotations):
        handles = {}
        for plot_coord, text in annotations.items():
            handles[plot_coord] = ax.annotate(text, xy=plot_coord,
                                              xycoords='data',
                                              horizontalalignment='center',
                                              verticalalignment='center')
        return handles


    def _annotate_values(self, element):
        val_dim = element.vdims[0]
        vals = element.dimension_values(2, flat=False)
        d1uniq, d2uniq = [element.dimension_values(i, False) for i in range(2)]
        if self.invert_axes:
            d1uniq, d2uniq = d2uniq, d1uniq
        else:
            vals = vals.T
        if self.invert_xaxis: vals = vals[::-1]
        if self.invert_yaxis: vals = vals[:, ::-1]
        vals = vals.flatten()
        num_x, num_y = len(d1uniq), len(d2uniq)
        xpos = np.linspace(0.5, num_x-0.5, num_x)
        ypos = np.linspace(0.5, num_y-0.5, num_y)
        plot_coords = product(xpos, ypos)
        annotations = {}
        for plot_coord, v in zip(plot_coords, vals):
            text = '-' if is_nan(v) else val_dim.pprint_value(v)
            annotations[plot_coord] = text
        return annotations


    def _compute_ticks(self, element, ranges):
        xdim, ydim = element.dimensions()[:2]
        agg = element.gridded
        dim1_keys, dim2_keys = [unique_array(agg.dimension_values(i, False))
                                for i in range(2)]
        if self.invert_axes:
            dim1_keys, dim2_keys = dim2_keys, dim1_keys
        num_x, num_y = len(dim1_keys), len(dim2_keys)
        xpos = np.linspace(.5, num_x-0.5, num_x)
        ypos = np.linspace(.5, num_y-0.5, num_y)
        xlabels = [xdim.pprint_value(k) for k in dim1_keys]
        ylabels = [ydim.pprint_value(k) for k in dim2_keys]
        return list(zip(xpos, xlabels)), list(zip(ypos, ylabels))


    def init_artists(self, ax, plot_args, plot_kwargs):
        ax.set_aspect(plot_kwargs.pop('aspect', 1))

        handles = {}
        annotations = plot_kwargs.pop('annotations', None)
        handles['artist'] = ax.imshow(*plot_args, **plot_kwargs)
        if self.show_values and annotations:
            handles['annotations'] = self._annotate_plot(ax, annotations)
        return handles


    def get_data(self, element, ranges, style):
        xticks, yticks = self._compute_ticks(element, ranges)

        data = np.flipud(element.gridded.dimension_values(2, flat=False))
        data = np.ma.array(data, mask=np.logical_not(np.isfinite(data)))
        if self.invert_axes: data = data.T[::-1, ::-1]
        if self.invert_xaxis: data = data[:, ::-1]
        if self.invert_yaxis: data = data[::-1]
        shape = data.shape
        style['aspect'] = shape[0]/shape[1]
        style['extent'] = (0, shape[1], 0, shape[0])
        style['annotations'] = self._annotate_values(element.gridded)
        style['origin'] = 'upper'
        vdim = element.vdims[0]
        self._norm_kwargs(element, ranges, style, vdim)
        return [data], style, {'xticks': xticks, 'yticks': yticks}


    def update_handles(self, key, axis, element, ranges, style):
        im = self.handles['artist']
        data, style, axis_kwargs = self.get_data(element, ranges, style)
        im.set_data(data[0])
        im.set_extent(style['extent'])
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

    style_opts = ['alpha', 'cmap', 'clims', 'edgecolors', 'norm', 'shading',
                  'linestyles', 'linewidths', 'hatch', 'visible']

    _plot_methods = dict(single='pcolormesh')

    def get_data(self, element, ranges, style):
        zdata = element.dimension_values(2, flat=False)
        data = np.ma.array(zdata, mask=np.logical_not(np.isfinite(zdata)))
        expanded = element.interface.irregular(element, element.kdims[0])
        edges = style.get('shading') != 'gouraud'
        coords = [element.interface.coords(element, d, ordered=True,
                                           expanded=expanded, edges=edges)
                  for d in element.kdims]
        if self.invert_axes:
            coords = coords[::-1]
            data = data.T
        cmesh_data = coords + [data]
        if expanded:
            style['locs'] = np.concatenate(coords)
        vdim = element.vdims[0]
        self._norm_kwargs(element, ranges, style, vdim)
        return tuple(cmesh_data), style, {}


    def init_artists(self, ax, plot_args, plot_kwargs):
        locs = plot_kwargs.pop('locs', None)
        artist = ax.pcolormesh(*plot_args, **plot_kwargs)
        return {'artist': artist, 'locs': locs}


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
    batched = param.Parameter(precedence=-1)
    bgcolor = param.Parameter(precedence=-1)
    invert_axes = param.Parameter(precedence=-1)
    invert_xaxis = param.Parameter(precedence=-1)
    invert_yaxis = param.Parameter(precedence=-1)
    invert_zaxis = param.Parameter(precedence=-1)
    labelled = param.Parameter(precedence=-1)
    legend_cols = param.Parameter(precedence=-1)
    legend_position = param.Parameter(precedence=-1)
    legend_limit = param.Parameter(precedence=-1)
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
        top_level = keys is None
        if top_level:
            dimensions, keys = traversal.unique_dimkeys(layout)
        MPLPlot.__init__(self, dimensions=dimensions, keys=keys, **params)
        if top_level:
            self.comm = self.init_comm()

        self.layout = layout
        self.cyclic_index = 0
        self.zorder = 0
        self.layout_num = layout_num
        self.overlaid = False
        self.hmap = layout
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
        # Note that streams are not supported on RasterGridPlot
        # until that is implemented this stub is needed
        self.streams = []

    def _finalize_artist(self, key):
        pass

    def get_extents(self, view, ranges):
        width, height, _, _, _, _ = self.border_extents
        return (0, 0, width, height)


    def _get_frame(self, key):
        return GridPlot._get_frame(self, key)


    @mpl_rc_context
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
                pane = vmap.last.values()[-1] if issubclass(vmap.type, CompositeOverlay) else vmap.last
                data = get_raster_array(pane) if pane else None
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

    @mpl_rc_context
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
                    img = element.values()[0] if isinstance(element, CompositeOverlay) else element
                    data = get_raster_array(img)
                    plot.set_data(data)
                else:
                    plot.set_visible(False)

        kwargs = self._get_axis_kwargs()
        return self._finalize_axis(key, ranges=ranges, **kwargs)


    def _get_axis_kwargs(self):
        xdim = self.layout.kdims[0]
        ydim = self.layout.kdims[1] if self.layout.ndims > 1 else None
        xticks = (self._xticks, [xdim.pprint_value(l) for l in self._xkeys])
        yticks = (self._yticks, [ydim.pprint_value(l) if ydim else ''
                                 for l in self._ykeys])
        return dict(xlabel=xdim.pprint_label, ylabel=ydim.pprint_label if ydim else '',
                    xticks=xticks, yticks=yticks)


    def _compute_borders(self):
        ndims = self.layout.ndims
        width_fn = lambda x: x.range(0)
        height_fn = lambda x: x.range(1)
        width_extents = [max_range(self.layout[x, :].traverse(width_fn, [Element]))
                         for x in unique_iterator(self.layout.dimension_values(0))]
        if ndims > 1:
            height_extents = [max_range(self.layout[:, y].traverse(height_fn, [Element]))
                              for y in unique_iterator(self.layout.dimension_values(1))]
        else:
            height_extents = [max_range(self.layout.traverse(height_fn, [Element]))]
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
