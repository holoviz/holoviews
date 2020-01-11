from __future__ import absolute_import, division, unicode_literals

from itertools import product

import numpy as np
import param

from matplotlib.patches import Wedge, Circle
from matplotlib.collections import LineCollection, PatchCollection

from ...core.data import GridInterface
from ...core.util import dimension_sanitizer, is_nan
from ...core.spaces import HoloMap
from .element import ColorbarPlot
from .raster import QuadMeshPlot
from .util import filter_styles


class HeatMapPlot(QuadMeshPlot):

    clipping_colors = param.Dict(default={'NaN': 'white'}, doc="""
        Dictionary to specify colors for clipped values, allows
        setting color for NaN values and for values above and below
        the min and max value. The min, max or NaN color may specify
        an RGB(A) color as a color hex string of the form #FFFFFF or
        #FFFFFFFF or a length 3 or length 4 tuple specifying values in
        the range 0-1 or a named HTML color.""")

    radial = param.Boolean(default=False, doc="""
        Whether the HeatMap should be radial""")

    show_values = param.Boolean(default=False, doc="""
        Whether to annotate each pixel with its value.""")

    xmarks = param.Parameter(default=None, doc="""
        Add separation lines to the heatmap for better readability. By
        default, does not show any separation lines. If parameter is of type
        integer, draws the given amount of separations lines spread across
        heatmap. If parameter is of type list containing integers, show
        separation lines at given indices. If parameter is of type tuple, draw
        separation lines at given categorical values. If parameter is of type
        function, draw separation lines where function returns True for passed
        heatmap category.""")

    ymarks = param.Parameter(default=None, doc="""
        Add separation lines to the heatmap for better readability. By
        default, does not show any separation lines. If parameter is of type
        integer, draws the given amount of separations lines spread across
        heatmap. If parameter is of type list containing integers, show
        separation lines at given indices. If parameter is of type tuple, draw
        separation lines at given categorical values. If parameter is of type
        function, draw separation lines where function returns True for passed
        heatmap category.""")

    xticks = param.Parameter(default=20, doc="""
        Ticks along x-axis/segments specified as an integer, explicit list of
        ticks or function. If `None`, no ticks are shown.""")

    yticks = param.Parameter(default=20, doc="""
        Ticks along y-axis/annulars specified as an integer, explicit list of
        ticks or function. If `None`, no ticks are shown.""")

    def get_extents(self, element, ranges, range_type='combined'):
        if range_type in ('data', 'combined'):
            agg = element.gridded
            xtype = agg.interface.dtype(agg, 0)
            shape = agg.interface.shape(agg, gridded=True)
            if xtype.kind in 'SUO':
                x0, x1 = (0-0.5, shape[1]-0.5)
            else:
                x0, x1 = element.range(0)
            ytype = agg.interface.dtype(agg, 1)
            if ytype.kind in 'SUO':
                y0, y1 = (-.5, shape[0]-0.5)
            else:
                y0, y1 = element.range(1)
            return (x0, y0, x1, y1)
        else:
            return super(HeatMapPlot, self).get_extents(element, ranges, range_type)


    @classmethod
    def is_radial(cls, heatmap):
        heatmap = heatmap.last if isinstance(heatmap, HoloMap) else heatmap
        opts = cls.lookup_options(heatmap, 'plot').options
        return ((any(o in opts for o in ('start_angle', 'radius_inner', 'radius_outer'))
                 and not (opts.get('radial') == False)) or opts.get('radial', False))

    def _annotate_plot(self, ax, annotations):
        for a in self.handles.get('annotations', {}).values():
            a.remove()
        handles = {}
        for plot_coord, text in annotations.items():
            handles[plot_coord] = ax.annotate(text, xy=plot_coord,
                                              xycoords='data',
                                              horizontalalignment='center',
                                              verticalalignment='center')
        return handles


    def _annotate_values(self, element, xvals, yvals):
        val_dim = element.vdims[0]
        vals = element.dimension_values(val_dim).flatten()
        xpos = xvals[:-1] + np.diff(xvals)/2.
        ypos = yvals[:-1] + np.diff(yvals)/2.
        plot_coords = product(xpos, ypos)
        annotations = {}
        for plot_coord, v in zip(plot_coords, vals):
            text = '-' if is_nan(v) else val_dim.pprint_value(v)
            annotations[plot_coord] = text
        return annotations


    def _compute_ticks(self, element, xvals, yvals):
        xdim, ydim = element.kdims
        if self.invert_axes:
            xdim, ydim = ydim, xdim

        opts = self.lookup_options(element, 'plot').options

        xticks = opts.get('xticks')
        if xticks is None:
            xpos = xvals[:-1] + np.diff(xvals)/2.
            xlabels = [xdim.pprint_value(k) for k in element.gridded.dimension_values(xdim, False)]
            xticks = list(zip(xpos, xlabels))

        yticks = opts.get('yticks')
        if yticks is None:
            ypos = yvals[:-1] + np.diff(yvals)/2.
            ylabels = [ydim.pprint_value(k) for k in element.gridded.dimension_values(ydim, False)]
            yticks = list(zip(ypos, ylabels))
        return xticks, yticks


    def _draw_markers(self, ax, element, marks, values, axis='x'):
        if marks is None:
            return
        style = self.style[self.cyclic_index]
        mark_opts = {k[7:]: v for k, v in style.items() if axis+'mark' in k}
        mark_opts = {k[4:] if 'edge' in k else k: v for k, v in mark_opts.items()}
        categories = values

        if marks == True:
            positions = values
        elif callable(marks):
            positions = [i for i, x in enumerate(categories) if marks(x)]
        elif isinstance(marks, int):
            nth_mark = np.ceil(len(categories) / marks).astype(int)
            positions = values[::nth_mark]
        elif isinstance(marks, tuple):
            positions = [categories.index(m) for m in marks if m in categories]
        else:
            positions = marks

        for m in self.handles.get(axis+'marks', []):
            m.remove()

        new_markers = []
        for p in positions:
            if axis == 'x':
                line = ax.axvline(p, **mark_opts)
            else:
                line = ax.axhline(p, **mark_opts)
            new_markers.append(line)
        self.handles[axis+'marks'] = new_markers


    def init_artists(self, ax, plot_args, plot_kwargs):
        annotations = plot_kwargs.pop('annotations', None)
        prefixes = ['annular', 'xmarks', 'ymarks']
        plot_kwargs = {k: v for k, v in plot_kwargs.items()
                       if not any(p in k for p in prefixes)}
        artist = ax.pcolormesh(*plot_args, **plot_kwargs)

        if self.show_values and annotations:
            self.handles['annotations'] = self._annotate_plot(ax, annotations)
        self._draw_markers(ax, self.current_frame, self.xmarks, plot_args[0], axis='x')
        self._draw_markers(ax, self.current_frame, self.ymarks, plot_args[1], axis='y')
        return {'artist': artist}


    def get_data(self, element, ranges, style):
        xdim, ydim = element.kdims
        aggregate = element.gridded

        data = aggregate.dimension_values(2, flat=False)
        data = np.ma.array(data, mask=np.logical_not(np.isfinite(data)))
        if self.invert_axes:
            xdim, ydim = ydim, xdim
            data = data.T[::-1, ::-1]

        xtype = aggregate.interface.dtype(aggregate, xdim)
        if xtype.kind in 'SUO':
            xvals = np.arange(data.shape[1]+1)-0.5
        else:
            xvals = aggregate.dimension_values(xdim, expanded=False)
            xvals = GridInterface._infer_interval_breaks(xvals)

        ytype = aggregate.interface.dtype(aggregate, ydim)
        if ytype.kind in 'SUO':
            yvals = np.arange(data.shape[0]+1)-0.5
        else:
            yvals = aggregate.dimension_values(ydim, expanded=False)
            yvals = GridInterface._infer_interval_breaks(yvals)

        xticks, yticks = self._compute_ticks(element, xvals, yvals)

        if self.show_values:
            style['annotations'] = self._annotate_values(element.gridded, xvals, yvals)
        vdim = element.vdims[0]
        self._norm_kwargs(element, ranges, style, vdim)
        return (xvals, yvals, data), style, {'xticks': xticks, 'yticks': yticks}



class RadialHeatMapPlot(ColorbarPlot):

    start_angle = param.Number(default=np.pi/2, doc="""
        Define starting angle of the first annulars. By default, beings
        at 12 o clock.""")

    max_radius = param.Number(default=0.5, doc="""
        Define the maximum radius which is used for the x and y range extents.
        """)

    radius_inner = param.Number(default=0.1, bounds=(0, 0.5), doc="""
        Define the radius fraction of inner, empty space.""")

    radius_outer = param.Number(default=0.05, bounds=(0, 1), doc="""
        Define the radius fraction of outer space including the labels.""")

    radial = param.Boolean(default=True, doc="""
        Whether the HeatMap should be radial""")

    show_values = param.Boolean(default=False, doc="""
        Whether to annotate each pixel with its value.""")

    xmarks = param.Parameter(default=None, doc="""
        Add separation lines between segments for better readability. By
        default, does not show any separation lines. If parameter is of type
        integer, draws the given amount of separations lines spread across
        radial heatmap. If parameter is of type list containing integers, show
        separation lines at given indices. If parameter is of type tuple, draw
        separation lines at given segment values. If parameter is of type
        function, draw separation lines where function returns True for passed
        segment value.""")

    ymarks = param.Parameter(default=None, doc="""
        Add separation lines between annulars for better readability. By
        default, does not show any separation lines. If parameter is of type
        integer, draws the given amount of separations lines spread across
        radial heatmap. If parameter is of type list containing integers, show
        separation lines at given indices. If parameter is of type tuple, draw
        separation lines at given annular values. If parameter is of type
        function, draw separation lines where function returns True for passed
        annular value.""")

    xticks = param.Parameter(default=4, doc="""
        Ticks along x-axis/segments specified as an integer, explicit list of
        ticks or function. If `None`, no ticks are shown.""")

    yticks = param.Parameter(default=4, doc="""
        Ticks along y-axis/annulars specified as an integer, explicit list of
        ticks or function. If `None`, no ticks are shown.""")

    projection = param.ObjectSelector(default='polar', objects=['polar'])

    _style_groups = ['annular', 'xmarks', 'ymarks']

    style_opts = ['annular_edgecolors', 'annular_linewidth',
                  'xmarks_linewidth', 'xmarks_edgecolor', 'cmap',
                  'ymarks_linewidth', 'ymarks_edgecolor']

    @staticmethod
    def _map_order_to_ticks(start, end, order, reverse=False):
        """Map elements from given `order` array to bins ranging from `start`
        to `end`.
        """
        size = len(order)
        bounds = np.linspace(start, end, size + 1)
        if reverse:
            bounds = bounds[::-1]
        mapping = list(zip(bounds[:-1]%(np.pi*2), order))
        return mapping

    @staticmethod
    def _compute_separations(inner, outer, angles):
        """Compute x and y positions for separation lines for given angles.

        """
        return [np.array([[a, inner], [a, outer]]) for a in angles]

    @staticmethod
    def _get_markers(ticks, marker):
        if callable(marker):
            marks = [v for v, l in ticks if marker(l)]
        elif isinstance(marker, int) and marker:
            nth_mark = max([np.ceil(len(ticks) / marker).astype(int), 1])
            marks = [v for v, l in ticks[::nth_mark]]
        elif isinstance(marker, tuple):
            marks = [v for v, l in ticks if l in marker]
        else:
            marks = []
        return marks

    @staticmethod
    def _get_ticks(ticks, ticker):
        if callable(ticker):
            ticks = [(v, l) for v, l in ticks if ticker(l)]
        elif isinstance(ticker, int):
            nth_mark = max([np.ceil(len(ticks) / ticker).astype(int), 1])
            ticks = ticks[::nth_mark]
        elif isinstance(ticker, (tuple, list)):
            nth_mark = max([np.ceil(len(ticks) / len(ticker)).astype(int), 1])
            ticks = [(v, tl) for (v, l), tl in zip(ticks[::nth_mark], ticker)]
        elif ticker:
            ticks = list(ticker)
        else:
            ticks = []
        return ticks


    def get_extents(self, view, ranges, range_type='combined'):
        if range_type == 'hard':
            return (np.nan,)*4
        return (0, 0, np.pi*2, self.max_radius+self.radius_outer)


    def get_data(self, element, ranges, style):
        # dimension labels
        dim_labels = element.dimensions(label=True)[:3]
        x, y, z = [dimension_sanitizer(d) for d in dim_labels]

        if self.invert_axes: x, y = y, x

        # get raw values
        aggregate = element.gridded
        xvals = aggregate.dimension_values(x, expanded=False)
        yvals = aggregate.dimension_values(y, expanded=False)
        zvals = aggregate.dimension_values(2, flat=False)

        # pretty print x and y dimension values if necessary
        def _pprint(dim_label, vals):
            if vals.dtype.kind not in 'SU':
                dim = aggregate.get_dimension(dim_label)
                return [dim.pprint_value(v) for v in vals]

            return vals

        xvals = _pprint(x, xvals)
        yvals = _pprint(y, yvals)

        # annular wedges
        start_angle = self.start_angle
        end_angle = self.start_angle + 2 * np.pi
        bins_segment = np.linspace(start_angle, end_angle, len(xvals)+1)
        segment_ticks = self._map_order_to_ticks(start_angle, end_angle,
                                                 xvals, True)

        radius_max = 0.5
        radius_min = radius_max * self.radius_inner
        bins_annular = np.linspace(radius_min, radius_max, len(yvals)+1)
        radius_ticks = self._map_order_to_ticks(radius_min, radius_max,
                                                yvals)

        patches = []
        for j in range(len(yvals)):
            ybin = bins_annular[j:j+2]
            for i in range(len(xvals))[::-1]:
                xbin = np.rad2deg(bins_segment[i:i+2])
                width = ybin[1]-ybin[0]
                wedge = Wedge((0.5, 0.5), ybin[1], xbin[0], xbin[1], width)
                patches.append(wedge)

        angles = self._get_markers(segment_ticks, self.xmarks)
        xmarks = self._compute_separations(radius_min, radius_max, angles)
        radii = self._get_markers(radius_ticks, self.ymarks)
        ymarks = [Circle((0.5, 0.5), r) for r in radii]

        style['array'] = zvals.flatten()
        self._norm_kwargs(element, ranges, style, element.vdims[0])
        if 'vmin' in style:
            style['clim'] = style.pop('vmin'), style.pop('vmax')

        data = {'annular': patches, 'xseparator': xmarks, 'yseparator': ymarks}
        xticks = self._get_ticks(segment_ticks, self.xticks)
        if not isinstance(self.xticks, int):
            xticks = [(v-((np.pi)/len(xticks)), l) for v, l in xticks]
        yticks = self._get_ticks(radius_ticks, self.yticks)
        ticks = {'xticks': xticks,  'yticks': yticks}
        return data, style, ticks


    def init_artists(self, ax, plot_args, plot_kwargs):
        # Draw edges
        color_opts = ['c', 'cmap', 'vmin', 'vmax', 'norm', 'array']
        groups = [g for g in self._style_groups if g != 'annular']
        edge_opts = filter_styles(plot_kwargs, 'annular', groups)
        annuli = plot_args['annular']
        edge_opts.pop('interpolation', None)
        annuli = PatchCollection(annuli, transform=ax.transAxes, **edge_opts)
        ax.add_collection(annuli)

        artists = {'artist': annuli}

        paths = plot_args['xseparator']
        if paths:
            groups = [g for g in self._style_groups if g != 'xmarks']
            xmark_opts = filter_styles(plot_kwargs, 'xmarks', groups, color_opts)
            xmark_opts.pop('interpolation', None)
            xseparators = LineCollection(paths, **xmark_opts)
            ax.add_collection(xseparators)
            artists['xseparator'] = xseparators

        paths = plot_args['yseparator']
        if paths:
            groups = [g for g in self._style_groups if g != 'ymarks']
            ymark_opts = filter_styles(plot_kwargs, 'ymarks', groups, color_opts)
            ymark_opts.pop('interpolation', None)
            yseparators = PatchCollection(paths, facecolor='none',
                                          transform=ax.transAxes, **ymark_opts)
            ax.add_collection(yseparators)
            artists['yseparator'] = yseparators

        return artists
