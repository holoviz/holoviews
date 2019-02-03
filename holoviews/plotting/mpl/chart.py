from __future__ import absolute_import, division, unicode_literals

from itertools import product

import param
import numpy as np
import matplotlib as mpl

from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.dates import DateFormatter, date2num

from ...core.dimension import Dimension, dimension_name
from ...core.options import Store, abbreviated_exception
from ...core.util import (
    OrderedDict, match_spec, unique_iterator, basestring, max_range,
    isfinite, datetime_types, dt_to_int, dt64_to_dt, search_indices,
    unique_array, isscalar
)
from ...element import Raster, HeatMap
from ...operation import interpolate_curve
from ...util.transform import dim
from ..plot import PlotSelector
from ..util import compute_sizes, get_sideplot_ranges, get_min_distance
from .element import ElementPlot, ColorbarPlot, LegendPlot
from .path  import PathPlot
from .plot import AdjoinedPlot, mpl_rc_context
from .util import mpl_version



class ChartPlot(ElementPlot):
    """
    Baseclass to plot Chart elements.
    """


class CurvePlot(ChartPlot):
    """
    CurvePlot can plot Curve and ViewMaps of Curve, which can be
    displayed as a single frame or animation. Axes, titles and legends
    are automatically generated from dim_info.

    If the dimension is set to cyclic in the dim_info it will rotate
    the curve so that minimum y values are at the minimum x value to
    make the plots easier to interpret.
    """

    autotick = param.Boolean(default=False, doc="""
        Whether to let matplotlib automatically compute tick marks
        or to allow the user to control tick marks.""")

    interpolation = param.ObjectSelector(objects=['linear', 'steps-mid',
                                                  'steps-pre', 'steps-post'],
                                         default='linear', doc="""
        Defines how the samples of the Curve are interpolated,
        default is 'linear', other options include 'steps-mid',
        'steps-pre' and 'steps-post'.""")

    relative_labels = param.Boolean(default=False, doc="""
        If plotted quantity is cyclic and center_cyclic is enabled,
        will compute tick labels relative to the center.""")

    show_grid = param.Boolean(default=False, doc="""
        Enable axis grid.""")

    show_legend = param.Boolean(default=True, doc="""
        Whether to show legend for the plot.""")

    style_opts = ['alpha', 'color', 'visible', 'linewidth', 'linestyle', 'marker', 'ms']

    _nonvectorized_styles = style_opts

    _plot_methods = dict(single='plot')

    def get_data(self, element, ranges, style):
        with abbreviated_exception():
            style = self._apply_transforms(element, ranges, style)

        if 'steps' in self.interpolation:
            element = interpolate_curve(element, interpolation=self.interpolation)
        xs = element.dimension_values(0)
        ys = element.dimension_values(1)
        dims = element.dimensions()
        if xs.dtype.kind == 'M' or (len(xs) and isinstance(xs[0], datetime_types)):
            dimtype = element.get_dimension_type(0)
            dt_format = Dimension.type_formatters.get(dimtype, '%Y-%m-%d %H:%M:%S')
            dims[0] = dims[0](value_format=DateFormatter(dt_format))
        coords = (ys, xs) if self.invert_axes else (xs, ys)
        return coords, style, {'dimensions': dims}

    def init_artists(self, ax, plot_args, plot_kwargs):
        xs, ys = plot_args
        if xs.dtype.kind == 'M' or (len(xs) and isinstance(xs[0], datetime_types)):
            artist = ax.plot_date(xs, ys, '-', **plot_kwargs)[0]
        else:
            artist = ax.plot(xs, ys, **plot_kwargs)[0]
        return {'artist': artist}

    def update_handles(self, key, axis, element, ranges, style):
        artist = self.handles['artist']
        (xs, ys), style, axis_kwargs = self.get_data(element, ranges, style)
        artist.set_xdata(xs)
        artist.set_ydata(ys)
        return axis_kwargs



class ErrorPlot(ColorbarPlot):
    """
    ErrorPlot plots the ErrorBar Element type and supporting
    both horizontal and vertical error bars via the 'horizontal'
    plot option.
    """

    style_opts = ['edgecolor', 'elinewidth', 'capsize', 'capthick',
                  'barsabove', 'lolims', 'uplims', 'xlolims',
                  'errorevery', 'xuplims', 'alpha', 'linestyle',
                  'linewidth', 'markeredgecolor', 'markeredgewidth',
                  'markerfacecolor', 'markersize', 'solid_capstyle',
                  'solid_joinstyle', 'dashes', 'color']

    _plot_methods = dict(single='errorbar')

    def init_artists(self, ax, plot_data, plot_kwargs):
        handles = ax.errorbar(*plot_data, **plot_kwargs)
        bottoms, tops = None, None
        if mpl_version >= str('2.0'):
            _, caps, verts = handles
            if caps:
                bottoms, tops = caps
        else:
            _, (bottoms, tops), verts = handles
        return {'bottoms': bottoms, 'tops': tops, 'verts': verts[0], 'artist': verts[0]}


    def get_data(self, element, ranges, style):
        with abbreviated_exception():
            style = self._apply_transforms(element, ranges, style)
        color = style.get('color')
        if isinstance(color, np.ndarray):
            style['ecolor'] = color
        if 'edgecolor' in style:
            style['ecolor'] = style.pop('edgecolor')
        c = style.get('c')
        if isinstance(c, np.ndarray):
            with abbreviated_exception():
                raise ValueError('Mapping a continuous or categorical '
                                 'dimension to a color on a ErrorBarPlot '
                                 'is not supported by the {backend} backend. '
                                 'To map a dimension to a color supply '
                                 'an explicit list of rgba colors.'.format(
                                     backend=self.renderer.backend
                                 )
                )

        style['fmt'] = 'none'
        dims = element.dimensions()
        xs, ys = (element.dimension_values(i) for i in range(2))
        yerr = element.array(dimensions=dims[2:4])

        if self.invert_axes:
            coords = (ys, xs)
            err_key = 'xerr'
        else:
            coords = (xs, ys)
            err_key = 'yerr'
        style[err_key] = yerr.T if len(dims) > 3 else yerr[:, 0]
        return coords, style, {}


    def update_handles(self, key, axis, element, ranges, style):
        bottoms = self.handles['bottoms']
        tops = self.handles['tops']
        verts = self.handles['verts']

        _, style, axis_kwargs = self.get_data(element, ranges, style)
        xs, ys, neg_error = (element.dimension_values(i) for i in range(3))
        samples = len(xs)
        pos_error = element.dimension_values(3) if len(element.dimensions()) > 3 else neg_error
        if self.invert_axes:
            bxs, bys = ys - neg_error, xs
            txs, tys = ys + pos_error, xs
            new_arrays = [np.array([[bxs[i], xs[i]], [txs[i], xs[i]]])
                          for i in range(samples)]
        else:
            bxs, bys = xs, ys - neg_error
            txs, tys = xs, ys + pos_error
            new_arrays = [np.array([[xs[i], bys[i]], [xs[i], tys[i]]])
                          for i in range(samples)]
        verts.set_paths(new_arrays)
        if bottoms:
            bottoms.set_xdata(bxs)
            bottoms.set_ydata(bys)
        if tops:
            tops.set_xdata(txs)
            tops.set_ydata(tys)
        if 'ecolor' in style:
            verts.set_edgecolors(style['ecolor'])
        if 'linewidth' in style:
            verts.set_linewidths(style['linewidth'])

        return axis_kwargs



class AreaPlot(ChartPlot):

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    style_opts = ['color', 'facecolor', 'alpha', 'edgecolor', 'linewidth',
                  'hatch', 'linestyle', 'joinstyle',
                  'fill', 'capstyle', 'interpolate']

    _nonvectorized_styles = style_opts

    _plot_methods = dict(single='fill_between')

    def get_data(self, element, ranges, style):
        with abbreviated_exception():
            style = self._apply_transforms(element, ranges, style)

        xs = element.dimension_values(0)
        ys = [element.dimension_values(vdim) for vdim in element.vdims]
        return tuple([xs]+ys), style, {}

    def init_artists(self, ax, plot_data, plot_kwargs):
        fill_fn = ax.fill_betweenx if self.invert_axes else ax.fill_between
        stack = fill_fn(*plot_data, **plot_kwargs)
        return {'artist': stack}

    def get_extents(self, element, ranges, range_type='combined'):
        vdims = element.vdims[:2]
        vdim = vdims[0].name
        if len(vdims) > 1:
            new_range = {}
            for r in ranges[vdim]:
                new_range[r] = max_range([ranges[vd.name][r] for vd in vdims])
            ranges[vdim] = new_range
        else:
            s0, s1 = ranges[vdim]['soft']
            s0 = min(s0, 0) if isfinite(s0) else 0
            s1 = max(s1, 0) if isfinite(s1) else 0
            ranges[vdim]['soft'] = (s0, s1)
        return super(AreaPlot, self).get_extents(element, ranges, range_type)




class SideAreaPlot(AdjoinedPlot, AreaPlot):

    bgcolor = param.Parameter(default=(1, 1, 1, 0), doc="""
        Make plot background invisible.""")

    border_size = param.Number(default=0, doc="""
        The size of the border expressed as a fraction of the main plot.""")

    xaxis = param.ObjectSelector(default='bare',
                                 objects=['top', 'bottom', 'bare', 'top-bare',
                                          'bottom-bare', None], doc="""
        Whether and where to display the xaxis, bare options allow suppressing
        all axis labels including ticks and xlabel. Valid options are 'top',
        'bottom', 'bare', 'top-bare' and 'bottom-bare'.""")

    yaxis = param.ObjectSelector(default='bare',
                                 objects=['left', 'right', 'bare', 'left-bare',
                                          'right-bare', None], doc="""
        Whether and where to display the yaxis, bare options allow suppressing
        all axis labels including ticks and ylabel. Valid options are 'left',
        'right', 'bare' 'left-bare' and 'right-bare'.""")



class SpreadPlot(AreaPlot):
    """
    SpreadPlot plots the Spread Element type.
    """

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    def __init__(self, element, **params):
        super(SpreadPlot, self).__init__(element, **params)

    def get_data(self, element, ranges, style):
        xs = element.dimension_values(0)
        mean = element.dimension_values(1)
        neg_error = element.dimension_values(2)
        pos_idx = 3 if len(element.dimensions()) > 3 else 2
        pos_error = element.dimension_values(pos_idx)
        return (xs, mean-neg_error, mean+pos_error), style, {}

    def get_extents(self, element, ranges, range_type='combined'):
        return ChartPlot.get_extents(self, element, ranges, range_type)



class HistogramPlot(ColorbarPlot):
    """
    HistogramPlot can plot DataHistograms and ViewMaps of
    DataHistograms, which can be displayed as a single frame or
    animation.
    """

    style_opts = ['alpha', 'color', 'align', 'visible', 'facecolor',
                  'edgecolor', 'log', 'capsize', 'error_kw', 'hatch',
                  'linewidth']

    _nonvectorized_styles = ['alpha', 'log', 'error_kw', 'hatch', 'visible', 'align']

    def __init__(self, histograms, **params):
        self.center = False
        self.cyclic = False

        super(HistogramPlot, self).__init__(histograms, **params)

        if self.invert_axes:
            self.axis_settings = ['ylabel', 'xlabel', 'yticks']
        else:
            self.axis_settings = ['xlabel', 'ylabel', 'xticks']
        val_dim = self.hmap.last.get_dimension(1)
        self.cyclic_range = val_dim.range if val_dim.cyclic else None


    @mpl_rc_context
    def initialize_plot(self, ranges=None):
        hist = self.hmap.last
        key = self.keys[-1]

        ranges = self.compute_ranges(self.hmap, key, ranges)
        el_ranges = match_spec(hist, ranges)

        # Get plot ranges and values
        dims = hist.dimensions()[:2]
        edges, hvals, widths, lims, isdatetime = self._process_hist(hist)
        if isdatetime and not dims[0].value_format:
            dt_format = Dimension.type_formatters[np.datetime64]
            dims[0] = dims[0](value_format=DateFormatter(dt_format))

        style = self.style[self.cyclic_index]
        if self.invert_axes:
            self.offset_linefn = self.handles['axis'].axvline
            self.plotfn = self.handles['axis'].barh
        else:
            self.offset_linefn = self.handles['axis'].axhline
            self.plotfn = self.handles['axis'].bar

        with abbreviated_exception():
            style = self._apply_transforms(hist, ranges, style)
            if 'vmin' in style:
                raise ValueError('Mapping a continuous dimension to a '
                                 'color on a HistogramPlot is not '
                                 'supported by the {backend} backend. '
                                 'To map a dimension to a color supply '
                                 'an explicit list of rgba colors.'.format(
                                     backend=self.renderer.backend
                                 )
                )

        # Plot bars and make any adjustments
        legend = hist.label if self.show_legend else ''
        bars = self.plotfn(edges, hvals, widths, zorder=self.zorder, label=legend, align='edge', **style)
        self.handles['artist'] = self._update_plot(self.keys[-1], hist, bars, lims, ranges) # Indexing top

        ticks = self._compute_ticks(hist, edges, widths, lims)
        ax_settings = self._process_axsettings(hist, lims, ticks)
        ax_settings['dimensions'] = dims

        return self._finalize_axis(self.keys[-1], ranges=el_ranges, element=hist, **ax_settings)


    def _process_hist(self, hist):
        """
        Get data from histogram, including bin_ranges and values.
        """
        self.cyclic = hist.get_dimension(0).cyclic
        x = hist.kdims[0]
        edges = hist.interface.coords(hist, x, edges=True)
        values = hist.dimension_values(1)
        hist_vals = np.array(values)
        xlim = hist.range(0)
        ylim = hist.range(1)
        isdatetime = False
        if edges.dtype.kind == 'M' or isinstance(edges[0], datetime_types):
            edges = np.array([dt64_to_dt(e) if isinstance(e, np.datetime64) else e for e in edges])
            edges = date2num(edges)
            xlim = tuple(dt_to_int(v, 'D') for v in xlim)
            isdatetime = True
        widths = np.diff(edges)
        return edges[:-1], hist_vals, widths, xlim+ylim, isdatetime


    def _compute_ticks(self, element, edges, widths, lims):
        """
        Compute the ticks either as cyclic values in degrees or as roughly
        evenly spaced bin centers.
        """
        if self.xticks is None or not isinstance(self.xticks, int):
            return None
        if self.cyclic:
            x0, x1, _, _ = lims
            xvals = np.linspace(x0, x1, self.xticks)
            labels = ["%.0f" % np.rad2deg(x) + '\N{DEGREE SIGN}' for x in xvals]
        elif self.xticks:
            dim = element.get_dimension(0)
            inds = np.linspace(0, len(edges), self.xticks, dtype=np.int)
            edges = list(edges) + [edges[-1] + widths[-1]]
            xvals = [edges[i] for i in inds]
            labels = [dim.pprint_value(v) for v in xvals]
        return [xvals, labels]


    def get_extents(self, element, ranges, range_type='combined'):
        ydim = element.get_dimension(1)
        s0, s1 = ranges[ydim.name]['soft']
        s0 = min(s0, 0) if isfinite(s0) else 0
        s1 = max(s1, 0) if isfinite(s1) else 0
        ranges[ydim.name]['soft'] = (s0, s1)
        return super(HistogramPlot, self).get_extents(element, ranges, range_type)


    def _process_axsettings(self, hist, lims, ticks):
        """
        Get axis settings options including ticks, x- and y-labels
        and limits.
        """
        axis_settings = dict(zip(self.axis_settings, [None, None, (None if self.overlaid else ticks)]))
        return axis_settings


    def _update_plot(self, key, hist, bars, lims, ranges):
        """
        Process bars can be subclassed to manually adjust bars
        after being plotted.
        """
        return bars


    def _update_artists(self, key, hist, edges, hvals, widths, lims, ranges):
        """
        Update all the artists in the histogram. Subclassable to
        allow updating of further artists.
        """
        plot_vals = zip(self.handles['artist'], edges, hvals, widths)
        for bar, edge, height, width in plot_vals:
            if self.invert_axes:
                bar.set_y(edge)
                bar.set_width(height)
                bar.set_height(width)
            else:
                bar.set_x(edge)
                bar.set_height(height)
                bar.set_width(width)


    def update_handles(self, key, axis, element, ranges, style):
        # Process values, axes and style
        edges, hvals, widths, lims, _ = self._process_hist(element)

        ticks = self._compute_ticks(element, edges, widths, lims)
        ax_settings = self._process_axsettings(element, lims, ticks)
        self._update_artists(key, element, edges, hvals, widths, lims, ranges)
        return ax_settings



class SideHistogramPlot(AdjoinedPlot, HistogramPlot):

    bgcolor = param.Parameter(default=(1, 1, 1, 0), doc="""
        Make plot background invisible.""")

    offset = param.Number(default=0.2, bounds=(0,1), doc="""
        Histogram value offset for a colorbar.""")

    show_grid = param.Boolean(default=False, doc="""
        Whether to overlay a grid on the axis.""")

    def _process_hist(self, hist):
        """
        Subclassed to offset histogram by defined amount.
        """
        edges, hvals, widths, lims, isdatetime = super(SideHistogramPlot, self)._process_hist(hist)
        offset = self.offset * lims[3]
        hvals *= 1-self.offset
        hvals += offset
        lims = lims[0:3] + (lims[3] + offset,)
        return edges, hvals, widths, lims, isdatetime


    def _update_artists(self, n, element, edges, hvals, widths, lims, ranges):
        super(SideHistogramPlot, self)._update_artists(n, element, edges, hvals, widths, lims, ranges)
        self._update_plot(n, element, self.handles['artist'], lims, ranges)


    def _update_plot(self, key, element, bars, lims, ranges):
        """
        Process the bars and draw the offset line as necessary. If a
        color map is set in the style of the 'main' ViewableElement object, color
        the bars appropriately, respecting the required normalization
        settings.
        """
        main = self.adjoined.main
        _, y1 = element.range(1)
        offset = self.offset * y1
        range_item, main_range, dim = get_sideplot_ranges(self, element, main, ranges)

        # Check if plot is colormapped
        plot_type = Store.registry['matplotlib'].get(type(range_item))
        if isinstance(plot_type, PlotSelector):
            plot_type = plot_type.get_plot_class(range_item)
        opts = self.lookup_options(range_item, 'plot')
        if plot_type and issubclass(plot_type, ColorbarPlot):
            cidx = opts.options.get('color_index', None)
            if cidx is None:
                opts = self.lookup_options(range_item, 'style')
                cidx = opts.kwargs.get('color', None)
                if cidx not in range_item:
                    cidx = None
            cdim = None if cidx is None else range_item.get_dimension(cidx)
        else:
            cdim = None

        # Get colormapping options
        if isinstance(range_item, (HeatMap, Raster)) or cdim:
            style = self.lookup_options(range_item, 'style')[self.cyclic_index]
            cmap = cm.get_cmap(style.get('cmap'))
            main_range = style.get('clims', main_range)
        else:
            cmap = None

        if offset and ('offset_line' not in self.handles):
            self.handles['offset_line'] = self.offset_linefn(offset,
                                                             linewidth=1.0,
                                                             color='k')
        elif offset:
            self._update_separator(offset)

        if cmap is not None:
            self._colorize_bars(cmap, bars, element, main_range, dim)
        return bars


    def _colorize_bars(self, cmap, bars, element, main_range, dim):
        """
        Use the given cmap to color the bars, applying the correct
        color ranges as necessary.
        """
        cmap_range = main_range[1] - main_range[0]
        lower_bound = main_range[0]
        colors = np.array(element.dimension_values(dim))
        colors = (colors - lower_bound) / (cmap_range)
        for c, bar in zip(colors, bars):
            bar.set_facecolor(cmap(c))
            bar.set_clip_on(False)


    def _update_separator(self, offset):
        """
        Compute colorbar offset and update separator line
        if map is non-zero.
        """
        offset_line = self.handles['offset_line']
        if offset == 0:
            offset_line.set_visible(False)
        else:
            offset_line.set_visible(True)
            if self.invert_axes:
                offset_line.set_xdata(offset)
            else:
                offset_line.set_ydata(offset)


class PointPlot(ChartPlot, ColorbarPlot):
    """
    Note that the 'cmap', 'vmin' and 'vmax' style arguments control
    how point magnitudes are rendered to different colors.
    """

    show_grid = param.Boolean(default=False, doc="""
      Whether to draw grid lines at the tick positions.""")

    # Deprecated parameters

    color_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
        Deprecated in favor of color style mapping, e.g. `color=dim('color')`""")

    size_index = param.ClassSelector(default=None, class_=(basestring, int),
                                     allow_None=True, doc="""
        Deprecated in favor of size style mapping, e.g. `size=dim('size')`""")

    scaling_method = param.ObjectSelector(default="area",
                                          objects=["width", "area"],
                                          doc="""
        Deprecated in favor of size style mapping, e.g.
        size=dim('size')**2.""")

    scaling_factor = param.Number(default=1, bounds=(0, None), doc="""
      Scaling factor which is applied to either the width or area
      of each point, depending on the value of `scaling_method`.""")

    size_fn = param.Callable(default=np.abs, doc="""
      Function applied to size values before applying scaling,
      to remove values lower than zero.""")

    style_opts = ['alpha', 'color', 'edgecolors', 'facecolors',
                  'linewidth', 'marker', 'size', 'visible',
                  'cmap', 'vmin', 'vmax', 'norm']

    _nonvectorized_styles = ['alpha', 'marker', 'cmap', 'vmin', 'vmax',
                      'norm', 'visible']

    _disabled_opts = ['size']
    _plot_methods = dict(single='scatter')

    def get_data(self, element, ranges, style):
        xs, ys = (element.dimension_values(i) for i in range(2))
        self._compute_styles(element, ranges, style)
        with abbreviated_exception():
            style = self._apply_transforms(element, ranges, style)
        return (ys, xs) if self.invert_axes else (xs, ys), style, {}


    def _compute_styles(self, element, ranges, style):
        cdim = element.get_dimension(self.color_index)
        color = style.pop('color', None)
        cmap = style.get('cmap', None)

        if cdim and ((isinstance(color, basestring) and color in element) or isinstance(color, dim)):
            self.param.warning(
                "Cannot declare style mapping for 'color' option and "
                "declare a color_index; ignoring the color_index.")
            cdim = None
        if cdim and cmap:
            cs = element.dimension_values(self.color_index)
            # Check if numeric otherwise treat as categorical
            if cs.dtype.kind in 'uif':
                style['c'] = cs
            else:
                style['c'] = search_indices(cs, unique_array(cs))
            self._norm_kwargs(element, ranges, style, cdim)
        elif color is not None:
            style['color'] = color
        style['edgecolors'] = style.pop('edgecolors', style.pop('edgecolor', 'none'))

        ms = style.get('s', mpl.rcParams['lines.markersize'])
        sdim = element.get_dimension(self.size_index)
        if sdim and ((isinstance(ms, basestring) and ms in element) or isinstance(ms, dim)):
            self.param.warning(
                "Cannot declare style mapping for 's' option and "
                "declare a size_index; ignoring the size_index.")
            sdim = None
        if sdim:
            sizes = element.dimension_values(self.size_index)
            sizes = compute_sizes(sizes, self.size_fn, self.scaling_factor,
                                  self.scaling_method, ms)
            if sizes is None:
                eltype = type(element).__name__
                self.param.warning(
                    '%s dimension is not numeric, cannot use to '
                    'scale %s size.' % (sdim.pprint_label, eltype))
            else:
                style['s'] = sizes
        style['edgecolors'] = style.pop('edgecolors', 'none')


    def update_handles(self, key, axis, element, ranges, style):
        paths = self.handles['artist']
        (xs, ys), style, _ = self.get_data(element, ranges, style)
        paths.set_offsets(np.column_stack([xs, ys]))
        if 's' in style:
            sizes = style['s']
            if isscalar(sizes):
                sizes = [sizes]
            paths.set_sizes(sizes)
        if 'vmin' in style:
            paths.set_clim((style['vmin'], style['vmax']))
        if 'c' in style:
            paths.set_array(style['c'])
        if 'norm' in style:
            paths.norm = style['norm']
        if 'linewidth' in style:
            paths.set_linewidths(style['linewidth'])
        if 'edgecolors' in style:
            paths.set_edgecolors(style['edgecolors'])
        if 'facecolors' in style:
            paths.set_edgecolors(style['facecolors'])



class VectorFieldPlot(ColorbarPlot):
    """
    Renders vector fields in sheet coordinates. The vectors are
    expressed in polar coordinates and may be displayed according to
    angle alone (with some common, arbitrary arrow length) or may be
    true polar vectors.

    The color or magnitude can be mapped onto any dimension using the
    color_index and size_index.

    The length of the arrows is controlled by the 'scale' style
    option. The scaling of the arrows may also be controlled via the
    normalize_lengths and rescale_lengths plot option, which will
    normalize the lengths to a maximum of 1 and scale them according
    to the minimum distance respectively.
    """

    arrow_heads = param.Boolean(default=True, doc="""
       Whether or not to draw arrow heads. If arrowheads are enabled,
       they may be customized with the 'headlength' and
       'headaxislength' style options.""")

    magnitude = param.ClassSelector(class_=(basestring, dim), doc="""
        Dimension or dimension value transform that declares the magnitude
        of each vector. Magnitude is expected to be scaled between 0-1,
        by default the magnitudes are rescaled relative to the minimum
        distance between vectors, this can be disabled with the
        rescale_lengths option.""")

    rescale_lengths = param.Boolean(default=True, doc="""
       Whether the lengths will be rescaled to take into account the
       smallest non-zero distance between two vectors.""")

    # Deprecated parameters

    color_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
        Deprecated in favor of dimension value transform on color option,
        e.g. `color=dim('Magnitude')`.
        """)

    size_index = param.ClassSelector(default=None, class_=(basestring, int),
                                     allow_None=True, doc="""
        Deprecated in favor of the magnitude option, e.g.
        `magnitude=dim('Magnitude')`.
        """)

    normalize_lengths = param.Boolean(default=True, doc="""
        Deprecated in favor of rescaling length using dimension value
        transforms using the magnitude option, e.g.
        `dim('Magnitude').norm()`.""")

    style_opts = ['alpha', 'color', 'edgecolors', 'facecolors',
                  'linewidth', 'marker', 'visible', 'cmap',
                  'scale', 'headlength', 'headaxislength', 'pivot',
                  'width', 'headwidth', 'norm']

    _nonvectorized_styles = ['alpha', 'marker', 'cmap', 'visible', 'norm',
                             'pivot', 'headlength', 'headaxislength',
                             'headwidth']

    _plot_methods = dict(single='quiver')

    def _get_magnitudes(self, element, style, ranges):
        size_dim = element.get_dimension(self.size_index)
        mag_dim = self.magnitude
        if size_dim and mag_dim:
            self.param.warning(
                "Cannot declare style mapping for 'magnitude' option "
                "and declare a size_index; ignoring the size_index.")
        elif size_dim:
            mag_dim = size_dim
        elif isinstance(mag_dim, basestring):
            mag_dim = element.get_dimension(mag_dim)
        if mag_dim is not None:
            if isinstance(mag_dim, dim):
                magnitudes = mag_dim.apply(element, flat=True)
            else:
                magnitudes = element.dimension_values(mag_dim)
                _, max_magnitude = ranges[dimension_name(mag_dim)]['combined']
                if self.normalize_lengths and max_magnitude != 0:
                    magnitudes = magnitudes / max_magnitude
        else:
            magnitudes = np.ones(len(element))
        return magnitudes

    def get_data(self, element, ranges, style):
        # Compute coordinates
        xidx, yidx = (1, 0) if self.invert_axes else (0, 1)
        xs = element.dimension_values(xidx) if len(element.data) else []
        ys = element.dimension_values(yidx) if len(element.data) else []

        # Compute vector angle and magnitude
        radians = element.dimension_values(2) if len(element.data) else []
        if self.invert_axes: radians = radians+1.5*np.pi
        angles = list(np.rad2deg(radians))
        magnitudes = self._get_magnitudes(element, style, ranges)
        input_scale = style.pop('scale', 1.0)
        if self.rescale_lengths:
            min_dist = get_min_distance(element)
            input_scale = input_scale / min_dist

        args = (xs, ys, magnitudes,  [0.0] * len(element))

        # Compute color
        cdim = element.get_dimension(self.color_index)
        color = style.get('color', None)
        if cdim and ((isinstance(color, basestring) and color in element) or isinstance(color, dim)):
            self.param.warning(
                "Cannot declare style mapping for 'color' option and "
                "declare a color_index; ignoring the color_index.")
            cdim = None
        if cdim:
            colors = element.dimension_values(self.color_index)
            style['c'] = colors
            cdim = element.get_dimension(self.color_index)
            self._norm_kwargs(element, ranges, style, cdim)
            style.pop('color', None)

        # Process style
        with abbreviated_exception():
            style = self._apply_transforms(element, ranges, style)
        style.update(dict(scale=input_scale, angles=angles, units='x', scale_units='x'))
        if 'vmin' in style:
            style['clim'] = (style.pop('vmin'), style.pop('vmax'))
        if 'c' in style:
            style['array'] = style.pop('c')
        if 'pivot' not in style:
            style['pivot'] = 'mid'
        if not self.arrow_heads:
            style['headaxislength'] = 0

        return args, style, {}

    def update_handles(self, key, axis, element, ranges, style):
        args, style, axis_kwargs = self.get_data(element, ranges, style)

        # Set magnitudes, angles and colors if supplied.
        quiver = self.handles['artist']
        quiver.set_offsets(np.column_stack(args[:2]))
        quiver.U = args[2]
        quiver.angles = style['angles']
        if 'color' in style:
            quiver.set_facecolors(style['color'])
            quiver.set_edgecolors(style['color'])
        if 'array' in style:
            quiver.set_array(style['array'])
        if 'clim' in style:
            quiver.set_clim(style['clim'])
        if 'linewidth' in style:
            quiver.set_linewidths(style['linewidth'])
        return axis_kwargs



class BarPlot(LegendPlot):

    padding = param.Number(default=0.2, doc="""
       Defines the padding between groups.""")

    show_legend = param.Boolean(default=True, doc="""
        Whether to show legend for the plot.""")

    stacked = param.Boolean(default=False, doc="""
       Whether the bars should be stacked or grouped.""")

    xticks = param.Integer(0, precedence=-1)

    # Deprecated parameters

    color_by = param.List(default=['category'], doc="""
       Defines how the Bar elements colored. Valid options include
       any permutation of 'group', 'category' and 'stack'.""")

    group_index = param.Integer(default=0, doc="""
       Index of the dimension in the supplied Bars
       Element, which will be laid out into groups.""")

    category_index = param.Integer(default=1, doc="""
       Index of the dimension in the supplied Bars
       Element, which will be laid out into categories.""")

    stack_index = param.Integer(default=2, doc="""
       Index of the dimension in the supplied Bars
       Element, which will stacked.""")

    style_opts = ['alpha', 'color', 'align', 'visible', 'edgecolor',
                  'log', 'facecolor', 'capsize', 'error_kw', 'hatch']

    _nonvectorized_styles = style_opts

    legend_specs = dict(LegendPlot.legend_specs, **{
        'top':    dict(bbox_to_anchor=(0., 1.02, 1., .102),
                       ncol=3, loc=3, mode="expand", borderaxespad=0.),
        'bottom': dict(ncol=3, mode="expand", loc=2,
                       bbox_to_anchor=(0., -0.4, 1., .102),
                       borderaxespad=0.1)})

    _dimensions = OrderedDict([('group', 0),
                               ('category',1),
                               ('stack',2)])

    def __init__(self, element, **params):
        super(BarPlot, self).__init__(element, **params)
        self.values, self.bar_dimensions = self._get_values()

    def _get_values(self):
        """
        Get unique index value for each bar
        """
        (gi, _), (ci, _), (si, _) = self._get_dims(self.hmap.last)
        ndims = self.hmap.last.ndims
        dims = self.hmap.last.kdims
        dimensions = []
        values = {}
        for vidx, vtype in zip([gi, ci, si], self._dimensions):
            if vidx < ndims:
                dim = dims[vidx]
                dimensions.append(dim)
                vals = self.hmap.dimension_values(dim.name)
            else:
                dimensions.append(None)
                vals = [None]
            values[vtype] = list(unique_iterator(vals))
        return values, dimensions


    def _compute_styles(self, element, style_groups):
        """
        Computes color and hatch combinations by
        any combination of the 'group', 'category'
        and 'stack'.
        """
        style = self.lookup_options(element, 'style')[0]
        sopts = []
        for sopt in ['color', 'hatch']:
            if sopt in style:
                sopts.append(sopt)
                style.pop(sopt, None)
        color_groups = []
        for sg in style_groups:
            color_groups.append(self.values[sg])
        style_product = list(product(*color_groups))
        wrapped_style = self.lookup_options(element, 'style').max_cycles(len(style_product))
        color_groups = {k:tuple(wrapped_style[n][sopt] for sopt in sopts)
                        for n,k in enumerate(style_product)}

        return style, color_groups, sopts


    def get_extents(self, element, ranges, range_type='combined'):
        ngroups = len(self.values['group'])
        vdim = element.vdims[0].name
        if self.stacked or self.stack_index == 1:
            return 0, 0, ngroups, np.NaN
        else:
            vrange = ranges[vdim]['combined']
            return 0, np.nanmin([vrange[0], 0]), ngroups, vrange[1]


    @mpl_rc_context
    def initialize_plot(self, ranges=None):
        element = self.hmap.last
        vdim = element.vdims[0]
        axis = self.handles['axis']
        key = self.keys[-1]

        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = match_spec(element, ranges)

        self.handles['artist'], self.handles['xticks'], xdims = self._create_bars(axis, element)
        return self._finalize_axis(key, ranges=ranges, xticks=self.handles['xticks'],
                                   element=element, dimensions=[xdims, vdim])


    def _finalize_ticks(self, axis, element, xticks, yticks, zticks):
        """
        Apply ticks with appropriate offsets.
        """
        yalignments = None
        if xticks is not None:
            ticks, labels, yalignments = zip(*sorted(xticks, key=lambda x: x[0]))
            xticks = (list(ticks), list(labels))
        super(BarPlot, self)._finalize_ticks(axis, element, xticks, yticks, zticks)
        if yalignments:
            for t, y in zip(axis.get_xticklabels(), yalignments):
                t.set_y(y)


    def _get_dims(self, element):
        ndims = len(element.dimensions())
        if element.ndims < 2:
            gdim, cdim, sdim = element.kdims[0], None, None
            gi, ci, si = 0, ndims+1, ndims+1
        elif element.ndims == 3:
            gdim, cdim, sdim = element.kdims
            gi, ci, si = 0, 1, 2
        elif self.stacked or self.stack_index == 1:
            gdim, cdim, sdim = element.kdims[0], None, element.kdims[1]
            gi, ci, si = 0, ndims+1, 1
        else:
            gdim, cdim, sdim = element.kdims[0], element.kdims[1], None
            gi, ci, si = 0, 1, ndims+1
        return (gi, gdim), (ci, cdim), (si, sdim)


    def _create_bars(self, axis, element):
        # Get style and dimension information
        values = self.values    
        if self.group_index != 0:
            self.warning('Bars group_index plot option is deprecated '
                         'and will be ignored, set stacked=True/False '
                         'instead.')
        if self.category_index != 1:
            self.warning('Bars category_index plot option is deprecated '
                         'and will be ignored, set stacked=True/False '
                         'instead.')
        if self.stack_index != 2 and not (self.stack_index == 1 and not self.stacked):
            self.warning('Bars stack_index plot option is deprecated '
                         'and will be ignored, set stacked=True/False '
                         'instead.')
        if self.color_by != ['category']:
            self.warning('Bars color_by plot option is deprecated '
                         'and will be ignored, in future it will '
                         'support color style mapping by dimension.')

        (gi, gdim), (ci, cdim), (si, sdim) = self._get_dims(element)
        indices = dict(zip(self._dimensions, (gi, ci, si)))
        color_by = ['category'] if cdim else ['stack']
        style_groups = [sg for sg in color_by if indices[sg] < element.ndims]
        style_opts, color_groups, sopts = self._compute_styles(element, style_groups)
        dims = element.dimensions('key', label=True)
        ndims = len(dims)
        xdims = [d for d in [cdim, gdim] if d is not None]

        # Compute widths
        width = (1-(2.*self.padding)) / len(values['category'])

        # Initialize variables
        xticks = []
        val_key = [None] * ndims
        style_key = [None] * len(style_groups)
        label_key = [None] * len(style_groups)
        labels = []
        bars = {}

        # Iterate over group, category and stack dimension values
        # computing xticks and drawing bars and applying styles
        for gidx, grp_name in enumerate(values['group']):
            if grp_name is not None:
                grp = gdim.pprint_value(grp_name)
                if 'group' in style_groups:
                    idx = style_groups.index('group')
                    label_key[idx] = str(grp)
                    style_key[idx] = grp_name
                val_key[gi] = grp_name
                if ci < ndims:
                    yalign = -0.04
                else:
                    yalign = 0
                xticks.append((gidx+0.5, grp, yalign))
            for cidx, cat_name in enumerate(values['category']):
                xpos = gidx+self.padding+(cidx*width)
                if cat_name is not None:
                    cat = gdim.pprint_value(cat_name)
                    if 'category' in style_groups:
                        idx = style_groups.index('category')
                        label_key[idx] = str(cat)
                        style_key[idx] = cat_name
                    val_key[ci] = cat_name
                    xticks.append((xpos+width/2., cat, 0))
                prev = 0
                for stk_name in values['stack']:
                    if stk_name is not None:
                        if 'stack' in style_groups:
                            idx = style_groups.index('stack')
                            stk = gdim.pprint_value(stk_name)
                            label_key[idx] = str(stk)
                            style_key[idx] = stk_name
                        val_key[si] = stk_name
                    vals = element.sample([tuple(val_key)]).dimension_values(element.vdims[0].name)
                    val = float(vals[0]) if len(vals) else np.NaN
                    label = ', '.join(label_key)
                    style = dict(style_opts, label='' if label in labels else label,
                                 **dict(zip(sopts, color_groups[tuple(style_key)])))
                    with abbreviated_exception():
                        style = self._apply_transforms(element, {}, style)
                    bar = axis.bar([xpos+width/2.], [val], width=width, bottom=prev,
                                   **style)

                    # Update variables
                    bars[tuple(val_key)] = bar
                    prev += val if isfinite(val) else 0
                    labels.append(label)
        title = [element.kdims[indices[cg]].pprint_label
                 for cg in color_by if indices[cg] < ndims]

        if self.show_legend and any(len(l) for l in labels) and color_by != ['category']:
            leg_spec = self.legend_specs[self.legend_position]
            if self.legend_cols: leg_spec['ncol'] = self.legend_cols
            axis.legend(title=', '.join(title), **leg_spec)
        return bars, xticks, xdims


    def update_handles(self, key, axis, element, ranges, style):
        dims = element.dimensions('key', label=True)
        ndims = len(dims)
        (gi, _), (ci, _), (si, _) = self._get_dims(element)
        val_key = [None] * ndims
        for g in self.values['group']:
            if g is not None: val_key[gi] = g
            for c in self.values['category']:
                if c is not None: val_key[ci] = c
                prev = 0
                for s in self.values['stack']:
                    if s is not None: val_key[si] = s
                    bar = self.handles['artist'].get(tuple(val_key))
                    if bar:
                        vals = element.sample([tuple(val_key)]).dimension_values(element.vdims[0].name)
                        height = float(vals[0]) if len(vals) else np.NaN
                        bar[0].set_height(height)
                        bar[0].set_y(prev)
                        prev += height if isfinite(height) else 0
        return {'xticks': self.handles['xticks']}


class SpikesPlot(PathPlot, ColorbarPlot):

    aspect = param.Parameter(default='square', doc="""
        The aspect ratio mode of the plot. Allows setting an
        explicit aspect ratio as width/height as well as
        'square' and 'equal' options.""")

    color_index = param.ClassSelector(default=None, allow_None=True,
                                      class_=(basestring, int), doc="""
      Index of the dimension from which the color will the drawn""")

    spike_length = param.Number(default=0.1, doc="""
      The length of each spike if Spikes object is one dimensional.""")

    position = param.Number(default=0., doc="""
      The position of the lower end of each spike.""")

    style_opts = PathPlot.style_opts + ['cmap']

    def init_artists(self, ax, plot_args, plot_kwargs):
        if 'c' in plot_kwargs:
            plot_kwargs['array'] = plot_kwargs.pop('c')
        if 'vmin' in plot_kwargs and 'vmax' in plot_kwargs:
            plot_kwargs['clim'] = plot_kwargs.pop('vmin'), plot_kwargs.pop('vmax')
        line_segments = LineCollection(*plot_args, **plot_kwargs)
        ax.add_collection(line_segments)
        return {'artist': line_segments}

    def get_extents(self, element, ranges, range_type='combined'):
        if len(element.dimensions()) > 1:
            ydim = element.get_dimension(1)
            s0, s1 = ranges[ydim.name]['soft']
            s0 = min(s0, 0) if isfinite(s0) else 0
            s1 = max(s1, 0) if isfinite(s1) else 0
            ranges[ydim.name]['soft'] = (s0, s1)
        l, b, r, t = super(SpikesPlot, self).get_extents(element, ranges, range_type)
        if len(element.dimensions()) == 1 and range_type != 'hard':
            if self.batched:
                bs, ts = [], []
                # Iterate over current NdOverlay and compute extents
                # from position and length plot options
                frame = self.current_frame or self.hmap.last
                for el in frame.values():
                    opts = self.lookup_options(el, 'plot').options
                    pos = opts.get('position', self.position)
                    length = opts.get('spike_length', self.spike_length)
                    bs.append(pos)
                    ts.append(pos+length)
                b, t = (np.nanmin(bs), np.nanmax(ts))
            else:
                b, t = self.position, self.position+self.spike_length
        return l, b, r, t

    def get_data(self, element, ranges, style):
        dimensions = element.dimensions(label=True)
        ndims = len(dimensions)

        pos = self.position
        if ndims > 1:
            data = [[(x, pos), (x, pos+y)] for x, y in element.array([0, 1])]
        else:
            height = self.spike_length
            data = [[(x[0], pos), (x[0], pos+height)] for x in element.array([0])]

        if self.invert_axes:
            data = [(line[0][::-1], line[1][::-1]) for line in data]

        dims = element.dimensions()
        clean_spikes = []
        for spike in data:
            xs, ys = zip(*spike)
            cols = []
            for i, vs in enumerate((xs, ys)):
                vs = np.array(vs)
                if (vs.dtype.kind == 'M' or (len(vs) and isinstance(vs[0], datetime_types))) and i < len(dims):
                    dt_format = Dimension.type_formatters[np.datetime64]
                    dims[i] = dims[i](value_format=DateFormatter(dt_format))
                    vs = np.array([dt_to_int(v, 'D') for v in vs])
                cols.append(vs)
            clean_spikes.append(np.column_stack(cols))

        cdim = element.get_dimension(self.color_index)
        color = style.get('color', None)
        if cdim and ((isinstance(color, basestring) and color in element) or isinstance(color, dim)):
            self.param.warning(
                "Cannot declare style mapping for 'color' option and "
                "declare a color_index; ignoring the color_index.")
            cdim = None
        if cdim:
            style['array'] = element.dimension_values(cdim)
            self._norm_kwargs(element, ranges, style, cdim)

        with abbreviated_exception():
            style = self._apply_transforms(element, ranges, style)
        return (clean_spikes,), style, {'dimensions': dims}


    def update_handles(self, key, axis, element, ranges, style):
        artist = self.handles['artist']
        (data,), kwargs, axis_kwargs = self.get_data(element, ranges, style)
        artist.set_paths(data)
        artist.set_visible(style.get('visible', True))
        if 'color' in kwargs:
            artist.set_edgecolors(kwargs['color'])
        if 'array' in kwargs or 'c' in kwargs:
            artist.set_array(kwargs.get('array', kwargs.get('c')))
        if 'vmin' in kwargs:
            artist.set_clim((kwargs['vmin'], kwargs['vmax']))
        if 'norm' in kwargs:
            artist.norm = kwargs['norm']
        if 'linewidth' in kwargs:
            artist.set_linewidths(kwargs['linewidth'])
        return axis_kwargs


class SideSpikesPlot(AdjoinedPlot, SpikesPlot):

    bgcolor = param.Parameter(default=(1, 1, 1, 0), doc="""
        Make plot background invisible.""")

    border_size = param.Number(default=0, doc="""
        The size of the border expressed as a fraction of the main plot.""")

    subplot_size = param.Number(default=0.1, doc="""
        The size subplots as expressed as a fraction of the main plot.""")

    spike_length = param.Number(default=1, doc="""
      The length of each spike if Spikes object is one dimensional.""")

    xaxis = param.ObjectSelector(default='bare',
                                 objects=['top', 'bottom', 'bare', 'top-bare',
                                          'bottom-bare', None], doc="""
        Whether and where to display the xaxis, bare options allow suppressing
        all axis labels including ticks and xlabel. Valid options are 'top',
        'bottom', 'bare', 'top-bare' and 'bottom-bare'.""")

    yaxis = param.ObjectSelector(default='bare',
                                      objects=['left', 'right', 'bare', 'left-bare',
                                               'right-bare', None], doc="""
        Whether and where to display the yaxis, bare options allow suppressing
        all axis labels including ticks and ylabel. Valid options are 'left',
        'right', 'bare' 'left-bare' and 'right-bare'.""")

