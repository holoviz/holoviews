from __future__ import unicode_literals
from itertools import product

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

import param

from ...core import OrderedDict
from ...core.util import (match_spec, unique_iterator, safe_unicode,
                          basestring, max_range, unicode)
from ...element import Points, Raster, Polygons, HeatMap
from ..util import compute_sizes, get_sideplot_ranges
from .element import ElementPlot, ColorbarPlot, LegendPlot
from .path  import PathPlot
from .plot import AdjoinedPlot


class ChartPlot(ElementPlot):

    show_legend = param.Boolean(default=True, doc="""
        Whether to show legend for the plot.""")


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

    relative_labels = param.Boolean(default=False, doc="""
        If plotted quantity is cyclic and center_cyclic is enabled,
        will compute tick labels relative to the center.""")

    show_frame = param.Boolean(default=False, doc="""
        Disabled by default for clarity.""")

    show_grid = param.Boolean(default=True, doc="""
        Enable axis grid.""")

    show_legend = param.Boolean(default=True, doc="""
        Whether to show legend for the plot.""")

    style_opts = ['alpha', 'color', 'visible', 'linewidth', 'linestyle', 'marker']

    def init_artists(self, ax, plot_data, plot_kwargs):
        return {'artist': ax.plot(*plot_data, **plot_kwargs)[0]}


    def get_data(self, element, ranges, style):
        xs = element.dimension_values(0)
        ys = element.dimension_values(1)
        return (xs, ys), style, {}


    def update_handles(self, key, axis, element, ranges, style):
        artist = self.handles['artist']
        (xs, ys), style, axis_kwargs = self.get_data(element, ranges, style)
        artist.set_xdata(xs)
        artist.set_ydata(ys)
        return axis_kwargs



class ErrorPlot(ChartPlot):
    """
    ErrorPlot plots the ErrorBar Element type and supporting
    both horizontal and vertical error bars via the 'horizontal'
    plot option.
    """

    style_opts = ['ecolor', 'elinewidth', 'capsize', 'capthick',
                  'barsabove', 'lolims', 'uplims', 'xlolims',
                  'errorevery', 'xuplims', 'alpha', 'linestyle',
                  'linewidth', 'markeredgecolor', 'markeredgewidth',
                  'markerfacecolor', 'markersize', 'solid_capstyle',
                  'solid_joinstyle', 'dashes', 'color']

    def init_artists(self, ax, plot_data, plot_kwargs):
        _, (bottoms, tops), verts = ax.errorbar(*plot_data, **plot_kwargs)
        return {'bottoms': bottoms, 'tops': tops, 'verts': verts[0]}


    def get_data(self, element, ranges, style):
        style['fmt'] = 'none'
        dims = element.dimensions()
        xs, ys = (element.dimension_values(i) for i in range(2))
        yerr = element.array(dimensions=dims[2:4])
        style['yerr'] = yerr.T if len(dims) > 3 else yerr
        return (xs, ys), style, {}


    def update_handles(self, key, axis, element, ranges, style):
        bottoms = self.handles['bottoms']
        tops = self.handles['tops']
        verts = self.handles['verts']
        paths = verts.get_paths()

        (xs, ys), style, axis_kwargs = self.get_data(element, ranges, style)

        neg_error = element.dimension_values(2)
        pos_error = element.dimension_values(3) if len(element.dimensions()) > 3 else neg_error
        if self.invert_axes:
            bdata = xs - neg_error
            tdata = xs + pos_error
            tops.set_xdata(bdata)
            tops.set_ydata(ys)
            bottoms.set_xdata(tdata)
            bottoms.set_ydata(ys)
            for i, path in enumerate(paths):
                path.vertices = np.array([[bdata[i], ys[i]],
                                          [tdata[i], ys[i]]])
        else:
            bdata = ys - neg_error
            tdata = ys + pos_error
            bottoms.set_xdata(xs)
            bottoms.set_ydata(bdata)
            tops.set_xdata(xs)
            tops.set_ydata(tdata)
            for i, path in enumerate(paths):
                path.vertices = np.array([[xs[i], bdata[i]],
                                          [xs[i], tdata[i]]])
        return axis_kwargs


class AreaPlot(ChartPlot):

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    style_opts = ['color', 'facecolor', 'alpha', 'edgecolor', 'linewidth',
                  'hatch', 'linestyle', 'joinstyle',
                  'fill', 'capstyle', 'interpolate']

    def get_data(self, element, ranges, style):
        xs = element.dimension_values(0)
        ys = [element.dimension_values(vdim) for vdim in element.vdims]
        return tuple([xs]+ys), style, {}

    def init_artists(self, ax, plot_data, plot_kwargs):
        fill_fn = ax.fill_betweenx if self.invert_axes else ax.fill_between
        stack = fill_fn(*plot_data, **plot_kwargs)
        return {'artist': stack}

    def get_extents(self, element, ranges):
        vdims = element.vdims
        vdim = vdims[0].name
        ranges[vdim] = max_range([ranges[vd.name] for vd in vdims])
        return super(AreaPlot, self).get_extents(element, ranges)




class SpreadPlot(AreaPlot):
    """
    SpreadPlot plots the Spread Element type.
    """

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    def __init__(self, element, **params):
        super(SpreadPlot, self).__init__(element, **params)
        self._extents = None

    def get_data(self, element, ranges, style):
        xs = element.dimension_values(0)
        mean = element.dimension_values(1)
        neg_error = element.dimension_values(2)
        pos_idx = 3 if len(element.dimensions()) > 3 else 2
        pos_error = element.dimension_values(pos_idx)
        return (xs, mean-neg_error, mean+pos_error), style, {}


class HistogramPlot(ChartPlot):
    """
    HistogramPlot can plot DataHistograms and ViewMaps of
    DataHistograms, which can be displayed as a single frame or
    animation.
    """

    show_frame = param.Boolean(default=False, doc="""
        Disabled by default for clarity.""")

    show_grid = param.Boolean(default=False, doc="""
        Whether to overlay a grid on the axis.""")

    style_opts = ['alpha', 'color', 'align', 'visible', 'facecolor',
                  'edgecolor', 'log', 'capsize', 'error_kw', 'hatch']

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


    def initialize_plot(self, ranges=None):
        hist = self.hmap.last
        key = self.keys[-1]

        ranges = self.compute_ranges(self.hmap, key, ranges)
        el_ranges = match_spec(hist, ranges)

        # Get plot ranges and values
        edges, hvals, widths, lims = self._process_hist(hist)

        if self.invert_axes:
            self.offset_linefn = self.handles['axis'].axvline
            self.plotfn = self.handles['axis'].barh
        else:
            self.offset_linefn = self.handles['axis'].axhline
            self.plotfn = self.handles['axis'].bar

        # Plot bars and make any adjustments
        style = self.style[self.cyclic_index]
        legend = hist.label if self.show_legend else ''
        bars = self.plotfn(edges, hvals, widths, zorder=self.zorder, label=legend, **style)
        self.handles['artist'] = self._update_plot(self.keys[-1], hist, bars, lims, ranges) # Indexing top

        ticks = self._compute_ticks(hist, edges, widths, lims)
        ax_settings = self._process_axsettings(hist, lims, ticks)

        return self._finalize_axis(self.keys[-1], ranges=el_ranges, **ax_settings)


    def _process_hist(self, hist):
        """
        Get data from histogram, including bin_ranges and values.
        """
        self.cyclic = hist.get_dimension(0).cyclic
        edges = hist.edges[:-1]
        hist_vals = np.array(hist.values)
        widths = [hist._width] * len(hist) if getattr(hist, '_width', None) else np.diff(hist.edges)
        lims = hist.range(0) + hist.range(1)
        return edges, hist_vals, widths, lims


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


    def get_extents(self, element, ranges):
        x0, y0, x1, y1 = super(HistogramPlot, self).get_extents(element, ranges)
        y0 = np.nanmin([0, y0])
        return (x0, y0, x1, y1)


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
        edges, hvals, widths, lims = self._process_hist(element)

        ticks = self._compute_ticks(element, edges, widths, lims)
        ax_settings = self._process_axsettings(element, lims, ticks)
        self._update_artists(key, element, edges, hvals, widths, lims, ranges)
        return ax_settings



class SideHistogramPlot(AdjoinedPlot, HistogramPlot):

    bgcolor = param.Parameter(default=(1, 1, 1, 0), doc="""
        Make plot background invisible.""")

    offset = param.Number(default=0.2, bounds=(0,1), doc="""
        Histogram value offset for a colorbar.""")

    show_grid = param.Boolean(default=True, doc="""
        Whether to overlay a grid on the axis.""")

    def _process_hist(self, hist):
        """
        Subclassed to offset histogram by defined amount.
        """
        edges, hvals, widths, lims = super(SideHistogramPlot, self)._process_hist(hist)
        offset = self.offset * lims[3]
        hvals *= 1-self.offset
        hvals += offset
        lims = lims[0:3] + (lims[3] + offset,)
        return edges, hvals, widths, lims


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
        if isinstance(range_item, (Raster, Points, Polygons, HeatMap)):
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


    def get_extents(self, element, ranges):
        x0, _, x1, _ = element.extents
        _, y1 = element.range(1)
        return (x0, 0, x1, y1)


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

    color_index = param.ClassSelector(default=3, class_=(basestring, int),
                                  allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    size_index = param.ClassSelector(default=2, class_=(basestring, int),
                                 allow_None=True, doc="""
      Index of the dimension from which the sizes will the drawn.""")

    scaling_method = param.ObjectSelector(default="area",
                                          objects=["width", "area"],
                                          doc="""
      Determines whether the `scaling_factor` should be applied to
      the width or area of each point (default: "area").""")

    scaling_factor = param.Number(default=1, bounds=(0, None), doc="""
      Scaling factor which is applied to either the width or area
      of each point, depending on the value of `scaling_method`.""")

    show_grid = param.Boolean(default=True, doc="""
      Whether to draw grid lines at the tick positions.""")

    size_fn = param.Callable(default=np.abs, doc="""
      Function applied to size values before applying scaling,
      to remove values lower than zero.""")

    style_opts = ['alpha', 'color', 'edgecolors', 'facecolors',
                  'linewidth', 'marker', 'size', 'visible',
                  'cmap', 'vmin', 'vmax']

    _disabled_opts = ['size']

    def init_artists(self, ax, plot_args, plot_kwargs):
        return {'artist': ax.scatter(*plot_args, **plot_kwargs)}


    def get_data(self, element, ranges, style):
        xs, ys = (element.dimension_values(i) for i in range(2))
        self._compute_styles(element, ranges, style)
        return (xs, ys), style, {}


    def _compute_styles(self, element, ranges, style):
        cdim = element.get_dimension(self.color_index)
        color = style.pop('color', None)
        if cdim:
            cs = element.dimension_values(self.color_index)
            style['c'] = cs
            self._norm_kwargs(element, ranges, style, cdim)
        elif color:
            style['c'] = color
        style['edgecolors'] = style.pop('edgecolors', style.pop('edgecolor', 'none'))

        if element.get_dimension(self.size_index):
            sizes = element.dimension_values(self.size_index)
            ms = style.pop('s') if 's' in style else plt.rcParams['lines.markersize']
            style['s'] = compute_sizes(sizes, self.size_fn, self.scaling_factor,
                                       self.scaling_method, ms)
        style['edgecolors'] = style.pop('edgecolors', 'none')


    def update_handles(self, key, axis, element, ranges, style):
        paths = self.handles['artist']
        (xs, ys), style, _ = self.get_data(element, ranges, style)
        paths.set_offsets(np.column_stack([xs, ys]))
        sdim = element.get_dimension(self.size_index)
        if sdim:
            paths.set_sizes(style['s'])

        cdim = element.get_dimension(self.color_index)
        if cdim:
            paths.set_clim((style['vmin'], style['vmax']))
            paths.set_array(style['c'])
            if 'norm' in style:
                paths.norm = style['norm']



class VectorFieldPlot(ElementPlot):
    """
    Renders vector fields in sheet coordinates. The vectors are
    expressed in polar coordinates and may be displayed according to
    angle alone (with some common, arbitrary arrow length) or may be
    true polar vectors.

    Optionally, the arrows may be colored but this dimension is
    redundant with either the specified angle or magnitudes. This
    choice is made by setting the color_dim parameter.

    Note that the 'cmap' style argument controls the color map used to
    color the arrows. The length of the arrows is controlled by the
    'scale' style option where a value of 1.0 is such that the largest
    arrow shown is no bigger than the smallest sampling distance.
    """

    color_dim = param.ObjectSelector(default=None,
                                     objects=['angle', 'magnitude', None], doc="""
       Which of the polar vector components is mapped to the color
       dimension (if any), valid values are 'angle' and 'magnitude'.""")

    arrow_heads = param.Boolean(default=True, doc="""
       Whether or not to draw arrow heads. If arrowheads are enabled,
       they may be customized with the 'headlength' and
       'headaxislength' style options.""")

    normalize_lengths = param.Boolean(default=True, doc="""
       Whether to normalize vector magnitudes automatically. If False,
       it will be assumed that the lengths have already been correctly
       normalized.""")

    style_opts = ['alpha', 'color', 'edgecolors', 'facecolors',
                  'linewidth', 'marker', 'visible', 'cmap',
                  'scale', 'headlength', 'headaxislength', 'pivot']

    def __init__(self, *args, **params):
        super(VectorFieldPlot, self).__init__(*args, **params)
        self._min_dist = self._get_map_info(self.hmap)


    def _get_map_info(self, vmap):
        """
        Get the minimum sample distance and maximum magnitude
        """
        return np.min([self._get_min_dist(vfield) for vfield in vmap])


    def _get_min_dist(self, vfield):
        "Get the minimum sampling distance."
        xys = vfield.array([0, 1]).view(dtype=np.complex128)
        m, n = np.meshgrid(xys, xys)
        distances = np.abs(m-n)
        np.fill_diagonal(distances, np.inf)
        return distances.min()


    def get_data(self, element, ranges, style):
        input_scale = style.pop('scale', 1.0)
        mag_dim = element.get_dimension(3)
        xs = element.dimension_values(0) if len(element.data) else []
        ys = element.dimension_values(1) if len(element.data) else []
        radians = element.dimension_values(2) if len(element.data) else []
        angles = list(np.rad2deg(radians))
        scale = input_scale / self._min_dist

        if mag_dim:
            magnitudes = element.dimension_values(3)
            _, max_magnitude = ranges[mag_dim.name]
            if self.normalize_lengths and max_magnitude != 0:
                magnitudes = magnitudes / max_magnitude
        else:
            magnitudes = np.ones(len(xs))

        args = (xs, ys, magnitudes,  [0.0] * len(element))
        if self.color_dim:
            colors = magnitudes if self.color_dim == 'magnitude' else radians
            args = args + (colors,)
            if self.color_dim == 'angle':
                style['clim'] = element.get_dimension(2).range
            elif self.color_dim == 'magnitude':
                magnitude_dim = element.get_dimension(3).name
                style['clim'] = ranges[magnitude_dim]
            style.pop('color', None)

        if 'pivot' not in style: style['pivot'] = 'mid'
        if not self.arrow_heads:
            style['headaxislength'] = 0
        style.update(dict(scale=scale, angles=angles))

        return args, style, {}


    def init_artists(self, ax, plot_args, plot_kwargs):
        quiver = ax.quiver(*plot_args, units='x', scale_units='x', **plot_kwargs)
        return {'artist': quiver}


    def update_handles(self, key, axis, element, ranges, style):
        args, style, axis_kwargs = self.get_data(element, ranges, style)

        # Set magnitudes, angles and colors if supplied.
        quiver = self.handles['artist']
        quiver.set_offsets(np.column_stack(args[:2]))
        quiver.U = args[2]
        quiver.angles = style['angles']
        if self.color_dim:
            quiver.set_array(args[-1])

        if self.color_dim == 'magnitude':
            quiver.set_clim(style['clim'])
        return axis_kwargs


class BarPlot(LegendPlot):

    group_index = param.Integer(default=0, doc="""
       Index of the dimension in the supplied Bars
       Element, which will be laid out into groups.""")

    category_index = param.Integer(default=1, doc="""
       Index of the dimension in the supplied Bars
       Element, which will be laid out into categories.""")

    stack_index = param.Integer(default=2, doc="""
       Index of the dimension in the supplied Bars
       Element, which will stacked.""")

    padding = param.Number(default=0.2, doc="""
       Defines the padding between groups.""")

    color_by = param.List(default=['category'], doc="""
       Defines how the Bar elements colored. Valid options include
       any permutation of 'group', 'category' and 'stack'.""")

    show_legend = param.Boolean(default=True, doc="""
        Whether to show legend for the plot.""")

    xticks = param.Integer(0, precedence=-1)

    style_opts = ['alpha', 'color', 'align', 'visible', 'edgecolor',
                  'log', 'facecolor', 'capsize', 'error_kw', 'hatch']

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
        gi, ci, si =self.group_index, self.category_index, self.stack_index
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


    def get_extents(self, element, ranges):
        ngroups = len(self.values['group'])
        vdim = element.vdims[0].name
        if self.stack_index in range(element.ndims):
            return 0, 0, ngroups, np.NaN
        else:
            vrange = ranges[vdim]
            return 0, np.nanmin([vrange[0], 0]), ngroups, vrange[1]


    def initialize_plot(self, ranges=None):
        element = self.hmap.last
        vdim = element.vdims[0]
        axis = self.handles['axis']
        key = self.keys[-1]

        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = match_spec(element, ranges)

        self.handles['artist'], self.handles['xticks'], xdims = self._create_bars(axis, element)
        return self._finalize_axis(key, ranges=ranges, xticks=self.handles['xticks'],
                                   dimensions=[xdims, vdim])


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


    def _create_bars(self, axis, element):
        # Get style and dimension information
        values = self.values
        gi, ci, si = self.group_index, self.category_index, self.stack_index
        gdim, cdim, sdim = [element.kdims[i] if i < element.ndims else None
                            for i in (gi, ci, si) ]
        indices = dict(zip(self._dimensions, (gi, ci, si)))
        style_groups = [sg for sg in self.color_by if indices[sg] < element.ndims]
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
                    bar = axis.bar([xpos], [val], width=width, bottom=prev,
                                   **style)

                    # Update variables
                    bars[tuple(val_key)] = bar
                    prev += val if np.isfinite(val) else 0
                    labels.append(label)
        title = [str(element.kdims[indices[cg]])
                 for cg in self.color_by if indices[cg] < ndims]

        if self.show_legend and any(len(l) for l in labels):
            leg_spec = self.legend_specs[self.legend_position]
            if self.legend_cols: leg_spec['ncol'] = self.legend_cols
            axis.legend(title=', '.join(title), **leg_spec)
        return bars, xticks, xdims


    def update_handles(self, key, axis, element, ranges, style):
        dims = element.dimensions('key', label=True)
        ndims = len(dims)
        ci, gi, si = self.category_index, self.group_index, self.stack_index
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
                        prev += height if np.isfinite(height) else 0
        return {'xticks': self.handles['xticks']}


class SpikesPlot(PathPlot, ColorbarPlot):

    aspect = param.Parameter(default='square', doc="""
        The aspect ratio mode of the plot. Allows setting an
        explicit aspect ratio as width/height as well as
        'square' and 'equal' options.""")

    color_index = param.ClassSelector(default=1, class_=(basestring, int), doc="""
      Index of the dimension from which the color will the drawn""")

    spike_length = param.Number(default=0.1, doc="""
      The length of each spike if Spikes object is one dimensional.""")

    position = param.Number(default=0., doc="""
      The position of the lower end of each spike.""")

    style_opts = PathPlot.style_opts + ['cmap']

    def init_artists(self, ax, plot_args, plot_kwargs):
        line_segments = LineCollection(*plot_args, **plot_kwargs)
        ax.add_collection(line_segments)
        return {'artist': line_segments}


    def get_extents(self, element, ranges):
        l, b, r, t = super(SpikesPlot, self).get_extents(element, ranges)
        ndims = len(element.dimensions(label=True))
        max_length = t if ndims > 1 else self.spike_length
        return (l, self.position, r, self.position+max_length)


    def get_data(self, element, ranges, style):
        dimensions = element.dimensions(label=True)
        ndims = len(dimensions)

        pos = self.position
        if ndims > 1:
            data = [[(x, pos), (x, pos+y)] for x, y in element.array()]
        else:
            height = self.spike_length
            data = [[(x[0], pos), (x[0], pos+height)] for x in element.array()]

        if self.invert_axes:
            data = [(line[0][::-1], line[1][::-1]) for line in data]

        cdim = element.get_dimension(self.color_index)
        if cdim:
            style['array'] = element.dimension_values(cdim)
            self._norm_kwargs(element, ranges, style, cdim)
            style['clim'] = style.pop('vmin'), style.pop('vmax')
        return (np.array(data),), style, {}


    def update_handles(self, key, axis, element, ranges, style):
        artist = self.handles['artist']
        (data,), kwargs, axis_kwargs = self.get_data(element, ranges, style)
        artist.set_paths(data)
        artist.set_visible(style.get('visible', True))
        if 'array' in kwargs:
            artist.set_clim((kwargs['vmin'], kwargs['vmax']))
            artist.set_array(kwargs['array'])
            if 'norm' in kwargs:
                artist.norm = kwargs['norm']
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



class BoxPlot(ChartPlot):
    """
    BoxPlot plots the ErrorBar Element type and supporting
    both horizontal and vertical error bars via the 'horizontal'
    plot option.
    """

    style_opts = ['notch', 'sym', 'whis', 'bootstrap',
                  'conf_intervals', 'widths', 'showmeans',
                  'show_caps', 'showfliers', 'boxprops',
                  'whiskerprops', 'capprops', 'flierprops',
                  'medianprops', 'meanprops', 'meanline']

    def get_extents(self, element, ranges):
        return (np.NaN,)*4


    def get_data(self, element, ranges, style):
        groups = element.groupby(element.kdims)

        data, labels = [], []

        groups = groups.data.items() if element.kdims else [(element.label, element)]
        for key, group in groups:
            if element.kdims:
                label = ','.join([unicode(safe_unicode(d.pprint_value(v)))
                                  for d, v in zip(element.kdims, key)])
            else:
                label = key
            data.append(group[group.vdims[0]])
            labels.append(label)
        style['labels'] = labels
        style.pop('zorder')
        style.pop('label')
        style['vert'] = not self.invert_axes
        format_kdims = [kd(value_format=None) for kd in element.kdims]
        return (data,), style, {'dimensions': [format_kdims,
                                               element.vdims[0]]}


    def init_artists(self, ax, plot_args, plot_kwargs):
        boxplot = ax.boxplot(*plot_args, **plot_kwargs)
        return {'artist': boxplot}


    def teardown_handles(self):
        for group in self.handles['artist'].values():
            for v in group:
                v.remove()


class SideBoxPlot(AdjoinedPlot, BoxPlot):

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

    def __init__(self, *args, **kwargs):
        super(SideBoxPlot, self).__init__(*args, **kwargs)
        if self.adjoined:
            self.invert_axes = not self.invert_axes
