from __future__ import unicode_literals
from itertools import product

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

import param

from ...core import OrderedDict, NdMapping, CompositeOverlay, HoloMap
from ...core.util import match_spec
from ...element import Points, Raster, Polygons
from ..util import compute_sizes, get_sideplot_ranges
from .element import ElementPlot, ColorbarPlot, LegendPlot


class ChartPlot(ElementPlot):

    def __init__(self, data, **params):
        super(ChartPlot, self).__init__(data, **params)
        key_dim = self.hmap.last.get_dimension(0)
        self.cyclic_range = key_dim.range if key_dim.cyclic else None


    def _cyclic_format_x_tick_label(self, x):
        if self.relative_labels:
            return str(x)
        return str(int(np.round(180*x/self.cyclic_range[1])))


    def _rotate(self, seq, n=1):
        n = n % len(seq) # n=hop interval
        return seq[n:] + seq[:n]

    def _cyclic_reduce_ticks(self, x_values):
        values = []
        labels = []
        step = self.cyclic_range[1] / (self.xticks - 1)
        if self.relative_labels:
            labels.append(-90)
            label_step = 180 / (self.xticks - 1)
        else:
            labels.append(x_values[0])
            label_step = step
        values.append(x_values[0])
        for i in range(0, self.xticks - 1):
            labels.append(labels[-1] + label_step)
            values.append(values[-1] + step)
        return values, [self._cyclic_format_x_tick_label(x) for x in labels]


    def _cyclic_curves(self, curveview):
        """
        Mutate the lines object to generate a rotated cyclic curves.
        """
        x_values = list(curveview.data[:, 0])
        y_values = list(curveview.data[:, 1])
        if self.center_cyclic:
            rotate_n = self.peak_argmax+len(x_values)/2
            y_values = self._rotate(y_values, n=rotate_n)
            ticks = self._rotate(x_values, n=rotate_n)
        else:
            ticks = list(x_values)

        ticks.append(ticks[0])
        x_values.append(x_values[0]+self.cyclic_range[1])
        y_values.append(y_values[0])

        self.xvalues = x_values
        return np.vstack([x_values, y_values]).T


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

    center_cyclic = param.Boolean(default=True, doc="""
        If enabled and plotted quantity is cyclic will center the
        plot around the peak.""")

    num_ticks = param.Integer(default=5, doc="""
        If autotick is disabled, this number of tickmarks will be drawn.""")

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

    def initialize_plot(self, ranges=None):
        element = self.hmap.last
        axis = self.handles['axis']
        key = self.keys[-1]

        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = match_spec(element, ranges)

        # Create xticks and reorder data if cyclic
        xticks = None
        data = element.data
        if self.cyclic_range is not None:
            if self.center_cyclic:
                self.peak_argmax = np.argmax(element.data[:, 1])
            data = self._cyclic_curves(element)
            if self.xticks is not None:
                xticks = self._cyclic_reduce_ticks(self.xvalues)

        # Create line segments and apply style
        style = self.style[self.cyclic_index]
        legend = element.label if self.show_legend else ''
        line_segment = axis.plot(data[:, 0], data[:, 1], label=legend,
                                 zorder=self.zorder, **style)[0]

        self.handles['artist'] = line_segment
        return self._finalize_axis(self.keys[-1], ranges=ranges, xticks=xticks)


    def update_handles(self, axis, element, key, ranges=None):
        data = element.data
        artist = self.handles['artist']
        if self.cyclic_range is not None:
            data = self._cyclic_curves(element)
        artist.set_xdata(data[:, 0])
        artist.set_ydata(data[:, 1])




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


    def initialize_plot(self, ranges=None):
        element = self.hmap.last
        axis = self.handles['axis']
        key = self.keys[-1]

        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = match_spec(element, ranges)

        error_kwargs = dict(self.style[self.cyclic_index], fmt='none',
                            zorder=self.zorder)
        error_kwargs['yerr'] = element.data[:, 2:4].T
        _, (bottoms, tops), verts = axis.errorbar(element.data[:, 0],
                                                  element.data[:, 1],
                                                  **error_kwargs)
        self.handles['bottoms'] = bottoms
        self.handles['tops'] = tops
        self.handles['verts'] = verts[0]

        return self._finalize_axis(self.keys[-1], ranges=ranges)


    def update_handles(self, axis, element, key, ranges=None):
        data = element.data
        bottoms = self.handles['bottoms']
        tops = self.handles['tops']
        verts = self.handles['verts']
        paths = verts.get_paths()
        if self.horizontal:
            bdata = data[:, 0] - data[:, 2]
            tdata = data[:, 0] + data[:, 3]
            tops.set_xdata(bdata)
            tops.set_ydata(data[:, 1])
            bottoms.set_xdata(tdata)
            bottoms.set_ydata(data[:, 1])
            for i, path in enumerate(paths):
                path.vertices = np.array([[bdata[i], data[i, 1]],
                                          [tdata[i], data[i, 1]]])
        else:
            bdata = data[:, 1] - data[:, 2]
            tdata = data[:, 1] + data[:, 3]
            bottoms.set_xdata(data[:, 0])
            bottoms.set_ydata(bdata)
            tops.set_xdata(data[:, 0])
            tops.set_ydata(tdata)
            for i, path in enumerate(paths):
                path.vertices = np.array([[data[i, 0], bdata[i]],
                                          [data[i, 0], tdata[i]]])


class SpreadPlot(ChartPlot):
    """
    SpreadPlot plots the Spread Element type.
    """

    style_opts = ['alpha', 'color', 'linestyle', 'linewidth',
                  'edgecolor', 'facecolor', 'hatch']

    def __init__(self, *args, **kwargs):
        super(SpreadPlot, self).__init__(*args, **kwargs)
        self._extent = None


    def initialize_plot(self, ranges=None):
        element = self.hmap.last
        axis = self.handles['axis']
        key = self.keys[-1]

        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = match_spec(element, ranges)
        self.update_handles(axis, element, key, ranges)

        return self._finalize_axis(self.keys[-1], ranges=ranges)


    def update_handles(self, axis, element, key, ranges=None):
        if 'paths' in self.handles:
            self.handles['paths'].remove()
        paths = axis.fill_between(element.data[:, 0],
                                  element.data[:, 1]-element.data[:, 2],
                                  element.data[:, 1]+element.data[:, 3],
                                  zorder=self.zorder,
                                  label=element.label if self.show_legend else None,
                                  **self.style[self.cyclic_index])
        self.handles['paths'] = paths



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

        if self.orientation == 'vertical':
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

        if self.orientation == 'vertical':
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
        return (y0, x0, y1, x1) if self.orientation == 'vertical' else (x0, y0, x1, y1)


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
            if self.orientation == 'vertical':
                bar.set_y(edge)
                bar.set_width(height)
                bar.set_height(width)
            else:
                bar.set_x(edge)
                bar.set_height(height)
                bar.set_width(width)


    def update_handles(self, axis, element, key, ranges=None):
        """
        Update the plot for an animation.
        :param axis:
        """
        # Process values, axes and style
        edges, hvals, widths, lims = self._process_hist(element)

        ticks = self._compute_ticks(element, edges, widths, lims)
        ax_settings = self._process_axsettings(element, lims, ticks)
        self._update_artists(key, element, edges, hvals, widths, lims, ranges)
        return ax_settings



class SideHistogramPlot(HistogramPlot):

    aspect = param.Parameter(default='auto', doc="""
        Aspect ratios on SideHistogramPlot should be determined by the
        AdjointLayoutPlot.""")

    offset = param.Number(default=0.2, bounds=(0,1), doc="""
        Histogram value offset for a colorbar.""")

    show_grid = param.Boolean(default=True, doc="""
        Whether to overlay a grid on the axis.""")

    show_title = param.Boolean(default=False, doc="""
        Titles should be disabled on all SidePlots to avoid clutter.""")

    show_xlabel = param.Boolean(default=False, doc="""
        Whether to show the x-label of the plot. Disabled by default
        because plots are often too cramped to fit the title correctly.""")

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


    def _process_axsettings(self, hist, lims, ticks):
        axsettings = super(SideHistogramPlot, self)._process_axsettings(hist, lims, ticks)
        label = 'ylabel' if self.orientation == 'vertical' else 'xlabel'
        if not self.show_xlabel:
            axsettings[label] = ''
        else:
            axsettings[label] = str(hist.kdims[0])
        return axsettings


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
        y0, y1 = element.range(1)
        offset = self.offset * y1
        range_item, main_range, dim = get_sideplot_ranges(self, element, main, ranges)
        if isinstance(range_item, (Raster, Points, Polygons)):
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
        return (0, x0, y1, x1) if self.orientation == 'vertical' else (x0, 0, x1, y1)


    def _colorize_bars(self, cmap, bars, element, main_range, dim):
        """
        Use the given cmap to color the bars, applying the correct
        color ranges as necessary.
        """
        vertical = (self.orientation == 'vertical')
        cmap_range = main_range[1] - main_range[0]
        lower_bound = main_range[0]
        colors = np.array(element.dimension_values(dim))
        colors = (colors - lower_bound) / (cmap_range)
        for c, bar in zip(colors, bars):
            bar_bin = bar.get_y() if vertical else bar.get_x()
            width = bar.get_height() if vertical else bar.get_width()
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
            if self.orientation == 'vertical':
                offset_line.set_xdata(offset)
            else:
                offset_line.set_ydata(offset)


class PointPlot(ChartPlot, ColorbarPlot):
    """
    Note that the 'cmap', 'vmin' and 'vmax' style arguments control
    how point magnitudes are rendered to different colors.
    """

    color_index = param.Integer(default=3, doc="""
      Index of the dimension from which the color will the drawn""")

    size_index = param.Integer(default=2, doc="""
      Index of the dimension from which the sizes will the drawn.""")

    scaling_factor = param.Number(default=1, bounds=(1, None), doc="""
      If values are supplied the area of the points is computed relative
      to the marker size. It is then multiplied by scaling_factor to the power
      of the ratio between the smallest point and all other points.
      For values of 1 scaling by the values is disabled, a factor of 2
      allows for linear scaling of the area and a factor of 4 linear
      scaling of the point width.""")

    show_grid = param.Boolean(default=True, doc="""
      Whether to draw grid lines at the tick positions.""")

    size_fn = param.Callable(default=np.abs, doc="""
      Function applied to size values before applying scaling,
      to remove values lower than zero.""")

    style_opts = ['alpha', 'color', 'edgecolors', 'facecolors',
                  'linewidth', 'marker', 'size', 'visible',
                  'cmap', 'vmin', 'vmax']

    def initialize_plot(self, ranges=None):
        points = self.hmap.last
        axis = self.handles['axis']

        ranges = self.compute_ranges(self.hmap, self.keys[-1], ranges)
        ranges = match_spec(points, ranges)

        ndims = points.data.shape[1]
        xs = points.data[:, 0] if len(points.data) else []
        ys = points.data[:, 1] if len(points.data) else []
        cs = points.data[:, self.color_index] if self.color_index < ndims else None

        style = self.style[self.cyclic_index]
        if self.size_index < ndims and self.scaling_factor > 1:
            style['s'] = self._compute_size(points, style)

        color = style.pop('color', None)
        if cs is not None:
            style['c'] = cs
        else:
            style['c'] = color
        edgecolor = style.pop('edgecolors', 'none')
        legend = points.label if self.show_legend else ''
        scatterplot = axis.scatter(xs, ys, zorder=self.zorder, label=legend,
                                   edgecolors=edgecolor, **style)
        self.handles['artist'] = scatterplot

        if cs is not None:
            val_dim = points.dimensions(label=True)[self.color_index]
            clims = ranges.get(val_dim)
            scatterplot.set_clim(clims)

        return self._finalize_axis(self.keys[-1], ranges=ranges)


    def _compute_size(self, element, opts):
        sizes = element.data[:, self.size_index]
        ms = opts.pop('s') if 's' in opts else plt.rcParams['lines.markersize']
        return compute_sizes(sizes, self.size_fn, self.scaling_factor, ms)


    def update_handles(self, axis, element, key, ranges=None):
        paths = self.handles['artist']
        paths.set_offsets(element.data[:, 0:2])
        ndims = element.data.shape[1]
        dims = element.dimensions(label=True)
        if self.size_index < ndims:
            opts = self.style[self.cyclic_index]
            paths.set_sizes(self._compute_size(element, opts))
        if self.color_index < ndims:
            cs = element.data[:, self.color_index]
            val_dim = dims[self.color_index]
            paths.set_clim(ranges[val_dim])
            paths.set_array(cs)



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
       dimension (if any)""")

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
        dists = []
        for vfield in vmap:
            dists.append(self._get_min_dist(vfield))
        return min(dists)


    def _get_info(self, vfield, input_scale, ranges):
        xs = vfield.data[:, 0] if len(vfield.data) else []
        ys = vfield.data[:, 1] if len(vfield.data) else []
        radians = vfield.data[:, 2] if len(vfield.data) else []
        magnitudes = vfield.data[:, 3] if vfield.data.shape[1]>=4 else np.array([1.0] * len(xs))
        colors = magnitudes if self.color_dim == 'magnitude' else radians

        if vfield.data.shape[1] >= 4:
            magnitude_dim = vfield.get_dimension(3).name
            _, max_magnitude = ranges[magnitude_dim]
        else:
            max_magnitude = 1.0

        min_dist =      self._min_dist if self._min_dist else self._get_min_dist(vfield)

        if self.normalize_lengths and max_magnitude != 0:
            magnitudes =  magnitudes / max_magnitude

        return (xs, ys, list((radians / np.pi) * 180),
                magnitudes, colors, input_scale / min_dist)


    def _get_min_dist(self, vfield):
        "Get the minimum sampling distance."
        xys = np.array([complex(x,y) for x,y in zip(vfield.data[:,0],
                                                    vfield.data[:,1])])
        m, n = np.meshgrid(xys, xys)
        distances = abs(m-n)
        np.fill_diagonal(distances, np.inf)
        return  distances.min()


    def initialize_plot(self, ranges=None):
        vfield = self.hmap.last
        axis = self.handles['axis']

        colorized = self.color_dim is not None
        kwargs = self.style[self.cyclic_index]
        input_scale = kwargs.pop('scale', 1.0)
        ranges = self.compute_ranges(self.hmap, self.keys[-1], ranges)
        ranges = match_spec(vfield, ranges)
        xs, ys, angles, lens, colors, scale = self._get_info(vfield, input_scale, ranges)

        args = (xs, ys, lens,  [0.0] * len(vfield.data))
        args = args + (colors,) if colorized else args

        if not self.arrow_heads:
            kwargs['headaxislength'] = 0

        if 'pivot' not in kwargs: kwargs['pivot'] = 'mid'

        legend = vfield.label if self.show_legend else ''
        quiver = axis.quiver(*args, zorder=self.zorder, units='x', label=legend,
                              scale_units='x', scale = scale, angles = angles ,
                              **({k:v for k,v in kwargs.items() if k!='color'}
                                 if colorized else kwargs))


        if self.color_dim == 'angle':
            clims = vfield.get_dimension(2).range
            quiver.set_clim(clims)
        elif self.color_dim == 'magnitude':
            magnitude_dim = vfield.get_dimension(3).name
            quiver.set_clim(ranges[magnitude_dim])

        self.handles['axis'].add_collection(quiver)
        self.handles['artist'] = quiver
        self.handles['input_scale'] = input_scale

        return self._finalize_axis(self.keys[-1], ranges=ranges)


    def update_handles(self, axis, element, key, ranges=None):
        artist = self.handles['artist']
        artist.set_offsets(element.data[:,0:2])
        input_scale = self.handles['input_scale']
        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = match_spec(element, ranges)

        xs, ys, angles, lens, colors, scale = self._get_info(element, input_scale, ranges)

        # Set magnitudes, angles and colors if supplied.
        quiver = self.handles['artist']
        quiver.U = lens
        quiver.angles = angles
        if self.color_dim is not None:
            quiver.set_array(colors)

        if self.color_dim == 'magnitude':
            magnitude_dim = element.get_dimension(3).name
            quiver.set_clim(ranges[magnitude_dim])


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
                params = dict(kdims=[dim])
            else:
                dimensions.append(None)
                vals = [None]
                params = {}
            values[vtype] = NdMapping([(v, None) for v in vals],
                                      **params).keys()
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

        self.handles['artist'], xticks, xlabel = self._create_bars(axis, element)
        return self._finalize_axis(key, ranges=ranges, xticks=xticks, xlabel=xlabel, ylabel=str(vdim))


    def _finalize_ticks(self, axis, element, xticks, yticks, zticks):
        """
        Apply ticks with appropriate offsets.
        """
        ticks, labels, yalignments = zip(*sorted(xticks, key=lambda x: x[0]))
        super(BarPlot, self)._finalize_ticks(axis, element, [ticks, labels], yticks, zticks)
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
        xlabel = ' / '.join([str(d) for d in [cdim, gdim] if d is not None])

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
                    label_key[idx] = grp
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
                        label_key[idx] = cat
                        style_key[idx] = cat_name
                    val_key[ci] = cat_name
                    xticks.append((xpos+width/2., cat, 0))
                prev = 0
                for sidx, stk_name in enumerate(values['stack']):
                    if stk_name is not None:
                        if 'stack' in style_groups:
                            idx = style_groups.index('stack')
                            stk = gdim.pprint_value(stk_name)
                            label_key[idx] = stk
                            style_key[idx] = stk_name
                        val_key[si] = stk_name
                    val = element.get(tuple(val_key), (np.NaN,))
                    label = ', '.join(label_key)
                    style = dict(style_opts, label='' if label in labels else label,
                                 **dict(zip(sopts, color_groups[tuple(style_key)])))
                    bar = axis.bar([xpos], val, width=width, bottom=prev,
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
        return bars, xticks, xlabel


    def update_handles(self, axis, element, key, ranges=None):
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
                    bar = self.handles['bars'].get(tuple(val_key))
                    if bar:
                        height = element.get(tuple(val_key), np.NaN)
                        height = height if np.isscalar(height) else height[0]
                        bar[0].set_height(height)
                        bar[0].set_y(prev)
                        prev += height if np.isfinite(height) else 0
