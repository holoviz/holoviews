from __future__ import unicode_literals
from itertools import product

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

import param

from ..core.options import Store
from ..core import OrderedDict, NdMapping, ViewableElement, CompositeOverlay, HoloMap
from ..core.util import match_spec
from ..element import Scatter, Curve, Histogram, Bars, Points, Raster, VectorField
from .element import ElementPlot
from .plot import Plot


class ChartPlot(ElementPlot):

    def __init__(self, data, **params):
        super(ChartPlot, self).__init__(data, **params)
        key_dim = self.map.last.get_dimension(0)
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


    def get_extents(self, element, ranges):
        l, r = self.xlim if self.rescale_individually else self.map.xlim
        b, t = self.ylim if self.rescale_individually else self.map.ylim
        dimensions = element.dimensions(label=True)
        xdim, ydim = dimensions[0], dimensions[1]
        if ranges is not None:
            xrange, yrange = ranges.get(xdim), ranges.get(ydim)
            if not xrange is None:
                l, r = (np.min([xrange[0], l]) if l else xrange[0],
                        np.max([xrange[1], r]) if r else xrange[1])
            if not yrange is None:
                b, t = (np.min([yrange[0], b]) if b else yrange[0],
                        np.max([yrange[1], t]) if t else yrange[1])
        return l, b, r, t


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

    style_opts = ['alpha', 'color', 'visible', 'linewidth',
                  'lw', 'ls', 'linestyle', 'marker']

    def __call__(self, ranges=None):
        element = self.map.last
        axis = self.handles['axis']
        key = self.keys[-1]

        ranges = self.compute_ranges(self.map, key, ranges)
        ranges = match_spec(element, ranges)

        # Create xticks and reorder data if cyclic
        xticks = None
        data = element.data
        if self.cyclic_range is not None:
            if self.center_cyclic:
                self.peak_argmax = np.argmax(element.data[:, 1])
            data = self._cyclic_curves(element)
            xticks = self._cyclic_reduce_ticks(self.xvalues)

        # Create line segments and apply style
        style = self.style[self.cyclic_index]
        line_segment = axis.plot(data[:, 0], data[:, 1],
                                 zorder=self.zorder, **style)[0]

        self.handles['line_segment'] = line_segment
        self.handles['legend_handle'] = line_segment
        return self._finalize_axis(self.keys[-1], ranges=ranges, xticks=xticks)


    def update_handles(self, axis, view, key, ranges=None):
        data = view.data
        if self.cyclic_range is not None:
            data = self._cyclic_curves(view)
        self.handles['line_segment'].set_xdata(data[:, 0])
        self.handles['line_segment'].set_ydata(data[:, 1])



class HistogramPlot(ChartPlot):
    """
    HistogramPlot can plot DataHistograms and ViewMaps of
    DataHistograms, which can be displayed as a single frame or
    animation.
    """

    num_ticks = param.Integer(default=5, doc="""
        If colorbar is enabled the number of labels will be overwritten.""")

    show_frame = param.Boolean(default=False, doc="""
        Disabled by default for clarity.""")

    show_grid = param.Boolean(default=False, doc="""
        Whether to overlay a grid on the axis.""")

    style_opts = ['alpha', 'color', 'align', 'visible',
                  'edgecolor', 'log', 'ecolor', 'capsize',
                  'error_kw', 'hatch', 'fc', 'ec']

    def __init__(self, histograms, **params):
        self.center = False
        self.cyclic = False

        super(HistogramPlot, self).__init__(histograms, **params)

        if self.orientation == 'vertical':
            self.axis_settings = ['ylabel', 'xlabel', 'yticks']
        else:
            self.axis_settings = ['xlabel', 'ylabel', 'xticks']
        val_dim = self.map.last.get_dimension(1)
        self.cyclic_range = val_dim.range if val_dim.cyclic else None


    def __call__(self, ranges=None):
        hist = self.map.last

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
        bars = self.plotfn(edges, hvals, widths, zorder=self.zorder, **style)
        self.handles['bars'] = self._update_plot(self.keys[-1], hist, bars, lims) # Indexing top
        self.handles['legend_handle'] = bars

        ticks = self._compute_ticks(hist, edges, widths, lims)
        ax_settings = self._process_axsettings(hist, lims, ticks)

        return self._finalize_axis(self.keys[-1], **ax_settings)


    def _process_hist(self, hist):
        """
        Get data from histogram, including bin_ranges and values.
        """
        self.cyclic = hist.get_dimension(0).cyclic
        edges = hist.edges[:-1]
        hist_vals = np.array(hist.values)
        widths = [hist._width] * len(hist) if getattr(hist, '_width', None) else np.diff(hist.edges)
        extents = None
        if extents is None:
            xlims = hist.xlim if self.rescale_individually else self.map.xlim
            ylims = hist.ylim
        else:
            l, b, r, t = extents
            xlims = (l, r)
            ylims = (b, t)
        lims = xlims + ylims
        return edges, hist_vals, widths, lims


    def _compute_ticks(self, view, edges, widths, lims):
        """
        Compute the ticks either as cyclic values in degrees or as roughly
        evenly spaced bin centers.
        """
        if self.cyclic:
            x0, x1, _, _ = lims
            xvals = np.linspace(x0, x1, self.num_ticks)
            labels = ["%.0f" % np.rad2deg(x) + '\N{DEGREE SIGN}' for x in xvals]
        else:
            dim_type = view.get_dimension_type(0)
            if dim_type in [str, type(None), np.string_]:
                xvals = [edges[i]+widths[i]/2. for i in range(len(edges))]
                labels = list(view.data[:, 0])
            else:
                edge_inds = list(range(len(edges)))
                step = len(edges)/float(self.num_ticks-1)
                inds = [0] + [edge_inds[int(i*step)-1] for i in range(1, self.num_ticks)]
                xvals = [edges[i]+widths[i]/2. for i in inds]
                labels = ["%g" % round(x, 2) for x in xvals]
        return [xvals, labels]


    def get_extents(self, view, ranges):
        x0, y0, x1, y1 = super(HistogramPlot, self).get_extents(view, ranges)
        return (0, x0, y1, x1) if self.orientation == 'vertical' else (x0, 0, x1, y1)


    def _process_axsettings(self, hist, lims, ticks):
        """
        Get axis settings options including ticks, x- and y-labels
        and limits.
        """
        axis_settings = dict(zip(self.axis_settings, [hist.xlabel, hist.ylabel, ticks]))
        return axis_settings


    def _update_plot(self, key, hist, bars, lims):
        """
        Process bars can be subclassed to manually adjust bars
        after being plotted.
        """
        return bars


    def _update_artists(self, key, hist, edges, hvals, widths, lims):
        """
        Update all the artists in the histogram. Subclassable to
        allow updating of further artists.
        """
        plot_vals = zip(self.handles['bars'], edges, hvals, widths)
        for bar, edge, height, width in plot_vals:
            if self.orientation == 'vertical':
                bar.set_y(edge)
                bar.set_width(height)
                bar.set_height(width)
            else:
                bar.set_x(edge)
                bar.set_height(height)
                bar.set_width(width)


    def update_handles(self, axis, view, key, ranges=None):
        """
        Update the plot for an animation.
        :param axis:
        """
        # Process values, axes and style
        edges, hvals, widths, lims = self._process_hist(view)

        ticks = self._compute_ticks(view, edges, widths, lims)
        ax_settings = self._process_axsettings(view, lims, ticks)
        self._update_artists(key, view, edges, hvals, widths, lims)
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
        if not self.show_xlabel:
            axsettings['ylabel' if self.orientation == 'vertical' else 'xlabel'] = ''
        return axsettings


    def _update_artists(self, n, view, edges, hvals, widths, lims):
        super(SideHistogramPlot, self)._update_artists(n, view, edges, hvals, widths, lims)
        self._update_plot(n, view, self.handles['bars'], lims)


    def _update_plot(self, key, view, bars, lims):
        """
        Process the bars and draw the offset line as necessary. If a
        color map is set in the style of the 'main' ViewableElement object, color
        the bars appropriately, respecting the required normalization
        settings.
        """
        hist = view
        main = self.adjoined.main
        y0, y1 = hist.ylim
        offset = self.offset * y1
        plot_options = Store.lookup_options(main, 'plot').options
        individually = plot_options.get('normalize_individually', False)

        hist_dim = hist.get_dimension(0).name
        range_item = main
        if isinstance(main, HoloMap):
            if issubclass(main.type, CompositeOverlay):
                range_item = [hm for hm in main.split_overlays()[1]
                              if hist_dim in hm.dimensions('value', label=True)][0]
                if individually:
                    range_item = range_item[key]
            else:
                range_item = main[key] if individually else main
        elif isinstance(main, ViewableElement):
            range_item = main
        main_range = range_item.range(hist_dim)

        if offset and ('offset_line' not in self.handles):
            self.handles['offset_line'] = self.offset_linefn(offset,
                                                             linewidth=1.0,
                                                             color='k')
        elif offset:
            self._update_separator(offset)


        # If .main is an NdOverlay or a HoloMap of Overlays get the correct style
        if isinstance(range_item, HoloMap):
            range_item = range_item.last
        elif isinstance(range_item, CompositeOverlay):
            range_item = [ov for ov in range_item
                          if hist_dim in ov.dimensions('value', label=True)][0]

        if isinstance(range_item, (Raster, Points)):
            style = Store.lookup_options(range_item, 'style')[self.cyclic_index]
            cmap = cm.get_cmap(style.get('cmap'))
            main_range = style.get('clims', main_range)
        else:
            cmap = None

        if cmap is not None:
            self._colorize_bars(cmap, bars, main_range)
        return bars


    def get_extents(self, element, ranges):
        (x0, x1), y1 = element.xlim, element.ylim[1]
        return (0, x0, y1, x1) if self.orientation == 'vertical' else (x0, 0, x1, y1)


    def _colorize_bars(self, cmap, bars, main_range):
        """
        Use the given cmap to color the bars, applying the correct
        color ranges as necessary.
        """
        vertical = (self.orientation == 'vertical')
        cmap_range = main_range[1] - main_range[0]
        lower_bound = main_range[0]
        for bar in bars:
            bar_bin = bar.get_y() if vertical else bar.get_x()
            width = bar.get_height() if vertical else bar.get_width()
            try:
                color_val = (bar_bin+width/2.-lower_bound)/cmap_range
            except:
                color_val = 0
            bar.set_facecolor(cmap(color_val))
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


class PointPlot(ChartPlot):
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

    style_opts = ['alpha', 'color', 'edgecolors', 'facecolors',
                  'linewidth', 'marker', 's', 'visible',
                  'cmap', 'vmin', 'vmax']

    def __call__(self, ranges=None):
        points = self.map.last
        axis = self.handles['axis']

        ranges = self.compute_ranges(self.map, self.keys[-1], ranges)
        ranges = match_spec(points, ranges)

        ndims = points.data.shape[1]
        xs = points.data[:, 0] if len(points.data) else []
        ys = points.data[:, 1] if len(points.data) else []
        sz = points.data[:, self.size_index] if self.size_index < ndims else None
        cs = points.data[:, self.color_index] if self.color_index < ndims else None

        style = self.style[self.cyclic_index]
        if sz is not None and self.scaling_factor > 1:
            style['s'] = self._compute_size(sz, style)
        color = style.pop('color', None)
        if cs is not None:
            style['c'] = cs
        else:
            style['c'] = color
        edgecolor = style.pop('edgecolors', 'none')
        scatterplot = axis.scatter(xs, ys, zorder=self.zorder, edgecolors=edgecolor, **style)
        self.handles['paths'] = scatterplot
        self.handles['legend_handle'] = scatterplot

        if cs is not None:
            val_dim = points.dimensions(label=True)[self.color_index]
            clims = ranges.get(val_dim)
            scatterplot.set_clim(clims)

        return self._finalize_axis(self.keys[-1], ranges=ranges)


    def _compute_size(self, sizes, opts):
        ms = opts.pop('s') if 's' in opts else plt.rcParams['lines.markersize']
        sizes = np.ma.array(sizes, mask=sizes<=0)
        return (ms*self.scaling_factor**sizes)


    def update_handles(self, axis, element, key, ranges=None):
        paths = self.handles['paths']
        paths.set_offsets(element.data[:, 0:2])
        ndims = element.data.shape[1]
        if ndims > 2:
            sz = element.data[:, self.size_index] if self.size_index < ndims else None
            cs = element.data[:, self.color_index] if self.color_index < ndims else None
            opts = self.style[0]

            if sz is not None and self.scaling_factor > 1:
                paths.set_sizes(self._compute_size(sz, opts))
            if cs is not None:
                val_dim = element.dimensions(label=True)[self.color_index]
                ranges = self.compute_ranges(self.map, key, ranges)
                ranges = match_spec(element, ranges)
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
                  'linewidth', 'marker', 's', 'visible', 'cmap',
                  'scale', 'headlength', 'headaxislength', 'pivot']

    def __init__(self, *args, **params):
        super(VectorFieldPlot, self).__init__(*args, **params)
        self._min_dist = self._get_map_info(self.map)


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


    def __call__(self, ranges=None):
        vfield = self.map.last
        axis = self.handles['axis']

        colorized = self.color_dim is not None
        kwargs = self.style[self.cyclic_index]
        input_scale = kwargs.pop('scale', 1.0)
        ranges = self.compute_ranges(self.map, self.keys[-1], ranges)
        ranges = match_spec(vfield, ranges)
        xs, ys, angles, lens, colors, scale = self._get_info(vfield, input_scale, ranges)

        args = (xs, ys, lens,  [0.0] * len(vfield.data))
        args = args + (colors,) if colorized else args

        if not self.arrow_heads:
            kwargs['headaxislength'] = 0

        if 'pivot' not in kwargs: kwargs['pivot'] = 'mid'

        quiver = axis.quiver(*args, zorder=self.zorder, units='x',
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
        self.handles['quiver'] = quiver
        self.handles['legend_handle'] = quiver
        self.handles['input_scale'] = input_scale

        return self._finalize_axis(self.keys[-1])


    def update_handles(self, axis, view, key, ranges=None):
        self.handles['quiver'].set_offsets(view.data[:,0:2])
        input_scale = self.handles['input_scale']
        ranges = self.compute_ranges(self.map, key, ranges)
        ranges = match_spec(view, ranges)

        xs, ys, angles, lens, colors, scale = self._get_info(view, input_scale, ranges)

        # Set magnitudes, angles and colors if supplied.
        quiver = self.handles['quiver']
        quiver.U = lens
        quiver.angles = angles
        if self.color_dim is not None:
            quiver.set_array(colors)

        if self.color_dim == 'magnitude':
            magnitude_dim = view.get_dimension(3).name
            quiver.set_clim(ranges[magnitude_dim])


class BarPlot(ElementPlot):

    group_index = param.Integer(default=0, doc="""
       Index of the dimension in the supplied MultiBars
       Element, which will be laid out into groups.""")

    category_index = param.Integer(default=1, doc="""
       Index of the dimension in the supplied MultiBars
       Element, which will be laid out into categories.""")

    stack_index = param.Integer(default=2, doc="""
       Index of the dimension in the supplied MultiBars
       Element, which will stacked.""")

    padding = param.Number(default=0.2, doc="""
       Defines the padding between groups.""")

    color_by = param.List(default=['category'], doc="""
       Defines how the Bar elements colored. Valid options include
       any permutation of 'group', 'category' and 'stack'.""")

    show_legend = param.Boolean(default=True, doc="""
        Whether to show legend for the plot.""")

    xticks = param.Integer(0, precedence=-1)

    style_opts = ['alpha', 'color', 'align', 'visible',
                  'edgecolor', 'log', 'ecolor', 'capsize',
                  'error_kw', 'hatch', 'fc', 'ec']

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
        ndims = self.map.last.ndims
        dims = self.map.last.key_dimensions
        dimensions = []
        values = {}
        for vidx, vtype in zip([gi, ci, si], self._dimensions):
            if vidx < ndims:
                dim = dims[vidx]
                dimensions.append(dim)
                vals = self.map.dimension_values(dim.name)
                params = dict(key_dimensions=[dim])
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
        style = Store.lookup_options(element, 'style')[0]
        sopts = []
        for sopt in ['color', 'hatch']:
            if sopt in style:
                sopts.append(sopt)
                style.pop(sopt, None)
        color_groups = []
        for sg in style_groups:
            color_groups.append(self.values[sg])
        style_product = list(product(*color_groups))
        wrapped_style = Store.lookup_options(element, 'style').max_cycles(len(style_product))
        color_groups = {k:tuple(wrapped_style[n][sopt] for sopt in sopts)
                        for n,k in enumerate(style_product)}
        return style, color_groups, sopts


    def get_extents(self, element, ranges):
        return 0, 0, len(self.values['group']), np.NaN


    def __call__(self, ranges=None):
        element = self.map.last
        vdim = element.value_dimensions[0]
        axis = self.handles['axis']
        key = self.keys[-1]

        ranges = self.compute_ranges(self.map, key, ranges)
        ranges = match_spec(element, ranges)
        dims = element.dimensions('key', label=True)

        self.handles['bars'], xticks = self._create_bars(axis, element)
        self.handles['legend_handle'] = self.handles['bars']
        self._set_ticks(axis, dims, xticks)
        return self._finalize_axis(key, ranges=ranges, ylabel=str(vdim))


    def _set_ticks(self, axis, dims, xticks):
        """
        Apply ticks with appropriate offsets.
        """
        ndims = len(dims)
        idx = self.group_index if self.group_index < ndims else self.category_index
        ticks, labels, yalignments = zip(*sorted(xticks, key=lambda x: x[0]))
        axis.set_xlabel(dims[idx], labelpad=-0.15 if idx < ndims else -0.05)
        axis.set_xticks(ticks)
        axis.set_xticklabels(labels)
        for t, y in zip(axis.get_xticklabels(), yalignments):
            t.set_y(y)


    def _create_bars(self, axis, element):
        # Get style and dimension information
        values = self.values
        gi, ci, si = self.group_index, self.category_index, self.stack_index
        gdim, cdim, sdim = [element.key_dimensions[i] if i < element.ndims else None
                            for i in (gi, ci, si) ]
        indices = dict(zip(self._dimensions, (gi, ci, si)))
        style_groups = [sg for sg in self.color_by if indices[sg] < element.ndims]
        style_opts, color_groups, sopts = self._compute_styles(element, style_groups)
        dims = element.dimensions('key', label=True)
        ndims = len(dims)

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
                if 'group' in style_groups:
                    idx = style_groups.index('group')
                    label_key[idx] = str(gdim.pprint_value(grp_name))
                    style_key[idx] = grp_name
                val_key[gi] = grp_name
                if ci < ndims:
                    yalign = -0.125
                    xticks.append((gidx+0.5, dims[ci], -0.05))
                else:
                    yalign = 0
                xticks.append((gidx+0.5, grp_name, yalign))
            for cidx, cat_name in enumerate(values['category']):
                xpos = gidx+self.padding+(cidx*width)
                if cat_name is not None:
                    if 'category' in style_groups:
                        idx = style_groups.index('category')
                        label_key[idx] = str(gdim.pprint_value(cat_name))
                        style_key[idx] = cat_name
                    val_key[ci] = cat_name
                    xticks.append((xpos+width/2., cat_name, 0))
                prev = 0
                for sidx, stk_name in enumerate(values['stack']):
                    if stk_name is not None:
                        if 'stack' in style_groups:
                            idx = style_groups.index('stack')
                            label_key[idx] = str(gdim.pprint_value(stk_name))
                            style_key[idx] = stk_name
                        val_key[si] = stk_name
                    val = element.get(tuple(val_key), (np.NaN,))
                    label = ', '.join(label_key)
                    style = dict(style_opts, label='' if label in labels else label,
                                 **dict(zip(sopts, color_groups[tuple(style_key)])))
                    bar = axis.bar([xpos], val, width=width , bottom=prev,
                                   **style)

                    # Update variables
                    bars[tuple(val_key)] = bar
                    prev += val[0] if np.isfinite(val[0]) else 0
                    labels.append(label)
        title = [str(element.key_dimensions[indices[cg]])
                 for cg in self.color_by if indices[cg] < ndims]
        if any(len(l) for l in labels):
            axis.legend(title=', '.join(title))
        return bars, xticks


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


Store.registry.update({Curve: CurvePlot,
                       Scatter: PointPlot,
                       Bars: BarPlot,
                       Histogram: HistogramPlot,
                       Points: PointPlot,
                       VectorField: VectorFieldPlot})

Plot.sideplots.update({Histogram: SideHistogramPlot})
