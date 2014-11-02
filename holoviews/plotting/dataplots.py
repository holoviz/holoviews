
from __future__ import unicode_literals

import numpy as np
from matplotlib import cm
from matplotlib.font_manager import FontProperties
from matplotlib.table import Table as mpl_Table

import param
from ..core import Map, View, Overlay
from ..view import Scatter, Curve, Histogram, ItemTable, Table
from .sheetplots import MatrixPlot
from .viewplots import Plot


class CurvePlot(Plot):
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

    style_opts = param.List(default=['alpha', 'color', 'visible'],
                            constant=True, doc="""
        The style options for CurvePlot match those of matplotlib's
        LineCollection object.""")

    _view_type = Curve

    def __init__(self, curves, **kwargs):
        super(CurvePlot, self).__init__(curves, **kwargs)
        self.cyclic_range = self._map.last.cyclic_range


    def _format_x_tick_label(self, x):
        return "%g" % round(x, 2)


    def _cyclic_format_x_tick_label(self, x):
        if self.relative_labels:
            return str(x)
        return str(int(np.round(180*x/self.cyclic_range)))


    def _rotate(self, seq, n=1):
        n = n % len(seq) # n=hop interval
        return seq[n:] + seq[:n]


    def _reduce_ticks(self, x_values):
        values = [x_values[0]]
        rangex = float(x_values[-1]) - x_values[0]
        for i in range(1, self.num_ticks+1):
            values.append(values[-1]+rangex/(self.num_ticks))
        return values, [self._format_x_tick_label(x) for x in values]


    def _cyclic_reduce_ticks(self, x_values):
        values = []
        labels = []
        step = self.cyclic_range / (self.num_ticks - 1)
        if self.relative_labels:
            labels.append(-90)
            label_step = 180 / (self.num_ticks - 1)
        else:
            labels.append(x_values[0])
            label_step = step
        values.append(x_values[0])
        for i in range(0, self.num_ticks - 1):
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
        x_values.append(x_values[0]+self.cyclic_range)
        y_values.append(y_values[0])

        curveview.data = np.vstack([x_values, y_values]).T
        self.xvalues = x_values


    def __call__(self, axis=None, cyclic_index=0, lbrt=None):
        curveview = self._map.last

        self.ax = self._init_axis(axis)

        # Create xticks and reorder data if cyclic
        xvals = curveview.data[:, 0]
        if self.autotick:
            xticks = None
        elif self.cyclic_range is not None:
            if self.center_cyclic:
                self.peak_argmax = np.argmax(curveview.data[:, 1])
            self._cyclic_curves(curveview)
            xticks = self._cyclic_reduce_ticks(self.xvalues)
        else:
            xticks = self._reduce_ticks(xvals)

        # Create line segments and apply style
        line_segment = self.ax.plot(curveview.data[:, 0], curveview.data[:, 1],
                                    zorder=self.zorder, label=curveview.label,
                                    **View.options.style(curveview)[cyclic_index])[0]

        self.handles['line_segment'] = line_segment

        return self._finalize_axis(self._keys[-1], xticks=xticks, lbrt=lbrt)


    def update_handles(self, view, key, lbrt=None):
        if self.cyclic_range is not None:
            self._cyclic_curves(view)
        self.handles['line_segment'].set_xdata(view.data[:, 0])
        self.handles['line_segment'].set_ydata(view.data[:, 1])



class ScatterPlot(CurvePlot):
    """
    ScatterPlot can plot Scatter and ViewMaps of Scatter, which can
    be displayed as a single frame or animation. Axes, titles and
    legends are automatically generated from dim_info.

    If the dimension is set to cyclic in the dim_info it will
    rotate the points curve so that minimum y values are at the minimum
    x value to make the plots easier to interpret.
    """

    style_opts = param.List(default=['alpha', 'color', 'edgecolors', 'facecolors',
                                     'linewidth', 'marker', 's', 'visible'],
                            constant=True, doc="""
       The style options for ScatterPlot match those of matplotlib's
       PolyCollection object.""")

    _view_type = Scatter

    def __call__(self, axis=None, cyclic_index=0, lbrt=None):
        scatterview = self._map.last
        self.cyclic_index = cyclic_index

        self.ax = self._init_axis(axis)

        # Create line segments and apply style
        paths = self.ax.scatter(scatterview.data[:, 0], scatterview.data[:, 1],
                                zorder=self.zorder, label=scatterview.label,
                                **View.options.style(scatterview)[cyclic_index])

        self.handles['paths'] = paths

        # Create xticks and reorder data if cyclic
        xvals = scatterview.data[:, 0]
        xticks = self._reduce_ticks(xvals)

        return self._finalize_axis(self._keys[-1], xticks=xticks, lbrt=lbrt)


    def update_handles(self, view, key, lbrt=None):
        self.handles['paths'].remove()

        paths = self.ax.scatter(view.data[:, 0], view.data[:, 1],
                                zorder=self.zorder, label=view.label,
                                **View.options.style(view)[self.cyclic_index])

        self.handles['paths'] = paths




class TablePlot(Plot):
    """
    A TablePlot can plot both TableViews and ViewMaps which display
    as either a single static table or as an animated table
    respectively.
    """

    border = param.Number(default=0.05, bounds=(0.0, 0.5), doc="""
        The fraction of the plot that should be empty around the
        edges.""")

    float_precision = param.Integer(default=3, doc="""
        The floating point precision to use when printing float
        numeric data types.""")

    max_value_len = param.Integer(default=20, doc="""
         The maximum allowable string length of a value shown in any
         table cell. Any strings longer than this length will be
         truncated.""")

    max_font_size = param.Integer(default=20, doc="""
       The largest allowable font size for the text in each table
       cell.""")

    font_types = param.Dict(default={'heading': FontProperties(weight='bold',
                                                               family='monospace')},
       doc="""The font style used for heading labels used for emphasis.""")


    style_opts = param.List(default=[], constant=True, doc="""
     TablePlot has specialized options which are controlled via plot
     options instead of matplotlib options.""")

    # Disable computing plot bounds from data.
    apply_databounds = False

    _view_type = ItemTable

    def pprint_value(self, value):
        """
        Generate the pretty printed representation of a value for
        inclusion in a table cell.
        """
        if isinstance(value, float):
            formatter = '{:.%df}' % self.float_precision
            formatted = formatter.format(value)
        else:
            formatted = str(value)

        if len(formatted) > self.max_value_len:
            return formatted[:(self.max_value_len-3)]+'...'
        else:
            return formatted


    def __call__(self, axis=None, cyclic_index=0, lbrt=None):

        tableview = self._map.last
        self.ax = self._init_axis(axis)

        self.ax.set_axis_off()
        size_factor = (1.0 - 2*self.border)
        table = mpl_Table(self.ax, bbox=[self.border, self.border,
                                         size_factor, size_factor])

        width = size_factor / tableview.cols
        height = size_factor / tableview.rows

        # Mapping from the cell coordinates to the dictionary key.

        for row in range(tableview.rows):
            for col in range(tableview.cols):
                value = tableview.cell_value(row, col)
                cell_text = self.pprint_value(value)

                cellfont = self.font_types.get(tableview.cell_type(row,col), None)
                font_kwargs = dict(fontproperties=cellfont) if cellfont else {}
                table.add_cell(row, col, width, height, text=cell_text,  loc='center',
                               **font_kwargs)

        table.set_fontsize(self.max_font_size)
        table.auto_set_font_size(True)
        self.ax.add_table(table)

        self.handles['table'] = table

        return self._finalize_axis(self._keys[-1])


    def update_handles(self, view, key, lbrt=None):
        table = self.handles['table']

        for coords, cell in table.get_celld().items():
            value = view.cell_value(*coords)
            cell.set_text_props(text=self.pprint_value(value))

        # Resize fonts across table as necessary
        table.set_fontsize(self.max_font_size)
        table.auto_set_font_size(True)



class HistogramPlot(Plot):
    """
    HistogramPlot can plot DataHistograms and ViewMaps of
    DataHistograms, which can be displayed as a single frame or
    animation.
    """

    style_opts = param.List(default=['alpha', 'color', 'align',
                                     'visible', 'edgecolor', 'log',
                                     'ecolor', 'capsize', 'error_kw',
                                     'hatch'], constant=True, doc="""
     The style options for HistogramPlot match those of
     matplotlib's bar command.""")

    num_ticks = param.Integer(default=5, doc="""
        If colorbar is enabled the number of labels will be overwritten.""")

    show_frame = param.Boolean(default=False, doc="""
        Disabled by default for clarity.""")

    show_grid = param.Boolean(default=False, doc="""
        Whether to overlay a grid on the axis.""")

    _view_type = Histogram

    def __init__(self, histograms, **kwargs):
        self.center = False
        self.cyclic = False
        self.cyclic_index = 0

        super(HistogramPlot, self).__init__(histograms, **kwargs)

        if self.orientation == 'vertical':
            self.axis_settings = ['ylabel', 'xlabel', 'yticks']
        else:
            self.axis_settings = ['xlabel', 'ylabel', 'xticks']


    def __call__(self, axis=None, cyclic_index=0, lbrt=None):
        hist = self._map.last
        self.cyclic_index = cyclic_index

        # Get plot ranges and values
        edges, hvals, widths, lims = self._process_hist(hist, lbrt)

        # Process and apply axis settings
        self.ax = self._init_axis(axis)

        if self.orientation == 'vertical':
            self.offset_linefn = self.ax.axvline
            self.plotfn = self.ax.barh
        else:
            self.offset_linefn = self.ax.axhline
            self.plotfn = self.ax.bar

        # Plot bars and make any adjustments
        style = View.options.style(hist)[cyclic_index]
        bars = self.plotfn(edges, hvals, widths, zorder=self.zorder, **style)
        self.handles['bars'] = self._update_plot(self._keys[-1], bars, lims) # Indexing top

        ticks = self._compute_ticks(edges, widths, lims)
        ax_settings = self._process_axsettings(hist, lims, ticks)

        return self._finalize_axis(self._keys[-1], **ax_settings)


    def _process_hist(self, hist, lbrt=None):
        """
        Get data from histogram, including bin_ranges and values.
        """
        self.cyclic = False if hist.cyclic_range is None else True
        edges = hist.edges[:-1]
        hist_vals = np.array(hist.values[:])
        widths = np.diff(hist.edges)
        if lbrt is None:
            xlims = hist.xlim if self.rescale_individually else self._map.xlim
            ylims = hist.ylim
        else:
            l, b, r, t = lbrt
            xlims = (l, r)
            ylims = (b, t)
        lims = xlims + ylims
        return edges, hist_vals, widths, lims


    def _compute_ticks(self, edges, widths, lims):
        """
        Compute the ticks either as cyclic values in degrees or as roughly
        evenly spaced bin centers.
        """
        if self.cyclic:
            x0, x1, _, _ = lims
            xvals = np.linspace(x0, x1, self.num_ticks)
            labels = ["%.0f" % np.rad2deg(x) + '\N{DEGREE SIGN}' for x in xvals]
        else:
            edge_inds = list(range(len(edges)))
            step = len(edges)/float(self.num_ticks-1)
            inds = [0] + [edge_inds[int(i*step)-1] for i in range(1, self.num_ticks)]
            xvals = [edges[i]+widths[i]/2. for i in inds]
            labels = ["%g" % round(x, 2) for x in xvals]
        return [xvals, labels]


    def _process_axsettings(self, hist, lims, ticks):
        """
        Get axis settings options including ticks, x- and y-labels
        and limits.
        """
        axis_settings = dict(zip(self.axis_settings, [hist.xlabel, hist.ylabel, ticks]))
        x0, x1, y0, y1 = lims
        axis_settings['lbrt'] = (0, x0, y1, x1) if self.orientation == 'vertical' else (x0, 0, x1, y1)

        return axis_settings


    def _update_plot(self, key, bars, lims):
        """
        Process bars can be subclassed to manually adjust bars
        after being plotted.
        """
        return bars


    def _update_artists(self, key, edges, hvals, widths, lims):
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


    def update_handles(self, view, key, lbrt=None):
        """
        Update the plot for an animation.
        """
        # Process values, axes and style
        edges, hvals, widths, lims = self._process_hist(view, lbrt)

        ticks = self._compute_ticks(edges, widths, lims)
        ax_settings = self._process_axsettings(view, lims, ticks)
        self._update_artists(key, edges, hvals, widths, lims)
        return ax_settings



class SideHistogramPlot(HistogramPlot):

    offset = param.Number(default=0.2, doc="""
        Histogram value offset for a colorbar.""")

    show_grid = param.Boolean(default=True, doc="""
        Whether to overlay a grid on the axis.""")

    show_title = param.Boolean(default=False, doc="""
        Titles should be disabled on all SidePlots to avoid clutter.""")

    show_xlabel = param.Boolean(default=False, doc="""
        Whether to show the x-label of the plot. Disabled by default
        because plots are often too cramped to fit the title correctly.""")

    def __init__(self, *args, **kwargs):
        self.layout = kwargs.pop('layout', None)
        super(SideHistogramPlot, self).__init__(*args, **kwargs)

    def _process_hist(self, hist, lbrt):
        """
        Subclassed to offset histogram by defined amount.
        """
        edges, hvals, widths, lims = super(SideHistogramPlot, self)._process_hist(hist, lbrt)
        offset = self.offset * lims[3]
        hvals += offset
        lims = lims[0:3] + (lims[3] + offset,)
        return edges, hvals, widths, lims


    def _process_axsettings(self, hist, lims, ticks):
        axsettings = super(SideHistogramPlot, self)._process_axsettings(hist, lims, ticks)
        if not self.show_xlabel:
            axsettings['ylabel' if self.orientation == 'vertical' else 'xlabel'] = ''
        return axsettings


    def _update_artists(self, n, edges, hvals, widths, lims):
        super(SideHistogramPlot, self)._update_artists(n, edges, hvals, widths, lims)
        self._update_plot(n, self.handles['bars'], lims)


    def _update_plot(self, key, bars, lims):
        """
        Process the bars and draw the offset line as necessary. If a
        color map is set in the style of the 'main' View object, color
        the bars appropriately, respecting the required normalization
        settings.
        """
        main = self.layout.main
        offset = self.offset * lims[3] * (1-self.offset)
        individually = View.options.plotting(main).opts.get('normalize_individually', False)

        if isinstance(main, Map):
            if issubclass(main.type, Overlay):
                top_map = main.split_overlays()[0]
                if individually:
                    main_range = top_map[key].range
                else:
                    main_range = top_map.range
            else:
                main_range = main[key].range if individually else main.range
        elif isinstance(main, View):
            main_range = main.range

        if offset and ('offset_line' not in self.handles):
            self.handles['offset_line'] = self.offset_linefn(offset,
                                                             linewidth=1.0,
                                                             color='k')
        elif offset:
            self._update_separator(lims, offset)


        # If .main is an Overlay or a Map of Overlays get the correct style
        if isinstance(main, Map) and issubclass(main.type, Overlay):
            style =  main.last[self.layout.main_layer].style
        elif isinstance(main, Overlay):
            style = main[self.layout.main_layer].style
        else:
            style = main.style

        cmap = cm.get_cmap(View.options.style(style).opts['cmap']) if self.offset else None
        main_range = View.options.style(style).opts.get('clims', main_range) if self.offset else None

        if cmap is not None:
            self._colorize_bars(cmap, bars, main_range)
        return bars


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


    def _update_separator(self, lims, offset):
        """
        Compute colorbar offset and update separator line
        if map is non-zero.
        """
        _, _, y0, y1 = lims
        offset_line = self.handles['offset_line']
        full_range = y1 - y0
        if full_range == 0:
            full_range = 1.
            y1 = y0 + 1.
        offset = (full_range*self.offset)*(1-self.offset)
        if y1 == 0:
            offset_line.set_visible(False)
        else:
            offset_line.set_visible(True)
            if self.orientation == 'vertical':
                offset_line.set_xdata(offset)
            else:
                offset_line.set_ydata(offset)

Plot.defaults.update({Curve: CurvePlot,
                      Scatter: ScatterPlot,
                      ItemTable: TablePlot,
                      Table: TablePlot,
                      Histogram: HistogramPlot})


Plot.sideplots.update({Histogram: SideHistogramPlot})
