import copy
from itertools import product, groupby

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.font_manager import FontProperties
from matplotlib.table import Table as mpl_Table

import param

from .. import DataStack, Matrix, DataLayer, HeatMap, View, DataOverlay, \
    Annotation, Curve, Scatter, TableStack, Table, Histogram, Stack, Overlay
from .viewplots import Plot

class MatrixPlot(Plot):

    normalize_individually = param.Boolean(default=False)

    show_values = param.Boolean(default=True, doc="""
        Whether to annotate the values when displaying a HeatMap.""")

    style_opts = param.List(default=['alpha', 'cmap', 'interpolation',
                                     'visible', 'filterrad', 'origin'],
                            constant=True, doc="""
        The style options for MatrixPlot are a subset of those used
        by matplotlib's imshow command. If supplied, the clim option
        will be ignored as it is computed from the input View.""")


    _stack_type = DataStack

    def __init__(self, view, zorder=0, **kwargs):
        self._stack = self._check_stack(view, (Matrix, DataLayer))
        super(MatrixPlot, self).__init__(zorder, **kwargs)


    def __call__(self, axis=None, cyclic_index=0, lbrt=None):
        view = self._stack.last
        xdim, ydim = view.dimensions
        (l, b, r, t) = (0, 0, 1, 1) if isinstance(view, HeatMap)\
            else self._stack.last.lbrt
        title = None if self.zorder > 0 else self._format_title(-1)
        xticks, yticks = self._compute_ticks(view)
        self.ax = self._axis(axis, title, str(xdim), str(ydim), (l, b, r, t),
                             xticks=xticks, yticks=yticks)

        opts = View.options.style(view)[cyclic_index]
        data = view.data
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
        clims = view.range if self.normalize_individually else self._stack.range
        im.set_clim(clims)
        self.handles['im'] = im

        if isinstance(view, HeatMap):
            self.ax.set_aspect(float(r - l)/(t-b))
            self.handles['annotations'] = {}
            self._annotate_values(view)

        if axis is None: plt.close(self.handles['fig'])
        return self.ax if axis else self.handles['fig']


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



    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        im = self.handles.get('im', None)

        view = list(self._stack.values())[n]
        im.set_data(view.data)

        if isinstance(view, HeatMap):
           self._annotate_values(view)

        if self.normalize_individually:
            im.set_clim(view.range)
        self._update_title(n)

        plt.draw()



class DataPlot(Plot):
    """
    A high-level plot, which will plot any DataView or DataStack type
    including DataOverlays.

    A generic plot that visualizes DataStacks containing DataOverlay or
    DataLayer objects.
    """

    _stack_type = DataStack

    style_opts = param.List(default=[], constant=True, doc="""
     DataPlot renders overlay layers which individually have style
     options but DataPlot itself does not.""")


    def __init__(self, overlays, **kwargs):
        self._stack = self._check_stack(overlays, DataOverlay)
        self.plots = []
        self.rescale = False
        super(DataPlot, self).__init__(**kwargs)


    def __call__(self, axis=None, lbrt=None, **kwargs):

        ax = self._axis(axis, None, self._stack.xlabel, self._stack.ylabel)

        stacks = self._stack.split_overlays()
        style_groups = dict((k, enumerate(list(v))) for k,v
                            in groupby(stacks, lambda s: s.style))

        for zorder, stack in enumerate(stacks):
            cyclic_index, _ = next(style_groups[stack.style])
            plotopts = View.options.plotting(stack).opts

            if zorder == 0:
                self.rescale = plotopts.get('rescale_individually', False)
                lbrt = self._stack.last.lbrt if self.rescale else self._stack.lbrt

            plotype = Plot.defaults[stack.type]
            plot = plotype(stack, size=self.size,
                           show_xaxis=self.show_xaxis, show_yaxis=self.show_yaxis,
                           show_legend=self.show_legend, show_title=self.show_title,
                           show_grid=self.show_grid, zorder=zorder,
                           **dict(plotopts, **kwargs))
            plot.aspect = self.aspect

            lbrt = None if stack.type == Annotation else lbrt
            plot(ax, cyclic_index=cyclic_index, lbrt=lbrt)
            self.plots.append(plot)

        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n if n < len(self) else len(self) - 1
        for zorder, plot in enumerate(self.plots):
            if zorder == 0:
                lbrt = list(self._stack.values())[n].lbrt if self.rescale else self._stack.lbrt
            plot.update_frame(n, lbrt)



class CurvePlot(Plot):
    """
    CurvePlot can plot Curve and DataStacks of Curve, which can be
    displayed as a single frame or animation. Axes, titles and legends
    are automatically generated from dim_info.

    If the dimension is set to cyclic in the dim_info it will rotate
    the curve so that minimum y values are at the minimum x value to
    make the plots easier to interpret.
    """

    center = param.Boolean(default=True)

    num_ticks = param.Integer(default=5)

    relative_labels = param.Boolean(default=False)

    rescale_individually = param.Boolean(default=False)

    show_frame = param.Boolean(default=False, doc="""
       Disabled by default for clarity.""")

    show_legend = param.Boolean(default=True, doc="""
      Whether to show legend for the plot.""")

    style_opts = param.List(default=['alpha', 'color', 'visible'],
                            constant=True, doc="""
       The style options for CurvePlot match those of matplotlib's
       LineCollection object.""")

    _stack_type = DataStack

    def __init__(self, curves, zorder=0, **kwargs):
        self._stack = self._check_stack(curves, Curve)
        self.cyclic_range = self._stack.last.cyclic_range
        self.ax = None

        super(CurvePlot, self).__init__(zorder, **kwargs)


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
        if self.center:
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
        curveview = self._stack.last

        # Create xticks and reorder data if cyclic
        xvals = curveview.data[:, 0]
        if self.cyclic_range is not None:
            if self.center:
                self.peak_argmax = np.argmax(curveview.data[:, 1])
            self._cyclic_curves(curveview)
            xticks = self._cyclic_reduce_ticks(self.xvalues)
        else:
            xticks = self._reduce_ticks(xvals)

        if lbrt is None:
            lbrt = curveview.lbrt if self.rescale_individually else self._stack.lbrt

        self.ax = self._axis(axis, self._format_title(-1), curveview.xlabel,
                             curveview.ylabel, xticks=xticks, lbrt=lbrt)

        # Create line segments and apply style
        line_segment = self.ax.plot(curveview.data[:, 0], curveview.data[:, 1],
                                    zorder=self.zorder, label=curveview.legend_label,
                                    **View.options.style(curveview)[cyclic_index])[0]

        self.handles['line_segment'] = line_segment

        # If legend enabled update handles and labels
        handles, labels = self.ax.get_legend_handles_labels()
        if len(handles) and self.show_legend:
            fontP = FontProperties()
            fontP.set_size('small')
            leg = self.ax.legend(handles[::-1], labels[::-1], prop=fontP)
            leg.get_frame().set_alpha(0.5)

        if axis is None: plt.close(self.handles['fig'])
        return self.ax if axis else self.handles['fig']


    def update_frame(self, n, lbrt=None):
        n = n  if n < len(self) else len(self) - 1
        curveview = list(self._stack.values())[n]
        if lbrt is None:
            lbrt = curveview.lbrt if self.rescale_individually else self._stack.lbrt

        if self.cyclic_range is not None:
            self._cyclic_curves(curveview)
        self.handles['line_segment'].set_xdata(curveview.data[:, 0])
        self.handles['line_segment'].set_ydata(curveview.data[:, 1])

        self._axis(self.ax, lbrt=lbrt)
        self._update_title(n)
        plt.draw()



class ScatterPlot(CurvePlot):
    """
    ScatterPlot can plot Scatter and DataStacks of Scatter, which can
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

    _stack_type = DataStack

    def __init__(self, points, zorder=0, **kwargs):
        self._stack = self._check_stack(points, Scatter)
        self.ax = None

        Plot.__init__(self, zorder, **kwargs)


    def __call__(self, axis=None, cyclic_index=0, lbrt=None):
        scatterview = self._stack.last
        self.cyclic_index = cyclic_index

        # Create xticks and reorder data if cyclic
        xvals = scatterview.data[:, 0]
        xticks = self._reduce_ticks(xvals)

        if lbrt is None:
            lbrt = scatterview.lbrt if self.rescale_individually else self._stack.lbrt

        self.ax = self._axis(axis, self._format_title(-1), scatterview.xlabel,
                             scatterview.ylabel, xticks=xticks, lbrt=lbrt)

        # Create line segments and apply style
        paths = self.ax.scatter(scatterview.data[:, 0], scatterview.data[:, 1],
                                zorder=self.zorder, label=scatterview.legend_label,
                                **View.options.style(scatterview)[cyclic_index])

        self.handles['paths'] = paths

        # If legend enabled update handles and labels
        handles, labels = self.ax.get_legend_handles_labels()
        if len(handles) and self.show_legend:
            fontP = FontProperties()
            fontP.set_size('small')
            leg = self.ax.legend(handles[::-1], labels[::-1], prop=fontP)
            leg.get_frame().set_alpha(0.5)

        if axis is None: plt.close(self.handles['fig'])
        return self.ax if axis else self.handles['fig']


    def update_frame(self, n, lbrt=None):
        n = n  if n < len(self) else len(self) - 1
        scatterview = list(self._stack.values())[n]
        if lbrt is None:
            lbrt = scatterview.lbrt if self.rescale_individually else self._stack.lbrt

        self.handles['paths'].remove()

        paths = self.ax.scatter(scatterview.data[:, 0], scatterview.data[:, 1],
                                zorder=self.zorder, label=scatterview.legend_label,
                                **View.options.style(scatterview)[self.cyclic_index])

        self.handles['paths'] = paths

        self._axis(self.ax, lbrt=lbrt)
        self._update_title(n)
        plt.draw()



class TablePlot(Plot):
    """
    A TablePlot can plot both TableViews and TableStacks which display
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


    _stack_type = TableStack

    def __init__(self, tables, zorder=0, **kwargs):
        self._stack = self._check_stack(tables, Table)
        super(TablePlot, self).__init__(zorder, **kwargs)


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


    def __call__(self, axis=None):

        tableview = self._stack.last

        ax = self._axis(axis, self._format_title(-1))

        ax.set_axis_off()
        size_factor = (1.0 - 2*self.border)
        table = mpl_Table(ax, bbox=[self.border, self.border,
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
        ax.add_table(table)

        self.handles['table'] = table
        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n if n < len(self) else len(self) - 1

        tableview = list(self._stack.values())[n]
        table = self.handles['table']

        for coords, cell in table.get_celld().items():
            value = tableview.cell_value(*coords)
            cell.set_text_props(text=self.pprint_value(value))

        # Resize fonts across table as necessary
        table.set_fontsize(self.max_font_size)
        table.auto_set_font_size(True)

        self._update_title(n)
        plt.draw()



class HistogramPlot(Plot):
    """
    HistogramPlot can plot DataHistograms and DataStacks of
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

    rescale_individually = param.Boolean(default=True, doc="""
        Whether to use redraw the axes per stack or per view.""")

    show_frame = param.Boolean(default=False, doc="""
       Disabled by default for clarity.""")

    _stack_type = DataStack

    def __init__(self, curves, zorder=0, **kwargs):
        self.center = False
        self.cyclic = False
        self.cyclic_index = 0
        self.ax = None

        self._stack = self._check_stack(curves, Histogram)
        super(HistogramPlot, self).__init__(zorder, **kwargs)

        if self.orientation == 'vertical':
            self.axis_settings = ['ylabel', 'xlabel', 'yticks']
        else:
            self.axis_settings = ['xlabel', 'ylabel', 'xticks']


    def __call__(self, axis=None, cyclic_index=0, lbrt=None):
        hist = self._stack.last
        self.cyclic_index = cyclic_index

        # Get plot ranges and values
        edges, hvals, widths, lims = self._process_hist(hist, lbrt)

        # Process and apply axis settings
        ticks = self._compute_ticks(edges, widths, lims)
        ax_settings = self._process_axsettings(hist, lims, ticks)
        if self.zorder == 0: ax_settings['title'] = self._format_title(-1)
        self.ax = self._axis(axis, **ax_settings)

        if self.orientation == 'vertical':
            self.offset_linefn = self.ax.axvline
            self.plotfn = self.ax.barh
        else:
            self.offset_linefn = self.ax.axhline
            self.plotfn = self.ax.bar

        # Plot bars and make any adjustments
        style = View.options.style(hist)[cyclic_index]
        bars = self.plotfn(edges, hvals, widths, zorder=self.zorder, **style)
        self.handles['bars'] = self._update_plot(-1, bars, lims) # Indexing top

        if not axis: plt.close(self.handles['fig'])
        return self.ax if axis else self.handles['fig']


    def _process_hist(self, hist, lbrt=None):
        """
        Get data from histogram, including bin_ranges and values.
        """
        self.cyclic = False if hist.cyclic_range is None else True
        edges = hist.edges[:-1]
        hist_vals = np.array(hist.values[:])
        widths = np.diff(hist.edges)
        if lbrt is None:
            xlims = hist.xlim if self.rescale_individually else self._stack.xlim
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
            labels = ["%.0f" % np.rad2deg(x) + '$^\circ$'
                      for x in xvals]
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


    def _update_plot(self, n, bars, lims):
        """
        Process bars is subclasses to manually adjust bars after
        being plotted.
        """
        for bar in bars:
            bar.set_clip_on(False)
        return bars


    def _update_artists(self, n, edges, hvals, widths, lims):
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
        plt.draw()


    def update_frame(self, n, lbrt=None):
        """
        Update the plot for an animation.
        """
        n = n if n < len(self) else len(self) - 1
        hist = list(self._stack.values())[n]

        # Process values, axes and style
        edges, hvals, widths, lims = self._process_hist(hist, lbrt)

        ticks = self._compute_ticks(edges, widths, lims)
        ax_settings = self._process_axsettings(hist, lims, ticks)
        self._axis(self.ax, **ax_settings)
        self._update_artists(n, edges, hvals, widths, lims)
        self._update_title(n)



class SideHistogramPlot(HistogramPlot):

    offset = param.Number(default=0.2, doc="""
        Histogram value offset for a colorbar.""")

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


    def _update_plot(self, n, bars, lims):
        """
        Process the bars and draw the offset line as necessary. If a
        color map is set in the style of the 'main' View object, color
        the bars appropriately, respecting the required normalization
        settings.
        """
        main = self.layout.main
        offset = self.offset * lims[3] * (1-self.offset)
        individually = View.options.plotting(main).opts.get('normalize_individually', False)

        if isinstance(main, Stack):
            if issubclass(main.type, Overlay):
                if individually:
                    main_range = list(main.split_stack()[0].values())[n].range
                else:
                    main_range = main.last[self.layout.main_layer].range
            else:
                main_range = list(main.values())[n].range if individually else main.range
        elif isinstance(main, View):
            main_range = main.range

        if offset and ('offset_line' not in self.handles):
            self.handles['offset_line'] = self.offset_linefn(offset,
                                                             linewidth=1.0,
                                                             color='k')
        elif offset:
            self._update_separator(lims, offset)


        # If .main is an Overlay or a Stack of Overlays get the correct style
        if isinstance(main, Stack) and issubclass(main.type, Overlay):
            style =  main.last[self.layout.main_layer].style
        elif isinstance(main, Overlay):
            style = main[self.layout.main_layer].style
        else:
            style = main.style

        cmap = cm.get_cmap(View.options.style(style).opts['cmap']) if self.offset else None
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
            color_val = (bar_bin+width/2.-lower_bound)/cmap_range
            bar.set_facecolor(cmap(color_val))
            bar.set_clip_on(False)



    def _update_separator(self, lims, offset):
        """
        Compute colorbar offset and update separator line
        if stack is non-zero.
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

Plot.defaults.update({Matrix: MatrixPlot,
                      HeatMap: MatrixPlot,
                      Curve: CurvePlot,
                      Scatter: ScatterPlot,
                      DataOverlay: DataPlot,
                      Table: TablePlot,
                      Histogram: HistogramPlot})

Plot.sideplots.update({Histogram: SideHistogramPlot})
