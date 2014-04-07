from itertools import groupby
import string

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.table import Table
import matplotlib.gridspec as gridspec

import param

from dataviews import Stack, TableView, TableStack
from dataviews import DataCurves, DataStack, DataOverlay, DataHistogram
from sheetviews import SheetView, SheetOverlay, SheetLines, \
                       SheetStack, SheetPoints, CoordinateGrid, DataGrid
from views import GridLayout, View
from styles import Styles

class Plot(param.Parameterized):
    """
    A Plot object returns either a matplotlib figure object (when
    called or indexed) or a matplotlib animation object as
    appropriate. Plots take view objects such as SheetViews,
    SheetLines or SheetPoints as inputs and plots them in the
    appropriate format. As views may vary over time, all plots support
    animation via the anim() method.
    """

    size = param.NumericTuple(default=(5,5), doc="""
      The matplotlib figure size in inches.""")

    show_axes = param.Boolean(default=True, doc="""
      Whether to show labelled axes for the plot.""")

    show_grid = param.Boolean(default=False, doc="""
      Whether to show a Cartesian grid on the plot.""")

    show_title = param.Boolean(default=True, doc="""
      Whether to display the plot title.""")

    _stack_type = Stack

    def __init__(self, **kwargs):
        super(Plot, self).__init__(**kwargs)
        # List of handles to matplotlib objects for animation update
        self.handles = {'fig':None}


    def _title_fields(self, stack):
        """
        Returns the formatting fields in the title string supplied by
        the view object.
        """
        if stack.title is None:  return []
        parse = list(string.Formatter().parse(stack.title))
        if parse == []: return []
        return [f for f in zip(*parse)[1] if f is not None]


    def _format_title(self, stack, index):
        """
        Format a title string based on the keys/values of the view
        stack.
        """
        if stack.values()[index].title is not None:
            return stack.values()[index].title
        labels = stack.dimension_labels
        vals = stack.keys()[index]
        if not isinstance(vals, tuple): vals = (vals,)
        fields = self._title_fields(stack)
        if fields == []:
            return stack.title if stack.title else ''
        label_map = dict(('label%d' % i, l) for (i,l) in enumerate(labels))
        val_map =   dict(('value%d' % i, float(l)) for (i,l) in enumerate(vals))
        format_items = dict(label_map,**val_map)
        if not set(fields).issubset(format_items):
            raise Exception("Cannot format")
        return stack.title.format(**format_items)


    def _check_stack(self, view, element_type=View):
        """
        Helper method that ensures a given view is always returned as
        an imagen.SheetStack object.
        """
        if not isinstance(view, self._stack_type):
            stack = self._stack_type(initial_items=(0, view), title=view.title)
            if self._title_fields(stack) != []:
                raise Exception('Can only format title string for animation and stacks.')
        else:
            stack = view

        if not issubclass(stack.type, element_type):
            raise TypeError("Requires View, Animation or Stack of type %s" % element_type)
        return stack


    def _axis(self, axis, title, xlabel=None, ylabel=None, lbrt=None, xticks=None, yticks=None):
        "Return an axis which may need to be initialized from a new figure."
        if axis is None:
            fig = plt.figure()
            self.handles['fig'] = fig
            fig.set_size_inches(list(self.size))
            axis = fig.add_subplot(111)
            axis.set_aspect('auto')

        if not self.show_axes:
            axis.set_axis_off()
        elif self.show_grid:
            axis.get_xaxis().grid(True)
            axis.get_yaxis().grid(True)

        if lbrt is not None:
            (l, b, r, t) = lbrt
            axis.set_xlim((l, r))
            axis.set_ylim((b, t))

        if xticks:
            axis.set_xticks(xticks[0])
            axis.set_xticklabels(xticks[1])

        if yticks:
            axis.set_yticks(yticks[0])
            axis.set_yticklabels(yticks[1])

        if self.show_title:
            self.handles['title'] = axis.set_title(title)

        if xlabel: axis.set_xlabel(xlabel)
        if ylabel: axis.set_ylabel(ylabel)
        return axis


    def __getitem__(self, frame):
        """
        Get the matplotlib figure at the given frame number.
        """
        if frame > len(self):
            self.warn("Showing last frame available: %d" % len(self))
        fig = self()
        self.update_frame(frame)
        return fig


    def anim(self, start=0, stop=None, fps=30):
        """
        Method to return an Matplotlib animation. The start and stop
        frames may be specified as well as the fps.
        """
        figure = self()
        frames = range(len(self))[slice(start, stop, 1)]
        anim = animation.FuncAnimation(figure, self.update_frame,
                                       frames=frames,
                                       interval = 1000.0/fps)
        # Close the figure handle
        plt.close(figure)
        return anim


    def update_frame(self, n):
        """
        Set the plot(s) to the given frame number.  Operates by
        manipulating the matplotlib objects held in the self._handles
        dictionary.

        If n is greater than the number of available frames, update
        using the last available frame.
        """
        n = n  if n < len(self) else len(self) - 1
        raise NotImplementedError


    def __len__(self):
        """
        Returns the total number of available frames.
        """
        raise NotImplementedError


    def __call__(self, ax=False, zorder=0):
        """
        Return a matplotlib figure.
        """
        raise NotImplementedError



class SheetLinesPlot(Plot):

    _stack_type = SheetStack

    def __init__(self, contours, **kwargs):
        self._stack = self._check_stack(contours, SheetLines)
        super(SheetLinesPlot, self).__init__(**kwargs)


    def __call__(self, axis=None, zorder=0):
        title = self._format_title(self._stack, -1)
        ax = self._axis(axis, title, 'x','y', self._stack.bounds.lbrt())
        lines = self._stack.top
        line_segments = LineCollection([], zorder=zorder, **Styles[lines].opts)
        line_segments.set_paths(lines.data)
        self.handles['line_segments'] = line_segments
        ax.add_collection(line_segments)
        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        contours = self._stack.values()[n]
        self.handles['line_segments'].set_paths(contours.data)
        if self.show_title:
            self.handles['title'].set_text(self._format_title(self._stack, n))
        plt.draw()


    def __len__(self):
        return len(self._stack)



class SheetPointsPlot(Plot):

    _stack_type = SheetStack

    def __init__(self, contours, **kwargs):
        self._stack = self._check_stack(contours, SheetPoints)
        super(SheetPointsPlot, self).__init__(**kwargs)


    def __call__(self, axis=None, zorder=0):
        title = self._format_title(self._stack, -1)
        ax = self._axis(axis, title, 'x','y', self._stack.bounds.lbrt())
        points = self._stack.top
        scatterplot = plt.scatter(points.data[:,0], points.data[:,1],
                                  zorder=zorder, **Styles[points].opts)
        self.handles['scatter'] = scatterplot
        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n if n < len(self) else len(self) - 1
        points = self._stack.values()[n]
        self.handles['scatter'].set_offsets(points.data)
        if self.show_title:
            self.handles['title'].set_text(self._format_title(self._stack, n))
        plt.draw()


    def __len__(self):
        return len(self._stack)



class SheetViewPlot(Plot):

    colorbar = param.ObjectSelector(default=None,
                                    objects=['horizontal','vertical', None],
        doc="""The style of the colorbar if applicable. """)

    _stack_type = SheetStack

    def __init__(self, sheetview, **kwargs):
        self._stack = self._check_stack(sheetview, SheetView)
        super(SheetViewPlot, self).__init__(**kwargs)


    def toggle_colorbar(self, bar, cmax):
        visible = not (cmax == 0.0)
        bar.set_clim(vmin=0.0, vmax=cmax if visible else 1.0)
        elements = (bar.ax.get_xticklines()
                    + bar.ax.get_ygridlines()
                    + bar.ax.get_children())
        for el in elements:
            el.set_visible(visible)
        bar.draw_all()


    def __call__(self, axis=None, zorder=0):
        sheetview = self._stack.top
        title = self._format_title(self._stack, -1)
        (l,b,r,t) = self._stack.bounds.lbrt()
        ax = self._axis(axis, title, 'x','y', (l,b,r,t))

        options = Styles[sheetview].opts
        if sheetview.depth!=1:
            options.pop('cmap',None)

        im = ax.imshow(sheetview.data, extent=[l,r,b,t],
                       zorder=zorder, interpolation='nearest', **options)
        self.handles['im'] = im

        normalization = sheetview.data.max()
        cyclic_range = sheetview.cyclic_range
        im.set_clim([0.0, cyclic_range if cyclic_range else normalization])

        if self.colorbar is not None:
            np.seterr(divide='ignore')
            bar = plt.colorbar(im, ax=ax,
                               orientation=self.colorbar)
            np.seterr(divide='raise')
            self.toggle_colorbar(bar, normalization)
            self.handles['bar'] = bar
        else:
            plt.tight_layout()

        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        im = self.handles.get('im',None)
        bar = self.handles.get('bar',None)

        sheetview = self._stack.values()[n]
        im.set_data(sheetview.data)
        normalization = sheetview.data.max()
        cmax = max([normalization, sheetview.cyclic_range])
        im.set_clim([0.0, cmax])
        if self.colorbar: self.toggle_colorbar(bar, cmax)

        if self.show_title:
            self.handles['title'].set_text(self._format_title(self._stack, n))
        plt.draw()


    def __len__(self):
        return len(self._stack)



class SheetPlot(Plot):
    """
    A generic plot that visualizes SheetOverlays which themselves may
    contain SheetLayers of type SheetView, SheetPoints or SheetLine
    objects.
    """

    _stack_type = SheetStack

    def __init__(self, overlays, **kwargs):
        self._stack = self._check_stack(overlays, SheetOverlay)
        self.plots = []
        super(SheetPlot, self).__init__(**kwargs)


    def __call__(self, axis=None, zorder=0):
        title = self._format_title(self._stack, -1)
        ax = self._axis(axis, title, 'x','y', self._stack.bounds.lbrt())

        for zorder, stack in enumerate(self._stack.split()):
            plotype = viewmap[stack.type]
            plot = plotype(stack, size=self.size, show_axes=self.show_axes)
            plot(ax, zorder=zorder)
            self.plots.append(plot)

        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        for plot in self.plots:
            plot.update_frame(n)


    def __len__(self):
        return len(self._stack)



class GridLayoutPlot(Plot):
    """
    Plot a group of views in a grid layout based on a GridLayout view
    object.
    """

    roi = param.Boolean(default=False, doc="""
      Whether to apply the ROI to each element of the grid.""")

    show_axes= param.Boolean(default=True, constant=True, doc="""
      Whether to show labelled axes for individual subplots.""")

    def __init__(self, grid, **kwargs):

        if not isinstance(grid, GridLayout):
            raise Exception("GridLayoutPlot only accepts GridLayouts.")

        self.grid = grid
        self.subplots = []
        self.rows, self.cols = grid.shape
        self._gridspec = gridspec.GridSpec(self.rows, self.cols)
        super(GridLayoutPlot, self).__init__(**kwargs)


    def __call__(self, axis=None):
        ax = self._axis(axis, '', '','', None)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        coords = [(r,c) for c in range(self.cols) for r in range(self.rows)]

        self.subplots = []
        for (r,c) in coords:
            view = self.grid.get((r,c),None)
            if view is not None:
                subax = plt.subplot(self._gridspec[r,c])
                subview = view.roi if self.roi else view
                vtype = subview.type if isinstance(subview,Stack) else subview.__class__
                subplot = viewmap[vtype](subview, show_axes=self.show_axes)
            self.subplots.append(subplot)
            subplot(subax)

        if not axis: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        for subplot in self.subplots:
            subplot.update_frame(n)


    def __len__(self):
        return len(self.grid)



class CoordinateGridPlot(Plot):
    """
    CoordinateGridPlot evenly spaces out plots of individual projections on
    a grid, even when they differ in size. The projections can be situated
    or an ROI can be applied to each element. Since this class uses a single
    axis to generate all the individual plots it is much faster than the
    equivalent using subplots.
    """

    border = param.Number(default=10, doc="""
        Aggregate border as a fraction of total plot size.""")

    situate = param.Boolean(default=False, doc="""
        Determines whether to situate the projection in the full bounds or
        apply the ROI.""")

    def __init__(self, grid, **kwargs):
        if not isinstance(grid, CoordinateGrid):
            raise Exception("CoordinateGridPlot only accepts ProjectionGrids.")
        self.grid = grid
        super(CoordinateGridPlot, self).__init__(**kwargs)


    def __call__(self, axis=None):
        ax = self._axis(axis, '', '','', None)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        grid_shape = [[v for (k,v) in col[1]] for col in groupby(self.grid.items(),
                                                                 lambda (k,v): k[0])]
        width, height, b_w, b_h = self._compute_borders(grid_shape)

        plt.xlim(0, width)
        plt.ylim(0, height)

        self.handles['projs'] = []
        x, y = b_w, b_h
        for row in grid_shape:
            for view in row:
                w, h = self._get_dims(view)
                if view.type == SheetOverlay:
                    data = view.top[-1].data if self.situate else view.top[-1].roi.data
                    cmap = {'cmap':view.top[-1].mode} if view.top[-1].depth==1 else {}
                else:
                    data = view.top.data if self.situate else view.top.roi.data
                    cmap = {'cmap':view.top.mode} if view.top.depth==1 else {}

                self.handles['projs'].append(plt.imshow(data, extent=(x,x+w, y, y+h),
                                                        interpolation='nearest',
                                                        **cmap))
                y += h + b_h
            y = b_h
            x += w + b_w

        if not axis: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        for i, plot in enumerate(self.handles['projs']):
            view = self.grid.values()[i].values()[n]
            if isinstance(view, SheetOverlay):
                data = view[-1].data if self.situate else view[-1].roi.data
            else:
                data = view.data if self.situate else view.roi.data

            plot.set_data(data)


    def _get_dims(self, view):
        l,b,r,t = view.bounds.lbrt() if self.situate else view.roi.bounds.lbrt()
        return (r-l, t-b)


    def _compute_borders(self, grid_shape):
        height = 0
        self.rows = 0
        for view in grid_shape[0]:
            height += self._get_dims(view)[1]
            self.rows += 1

        width = 0
        self.cols = 0
        for view in [row[0] for row in grid_shape]:
            width += self._get_dims(view)[0]
            self.cols += 1

        border_width = (width/10)/(self.cols+1)
        border_height = (height/10)/(self.rows+1)
        width += width/10
        height += height/10

        return width, height, border_width, border_height


    def __len__(self):
        return len(self.grid)



class DataCurvePlot(Plot):
    """
    DataCurvePlot can plot DataCurves and DataStacks of DataCurves,
    which can be displayed as a single frame or animation. Axes,
    titles and legends are automatically generated from the metadata
    and dim_info.

    If the dimension is set to cyclic in the dim_info it will
    rotate the curve so that minimum y values are at the minimum
    x value to make the plots easier to interpret.
    """

    center = param.Boolean(default=True)

    num_ticks = param.Integer(default=5)

    relative_labels = param.Boolean(default=False)

    _stack_type = DataStack

    def __init__(self, curves, **kwargs):
        self._stack = self._check_stack(curves, DataCurves)
        self.cyclic_range = self._stack.top.cyclic_range

        super(DataCurvePlot, self).__init__(**kwargs)


    def _format_x_tick_label(self, x):
        return "%g" % round(x, 2)


    def _cyclic_format_x_tick_label(self, x):
        if self.relative_labels:
            return str(x)
        return str(int(np.round(180*x/self.cyclic_range)))


    def _rotate(self, seq, n=1):
        n = n % len(seq) # n=hop interval
        return seq[n:] + seq[:n]


    def _curve_values(self, coord, curve):
        """Return the x, y, and x ticks values for the specified curve from the curve_dict"""
        x, y = coord
        x_values = curve.keys()
        y_values = [curve[k, x, y] for k in x_values]
        self.x_values = x_values
        return x_values, y_values, x_values


    def _reduce_ticks(self, x_values):
        values = [x_values[0]]
        rangex = x_values[-1] - x_values[0]
        for i in xrange(1, self.num_ticks+1):
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
        for i in xrange(0, self.num_ticks - 1):
            labels.append(labels[-1] + label_step)
            values.append(values[-1] + step)
        return values, [self._cyclic_format_x_tick_label(x) for x in labels]


    def _cyclic_curves(self, lines):
        """
        Mutate the lines object to generate a rotated cyclic curves.
        """
        for idx, line in enumerate(lines.data):
            x_values = list(line[:, 0])
            y_values = list(line[:, 1])
            if self.center:
                rotate_n = self.peak_argmax+len(x_values)/2
                y_values = self._rotate(y_values, n=rotate_n)
                ticks = self._rotate(x_values, n=rotate_n)
            else:
                ticks = list(x_values)

            ticks.append(ticks[0])
            x_values.append(x_values[0]+self.cyclic_range)
            y_values.append(y_values[0])

            lines.data[idx] = np.vstack([x_values, y_values]).T
        self.xvalues = x_values


    def _find_peak(self, lines):
        """
        Finds the peak value in the supplied lines object to center the
        relative labels around if the relative_labels option is enabled.
        """
        self.peak_argmax = 0
        max_y = 0.0
        for line in lines:
            y_values = line[:, 1]
            if np.max(y_values) > max_y:
                max_y = np.max(y_values)
                self.peak_argmax = np.argmax(y_values)


    def __call__(self, axis=None, zorder=0, cyclic_index=0, lbrt=None):
        title = self._format_title(self._stack, -1)
        lines = self._stack.top

        # Create xticks and reorder data if cyclic
        xvals = lines[0][:, 0]
        if self.cyclic_range is not None:
            if self.center:
                self._find_peak(lines)
            self._cyclic_curves(lines)
            xticks = self._cyclic_reduce_ticks(self.xvalues)
        else:
            xticks = self._reduce_ticks(xvals)

        if lbrt is None:
            lbrt = lines.lbrt

        ax = self._axis(axis, title, lines.xlabel, lines.ylabel,
                        xticks=xticks, lbrt=lbrt)

        # Create line segments and apply style
        line_segments = LineCollection([], zorder=zorder, **Styles[lines][cyclic_index])
        line_segments.set_paths(lines.data)

        # Add legend
        line_segments.set_label(lines.legend_label)

        self.handles['line_segments'] = line_segments
        ax.add_collection(line_segments)

        # If legend enabled update handles and labels
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) and self.show_legend:
            fontP = FontProperties()
            fontP.set_size('small')
            leg = ax.legend(handles[::-1], labels[::-1], prop=fontP)
            leg.get_frame().set_alpha(0.5)

        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        contours = self._stack.values()[n]
        if self.cyclic_range is not None:
            self._cyclic_curves(contours)
        self.handles['line_segments'].set_paths(contours.data)
        if self.show_title:
            self.handles['title'].set_text(self._format_title(self._stack, n))
        plt.draw()


    def __len__(self):
        return len(self._stack)



class DataPlot(Plot):
    """
    A high-level plot, which will plot any DataView or DataStack type
    including DataOverlays.

    A generic plot that visualizes DataStacks containing DataOverlay or
    DataLayer objects.
    """

    show_legend = param.Boolean(default=True, doc="""
      Whether to show legend for the plot.""")

    _stack_type = DataStack

    def __init__(self, overlays, **kwargs):
        self._stack = self._check_stack(overlays, DataOverlay)
        self.plots = []
        super(DataPlot, self).__init__(**kwargs)


    def __call__(self, axis=None, zorder=0, **kwargs):
        title = self._format_title(self._stack, -1)

        ax = self._axis(axis, title, self._stack.xlabel, self._stack.ylabel, self._stack.lbrt)


        stacks = self._stack.split()
        style_groups = dict((k, enumerate(list(v))) for k,v
                            in groupby(stacks, lambda s: s.style))

        for zorder, stack in enumerate(stacks):
            cyclic_index, _ = style_groups[stack.style].next()

            plotype = viewmap[stack.type]
            plot = plotype(stack, size=self.size, show_axes=self.show_axes,
                           show_legend=self.show_legend, show_title=self.show_title,
                           **kwargs)
            plot(ax, zorder=zorder, lbrt=self._stack.lbrt, cyclic_index=cyclic_index)
            self.plots.append(plot)

        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n if n < len(self) else len(self) - 1
        for plot in self.plots:
            plot.update_frame(n)


    def __len__(self):
        return len(self._stack)



class DataGridPlot(Plot):
    """
    Plot a group of views in a grid layout based on a DataGrid view
    object.
    """

    show_axes= param.Boolean(default=False, constant=True, doc="""
      Whether to show labelled axes for individual subplots.""")

    show_legend = param.Boolean(default=False, doc="""
      Legends add to much clutter in a grid and are disabled by default.""")

    show_title = param.Boolean(default=False)

    def __init__(self, grid, **kwargs):

        if not isinstance(grid, DataGrid):
            raise Exception("DataGridPlot only accepts DataGrids.")

        self.grid = grid
        self.subplots = []
        x, y = zip(*grid.keys())
        self.rows, self.cols = (len(set(x)), len(set(y)))
        self._gridspec = gridspec.GridSpec(self.rows, self.cols)
        super(DataGridPlot, self).__init__(**kwargs)


    def __call__(self, axis=None):
        ax = self._axis(axis, '', '','', None)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        self.subplots = []
        r, c = (self.rows-1, 0)
        for coord in self.grid.keys():
            view = self.grid.get(coord, None)
            if view is not None:
                subax = plt.subplot(self._gridspec[c, r])
                vtype = view.type if isinstance(view, DataStack) else view.__class__
                subplot = viewmap[vtype](view, show_axes=self.show_axes,
                                         show_legend=self.show_legend,
                                         show_title=self.show_title)
            self.subplots.append(subplot)
            subplot(subax)
            if c != self.cols-1:
                c += 1
            else:
                c = 0
                r -= 1

        if not axis: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        for subplot in self.subplots:
            subplot.update_frame(n)


    def __len__(self):
        return len(self.grid)



class TablePlot(Plot):
    """
    A TablePlot can plot both TableViews and TableStacks which display
    as either a single static table or as an animated table
    respectively.
    """

    border = param.Number(default = 0.05, bounds=(0.0, 0.5), doc="""
        The fraction of the plot that should be empty around the
        edges.""")

    float_precision = param.Integer(default=3, doc="""
        The floating point precision to use when printing float
        numeric data types.""")

    max_value_len = param.Integer(default=20, doc="""
         The maximum allowable string length of a value shown in any
         table cell. Any strings longer than this length will be
         truncated.""")

    max_font_size = param.Integer(default = 20, doc="""
       The largest allowable font size for the text in each table
       cell.""")

    font_types = param.Dict(default = {'heading':FontProperties(weight='bold',
                                                                family='monospace')},
       doc="""The font style used for heading labels used for emphasis.""")

    _stack_type = TableStack

    def __init__(self, contours, **kwargs):
        self._stack = self._check_stack(contours, TableView)
        super(TablePlot, self).__init__(**kwargs)


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


    def __call__(self, axis=None, zorder=0):
        title = self._format_title(self._stack, -1)
        ax = self._axis(axis, title)

        tableview = self._stack.top
        ax.set_axis_off()
        size_factor = (1.0 - 2*self.border)
        table = Table(ax, bbox=[self.border, self.border,
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

        tableview = self._stack.values()[n]
        table = self.handles['table']

        for coords, cell in table.get_celld().items():
            value = tableview.cell_value(*coords)
            cell.set_text_props(text=self.pprint_value(value))

        # Resize fonts across table as necessary
        table.set_fontsize(self.max_font_size)
        table.auto_set_font_size(True)

        if self.show_title:
            self.handles['title'].set_text(self._format_title(self._stack, n))
        plt.draw()

    def __len__(self):
        return len(self._stack)



class DataHistogramPlot(Plot):
    """
    DataHistogramPlot can plot DataHistograms and DataStacks of
    DataHistograms, which can be displayed as a single frame or
    animation.
    """
    _stack_type = DataStack

    def __init__(self, curves, **kwargs):
        self._stack = self._check_stack(curves, DataHistogram)
        super(DataHistogramPlot, self).__init__(**kwargs)


    def __call__(self, axis=None, zorder=0, color='b', cyclic_index=0, lbrt=None):

        hist = self._stack.top
        title = self._format_title(self._stack, -1)
        ax = self._axis(axis, title, hist.xlabel, hist.ylabel)


        bars = plt.bar(hist.edges, hist.hist, width=1.0, fc='w', zorder=zorder) # Custom color and width
        self.handles['bars'] = bars

        if not axis: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        hist = self._stack.values()[n]
        bars = self.handles['bars']
        if hist.ndims != len(bars):
            raise Exception("Histograms must all have the same bin edges.")

        for i, bar in enumerate(bars):
            height = hist.hist[i]
            bar.set_height(height)

        if self.show_title:
            self.handles['title'].set_text(self._format_title(self._stack, n))
        plt.draw()


    def __len__(self):
        return len(self._stack)


viewmap = {SheetView: SheetViewPlot,
           SheetPoints: SheetPointsPlot,
           SheetLines: SheetLinesPlot,
           SheetOverlay: SheetPlot,
           CoordinateGrid: CoordinateGridPlot,
           DataCurves: DataCurvePlot,
           DataOverlay: DataPlot,
           DataGrid: DataGridPlot,
           TableView: TablePlot,
           DataHistogram:DataHistogramPlot
}


__all__ = ['viewmap'] + list(set([_k for _k,_v in locals().items()
                                  if isinstance(_v, type) and issubclass(_v, Plot)]))
