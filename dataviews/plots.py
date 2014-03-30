from itertools import groupby, cycle
import string

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec

import param

from dataviews import DataCurves, DataStack, DataOverlay
from sheetviews import SheetView, SheetOverlay, SheetLines, SheetStack, SheetPoints, CoordinateGrid, DataGrid
from views import GridLayout, View, Stack



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

    show_legend = param.Boolean(default=True, doc="""
      Whether to show legend for the plot.""")

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

        if not stack.type == element_type:
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

            if lbrt is not None:
                (l, b, r, t) = lbrt
                axis.set_xlim((l, r))
                axis.set_ylim((b, t))

        if not self.show_axes:
            axis.set_axis_off()
        elif self.show_grid:
            axis.get_xaxis().grid(True)
            axis.get_yaxis().grid(True)

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
        line_segments = LineCollection([], zorder=zorder, **lines.style)
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
                                  zorder=zorder, **points.style)
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

        cmap = {'cmap':sheetview.mode} if sheetview.depth==1 else {}
        im = ax.imshow(sheetview.data, extent=[l,r,b,t],
                       zorder=zorder, interpolation='nearest', **cmap)
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
        cmap = 'hsv' if (sheetview.cyclic_range is not None) else 'gray'
        im.set_cmap(sheetview.style.get('cmap', cmap))
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
                vtype = subview.type if isinstance(subview,SheetStack) else subview.__class__
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


    def _format_legend(self, lines):
        units = dict([(dim, info.get('unit', ''))
                      for dim, info in self._stack._dimensions])
        return ', '.join(["%s = %.2f%s" % (l, v, units[l]) for l, v in lines.labels])


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


    def __call__(self, axis=None, zorder=0, color='b'):
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

        l, r = lines.xlim
        b, t = lines.ylim

        ax = self._axis(axis, title, lines.xlabel, lines.ylabel, xticks=xticks, lbrt=(l, b, r, t))
        
        # Create line segments and apply style
        line_segments = LineCollection([], zorder=zorder, **lines.style)
        line_segments.set_paths(lines.data)
        line_segments.set_linewidth(2.0)
        if 'color' not in lines.style:
            line_segments.set_color(color)

        # Add legend
        label = self._format_legend(lines)
        line_segments.set_label(label)

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

    color_cycle = param.List(default=['b', 'g', 'r', 'y', 'm'])

    _stack_type = DataStack

    def __init__(self, overlays, **kwargs):
        self._stack = self._check_stack(overlays, DataOverlay)
        self.plots = []
        super(DataPlot, self).__init__(**kwargs)
        self._color = cycle(self.color_cycle)


    def __call__(self, axis=None, zorder=0, **kwargs):
        title = self._format_title(self._stack, -1)

        ax = self._axis(axis, title, self._stack.xlabel, self._stack.ylabel, self._stack.lbrt)

        for zorder, stack in enumerate(self._stack.split()):
            plotype = viewmap[stack.type]
            plot = plotype(stack, size=self.size, show_axes=self.show_axes,
                           show_legend=self.show_legend, show_title=self.show_title,
                           **kwargs)
            plot(ax, zorder=zorder, color=self._color.next())
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



viewmap = {SheetView: SheetViewPlot,
           SheetPoints: SheetPointsPlot,
           SheetLines: SheetLinesPlot,
           SheetOverlay: SheetPlot,
           CoordinateGrid: CoordinateGridPlot,
           DataCurves: DataCurvePlot,
           DataOverlay: DataPlot,
           DataStack: DataPlot,
           DataGrid: DataGridPlot}


__all__ = ['viewmap'] + list(set([_k for _k,_v in locals().items()
                                  if isinstance(_v, type) and issubclass(_v, Plot)]))
