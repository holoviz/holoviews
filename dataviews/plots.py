import copy
from itertools import groupby

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.table import Table
import matplotlib.gridspec as gridspec

import param

from dataviews import NdMapping, Stack, TableView, TableStack
from dataviews import DataStack, DataOverlay, DataLayer, DataCurves, DataHistogram
from sheetviews import SheetView, SheetOverlay, SheetLines, \
                       SheetStack, SheetPoints, CoordinateGrid, DataGrid
from views import GridLayout, Layout, Overlay, View, Annotation

from options import options, channels
from operation import RGBA, HCS, AlphaOverlay


class Plot(param.Parameterized):
    """
    A Plot object returns either a matplotlib figure object (when
    called or indexed) or a matplotlib animation object as
    appropriate. Plots take view objects such as SheetViews,
    SheetLines or SheetPoints as inputs and plots them in the
    appropriate format. As views may vary over time, all plots support
    animation via the anim() method.
    """

    size = param.NumericTuple(default=(5, 5), doc="""
      The matplotlib figure size in inches.""")

    show_grid = param.Boolean(default=False, doc="""
      Whether to show a Cartesian grid on the plot.""")

    show_title = param.Boolean(default=True, doc="""
      Whether to display the plot title.""")

    show_xaxis = param.ObjectSelector(default='bottom',
                                      objects=['top', 'bottom', None], doc="""
      Whether and where to display the xaxis.""")

    show_yaxis = param.ObjectSelector(default='left',
                                      objects=['left', 'right', None], doc="""
      Whether and where to display the yaxis.""")

    style_opts = param.List(default=[], constant=True, doc="""
     A list of matplotlib keyword arguments that may be supplied via a
     style options object. Each subclass should override this
     parameter to list every option that works correctly.""")

    aspect = param.ObjectSelector(default=None,
                                  objects=['auto', 'equal','square', None],
                                  doc="""
    The aspect ratio mode of the plot. By default, a plot may select
    its own appropriate aspect ratio but sometimes it may be necessary
    to force a square aspect ratio (e.g. to display the plot as an
    element of a grid). The modes 'auto' and 'equal' correspond to the
    axis modes of the same name in matplotlib.""" )

    _stack_type = Stack

    def __init__(self, **kwargs):
        super(Plot, self).__init__(**kwargs)
        # List of handles to matplotlib objects for animation update
        self.handles = {'fig':None}

    def _check_stack(self, view, element_type=View):
        """
        Helper method that ensures a given view is always returned as
        an imagen.SheetStack object.
        """
        if not isinstance(view, self._stack_type):
            stack = self._stack_type(initial_items=(0, view))
        else:
            stack = view

        if not issubclass(stack.type, element_type):
            raise TypeError("Requires View, Animation or Stack of type %s" % element_type)
        return stack


    def _axis(self, axis, title=None, xlabel=None, ylabel=None,
              lbrt=None, xticks=None, yticks=None):
        "Return an axis which may need to be initialized from a new figure."
        if axis is None:
            fig = plt.figure()
            self.handles['fig'] = fig
            fig.set_size_inches(list(self.size))
            axis = fig.add_subplot(111)
            axis.set_aspect('auto')

        if self.show_grid:
            axis.get_xaxis().grid(True)
            axis.get_yaxis().grid(True)

        if xlabel: axis.set_xlabel(xlabel)
        if ylabel: axis.set_ylabel(ylabel)

        if self.show_xaxis is not None:
            if self.show_xaxis == 'top':
                axis.spines['bottom'].set_visible(False)
                axis.xaxis.tick_top()
                axis.xaxis.set_label_position("top")
            elif self.show_xaxis == 'bottom':
                axis.spines['top'].set_visible(False)
                axis.xaxis.tick_bottom()
        else:
            axis.xaxis.set_visible(False)

        if self.show_yaxis is not None:
            if self.show_yaxis == 'left':
                axis.spines['right'].set_visible(False)
                axis.yaxis.tick_left()
            elif self.show_yaxis == 'right':
                axis.spines['left'].set_visible(False)
                axis.yaxis.tick_right()
                axis.yaxis.set_label_position("right")
        else:
            axis.yaxis.set_visible(False)

        if not any([self.show_xaxis, self.show_yaxis]):
            axis.set_frame_on(False)

        if lbrt is not None:
            (l, b, r, t) = lbrt
            axis.set_xlim((l, r))
            axis.set_ylim((b, t))

        if self.aspect == 'square':
            xrange = lbrt[2] - lbrt[0]
            yrange = lbrt[3] - lbrt[1]
            axis.set_aspect(xrange/yrange)
        elif self.aspect is not None:
            axis.set_aspect(self.aspect)

        if xticks:
            axis.set_xticks(xticks[0])
            axis.set_xticklabels(xticks[1])

        if yticks:
            axis.set_yticks(yticks[0])
            axis.set_yticklabels(yticks[1])

        if self.show_title:
            title = '' if title is None else title
            self.handles['title'] = axis.set_title(title)

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
        return len(self._stack)


    def __call__(self, ax=False, zorder=0):
        """
        Return a matplotlib figure.
        """
        raise NotImplementedError



class SheetLinesPlot(Plot):


    style_opts = param.List(default=['alpha', 'color', 'linestyle',
                                     'linewidth', 'visible'],
                            constant=True, doc="""
        The style options for SheetLinesPlot match those of matplotlib's
        LineCollection class.""")

    _stack_type = SheetStack

    def __init__(self, contours, zorder=0, **kwargs):
        self.zorder = zorder
        self._stack = self._check_stack(contours, SheetLines)
        super(SheetLinesPlot, self).__init__(**kwargs)


    def __call__(self, axis=None, cyclic_index=0):
        lines = self._stack.top
        title = None if self.zorder > 0 else lines.title
        ax = self._axis(axis, title, 'x', 'y', self._stack.bounds.lbrt())
        line_segments = LineCollection(lines.data, zorder=self.zorder, **options.style[lines][cyclic_index])
        self.handles['line_segments'] = line_segments
        ax.add_collection(line_segments)
        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        contours = self._stack.values()[n]
        self.handles['line_segments'].set_paths(contours.data)
        if self.show_title and self.zorder == 0:
            self.handles['title'].set_text(contours.title)
        plt.draw()



class AnnotationPlot(Plot):
    """
    Draw the Annotation view on the supplied axis. Supports axis
    vlines, hlines, arrows (with or without labels), boxes and
    arbitrary polygonal lines. Note, unlike other Plot types,
    AnnotationPlot must always operate on a supplied axis as
    Annotations may only be used as part of Overlays.
    """

    style_opts = param.List(default=['alpha', 'color', 'linewidth',
                                     'linestyle', 'rotation', 'family',
                                     'weight', 'fontsize', 'visible'],
                            constant=True, doc="""
     Box annotations, hlines and vlines and lines all accept
     matplotlib line style options. Arrow annotations also accept
     additional text options.""")

    def __init__(self, annotation, zorder=0, **kwargs):
        self.zorder = zorder
        self._annotation = annotation
        self._stack = self._check_stack(annotation, Annotation)
        self._warn_invalid_intervals(self._stack)
        super(AnnotationPlot, self).__init__(**kwargs)
        self.handles['annotations'] = []

        line_only = ['linewidth', 'linestyle']
        arrow_opts = [opt for opt in self.style_opts if opt not in line_only]
        line_opts = line_only + ['color']
        self.opt_filter = {'hline':line_opts, 'vline':line_opts, 'line':line_opts,
                           '<':arrow_opts, '^':arrow_opts,
                           '>':arrow_opts, 'v':arrow_opts}


    def _warn_invalid_intervals(self, stack):
        "Check if the annotated intervals have appropriate keys"
        dim_labels = self._stack.dimension_labels

        mismatch_set = set()
        for annotation in stack.values():
            for spec in annotation.data:
                interval = spec[-1]
                if interval is None or dim_labels == ['Default']:
                    continue
                mismatches = set(dict(interval).keys()) - set(dim_labels)
                mismatch_set = mismatch_set | mismatches

        if mismatch_set:
            mismatch_list= ', '.join('%r' % el for el in mismatch_set)
            print "<WARNING>: Invalid annotation interval key(s) ignored: %r" % mismatch_list


    def _active_interval(self, key, interval):
        """
        Given an interval specification, determine whether the
        annotation should be shown or not.
        """
        dim_labels = self._stack.dimension_labels
        if (interval is None) or dim_labels == ['Default']:
            return True

        key = key if isinstance(key, tuple) else (key,)
        key_dict = dict(zip(dim_labels, key))
        for key, (start, end) in dict(interval).items():
            if (start is not None) and key_dict.get(key, -float('inf')) <= start:
                return False
            if (end is not None) and key_dict.get(key, float('inf')) > end:
                return False

        return True


    def _draw_annotations(self, annotation, axis, key):
        """
        Draw the elements specified by the Annotation View on the
        axis, return a list of handles.
        """
        handles = []
        opts = options.style[annotation].opts
        color = opts.get('color', 'k')

        for spec in annotation.data:
            mode, info, interval = spec[0], spec[1:-1], spec[-1]
            opts = dict(el for el in opts.items()
                        if el[0] in self.opt_filter[mode])

            if not self._active_interval(key, interval):
                continue
            if mode == 'vline':
                handles.append(axis.axvline(spec[1], **opts))
                continue
            elif mode == 'hline':
                handles.append(axis.axhline(spec[1], **opts))
                continue
            elif mode == 'line':
                line = LineCollection([np.array(info[0])], **opts)
                axis.add_collection(line)
                handles.append(line)
                continue

            text, xy, points, arrowstyle = info
            arrowprops = dict(arrowstyle=arrowstyle, color=color)
            if mode in ['v', '^']:
                xytext = (0, points if mode=='v' else -points)
            elif mode in ['>', '<']:
                xytext = (points if mode=='<' else -points, 0)
            arrow = axis.annotate(text, xy=xy,
                                  textcoords='offset points',
                                  xytext=xytext,
                                  ha="center", va="center",
                                  arrowprops=arrowprops,
                                  **opts)
            handles.append(arrow)
        return handles


    def __call__(self, axis=None, cyclic_index=0, lbrt=None):

        if axis is None:
            raise Exception("Annotations can only be plotted as part of overlays.")

        self.handles['axis'] = axis
        handles = self._draw_annotations(self._stack.top, axis, self._stack.keys()[-1])
        self.handles['annotations'] = handles
        return axis


    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        annotation = self._stack.values()[n]
        key = self._stack.keys()[n]

        axis = self.handles['axis']
        # Cear all existing annotations
        for element in self.handles['annotations']:
            element.remove()

        self.handles['annotations'] = self._draw_annotations(annotation, axis, key)
        plt.draw()



class SheetPointsPlot(Plot):

    style_opts = param.List(default=['alpha', 'color', 'marker', 's', 'visible'],
                            constant=True, doc="""
     The style options for SheetPointsPlot match those of matplotlib's
     scatter plot command.""")

    _stack_type = SheetStack

    def __init__(self, contours, zorder=0, **kwargs):
        self.zorder = zorder
        self._stack = self._check_stack(contours, SheetPoints)
        super(SheetPointsPlot, self).__init__(**kwargs)


    def __call__(self, axis=None, cyclic_index=0):
        points = self._stack.top
        title = None if self.zorder > 0 else points.title
        ax = self._axis(axis, title, 'x', 'y', self._stack.bounds.lbrt())

        scatterplot = plt.scatter(points.data[:, 0], points.data[:, 1],
                                  zorder=self.zorder, **options.style[points][cyclic_index])
        self.handles['scatter'] = scatterplot
        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n if n < len(self) else len(self) - 1
        points = self._stack.values()[n]
        self.handles['scatter'].set_offsets(points.data)
        if self.show_title and self.zorder == 0:
            self.handles['title'].set_text(points.title)
        plt.draw()



class SheetViewPlot(Plot):

    normalize_individually = param.Boolean(default=False)

    style_opts = param.List(default=['alpha', 'cmap', 'interpolation',
                                     'visible', 'filterrad', 'origin'],
                            constant=True, doc="""
        The style options for SheetViewPlot are a subset of those used
        by matplotlib's imshow command. If supplied, the clim option
        will be ignored as it is computed from the input SheetView.""")


    _stack_type = SheetStack

    def __init__(self, sheetview, zorder=0, **kwargs):
        self.zorder = zorder
        self._stack = self._check_stack(sheetview, SheetView)
        super(SheetViewPlot, self).__init__(**kwargs)


    def __call__(self, axis=None, cyclic_index=0):
        sheetview = self._stack.top
        (l, b, r, t) = self._stack.bounds.lbrt()
        title = None if self.zorder > 0 else sheetview.title
        ax = self._axis(axis, title, 'x', 'y', (l, b, r, t))

        opts = options.style[sheetview][cyclic_index]
        if sheetview.depth != 1:
            opts.pop('cmap', None)

        im = ax.imshow(sheetview.data, extent=[l, r, b, t],
                       zorder=self.zorder, **opts)
        clims = sheetview.range if self.normalize_individually else self._stack.range
        im.set_clim(clims)
        self.handles['im'] = im

        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        im = self.handles.get('im', None)

        sheetview = self._stack.values()[n]
        im.set_data(sheetview.data)

        if self.normalize_individually:
            im.set_clim(sheetview.range)

        if self.show_title and self.zorder == 0:
            self.handles['title'].set_text(sheetview.title)
        plt.draw()



class SheetPlot(Plot):
    """
    A generic plot that visualizes SheetOverlays which themselves may
    contain SheetLayers of type SheetView, SheetPoints or SheetLine
    objects.
    """


    style_opts = param.List(default=[], constant=True, doc="""
     SheetPlot renders overlay layers which individually have style
     options but SheetPlot itself does not.""")

    _stack_type = SheetStack

    def __init__(self, overlays, **kwargs):
        stack = self._check_stack(overlays, SheetOverlay)
        self._stack = self._collapse_channels(stack)
        self.plots = []
        super(SheetPlot, self).__init__(**kwargs)



    def _collapse(self, overlay, pattern, fn, style_key):
        """
        Given an overlay object collapse the channels according to
        pattern using the supplied function. Any collapsed View is
        then given the supplied style key.
        """
        pattern = [el.strip() for el in pattern.rsplit('*')]
        if len(pattern) > len(overlay): return

        skip=0
        collapsed_views = []
        for i in range(len(overlay)):
            layer_labels = overlay.labels[i:len(pattern)+i]
            matching = all(l.endswith(p) for l, p in zip(layer_labels, pattern))
            if matching and len(layer_labels)==len(pattern):
                views = [overlay[label] for label in layer_labels]
                overlay_slice = SheetOverlay(views, overlay.bounds)
                collapsed_view = fn(overlay_slice)
                collapsed_view.style = style_key
                collapsed_views.append(collapsed_view)
                skip = len(views)-1
            elif skip:
                skip = 0 if skip <= 0 else (skip - 1)
            else:
                collapsed_views.append(overlay[i])
        overlay.data = collapsed_views


    def _collapse_channels(self, stack):
        """
        Given a stack of Overlays, apply all applicable channel
        reductions.
        """
        if not issubclass(stack.type, Overlay):
            return stack
        elif not channels.keys(): # No potential channel reductions
            return stack
        else:
            # The original stack should not be mutated by this operation
            stack = copy.deepcopy(stack)

        # Apply all customized channel operations
        for overlay in stack:
            customized = [k for k in channels.keys() if overlay.label and k.startswith(overlay.label)]
            # Largest reductions should be applied first
            sorted_customized = sorted(customized, key=lambda k: -channels[k].size)
            sorted_reductions = sorted(channels.options(), key=lambda k: -channels[k].size)
            # Collapse the customized channel before the other definitions
            for key in sorted_customized + sorted_reductions:
                channel = channels[key]
                collapse_fn = channel_modes[channel.mode]
                fn = collapse_fn.instance(**channel.opts)
                self._collapse(overlay, channel.pattern, fn, key)
        return stack


    def __call__(self, axis=None):
        ax = self._axis(axis, None, 'x','y', self._stack.bounds.lbrt())
        stacks = self._stack.split()

        sorted_stacks = sorted(stacks, key=lambda x: x.style)
        style_groups = dict((k, enumerate(list(v))) for k,v
                            in groupby(sorted_stacks, lambda s: s.style))

        for zorder, stack in enumerate(stacks):
            cyclic_index, _ = style_groups[stack.style].next()
            plotype = viewmap[stack.type]
            plot = plotype(stack, **dict(options.plotting[stack].opts, zorder=zorder))

            plot(ax, cyclic_index=cyclic_index)
            self.plots.append(plot)

        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        for plot in self.plots:
            plot.update_frame(n)


class LayoutPlot(Plot):
    """
    LayoutPlot allows placing up to three Views in a number of
    predefined and fixed layouts, which are defined by the layout_dict
    class attribute. This allows placing subviews next to a main plot
    in either a 'top' or 'right' position.

    Initially, a LayoutPlot computes an appropriate layout based for
    the number of Views in the Layout object it has been given, but
    when embedded in a GridLayout, it can recompute the layout to
    match the number of rows and columns as part of a larger grid.
    """

    layout_dict = {'Single':          {'width_ratios': [4],
                                       'height_ratios': [4],
                                       'positions': ['main']},
                   'Dual':            {'width_ratios': [4, 1],
                                       'height_ratios': [4],
                                       'positions': ['main', 'right']},
                   'Triple':          {'width_ratios': [4, 1],
                                       'height_ratios': [1, 4],
                                       'positions': ['top',   None,
                                                     'main', 'right']},
                   'Embedded Dual':   {'width_ratios': [4],
                                       'height_ratios': [1, 4],
                                       'positions': [None, 'main']}}

    border_size = param.Number(default=0.25, doc="""
        The size of the border expressed as a fraction of the main plot.""")

    subplot_size = param.Number(default=0.25, doc="""
        The size subplots as expressed as a fraction of the main plot.""")


    def __init__(self, layout, **params):
        # The Layout View object
        self.layout = layout
        layout_lens = {1:'Single', 2:'Dual', 3:'Triple'}
        # Type may be set to 'Embedded Dual' by a call it grid_situate
        self.layout_type = layout_lens[len(self.layout)]
        # Handles on subplots by position: 'main', 'top' or 'right'
        self.subplots = {}

        # The supplied (axes, view) objects as indexed by position
        self.plot_axes = {} # Populated by call, used in adjust_positions
        super(LayoutPlot, self).__init__(**params)


    @property
    def shape(self):
        """
        Property used by GridLayoutPlot to compute an overall grid
        structure in which to position LayoutPlots.
        """
        return (len(self.height_ratios), len(self.width_ratios))


    @property
    def width_ratios(self):
        """
        The relative distances for horizontal divisions between the
        primary plot and associated  subplots (if any).
        """
        return self.layout_dict[self.layout_type]['width_ratios']

    @property
    def height_ratios(self):
        """
        The relative distances for the vertical divisions between the
        primary plot and associated subplots (if any).
        """
        return self.layout_dict[self.layout_type]['height_ratios']

    @property
    def view_positions(self):
        """
        A list of position names used in the plot, matching the
        corresponding properties of Layouts. Valid positions are
        'main', 'top', 'right' or None.
        """
        return self.layout_dict[self.layout_type]['positions']


    def __call__(self, subaxes=[]):
        """
        Plot all the views contained in the Layout Object using axes
        appropriate to the layout configuration. All the axes are
        supplied by GridLayoutPlot - the purpose of the call is to
        invoke subplots with correct options and styles and hide any
        empty axes as necessary.
        """
        for ax, pos in zip(subaxes, self.view_positions):
            # Pos will be one of 'main', 'top' or 'right' or None
            view = self.layout.get(pos, None)
            # Record the axis and view at this position
            self.plot_axes[pos] = (ax, view)
            # If no view object or empty position, disable the axis
            if None in [view, pos]:
                ax.set_axis_off()
                continue
            # Customize plotopts depending on position.
            plotopts = options.plotting[view].opts
            # Options common for any subplot
            subplot_opts = dict(show_title=False, main=self.layout.main)
            override_opts = {}

            if pos == 'right':
                right_opts = dict(orientation='vertical', show_xaxis=None, show_yaxis='left')
                override_opts = dict(subplot_opts, **right_opts)
            elif pos == 'top':
                top_opts = dict(show_xaxis='bottom', show_yaxis=None)
                override_opts = dict(subplot_opts, **top_opts)

            # Override the plotopts as required
            plotopts.update(override_opts)
            vtype = view.type if isinstance(view, Stack) else view.__class__
            if pos == 'main':
                subplot = viewmap[vtype](view, **plotopts)
            else:
                subplot = sideviewmap[vtype](view, **plotopts)

            # 'Main' views that should be displayed with square aspect
            if pos == 'main' and issubclass(vtype, (DataOverlay, DataLayer)):
                subplot.aspect='square'

            subplot(ax)
            # Save subplot handles and the axis/views pairs by position
            self.subplots[pos] = subplot


    def adjust_positions(self):
        """
        Make adjustments to the positions of subplots (if available)
        relative to the main plot axes as required.

        This method is called by GridLayoutPlot after an initial pass
        used to position all the Layouts together. This method allows
        LayoutPlots to make final adjustments to the axis positions.
        """
        main_ax, _ = self.plot_axes['main']
        bbox = main_ax.get_position()
        if 'right' in self.view_positions:
            ax, _ = self.plot_axes['right']
            ax.set_position([bbox.x1 + bbox.width * self.border_size,
                             bbox.y0,
                             bbox.width * self.subplot_size, bbox.height])
        if 'top' in self.view_positions:
            ax, _ = self.plot_axes['top']
            ax.set_position([bbox.x0,
                             bbox.y1 + bbox.height * self.border_size,
                             bbox.width, bbox.height * self.subplot_size])


    def grid_situate(self, current_idx, layout_type, subgrid_width):
        """
        Situate the current LayoutPlot in a GridLayoutPlot. The
        GridLayout specifies a layout_type into which the LayoutPlot
        must be embedded. This enclosing layout is guaranteed to have
        enough cells to display all the views.

        Based on this enforced layout format, a starting index
        supplied by GridLayoutPlot (indexing into a large gridspec
        arrangement) is updated to the appropriate embedded value. It
        will also return a list of gridspec indices associated with
        the all the required layout axes.
        """
        # Set the layout configuration as situated in a GridLayout
        self.layout_type = layout_type

        if layout_type == 'Single':
            return current_idx+1, [current_idx]
        elif layout_type == 'Dual':
            return current_idx+2, [current_idx, current_idx+1]

        bottom_idx = current_idx + subgrid_width
        if layout_type == 'Embedded Dual':
            bottom = ((current_idx+1) % subgrid_width) == 0
            grid_idx = (bottom_idx if bottom else current_idx)+1
            return grid_idx, [current_idx, bottom_idx]
        elif layout_type == 'Triple':
            bottom = ((current_idx+2) % subgrid_width) == 0
            grid_idx = (bottom_idx if bottom else current_idx) + 2
            return grid_idx, [current_idx, current_idx+1,
                              bottom_idx, bottom_idx+1]


    def update_frame(self, n):
        for pos, subplot in self.subplots.items():
            if subplot is not None:
                subplot.update_frame(n)


    def __len__(self):
        return max([len(v) for v in self.layout if isinstance(v, NdMapping)]+[1])



class GridLayoutPlot(Plot):
    """
    Plot a group of views in a grid layout based on a GridLayout view
    object.
    """

    roi = param.Boolean(default=False, doc="""
      Whether to apply the ROI to each element of the grid.""")

    style_opts = param.List(default=[], constant=True, doc="""
      GridLayoutPlot renders a group of views which individually have
      style options but GridLayoutPlot itself does not.""")

    horizontal_spacing = param.Number(default=0.5, doc="""
      Specifies the space between horizontally adjacent elements in the grid.
      Default value is set conservatively to avoid overlap of subplots.""")

    vertical_spacing = param.Number(default=0.2, doc="""
      Specifies the space between vertically adjacent elements in the grid.
      Default value is set conservatively to avoid overlap of subplots.""")


    def __init__(self, grid, **kwargs):
        if not isinstance(grid, GridLayout):
            raise Exception("GridLayoutPlot only accepts GridLayouts.")

        self.grid = grid
        # LayoutPlots indexed by their row and column indices
        self.subplots = {}
        self.rows, self.cols = grid.shape
        self.coords = [(r, c) for r in range(self.rows)
                       for c in range(self.cols)]

        super(GridLayoutPlot, self).__init__(**kwargs)
        self.subplots, self.grid_indices = self._compute_gridspecs()


    def _compute_gridspecs(self):
        """
        Computes the tallest and widest cell for each row and column
        by examining the Layouts in the Grid. The GridSpec is then
        instantiated and the LayoutPlots are configured with the
        appropriate embedded layout_types. The first element of the
        returned tuple is a dictionary of all the LayoutPlots indexed
        by row and column. The second dictionary in the tuple supplies
        the grid indicies needed to instantiate the axes for each
        LayoutPlot.
        """
        subplots, grid_indices = {}, {}
        row_heightratios, col_widthratios = {}, {}
        for (r, c) in self.coords:
            view = self.grid.get((r, c), None)
            layout_view = view if isinstance(view, Layout) else Layout([view])
            layout = LayoutPlot(layout_view)
            subplots[(r, c)] = layout
            # For each row and column record the width and height ratios
            # of the LayoutPlot with the most horizontal or vertical splits
            if layout.shape[0] > row_heightratios.get(r, (0, None))[0]:
                row_heightratios[r] = (layout.shape[1], layout.height_ratios)
            if layout.shape[1] > col_widthratios.get(c, (0, None))[0]:
                col_widthratios[c] = (layout.shape[0], layout.width_ratios)

        # In order of row/column collect the largest width and height ratios
        height_ratios = [v[1] for k, v in sorted(row_heightratios.items())]
        width_ratios = [v[1] for k, v in sorted(col_widthratios.items())]
        # Compute the number of rows and cols
        cols = np.sum([len(wr) for wr in width_ratios])
        rows = np.sum([len(hr) for hr in height_ratios])
        # Flatten the width and height ratio lists
        wr_list = [wr for wrs in width_ratios for wr in wrs]
        hr_list = [hr for hrs in height_ratios for hr in hrs]

        self.gs = gridspec.GridSpec(rows, cols,
                                    width_ratios=wr_list,
                                    height_ratios=hr_list,
                                    wspace=self.horizontal_spacing,
                                    hspace=self.vertical_spacing)

        # Situate all the Layouts in the grid and compute the gridspec
        # indices for all the axes required by each LayoutPlot.
        gidx = 0
        for (r, c) in self.coords:
            wsplits = len(width_ratios[c])
            hsplits = len(height_ratios[r])
            if (wsplits, hsplits) == (1,1):
                layout_type = 'Single'
            elif (wsplits, hsplits) == (2,1):
                layout_type = 'Dual'
            elif (wsplits, hsplits) == (1,2):
                layout_type = 'Embedded Dual'
            elif (wsplits, hsplits) == (2,2):
                layout_type = 'Triple'

            gidx, gsinds = subplots[(r, c)].grid_situate(gidx, layout_type, cols)
            grid_indices[(r, c)] = gsinds

        return subplots, grid_indices


    def __call__(self, axis=None):
        ax = self._axis(axis, None, '', '', None)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        for (r, c) in self.coords:
            layout_plot = self.subplots.get((r, c), None)
            subaxes = [plt.subplot(self.gs[ind]) for ind in self.grid_indices[(r, c)]]
            layout_plot(subaxes)
        plt.draw()

        # Adjusts the Layout subplot positions
        for (r, c) in self.coords:
            self.subplots.get((r, c), None).adjust_positions()

        if not axis: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        for subplot in self.subplots.values():
            subplot.update_frame(n)


    def __len__(self):
        return max([len(v) for v in self.subplots.values()]+[1])



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

    style_opts = param.List(default=['alpha', 'cmap', 'interpolation',
                                     'visible', 'filterrad', 'origin'],
                            constant=True, doc="""
       The style options for CoordinateGridPlot match those of
       matplotlib's imshow command.""")


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
                    opts = options.style[view.top[0]].opts
                else:
                    data = view.top.data if self.situate else view.top.roi.data
                    opts = options.style[view.top].opts

                self.handles['projs'].append(plt.imshow(data, extent=(x,x+w, y, y+h), **opts))
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
        return max([len(v) for v in self.grid if isinstance(v, NdMapping)]+[1])



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

    style_opts = param.List(default=[], constant=True, doc="""
     DataPlot renders overlay layers which individually have style
     options but DataPlot itself does not.""")


    def __init__(self, overlays, **kwargs):
        self._stack = self._check_stack(overlays, DataOverlay)
        self.plots = []
        super(DataPlot, self).__init__(**kwargs)


    def __call__(self, axis=None, **kwargs):
        ax = self._axis(axis, None, self._stack.xlabel, self._stack.ylabel, self._stack.lbrt)

        stacks = self._stack.split()
        style_groups = dict((k, enumerate(list(v))) for k,v
                            in groupby(stacks, lambda s: s.style))

        for zorder, stack in enumerate(stacks):
            cyclic_index, _ = style_groups[stack.style].next()

            plotype = viewmap[stack.type]
            plot = plotype(stack, size=self.size,
                           show_xaxis=self.show_xaxis, show_yaxis=self.show_yaxis,
                           show_legend=self.show_legend, show_title=self.show_title,
                           show_grid=self.show_grid, zorder=zorder, **kwargs)
            plot.aspect = self.aspect

            lbrt= None if stack.type == Annotation else self._stack.lbrt
            plot(ax, cyclic_index=cyclic_index, lbrt=lbrt)
            self.plots.append(plot)

        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n if n < len(self) else len(self) - 1
        for plot in self.plots:
            plot.update_frame(n)


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

    show_xaxis = param.String(default='bottom', allow_None=True, doc="""
      Whether to display the right axis.""")

    show_yaxis = param.String(default='left', allow_None=True, doc="""
      Whether to display the right axis.""")

    style_opts = param.List(default=['alpha', 'color', 'linestyle', 'linewidth',
                                     'visible'], constant=True, doc="""
       The style options for DataCurvePlot match those of matplotlib's
       LineCollection object.""")

    _stack_type = DataStack

    def __init__(self, curves, zorder=0, **kwargs):
        self.zorder = zorder
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
        rangex = float(x_values[-1]) - x_values[0]
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


    def __call__(self, axis=None, cyclic_index=0, lbrt=None):
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

        ax = self._axis(axis, lines.title, lines.xlabel, lines.ylabel,
                        xticks=xticks, lbrt=lbrt)

        # Create line segments and apply style
        line_segments = LineCollection(lines.data, zorder=self.zorder,
                                       **options.style[lines][cyclic_index])

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
        lines = self._stack.values()[n]
        if self.cyclic_range is not None:
            self._cyclic_curves(lines)
        self.handles['line_segments'].set_paths(lines.data)
        if self.show_title and self.zorder == 0:
            self.handles['title'].set_text(lines.title)
        plt.draw()


class DataGridPlot(Plot):
    """
    Plot a group of views in a grid layout based on a DataGrid view
    object.
    """

    show_legend = param.Boolean(default=False, doc="""
      Legends add to much clutter in a grid and are disabled by default.""")

    show_title = param.Boolean(default=False)

    style_opts = param.List(default=[], constant=True, doc="""
     DataGridPlot renders groups of DataLayers which individually have
     style options but DataGridPlot itself does not.""")

    show_xaxis = param.ObjectSelector(default=None,
                                      objects=['both','top', 'bottom', None], doc="""
      Whether and where to display the xaxis.""")

    show_yaxis = param.ObjectSelector(default=None,
                                      objects=['both', 'left', 'right', None], doc="""
      Whether and where to display the yaxis.""")


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
        ax = self._axis(axis)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        self.subplots = []
        r, c = (self.rows-1, 0)
        for coord in self.grid.keys():
            view = self.grid.get(coord, None)
            if view is not None:
                subax = plt.subplot(self._gridspec[c, r])
                vtype = view.type if isinstance(view, DataStack) else view.__class__
                subplot = viewmap[vtype](view, show_legend=self.show_legend,
                                         show_xaxis=self.show_xaxis,
                                         show_yaxis=self.show_yaxis,
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
        return max([len(v) for v in self.grid ]+[1])



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


    style_opts = param.List(default=[], constant=True, doc="""
     TablePlot has specialized options which are controlled via plot
     options instead of matplotlib options.""")


    _stack_type = TableStack

    def __init__(self, tables, zorder=0, **kwargs):
        self.zorder = zorder
        self._stack = self._check_stack(tables, TableView)
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


    def __call__(self, axis=None):
        tableview = self._stack.top

        ax = self._axis(axis, tableview.title)

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

        if self.show_title and self.zorder == 0:
            self.handles['title'].set_text(tableview.title)
        plt.draw()



class DataHistogramPlot(Plot):
    """
    DataHistogramPlot can plot DataHistograms and DataStacks of
    DataHistograms, which can be displayed as a single frame or
    animation.
    """

    style_opts = param.List(default=['alpha', 'color', 'align',
                                     'visible', 'edgecolor', 'log',
                                     'ecolor', 'capsize', 'error_kw',
                                     'hatch'], constant=True, doc="""
     The style options for DataHistogramPlot match those of
     matplotlib's bar command.""")

    num_ticks = param.Integer(default=5, doc="""
        If colorbar is enabled the number of labels will be overwritten.""")

    rescale_individually = param.Boolean(default=True, doc="""
        Whether to use redraw the axes per stack or per view.""")

    orientation = param.ObjectSelector(default='horizontal',
                                       objects=['horizontal', 'vertical'])

    _stack_type = DataStack

    def __init__(self, curves, zorder=0, **kwargs):
        self.zorder = zorder
        self.center = False
        self.cyclic = False
        self.cyclic_index = 0
        self.ax = None

        self._stack = self._check_stack(curves, DataHistogram)
        super(DataHistogramPlot, self).__init__(**kwargs)

        if self.orientation == 'vertical':
            self.axis_settings = ['ylabel', 'xlabel', 'yticks']
            self.offset_linefn = plt.axvline
            self.plotfn = plt.barh
        else:
            self.axis_settings = ['xlabel', 'ylabel', 'xticks']
            self.offset_linefn = plt.axhline
            self.plotfn = plt.bar


    def __call__(self, axis=None, cyclic_index=0, lbrt=None):
        hist = self._stack.top
        self.cyclic_index = cyclic_index

        # Get plot ranges and values
        edges, hvals, widths, lims = self._process_hist(hist)

        # Process and apply axis settings
        ticks = self._compute_ticks(edges, widths, lims)
        ax_settings = self._process_axsettings(hist, lims, ticks)
        self.ax = self._axis(axis, **ax_settings)

        # Plot bars and make any adjustments
        style = options.style[hist][cyclic_index]
        bars = self.plotfn(edges, hvals, widths, zorder=self.zorder, **style)
        self.handles['bars'] = self._update_plot(-1, bars, lims) # Indexing top

        if not axis: plt.close(self.handles['fig'])
        return self.ax if axis else self.handles['fig']


    def _process_hist(self, hist):
        """
        Get data from histogram, including bin_ranges and values.
        """
        self.cyclic = False if hist.cyclic_range is None else True
        edges = hist.edges[:-1]
        hist_vals = np.array(hist.hist[:])
        widths = np.diff(hist.edges)
        xlims = hist.xlim if self.rescale_individually else self._stack.xlim
        lims = xlims + hist.ylim
        return edges, hist_vals, widths, lims


    def _compute_ticks(self, edges, widths, lims):
        """
        Compute the ticks either as cyclic values in degrees or as roughly
        evenly spaced bin centers.
        """
        if self.cyclic:
            x0, x1, _, _ = lims
            xvals = np.linspace(x0, x1, self.num_ticks)
            labels = ["%.0f" % np.rad2deg(x) + u'\N{DEGREE SIGN}'
                      for x in xvals]
        else:
            edge_inds = range(len(edges))
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
        if self.zorder == 0: axis_settings['title'] = hist.title

        return axis_settings


    def _update_plot(self, n, bars, lims):
        """
        Process bars is subclasses to manually adjust bars after
        being plotted.
        """
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
                bar.set_clip_on(False)
        plt.draw()


    def update_frame(self, n):
        """
        Update the plot for an animation.
        """
        n = n if n < len(self) else len(self) - 1
        hist = self._stack.values()[n]

        # Process values, axes and style
        edges, hvals, widths, lims = self._process_hist(hist)

        ticks = self._compute_ticks(edges, widths, lims)
        ax_settings = self._process_axsettings(hist, lims, ticks)
        self._axis(self.ax, **ax_settings)
        self._update_artists(n, edges, hvals, widths, lims)
        if self.show_title: self.handles['title'] = self.ax.set_title(hist.title)



class SideHistogramPlot(DataHistogramPlot):

    main = param.Parameterized(doc="""
        The main View or Stack this SideHistogramPlot is attached to.""")

    offset = param.Number(default=0.2, doc="""
        Histogram value offset for a colorbar.""")

    show_title = param.Boolean(default=False, doc="""
        Titles should be disabled on all SidePlots to avoid clutter.""")


    def _process_hist(self, hist):
        """
        Subclassed to offset histogram by defined amount.
        """
        edges, hvals, widths, lims = super(SideHistogramPlot, self)._process_hist(hist)
        offset = self.offset * lims[3]
        hvals += offset
        lims = lims[0:3] + (lims[3] + offset,)
        return edges, hvals, widths, lims


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
        offset = self.offset * lims[3] * (1-self.offset)
        main_style = options.style[self.main].opts
        individually = options.plotting[self.main].opts.get('normalize_individually', False)

        if isinstance(self.main, Stack):
            main_range = self.main.values()[n].range if individually else self.main.range
        elif isinstance(self.main, View):
            main_range = self.main.range

        if offset and ('offset_line' not in self.handles):
            self.handles['offset_line'] = self.offset_linefn(offset,
                                                             linewidth=1.0,
                                                             color='k')
        elif offset:
            self._update_separator(lims, offset)

        cmap = cm.get_cmap(main_style['cmap']) if self.offset else None
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



sideviewmap = {DataHistogram: SideHistogramPlot,
               TableView: TablePlot,
               CoordinateGrid: CoordinateGridPlot}


viewmap = {SheetView: SheetViewPlot,
           SheetPoints: SheetPointsPlot,
           SheetLines: SheetLinesPlot,
           SheetOverlay: SheetPlot,
           CoordinateGrid: CoordinateGridPlot,
           DataCurves: DataCurvePlot,
           DataOverlay: DataPlot,
           DataGrid: DataGridPlot,
           TableView: TablePlot,
           DataHistogram: DataHistogramPlot,
           Layout: GridLayoutPlot,
           Annotation: AnnotationPlot
}


# The channel_modes dictionary contains the available channel processing
# modes. These modes are ViewOperations that accept Sheet Overlays as
# input and process them in some way to return a single RGB(A)
# SheetView.

channel_modes={'RGBA':RGBA,
               'HCS':HCS,
               'AlphaOverlay':AlphaOverlay}


__all__ = ['viewmap'] + list(set([_k for _k,_v in locals().items()
                                  if isinstance(_v, type) and issubclass(_v, Plot)]))
