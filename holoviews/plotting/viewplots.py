import copy
from itertools import groupby, product

import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec, animation
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
import matplotlib.patches as patches

import param

from ..core import DataElement, UniformNdMapping, Element, HoloMap, CompositeOverlay,\
    NdOverlay, Overlay, AdjointLayout, GridLayout, AxisLayout, ViewTree
from ..element import Annotation, Raster


class Plot(param.Parameterized):
    """
    A Plot object returns either a matplotlib figure object (when
    called or indexed) or a matplotlib animation object as
    appropriate. Plots take element objects such as Matrix,
    Contours or Points as inputs and plots them in the
    appropriate format. As views may vary over time, all plots support
    animation via the anim() method.
    """

    apply_databounds = param.Boolean(default=True, doc="""
        Whether to compute the plot bounds from the data itself.""")

    aspect = param.Parameter(default=None, doc="""
        The aspect ratio mode of the plot. By default, a plot may
        select its own appropriate aspect ratio but sometimes it may
        be necessary to force a square aspect ratio (e.g. to display
        the plot as an element of a grid). The modes 'auto' and
        'equal' correspond to the axis modes of the same name in
        matplotlib, a numeric value may also be passed.""")

    orientation = param.ObjectSelector(default='horizontal',
                                       objects=['horizontal', 'vertical'], doc="""
        The orientation of the plot. Note that this parameter may not
        always be respected by all plots but should be respected by
        adjoined plots when appropriate.""")

    rescale_individually = param.Boolean(default=False, doc="""
        Whether to use redraw the axes per map or per element.""")

    show_frame = param.Boolean(default=True, doc="""
        Whether or not to show a complete frame around the plot.""")

    show_grid = param.Boolean(default=False, doc="""
        Whether to show a Cartesian grid on the plot.""")

    show_legend = param.Boolean(default=True, doc="""
        Whether to show legend for the plot.""")

    show_title = param.Boolean(default=True, doc="""
        Whether to display the plot title.""")

    show_xaxis = param.ObjectSelector(default='bottom',
                                      objects=['top', 'bottom', None], doc="""
        Whether and where to display the xaxis.""")

    show_yaxis = param.ObjectSelector(default='left',
                                      objects=['left', 'right', None], doc="""
        Whether and where to display the yaxis.""")

    size = param.NumericTuple(default=(4, 4), doc="""
        The matplotlib figure size in inches.""")

    style_opts = param.List(default=[], constant=True, doc="""
        A list of matplotlib keyword arguments that may be supplied via a
        style options object. Each subclass should override this
        parameter to list every option that works correctly.""")

    # A mapping from DataElement types to their corresponding plot types
    defaults = {}

    # A mapping from DataElement types to their corresponding side plot types
    sideplots = {}

    def __init__(self, view=None, zorder=0, all_keys=None, **params):
        if view is not None:
            self._map = self._check_map(view)
            self._keys = all_keys if all_keys else self._map.keys()
        super(Plot, self).__init__(**params)
        self.zorder = zorder
        self.ax = None
        self._create_fig = True
        # List of handles to matplotlib objects for animation update
        self.handles = {}


    def _check_map(self, view, element_type=Element):
        """
        Helper method that ensures a given element is always returned as
        an HoloMap object.
        """
        if not isinstance(view, HoloMap):
            vmap = HoloMap(initial_items=(0, view))
        else:
            vmap = view

        return vmap


    def _format_title(self, key):
        view = self._map.get(key, None)
        if view is None: return None
        title_format = self._map.get_title(key if isinstance(key, tuple) else (key,), view)
        if title_format is None:
            return None
        return title_format.format(label=view.label, value=view.value,
                                   type=view.__class__.__name__)


    def _init_axis(self, axis):
        """
        Return an axis which may need to be initialized from
        a new figure.
        """
        if axis is None and self._create_fig:
            fig = plt.figure()
            self.handles['fig'] = fig
            fig.set_size_inches(list(self.size))
            axis = fig.add_subplot(111)
            axis.set_aspect('auto')

        return axis


    def _finalize_axis(self, key, title=None, lbrt=None, xticks=None, yticks=None,
                       xlabel=None, ylabel=None):
        """
        Applies all the axis settings before the axis or figure is returned.
        Only plots with zorder 0 get to apply their settings.

        When the number of the frame is supplied as n, this method looks
        up and computes the appropriate title, axis labels and axis bounds.
        """

        axis = self.ax

        if self.zorder == 0 and axis is not None and key is not None:
            view = self._map.get(key, None) if hasattr(self, '_map') else None
            if view is not None:
                title = None if self.zorder > 0 else self._format_title(key)
                if hasattr(view, 'xlabel') and xlabel is None:
                    xlabel = view.xlabel
                if hasattr(view, 'ylabel') and ylabel is None:
                    ylabel = view.ylabel
                if lbrt is None and self.apply_databounds:
                    lbrt = view.lbrt if self.rescale_individually else self._map.lbrt

            if self.show_grid:
                axis.get_xaxis().grid(True)
                axis.get_yaxis().grid(True)

            if xlabel: axis.set_xlabel(xlabel)
            if ylabel: axis.set_ylabel(ylabel)

            disabled_spines = []
            if self.show_xaxis is not None:
                if self.show_xaxis == 'top':
                    axis.xaxis.set_ticks_position("top")
                    axis.xaxis.set_label_position("top")
                elif self.show_xaxis == 'bottom':
                    axis.xaxis.set_ticks_position("bottom")
            else:
                axis.xaxis.set_visible(False)
                disabled_spines.extend(['top', 'bottom'])

            if self.show_yaxis is not None:
                if self.show_yaxis == 'left':
                    axis.yaxis.set_ticks_position("left")
                elif self.show_yaxis == 'right':
                    axis.yaxis.set_ticks_position("right")
                    axis.yaxis.set_label_position("right")
            else:
                axis.yaxis.set_visible(False)
                disabled_spines.extend(['left', 'right'])

            for pos in disabled_spines:
                axis.spines[pos].set_visible(False)

            if not self.show_frame:
                axis.spines['right' if self.show_yaxis == 'left' else 'left'].set_visible(False)
                axis.spines['bottom' if self.show_xaxis == 'top' else 'top'].set_visible(False)

            if lbrt and self.apply_databounds:
                (l, b, r, t) = [coord if np.isreal(coord) else np.NaN for coord in lbrt]
                if not np.NaN in (l, r): axis.set_xlim((l, r))
                if b == t: t += 1. # Arbitrary y-extent if zero range
                if not np.NaN in (b, t): axis.set_ylim((b, t))

            if self.aspect == 'square':
                axis.set_aspect((1./axis.get_data_ratio()))
            elif self.aspect not in [None, 'square']:
                axis.set_aspect(self.aspect)

            if xticks:
                axis.set_xticks(xticks[0])
                axis.set_xticklabels(xticks[1])

            if yticks:
                axis.set_yticks(yticks[0])
                axis.set_yticklabels(yticks[1])

            if self.show_title and title is not None:
                self.handles['title'] = axis.set_title(title)


        if 'fig' in self.handles:
            plt.draw()
            fig = self.handles['fig']
            plt.close(fig)
            return fig
        else:
            return axis


    def __getitem__(self, frame):
        """
        Get the matplotlib figure at the given frame number.
        """
        if frame > len(self):
            self.warning("Showing last frame available: %d" % len(self))
        if self.handles.get('fig') is None: self.handles['fig'] = self()
        self.update_frame(frame)
        return self.handles['fig']


    def anim(self, start=0, stop=None, fps=30):
        """
        Method to return an Matplotlib animation. The start and stop
        frames may be specified as well as the fps.
        """
        figure = self()
        frames = list(range(len(self)))[slice(start, stop, 1)]
        anim = animation.FuncAnimation(figure, self.update_frame,
                                       frames=frames,
                                       interval = 1000.0/fps)
        # Close the figure handle
        plt.close(figure)
        return anim


    def update_frame(self, n, lbrt=None):
        """
        Set the plot(s) to the given frame number.  Operates by
        manipulating the matplotlib objects held in the self._handles
        dictionary.

        If n is greater than the number of available frames, update
        using the last available frame.
        """
        n = n if n < len(self) else len(self) - 1
        key = self._keys[n]
        view = self._map.get(key, None)
        self.ax.set_visible(view is not None)
        axis_kwargs = self.update_handles(view, key, lbrt) if view is not None else {}
        self._finalize_axis(key, **dict({'lbrt': lbrt}, **(axis_kwargs if axis_kwargs else {})))


    def update_handles(self, view, key, lbrt=None):
        """
        Update the elements of the plot.
        """
        raise NotImplementedError


    def __len__(self):
        """
        Returns the total number of available frames.
        """
        return len(self._keys)


    def __call__(self, ax=False, zorder=0):
        """
        Return a matplotlib figure.
        """
        raise NotImplementedError



class GridPlot(Plot):
    """
    Plot a group of elements in a grid layout based on a AxisLayout element
    object.
    """

    joint_axes = param.Boolean(default=True, doc="""
        Share axes between all elements in the AxisLayout.""")

    show_legend = param.Boolean(default=False, doc="""
        Legends add to much clutter in a grid and are disabled by default.""")

    show_title = param.Boolean(default=False)

    style_opts = param.List(default=[], constant=True, doc="""
        GridPlot renders groups of DataLayers which individually have
        style options but GridPlot itself does not.""")

    def __init__(self, grid, **params):
        if not isinstance(grid, AxisLayout):
            raise Exception("GridPlot only accepts AxisLayout.")

        self.grid = copy.deepcopy(grid)
        for k, vmap in self.grid.data.items():
            self.grid[k] = self._check_map(self.grid[k])

        self.subaxes = []
        if grid.ndims == 1:
            self.rows, self.cols = (1, len(grid.keys()))
        else:
            x, y = list(zip(*list(grid.keys())))
            self.cols, self.rows = (len(set(x)), len(set(y)))
        self._gridspec = gridspec.GridSpec(self.rows, self.cols)
        self.subplots = self._create_subplots()

        extra_opts = DataElement.options.plotting(self.grid).opts
        super(GridPlot, self).__init__(show_xaxis=None, show_yaxis=None,
                                       show_frame=False,
                                       **dict(params, **extra_opts))
        self._keys = self.grid.all_keys


    def _create_subplots(self):
        subplots, subaxes = {}, {}
        r, c = (0, 0)
        for coord in self.grid.keys(full_grid=True):
            view = self.grid.data.get(coord, None)
            if view is not None:
                vtype = view.type if isinstance(view, HoloMap) else view.__class__
                opts = Element.options.plotting(view).opts
                opts.update(show_legend=self.show_legend, show_xaxis=self.show_xaxis,
                            show_yaxis=self.show_yaxis, show_title=self.show_title)
                subplot = Plot.defaults[vtype](view, **opts)
                subplots[(r, c)] = subplot
            if r != self.rows-1:
                r += 1
            else:
                r = 0
                c += 1
        return subplots


    def __call__(self, axis=None):
        ax = self._init_axis(None)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Get the lbrt of the grid elements (not the whole grid)
        subplot_kwargs = dict()
        if self.joint_axes:
            try:
                l, r = self.grid.xlim
                b, t = self.grid.ylim
                subplot_kwargs = dict(lbrt=(l, b, r, t))
            except:
                pass

        for (r, c), subplot in self.subplots.items():
            subax = plt.subplot(self._gridspec[r, c])
            self.subaxes[(r, c)] = subax
            subplot(subax, **subplot_kwargs)
        self._grid_axis()
        self._adjust_subplots()

        if not axis: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def _format_title(self, key):
        view = self.grid.values()[0]
        key = key if isinstance(key, tuple) else (key,)
        if len(self) > 1:
            title_format = view.get_title(key, self.grid)
        else:
            title_format = self.grid.title
        view = view.last
        return title_format.format(label=view.label, value=view.value,
                                   type=self.grid.__class__.__name__)


    def _grid_axis(self):
        fig = self.handles['fig']
        grid_axis = fig.add_subplot(111)
        grid_axis.patch.set_visible(False)

        # Set labels and titles
        key = self._keys[-1]
        grid_axis.set_xlabel(str(self.grid.dimensions[0]))
        grid_axis.set_title(self._format_title(key))

        # Compute and set x- and y-ticks
        keys = self.grid.keys()
        if self.grid.ndims == 1:
            dim1_keys = keys
            dim2_keys = [0]
            grid_axis.get_yaxis().set_visible(False)
        else:
            dim1_keys, dim2_keys = zip(*keys)
            grid_axis.set_ylabel(str(self.grid.dimensions[1]))
            grid_axis.set_aspect(float(self.rows)/self.cols)
        plot_width = 1.0 / self.cols
        xticks = [(plot_width/2)+(r*plot_width) for r in range(self.cols)]
        plot_height = 1.0 / self.rows
        yticks = [(plot_height/2)+(r*plot_height) for r in range(self.rows)]
        grid_axis.set_xticks(xticks)
        grid_axis.set_xticklabels(self._process_ticklabels(sorted(set(dim1_keys))))
        grid_axis.set_yticks(yticks)
        grid_axis.set_yticklabels(self._process_ticklabels(sorted(set(dim2_keys))))

        self.handles['grid_axis'] = grid_axis
        plt.draw()


    def _process_ticklabels(self, labels):
        return [k if isinstance(k, str) else np.round(float(k), 3) for k in labels]


    def _adjust_subplots(self):
        bbox = self.handles['grid_axis'].get_position()
        l, b, w, h = bbox.x0, bbox.y0, bbox.width, bbox.height

        if self.cols == 1:
            b_w = 0
        else:
            b_w = (w/10.) / (self.cols - 1)

        if self.rows == 1:
            b_h = 0
        else:
            b_h = (h/10.) / (self.rows - 1)
        ax_w = (w - ((w/10.) if self.cols > 1 else 0)) / self.cols
        ax_h = (h - ((h/10.) if self.rows > 1 else 0)) / self.rows

        r, c = (0, 0)
        for ax in self.subaxes:
            xpos = l + (c*ax_w) + (c * b_w)
            ypos = b + (r*ax_h) + (r * b_h)
            if r != self.rows-1:
                r += 1
            else:
                r = 0
                c += 1
            if not ax is None:
                ax.set_position([xpos, ypos, ax_w, ax_h])


    def update_frame(self, n):
        key = self._keys[n]
        for subplot in self.subplots.values():
            subplot.update_frame(n)
        self.handles['grid_axis'].set_title(self._format_title(key))


    def __len__(self):
        return max([len(self._keys), 1])


class AdjointLayoutPlot(Plot):
    """
    LayoutPlot allows placing up to three Views in a number of
    predefined and fixed layouts, which are defined by the layout_dict
    class attribute. This allows placing subviews next to a main plot
    in either a 'top' or 'right' position.

    Initially, a LayoutPlot computes an appropriate layout based for
    the number of Views in the AdjointLayout object it has been given, but
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
        # The AdjointLayout DataElement object
        self.layout = layout
        layout_lens = {1:'Single', 2:'Dual', 3:'Triple'}
        # Type may be set to 'Embedded Dual' by a call it grid_situate
        self.layout_type = layout_lens[len(self.layout)]

        # The supplied (axes, view) objects as indexed by position
        self.plot_axes = {} # Populated by call, used in adjust_positions
        super(AdjointLayoutPlot, self).__init__(**params)

        # Handles on subplots by position: 'main', 'top' or 'right'
        self.subplots = self._create_subplots()


    def _create_subplots(self):
        """
        Plot all the views contained in the AdjointLayout Object using axes
        appropriate to the layout configuration. All the axes are
        supplied by LayoutPlot - the purpose of the call is to
        invoke subplots with correct options and styles and hide any
        empty axes as necessary.
        """
        subplots = {}
        for pos in self.view_positions:
            # Pos will be one of 'main', 'top' or 'right' or None
            view = self.layout.get(pos, None)
            # Customize plotopts depending on position.
            plotopts = Element.options.plotting(view).opts
            # Options common for any subplot
            subplot_opts = dict(show_title=False, layout=self.layout)
            override_opts = {}

            if pos == 'right':
                right_opts = dict(orientation='vertical', show_xaxis=None, show_yaxis='left')
                override_opts = dict(subplot_opts, **right_opts)
            elif pos == 'top':
                top_opts = dict(show_xaxis='bottom', show_yaxis=None)
                override_opts = dict(subplot_opts, **top_opts)

            # Override the plotopts as required
            plotopts.update(override_opts)
            vtype = view.type if isinstance(view, HoloMap) else view.__class__
            layer_types = (vtype,) if isinstance(view, DataElement) else view.layer_types
            if isinstance(view, AxisLayout):
                if len(layer_types) == 1 and issubclass(layer_types[0], Raster):
                    from .sheetplots import MatrixGridPlot
                    plot_type = MatrixGridPlot
                else:
                    plot_type = GridPlot
            else:
                if pos == 'main':
                    plot_type = Plot.defaults[vtype]
                else:
                    plot_type = Plot.sideplots[vtype]
            subplots[pos] = plot_type(view, **plotopts)
        return subplots


    @property
    def shape(self):
        """
        Property used by LayoutPlot to compute an overall grid
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
        Plot all the views contained in the AdjointLayout Object using axes
        appropriate to the layout configuration. All the axes are
        supplied by LayoutPlot - the purpose of the call is to
        invoke subplots with correct options and styles and hide any
        empty axes as necessary.
        """
        for ax, pos in zip(subaxes, self.view_positions):
            # Pos will be one of 'main', 'top' or 'right' or None
            view = self.layout.get(pos, None)
            subplot = self.subplots.get(pos, None)
            # Record the axis and view at this position
            self.plot_axes[pos] = (ax, view)
            # If no view object or empty position, disable the axis
            if None in [view, pos, subplot]:
                ax.set_axis_off()
                continue

            vtype = view.type if isinstance(view, HoloMap) else view.__class__
            # 'Main' views that should be displayed with square aspect
            if pos == 'main' and issubclass(vtype, DataElement):
                subplot.aspect='square'
            subplot(ax)


    def adjust_positions(self):
        """
        Make adjustments to the positions of subplots (if available)
        relative to the main plot axes as required.

        This method is called by LayoutPlot after an initial pass
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
        Situate the current LayoutPlot in a LayoutPlot. The
        GridLayout specifies a layout_type into which the LayoutPlot
        must be embedded. This enclosing layout is guaranteed to have
        enough cells to display all the views.

        Based on this enforced layout format, a starting index
        supplied by LayoutPlot (indexing into a large gridspec
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
        return max([len(v) if isinstance(v, UniformNdMapping) else len(v.all_keys)
                    for v in self.layout if isinstance(v, (UniformNdMapping, AxisLayout))]+[1])


class LayoutPlot(Plot):
    """
    A LayoutPlot accepts either a ViewTree or a GridLayout and
    displays the elements in a cartesian grid in scanline order.
    """

    style_opts = param.List(default=[], constant=True, doc="""
      LayoutPlot renders a group of views which individually have
      style options but LayoutPlot itself does not.""")

    horizontal_spacing = param.Number(default=0.5, doc="""
      Specifies the space between horizontally adjacent elements in the grid.
      Default value is set conservatively to avoid overlap of subplots.""")

    vertical_spacing = param.Number(default=0.2, doc="""
      Specifies the space between vertically adjacent elements in the grid.
      Default value is set conservatively to avoid overlap of subplots.""")

    def __init__(self, layout, **params):
        if not isinstance(layout, (GridLayout, ViewTree)):
            raise Exception("LayoutPlot only accepts ViewTree objects.")

        self.layout = layout
        self.subplots = {}
        self.rows, self.cols = layout.shape
        self.coords = list(product(range(self.rows),
                                   range(self.cols)))

        super(LayoutPlot, self).__init__(**params)
        self.subplots, self.grid_indices = self._compute_gridspecs()

    def _compute_gridspecs(self):
        """
        Computes the tallest and widest cell for each row and column
        by examining the Layouts in the AxisLayout. The GridSpec is then
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
            view = self.layout.grid_items.get((r, c), None)
            layout_view = view if isinstance(view, AdjointLayout) else AdjointLayout([view])
            layout = AdjointLayoutPlot(layout_view)
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
        ax = self._init_axis(axis)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        for (r, c) in self.coords:
            layout_plot = self.subplots.get((r, c), None)
            subaxes = [plt.subplot(self.gs[ind]) for ind in self.grid_indices[(r, c)]]

            rcopts = Element.options.style(self.layout).opts
            with matplotlib.rc_context(rcopts):
                layout_plot(subaxes)
        plt.draw()

        # Adjusts the AdjointLayout subplot positions
        for (r, c) in self.coords:
            self.subplots.get((r, c), None).adjust_positions()

        if not axis: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        for subplot in self.subplots.values():
            subplot.update_frame(n)


    def __len__(self):
        return max([len(v) for v in self.subplots.values()]+[1])



class LayersPlot(Plot):
    """
    LayersPlot supports processing of channel operations on Overlays
    across maps. SheetPlot and MatrixGridPlot are examples of
    LayersPlots.
    """

    style_opts = param.List(default=[], constant=True, doc="""
     LayersPlot renders layers which individually have style and plot
     options but LayersPlot itself does not.""")

    _abstract = True

    def __init__(self, overlay, **params):
        super(LayersPlot, self).__init__(overlay, **params)
        self.subplots = self._create_subplots()


    def _create_subplots(self):
        subplots = {}

        collapsed = self._collapse_channels(self._map)
        vmaps = collapsed.split_overlays()
        style_groups = dict((k, enumerate(list(v))) for k,v
                            in groupby(vmaps, lambda s: s.style))

        for zorder, vmap in enumerate(vmaps):
            cyclic_index, _ = next(style_groups[vmap.style])
            plotopts = Element.options.plotting(vmap.style).opts
            plotype = Plot.defaults[type(vmap.last)]
            subplots[zorder] = plotype(vmap, **dict(plotopts, size=self.size, all_keys=self._keys,
                                                    show_legend=self.show_legend, zorder=zorder,
                                                    aspect=self.aspect))

        return subplots


    def _collapse(self, overlay, pattern, fn, style_key):
        """
        Given an overlay object collapse the channels according to
        pattern using the supplied function. Any collapsed DataElement is
        then given the supplied style key.
        """
        pattern = [el.strip() for el in pattern.rsplit('*')]
        if len(pattern) > len(overlay): return

        skip=0
        collapsed_overlay = overlay.clone(None)
        for i, key in enumerate(overlay.keys()):
            layer_labels = overlay.labels[i:len(pattern)+i]
            matching = all(l.endswith(p) for l, p in zip(layer_labels, pattern))
            if matching and len(layer_labels)==len(pattern):
                views = [el for el in overlay if el.label in layer_labels]
                if isinstance(overlay, Overlay):
                    views = np.product([Overlay.from_view(el) for el in overlay])
                else:
                    overlay_slice = overlay.clone(views)
                collapsed_view = fn(overlay_slice)
                if isinstance(overlay, ViewTree):
                    collapsed_overlay *= collapsed_view
                else:
                    collapsed_overlay[key] = collapsed_view
                skip = len(views)-1
            elif skip:
                skip = 0 if skip <= 0 else (skip - 1)
            else:
                collapsed_overlay[key] = overlay[key]
        return collapsed_overlay


    def _collapse_channels(self, vmap):
        """
        Given a map of Overlays, apply all applicable channel
        reductions.
        """
        if not issubclass(vmap.type, CompositeOverlay):
            return vmap
        elif not CompositeOverlay.channels.keys(): # No potential channel reductions
            return vmap

        # Apply all customized channel operations
        collapsed_vmap = vmap.clone()
        for key, overlay in vmap.items():
            customized = [k for k in CompositeOverlay.channels.keys()
                          if overlay.label and k.startswith(overlay.label)]
            # Largest reductions should be applied first
            sorted_customized = sorted(customized, key=lambda k: -CompositeOverlay.channels[k].size)
            sorted_reductions = sorted(CompositeOverlay.channels.options(),
                                       key=lambda k: -CompositeOverlay.channels[k].size)
            # Collapse the customized channel before the other definitions
            for key in sorted_customized + sorted_reductions:
                channel = CompositeOverlay.channels[key]
                if channel.mode is None: continue
                collapse_fn = channel.operation
                fn = collapse_fn.instance(**channel.opts)
                collapsed_vmap[k] = self._collapse(overlay, channel.pattern, fn, key)
        return vmap


    def _adjust_legend(self):
        # If legend enabled update handles and labels
        if not self.ax or not self.ax.get_legend(): return
        handles, _ = self.ax.get_legend_handles_labels()
        labels = self._map.last.legend
        if len(handles) and self.show_legend:
            fontP = FontProperties()
            fontP.set_size('medium')
            leg = self.ax.legend(handles[::-1], labels[::-1], prop=fontP)
            leg.get_frame().set_alpha(1.0)
        frame = self.ax.get_legend().get_frame()
        frame.set_facecolor('1.0')
        frame.set_edgecolor('0.0')
        frame.set_linewidth('1.5')

    def _format_title(self, key):
        view = self._map.get(key, None)
        if view is None: return None
        title_format = self._map.get_title(key if isinstance(key, tuple) else (key,), view)
        if title_format is None: return None

        values = [v.value for v in view]
        value = values[0] if len(set(values)) == 1 else ""
        return title_format.format(label=view.label, value=value,
                                   type=view.__class__.__name__)


    def __call__(self, axis=None, cyclic_index=0, ranges={}):
        key = self._keys[-1]
        self.ax = self._init_axis(axis)
        overlay = self._collapse_channels(HoloMap([((0,), self._map.last)])).last
        style_groups = dict((k, enumerate(list(v))) for k,v
                            in groupby(overlay, lambda s: s.style))

        for zorder, vmap in enumerate(overlay):
            new_index, _ = next(style_groups[vmap.style])
            self.subplots[zorder](self.ax, cyclic_index=cyclic_index+new_index, ranges=ranges)

        self._adjust_legend()

        return self._finalize_axis(None, title=self._format_title(key))


    def update_frame(self, n, ranges={}):
        n = n if n < len(self) else len(self) - 1
        ranges = ranges if ranges else self._get_range(n)
        for zorder, plot in enumerate(self.subplots.values()):
            plot.update_frame(n, ranges)
        self._finalize_axis(None)



class AnnotationPlot(Plot):
    """
    Draw the Annotation element on the supplied axis. Supports axis
    vlines, hlines, arrows (with or without labels), boxes and
    arbitrary polygonal lines. Note, unlike other Plot types,
    AnnotationPlot must always operate on a supplied axis as
    Annotations may only be used as part of Overlays.
    """

    style_opts = param.List(default=['alpha', 'color', 'edgecolors',
                                     'facecolors', 'linewidth',
                                     'linestyle', 'rotation', 'family',
                                     'weight', 'fontsize', 'visible',
                                     'edgecolor'],
                            constant=True, doc="""
     Box annotations, hlines and vlines and lines all accept
     matplotlib line style options. Arrow annotations also accept
     additional text options.""")

    def __init__(self, annotation, **params):
        self._annotation = annotation
        super(AnnotationPlot, self).__init__(annotation, **params)
        self._warn_invalid_intervals(self._map)
        self.handles['annotations'] = []

        line_only = ['linewidth', 'linestyle']
        arrow_opts = [opt for opt in self.style_opts if opt not in line_only]
        line_opts = line_only + ['color']
        self.opt_filter = {'hline': line_opts, 'vline': line_opts,
                           'line': line_opts,
                           '<': arrow_opts, '^': arrow_opts,
                           '>': arrow_opts, 'v': arrow_opts,
                           'spline': line_opts + ['edgecolor']}


    def _warn_invalid_intervals(self, vmap):
        "Check if the annotated intervals have appropriate keys"
        dim_labels = [d.name for d in self._map.key_dimensions]

        mismatch_set = set()
        for annotation in vmap.values():
            for spec in annotation.data:
                interval = spec[-1]
                if interval is None or dim_labels == ['Default']:
                    continue
                mismatches = set(interval.keys()) - set(dim_labels)
                mismatch_set = mismatch_set | mismatches

        if mismatch_set:
            mismatch_list= ', '.join('%r' % el for el in mismatch_set)
            self.warning("Invalid annotation interval key(s) ignored: %r" % mismatch_list)


    def _active_interval(self, key, interval):
        """
        Given an interval specification, determine whether the
        annotation should be shown or not.
        """
        dim_labels = [d.name for d in self._map.key_dimensions]
        if (interval is None) or dim_labels == ['Default']:
            return True

        key = key if isinstance(key, tuple) else (key,)
        key_dict = dict(zip(dim_labels, key))
        for key, (start, end) in interval.items():
            if (start is not None) and key_dict.get(key, -float('inf')) <= start:
                return False
            if (end is not None) and key_dict.get(key, float('inf')) > end:
                return False

        return True


    def _draw_annotations(self, annotation, key):
        """
        Draw the elements specified by the Annotation DataElement on the
        axis, return a list of handles.
        """
        handles = []
        opts = Element.options.style(annotation).opts
        color = opts.get('color', 'k')

        for spec in annotation.data:
            mode, info, interval = spec[0], spec[1:-1], spec[-1]
            opts = dict(el for el in opts.items()
                        if el[0] in self.opt_filter[mode])

            if not self._active_interval(key, interval):
                continue
            if mode == 'vline':
                handles.append(self.ax.axvline(spec[1], **opts))
                continue
            elif mode == 'hline':
                handles.append(self.ax.axhline(spec[1], **opts))
                continue
            elif mode == 'line':
                line = LineCollection([np.array(info[0])], **opts)
                self.ax.add_collection(line)
                handles.append(line)
                continue
            elif mode == 'spline':
                verts, codes = info
                patch = patches.PathPatch(Path(verts, codes),
                                          facecolor='none', **opts)
                self.ax.add_patch(patch)
                continue


            text, xy, points, arrowstyle = info
            arrowprops = dict(arrowstyle=arrowstyle, color=color)
            if mode in ['v', '^']:
                xytext = (0, points if mode=='v' else -points)
            elif mode in ['>', '<']:
                xytext = (points if mode=='<' else -points, 0)
            arrow = self.ax.annotate(text, xy=xy, textcoords='offset points',
                                     xytext=xytext, ha="center", va="center",
                                     arrowprops=arrowprops, **opts)
            handles.append(arrow)
        return handles


    def __call__(self, axis=None, cyclic_index=0, lbrt=None):
        self.ax = self._init_axis(axis)
        handles = self._draw_annotations(self._map.last, list(self._map.keys())[-1])
        self.handles['annotations'] = handles
        return self._finalize_axis(self._keys[-1])


    def update_handles(self, annotation, key, lbrt=None):
        # Clear all existing annotations
        for element in self.handles['annotations']:
            element.remove()

        self.handles['annotations'] = self._draw_annotations(annotation, key)


Plot.defaults.update({AxisLayout: GridPlot,
                      GridLayout: LayoutPlot,
                      ViewTree: LayoutPlot,
                      AdjointLayout: AdjointLayoutPlot,
                      NdOverlay: LayersPlot,
                      Overlay: LayersPlot,
                      Annotation: AnnotationPlot})
