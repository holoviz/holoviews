from itertools import product, groupby
from collections import OrderedDict

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec, animation
from matplotlib.font_manager import FontProperties

import param
from ..core import UniformNdMapping, ViewableElement, CompositeOverlay, NdOverlay, Overlay, HoloMap, \
    AdjointLayout, NdLayout, AxisLayout, LayoutTree, Element, Element3D
from ..core.options import Options, OptionTree
from ..core.util import find_minmax, valid_identifier
from ..element.raster import Raster


class Plot(param.Parameterized):
    """
    A Plot object returns either a matplotlib figure object (when
    called or indexed) or a matplotlib animation object as
    appropriate. Plots take element objects such as Matrix,
    Contours or Points as inputs and plots them in the
    appropriate format. As views may vary over time, all plots support
    animation via the anim() method.
    """

    aspect = param.Parameter(default=None, doc="""
        The aspect ratio mode of the plot. By default, a plot may
        select its own appropriate aspect ratio but sometimes it may
        be necessary to force a square aspect ratio (e.g. to display
        the plot as an element of a grid). The modes 'auto' and
        'equal' correspond to the axis modes of the same name in
        matplotlib, a numeric value may also be passed.""")

    figure_bounds = param.NumericTuple(default=(0.15, 0.15, 0.85, 0.85),
                                       doc="""
        The bounds of the figure as a 4-tuple of the form
        (left, bottom, right, top), defining the size of the border
        around the subplots.""")

    finalize_hooks = param.HookList(default=[], doc="""
        Optional list of hooks called when finalizing an axis.
        The hook is passed the full set of plot handles and the
        displayed object.""")

    normalization = param.Integer(default=0, bounds=(0, 1), doc="""
        Normalization options defining how to normalize the values in
        each plot. Valid options include:

        0 - Normalize by group over all frames.
        1 - Normalize by group for each frame individually.
        2 - Normalize per element over all frames.
        3 - Normalize per element for each frame""")

    projection = param.ObjectSelector(default=None,
                                      objects=['3d', 'polar', None], doc="""
        The projection of the plot axis, default of None is equivalent to
        2D plot, 3D and polar plots are also supported.""")

    show_frame = param.Boolean(default=True, doc="""
        Whether or not to show a complete frame around the plot.""")

    show_title = param.Boolean(default=True, doc="""
        Whether to display the plot title.""")

    size = param.NumericTuple(default=(4, 4), doc="""
        The matplotlib figure size in inches.""")

    # A list of matplotlib keyword arguments that may be supplied via a
    # style options object. Each subclass should override this
    # parameter to list every option that works correctly.
    style_opts = []

    # A mapping from ViewableElement types to their corresponding plot types
    defaults = {}

    # A mapping from ViewableElement types to their corresponding side plot types
    sideplots = {}

    # Once register_options is called, this OptionTree is populated
    options = OptionTree(groups={'plot': Options(), 'style': Options()})

    # A dictionary of custom OptionTree by custom id
    custom_options = {}


    def __init__(self, figure=None, axis=None, dimensions=None,
                 subplots=None, keys=None, **params):
        self.subplots = subplots
        self.subplot = figure is not None
        self._create_fig = True
        self.drawn = False
        self.dimensions = dimensions
        self._keys = keys
        self.ranges = None
        # List of handles to matplotlib objects for animation update
        self.handles = {} if figure is None else {'fig': figure}
        super(Plot, self).__init__(**params)
        self.ax = self._init_axis(axis)


    @classmethod
    def lookup_options(cls, obj, group):
        if obj.id is None:
            return cls.options.closest(obj, group)
        elif obj.id in cls.custom_options:
            return cls.custom_options[obj.id].closest(obj, group)
        else:
            raise KeyError("No custom settings defined for object with id %d" % obj.id)


    def compute_ranges(self, obj, key, ranges, norm_opt):
        """
        Given an object, a specific key and the normalization options
        this method will find the specified normalization options on
        the appropriate OptionTree, group the elements according to
        the selected normalization option (i.e. either per frame or
        over the whole animation) and finally compute the dimension
        ranges in each group. The new set of ranges is returned.
        """
        if obj is None: return None
        # Get inherited ranges
        ranges = dict((ranges if ranges else {}),
                      **(self.ranges if self.ranges else {}))
        # If only one norm option was requested then wrap it in a list
        if not isinstance(norm_opt, list): norm_opt = [norm_opt]

        # Get element identifiers from current object and resolve
        # with selected normalization options
        norm_opts = self._get_norm_opts(obj)

        # Traverse displayed object if normalization applies
        # at this level, and ranges for the group have not
        # been supplied from a composite plot
        return_fn = lambda x: x if isinstance(x, Element) else None
        for group, nopt in norm_opts.items():
            if nopt not in norm_opt or group in ranges:
                continue
            elif nopt in [1, 3]: # Traverse frame
                elements = self._get_frame(key, obj).traverse(return_fn, [group])
            elif nopt in [0, 2]: # Traverse full animation
                elements = obj.traverse(return_fn, [group])
            self._compute_group_range(group, elements, ranges)
        return ranges


    def _get_norm_opts(self, obj):
        """
        Gets the normalization options for a LabelledData object by
        traversing the object for to find elements and their ids.
        The id is then used to select the appropriate OptionsTree,
        accumulating the normalization options into a dictionary.
        Returns a dictionary of normalization options for each
        element in the tree.
        """
        norm_opts = {}
        type_val_fn = lambda x: (x.id, (type(x).__name__, valid_identifier(x.value),
                                        valid_identifier(x.label)))\
                                        if isinstance(x, Element) else None
        element_specs = {(idspec[0], idspec[1]) for idspec in obj.traverse(type_val_fn)
                         if idspec is not None}
        id_groups = sorted(groupby(element_specs, lambda x: x[0]))
        for id, element_spec_group in id_groups:
            group_specs = [el for _,el in element_spec_group]
            optstree = self.custom_options.get(obj.id, Plot.options)
            custom_opts = [(tuple(opts.path.split('.')[1:]), opts['plot'].options.get('normalization'))
                           for opts in optstree]
            norm_opts.update({path: nopt for path, nopt in custom_opts if nopt is not None and
                              any(path == spec[:i] for spec in group_specs for i in range(1, 4))})
        element_specs = [spec for eid, spec in element_specs]
        norm_opts.update({spec: self.normalization for spec in element_specs
                          if not any(spec[:i] in norm_opts.keys() for i in range(1, 4))})
        return norm_opts


    @staticmethod
    def _compute_group_range(group, elements, ranges):
        # Iterate over all elements in a normalization group
        # and accumulate their ranges into the supplied dictionary.
        elements = [el for el in elements if el is not None]
        for el in elements:
            for dim in el.dimensions():
                dim_range = el.range(dim.name)
                if group not in ranges: ranges[group] = OrderedDict()
                if dim.name in ranges[group]:
                    ranges[group][dim.name] = find_minmax(ranges[group][dim.name], dim_range)
                else:
                    ranges[group][dim.name] = dim_range


    def _get_frame(self, key, obj):
        if not isinstance(key, tuple): key = (key,)
        key_dims = obj.traverse(lambda x: x.key_dimensions, ('HoloMap',))[0]
        return obj.select(**dict(zip([d.name for d in key_dims], key)))


    @classmethod
    def register_options(cls):
        path_items = {}
        for view_class, plot in Plot.defaults.items():
            name = view_class.__name__
            plot_opts = [k for k in plot.params().keys() if k not in ['name']]
            style_opts = plot.style_opts
            opt_groups = {'plot': Options(allowed_keywords=plot_opts)}
            if style_opts:
                opt_groups.update({'style': Options(allowed_keywords=style_opts)})
            path_items[name] = opt_groups
        cls.options = OptionTree(sorted(path_items.items()),
                                  groups={'style': Options(), 'plot': Options()})


    def _check_map(self, view, element_type=Element):
        """
        Helper method that ensures a given element is always returned as
        an HoloMap object.
        """
        if not isinstance(view, HoloMap):
            vmap = HoloMap(initial_items=(0, view), id=view.id)
        else:
            vmap = view

        check = vmap.last
        if issubclass(vmap.type, CompositeOverlay):
            check = vmap.last.values()[0]
        if isinstance(check, Element3D):
            self.projection = '3d'

        return vmap


    def _format_title(self, key):
        view = self._map.get(key, None)
        if view is None: return None
        title_format = self._map.get_title(key if isinstance(key, tuple) else (key,), view)
        if title_format is None:
            return None
        return title_format.format(label=view.label, value=view.value,
                                   type=view.__class__.__name__)


    def _finalize_axis(self, key):
        """
        General method to finalize the axis and plot.
        """

        self.drawn = True
        if self.subplot:
            return self.ax
        else:
            plt.draw()
            fig = self.handles['fig']
            plt.close(fig)
            return fig


    def _init_axis(self, axis):
        """
        Return an axis which may need to be initialized from
        a new figure.
        """
        if not self.subplot and self._create_fig:
            fig = plt.figure()
            self.handles['fig'] = fig
            l, b, r, t = self.figure_bounds
            fig.subplots_adjust(left=l, bottom=b, right=r, top=t)
            fig.set_size_inches(list(self.size))
            axis = fig.add_subplot(111, projection=self.projection)
            axis.set_aspect('auto')

        return axis


    def __getitem__(self, frame):
        """
        Get the matplotlib figure at the given frame number.
        """
        if frame > len(self):
            self.warning("Showing last frame available: %d" % len(self))
        if not self.drawn: self.handles['fig'] = self()
        self.update_frame(frame)
        return self.handles['fig']


    def anim(self, start=0, stop=None, fps=30):
        """
        Method to return an Matplotlib animation. The start and stop
        frames may be specified as well as the fps.
        """

        frames = list(range(len(self)))[slice(start, stop, 1)]

        figure = self()
        anim = animation.FuncAnimation(figure, self.update_frame,
                                       frames=frames,
                                       interval = 1000.0/fps)
        # Close the figure handle
        plt.close(figure)
        return anim

    def __len__(self):
        """
        Returns the total number of available frames.
        """
        return len(self._keys)


    def __call__(self, ranges=None):
        """
        Return a matplotlib figure.
        """
        raise NotImplementedError


    def update_frame(self, n, ranges=None):
        """
        Updates the current frame of the plot.
        """
        raise NotImplementedError



class ElementPlot(Plot):

    apply_databounds = param.Boolean(default=True, doc="""
        Whether to compute the plot bounds from the data itself.""")

    normalization = param.Integer(default=2, bounds=(0, 3), doc="""
        Normalization options defining how to normalize the values in
        each plot. Valid options include:

        0 - Normalize by group over all frames.
        1 - Normalize by group for each frame individually.
        2 - Normalize per element over all frames.
        3 - Normalize per element for each frame""")

    orientation = param.ObjectSelector(default='horizontal',
                                       objects=['horizontal', 'vertical'], doc="""
        The orientation of the plot. Note that this parameter may not
        always be respected by all plots but should be respected by
        adjoined plots when appropriate.""")

    rescale_individually = param.Boolean(default=False, doc="""
        Whether to use redraw the axes per map or per element.""")

    show_grid = param.Boolean(default=False, doc="""
        Whether to show a Cartesian grid on the plot.""")

    show_xaxis = param.ObjectSelector(default='bottom',
                                      objects=['top', 'bottom', None], doc="""
        Whether and where to display the xaxis.""")

    show_yaxis = param.ObjectSelector(default='left',
                                      objects=['left', 'right', None], doc="""
        Whether and where to display the yaxis.""")


    def __init__(self, element, keys=None, cyclic_index=0, zorder=0, **params):
        self._map = self._check_map(element)
        self.cyclic_index = cyclic_index
        self.zorder = zorder
        keys = keys if keys else self._map.keys()
        super(ElementPlot, self).__init__(keys=keys, **params)


    def _get_frame(self, key, obj):
        return self._map[key]


    def get_extents(self, view, ranges):
        """
        Gets the extents for the axes from the current View. The globally
        computed ranges can optionally override the extents.
        """
        return view.extents if self.rescale_individually else self._map.extents


    def _finalize_axis(self, key, title=None, ranges=None, xticks=None, yticks=None,
                       xlabel=None, ylabel=None):
        """
        Applies all the axis settings before the axis or figure is returned.
        Only plots with zorder 0 get to apply their settings.

        When the number of the frame is supplied as n, this method looks
        up and computes the appropriate title, axis labels and axis bounds.
        """

        axis = self.ax

        view = self._map.get(key, None)
        if self.zorder == 0 and key is not None:
            if view is not None:
                title = None if self.zorder > 0 else self._format_title(key)
                if hasattr(view, 'xlabel') and xlabel is None:
                    xlabel = view.xlabel
                if hasattr(view, 'ylabel') and ylabel is None:
                    ylabel = view.ylabel
                if self.apply_databounds:
                    extents = self.get_extents(view, ranges)
                    l, b, r, t = [coord if np.isreal(coord) else np.NaN for coord in extents]
                    if not np.NaN in (l, r): axis.set_xlim((l, r))
                    if b == t: t += 1. # Arbitrary y-extent if zero range
                    if not np.NaN in (b, t): axis.set_ylim((b, t))

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

        for hook in self.finalize_hooks:
            hook(self.subplots, self.handles, view)

        return super(ElementPlot, self)._finalize_axis(key)


    def match_range(self, element, ranges):
        match_tuple = ()
        match = ranges.get((), {})
        for spec in [type(element).__name__,
                     valid_identifier(element.value),
                     valid_identifier(element.label)]:
            match_tuple += (spec,)
            if match_tuple in ranges:
                match = ranges[match_tuple]
        return match


    def update_frame(self, n, ranges=None):
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
        ranges = self.compute_ranges(self._map, key, ranges, [0, 1, 2, 3])
        ranges = self.match_range(view, ranges)
        axis_kwargs = self.update_handles(view, key, ranges) if view is not None else {}
        self._finalize_axis(key, ranges=ranges, **(axis_kwargs if axis_kwargs else {}))


    def update_handles(self, view, key, ranges=None):
        """
        Update the elements of the plot.
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

    def __init__(self, grid, ranges=None, **params):
        if not isinstance(grid, AxisLayout):
            raise Exception("GridPlot only accepts AxisLayout.")
        items = [(k, self._check_map(v)) for k, v in grid.data.items()]
        self.grid = grid.clone(items, id=grid.id)

        if grid.ndims == 1:
            self.rows, self.cols = (1, len(grid.keys()))
        else:
            x, y = list(zip(*list(grid.keys())))
            self.cols, self.rows = (len(set(x)), len(set(y)))

        extra_opts = self.lookup_options(self.grid, 'plot').options
        super(GridPlot, self).__init__(show_xaxis=None, show_yaxis=None,
                                       show_frame=False, keys=self.grid.all_keys,
                                       **dict(params, **extra_opts))
        # Compute ranges gridwise
        self.ranges = self.compute_ranges(self.grid, None, ranges, 0)
        self._gridspec = gridspec.GridSpec(self.rows, self.cols)
        self.subplots, self.subaxes = self._create_subplots()


    def _get_frame(self, key, obj):
        if not isinstance(key, tuple): key = (key,)
        key_dims = obj.traverse(lambda x: x.key_dimensions, ('HoloMap',))[0]
        return obj.select(**dict(zip([d.name for d in key_dims], key)))


    def _create_subplots(self):
        subplots, subaxes = OrderedDict(), OrderedDict()
        r, c = (0, 0)
        for coord in self.grid.keys(full_grid=True):
            # Create axes
            subax = plt.subplot(self._gridspec[r, c])
            subaxes[(r, c)] = subax

            # Create subplot
            view = self.grid.data.get(coord, None)
            if view is not None:
                grid_dimvals = dict(AxisLayout=zip(zip(self.grid.key_dimensions, coord)))
                vtype = view.type if isinstance(view, HoloMap) else view.__class__
                opts = self.lookup_options(view, 'plot').options
                opts.update(show_legend=self.show_legend, show_xaxis=self.show_xaxis,
                            show_yaxis=self.show_yaxis, show_title=self.show_title,
                            figure=self.handles['fig'], axis=subax,
                            dimensions=grid_dimvals)
                subplot = Plot.defaults[vtype](view, **opts)
                subplots[(r, c)] = subplot
            if r != self.rows-1:
                r += 1
            else:
                r = 0
                c += 1
        return subplots, subaxes


    def __call__(self, ranges=None):
        # Get the extent of the grid elements (not the whole grid)
        subplot_kwargs = dict()

        ranges = self.compute_ranges(self.grid, self._keys[-1], ranges, [0, 1])
        for subplot in self.subplots.values():
            subplot(ranges=ranges, **subplot_kwargs)
        self._grid_axis()
        self._adjust_subplots()

        self.drawn = True
        if self.subplot: return self.ax
        plt.close(self.handles['fig'])
        return self.handles['fig']


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
        grid_axis.set_xlabel(str(self.grid.key_dimensions[0]))
        grid_axis.set_title(self._format_title(key))

        # Compute and set x- and y-ticks
        keys = self.grid.keys()
        if self.grid.ndims == 1:
            dim1_keys = keys
            dim2_keys = [0]
            grid_axis.get_yaxis().set_visible(False)
        else:
            dim1_keys, dim2_keys = zip(*keys)
            grid_axis.set_ylabel(str(self.grid.key_dimensions[1]))
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
        for ax in self.subaxes.values():
            xpos = l + (c*ax_w) + (c * b_w)
            ypos = b + (r*ax_h) + (r * b_h)
            if r != self.rows-1:
                r += 1
            else:
                r = 0
                c += 1
            if not ax is None:
                ax.set_position([xpos, ypos, ax_w, ax_h])


    def update_frame(self, n, ranges=None):
        key = self._keys[n]
        ranges = self.compute_ranges(self.grid, key, ranges, 1)
        for subplot in self.subplots.values():
            subplot.update_frame(n, ranges)
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
    when embedded in a NdLayout, it can recompute the layout to
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


    def __init__(self, layout, layout_type, subaxes, subplots, **params):
        # The AdjointLayout ViewableElement object
        self.layout = layout
        # Type may be set to 'Embedded Dual' by a call it grid_situate
        self.layout_type = layout_type
        self.view_positions = self.layout_dict[self.layout_type]['positions']

        # The supplied (axes, view) objects as indexed by position
        self.subaxes = {pos: ax for ax, pos in zip(subaxes, self.view_positions)}
        super(AdjointLayoutPlot, self).__init__(subplots=subplots, **params)


    def __call__(self, ranges=None):
        """
        Plot all the views contained in the AdjointLayout Object using axes
        appropriate to the layout configuration. All the axes are
        supplied by LayoutPlot - the purpose of the call is to
        invoke subplots with correct options and styles and hide any
        empty axes as necessary.
        """
        for pos in self.view_positions:
            # Pos will be one of 'main', 'top' or 'right' or None
            view = self.layout.get(pos, None)
            subplot = self.subplots.get(pos, None)
            ax = self.subaxes.get(pos, None)
            # If no view object or empty position, disable the axis
            if None in [view, pos, subplot]:
                ax.set_axis_off()
                continue

            vtype = view.type if isinstance(view, HoloMap) else view.__class__
            # 'Main' views that should be displayed with square aspect
            if pos == 'main' and issubclass(vtype, ViewableElement):
                subplot.aspect='square'
            subplot(ranges=ranges)
        self.drawn = True


    def adjust_positions(self):
        """
        Make adjustments to the positions of subplots (if available)
        relative to the main plot axes as required.

        This method is called by LayoutPlot after an initial pass
        used to position all the Layouts together. This method allows
        LayoutPlots to make final adjustments to the axis positions.
        """
        main_ax = self.subaxes['main']
        bbox = main_ax.get_position()
        if 'right' in self.view_positions:
            ax = self.subaxes['right']
            ax.set_position([bbox.x1 + bbox.width * self.border_size,
                             bbox.y0,
                             bbox.width * self.subplot_size, bbox.height])
        if 'top' in self.view_positions:
            ax = self.subaxes['top']
            ax.set_position([bbox.x0,
                             bbox.y1 + bbox.height * self.border_size,
                             bbox.width, bbox.height * self.subplot_size])


    def update_frame(self, n, ranges=None):
        for pos, subplot in self.subplots.items():
            if subplot is not None:
                subplot.update_frame(n, ranges)


    def __len__(self):
        return max([len(v) if isinstance(v, UniformNdMapping) else len(v.all_keys)
                    for v in self.layout if isinstance(v, (UniformNdMapping, AxisLayout))]+[1])


class LayoutPlot(Plot):
    """
    A LayoutPlot accepts either a LayoutTree or a NdLayout and
    displays the elements in a cartesian grid in scanline order.
    """

    horizontal_spacing = param.Number(default=0.5, doc="""
      Specifies the space between horizontally adjacent elements in the grid.
      Default value is set conservatively to avoid overlap of subplots.""")

    vertical_spacing = param.Number(default=0.2, doc="""
      Specifies the space between vertically adjacent elements in the grid.
      Default value is set conservatively to avoid overlap of subplots.""")

    def __init__(self, layout, **params):
        if not isinstance(layout, (NdLayout, LayoutTree)):
            raise Exception("LayoutPlot only accepts LayoutTree objects.")

        self.layout = layout
        self.subplots = {}
        self.rows, self.cols = layout.shape
        self.coords = list(product(range(self.rows),
                                   range(self.cols)))
        keys = None
        if isinstance(layout, NdLayout):
            keys = layout.traverse(lambda x: x.keys(), ('HoloMap',),
                                   full_breadth=False)[0]

        super(LayoutPlot, self).__init__(keys=keys, **params)
        self.ax, self.subplots, self.subaxes = self._compute_gridspec()


    def _get_frame(self, n, obj):
        """
        Creates a clone of the Layout with the nth-frame for each
        Element.
        """
        layout_frame = obj.clone()
        nthkey_fn = lambda x: zip(tuple(x.name for x in x.key_dimensions),
                                  x.data.keys()[max([n, len(x)-1])])
        for path, item in obj.items():
            dim_keys = item.traverse(nthkey_fn, ('HoloMap',))
            if dim_keys:
                layout_frame[path] = item.select(**dict(dim_keys[0]))
            else:
                layout_frame[path] = item
        return layout_frame


    def _compute_gridspec(self):
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
        axis = self._init_axis(None)

        layouts, grid_indices = {}, {}
        row_heightratios, col_widthratios = {}, {}
        for (r, c) in self.coords:
            # Get view at layout position and wrap in AdjointLayout
            view = self.layout.grid_items.get((r, c), None)
            layout_view = view if isinstance(view, AdjointLayout) else AdjointLayout([view])
            layouts[(r, c)] = layout_view

            # Compute shape of AdjointLayout element
            layout_lens = {1:'Single', 2:'Dual', 3:'Triple'}
            layout_type = layout_lens[len(layout_view)]
            width_ratios = AdjointLayoutPlot.layout_dict[layout_type]['width_ratios']
            height_ratios = AdjointLayoutPlot.layout_dict[layout_type]['height_ratios']
            layout_shape = (len(width_ratios), len(height_ratios))

            # For each row and column record the width and height ratios
            # of the LayoutPlot with the most horizontal or vertical splits
            if layout_shape[0] > row_heightratios.get(r, (0, None))[0]:
                row_heightratios[r] = (layout_shape[1], height_ratios)
            if layout_shape[1] > col_widthratios.get(c, (0, None))[0]:
                col_widthratios[c] = (layout_shape[0], width_ratios)

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
        layout_subplots, layout_axes = {}, {}
        for (r, c) in self.coords:
            # Compute the layout type from shape
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

            # Get the AdjoinLayout at the specified coordinate
            view = layouts[(r, c)]
            positions = AdjointLayoutPlot.layout_dict[layout_type]['positions']

            # Create temporary subplots to get projections types
            # to create the correct subaxes for all plots in the layout
            temp_subplots = self._create_subplots(layouts[(r, c)], positions)
            gidx, gsinds, projs = self.grid_situate(temp_subplots, gidx, layout_type, cols)

            # Generate the axes and create the subplots with the appropriate
            # axis objects
            subaxes = [plt.subplot(self.gs[ind], projection=proj)
                       for ind, proj in zip(gsinds, projs)]
            subplots = self._create_subplots(layouts[(r, c)], positions,
                                             dict(zip(positions, subaxes)))
            layout_axes[(r, c)] = subaxes

            # Generate the AdjointLayoutsPlot which will coordinate
            # plotting of AdjointLayouts in the larger grid
            plotopts = self.lookup_options(view, 'plot').options
            layout_plot = AdjointLayoutPlot(view, layout_type, subaxes, subplots,
                                            figure=self.handles['fig'], **plotopts)
            layout_subplots[(r, c)] = layout_plot

        return axis, layout_subplots, layout_axes


    def grid_situate(self, subplots, current_idx, layout_type, subgrid_width):
        """
        Situate the current AdjointLayoutPlot in a LayoutPlot. The
        LayoutPlot specifies a layout_type into which the AdjointLayoutPlot
        must be embedded. This enclosing layout is guaranteed to have
        enough cells to display all the views.

        Based on this enforced layout format, a starting index
        supplied by LayoutPlot (indexing into a large gridspec
        arrangement) is updated to the appropriate embedded value. It
        will also return a list of gridspec indices associated with
        the all the required layout axes.
        """
        # Set the layout configuration as situated in a NdLayout

        if layout_type == 'Single':
            positions = ['main']
            start, inds = current_idx+1, [current_idx]
        elif layout_type == 'Dual':
            positions = ['main', 'right']
            start, inds = current_idx+2, [current_idx, current_idx+1]

        bottom_idx = current_idx + subgrid_width
        if layout_type == 'Embedded Dual':
            positions = [None, None, 'main', 'right']
            bottom = ((current_idx+1) % subgrid_width) == 0
            grid_idx = (bottom_idx if bottom else current_idx)+1
            start, inds = grid_idx, [current_idx, bottom_idx]
        elif layout_type == 'Triple':
            positions = ['top', None, 'main', 'right']
            bottom = ((current_idx+2) % subgrid_width) == 0
            grid_idx = (bottom_idx if bottom else current_idx) + 2
            start, inds = grid_idx, [current_idx, current_idx+1,
                              bottom_idx, bottom_idx+1]
        projs = [subplots.get(pos, Plot).projection for pos in positions]

        return start, inds, projs


    def _create_subplots(self, layout, positions, axes={}):
        """
        Plot all the views contained in the AdjointLayout Object using axes
        appropriate to the layout configuration. All the axes are
        supplied by LayoutPlot - the purpose of the call is to
        invoke subplots with correct options and styles and hide any
        empty axes as necessary.
        """
        subplots = {}
        subplot_opts = dict(show_title=False, layout=layout)
        for pos in positions:
            # Pos will be one of 'main', 'top' or 'right' or None
            view = layout.get(pos, None)
            ax = axes.get(pos, None)
            if view is None:
                continue
            # Customize plotopts depending on position.
            plotopts = self.lookup_options(view, 'plot').options
            # Options common for any subplot

            override_opts = {}
            if pos == 'right':
                right_opts = dict(orientation='vertical', show_xaxis=None, show_yaxis='left')
                override_opts = dict(subplot_opts, **right_opts)
            elif pos == 'top':
                top_opts = dict(show_xaxis='bottom', show_yaxis=None)
                override_opts = dict(subplot_opts, **top_opts)

            # Override the plotopts as required
            plotopts.update(override_opts, figure=self.handles['fig'])
            vtype = view.type if isinstance(view, HoloMap) else view.__class__
            layer_types = (vtype,) if isinstance(view, ViewableElement) else view.layer_types
            if isinstance(view, AxisLayout):
                if len(layer_types) == 1 and issubclass(layer_types[0], Raster):
                    from .raster import MatrixGridPlot
                    plot_type = MatrixGridPlot
                else:
                    plot_type = GridPlot
            else:
                if pos == 'main':
                    plot_type = Plot.defaults[vtype]
                else:
                    plot_type = Plot.sideplots[vtype]

            subplots[pos] = plot_type(view, axis=ax, keys=self._keys, **plotopts)
        return subplots


    def __call__(self):
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        ranges = self.compute_ranges(self.layout, -1, None, [0, 1])
        rcopts = self.lookup_options(self.layout, 'style').options
        for subplot in self.subplots.values():
            with matplotlib.rc_context(rcopts):
                subplot(ranges=ranges)
        plt.draw()

        # Adjusts the AdjointLayout subplot positions
        for (r, c) in self.coords:
            self.subplots.get((r, c), None).adjust_positions()

        return self._finalize_axis(None)


    def update_frame(self, n):
        ranges = self.compute_ranges(self.layout, n, None, [0, 1])
        for subplot in self.subplots.values():
            subplot.update_frame(n, ranges=ranges)


    def __len__(self):
        return max([len(v) for v in self.subplots.values()]+[1])



class OverlayPlot(ElementPlot):
    """
    OverlayPlot supports processing of channel operations on Overlays
    across maps.
    """

    show_legend = param.Boolean(default=True, doc="""
        Whether to show legend for the plot.""")

    def __init__(self, overlay, **params):
        super(OverlayPlot, self).__init__(overlay, **params)
        self.subplots = self._create_subplots()


    def _create_subplots(self):
        subplots = OrderedDict()

        #collapsed = self._collapse_channels(self._map)
        keys, vmaps = self._map.split_overlays()
        style_groups = dict((k, enumerate(list(v))) for k,v
                            in groupby(vmaps, lambda s: (s.last.value)))
        for zorder, (key, vmap) in enumerate(zip(keys, vmaps)):
            cyclic_index, _ = next(style_groups[(vmap.last.value)])
            plotopts = self.lookup_options(vmap.last, 'plot').options
            if issubclass(vmap.type, NdOverlay):
                plotopts['dimensions'] = zip(vmap.last.key_dimensions, key)
            plotopts = dict(keys=self._keys, axis=self.ax,
                            cyclic_index=cyclic_index, figure=self.handles['fig'],
                            zorder=zorder, **plotopts)
            plotype = Plot.defaults[type(vmap.last)]
            subplots[key] = plotype(vmap, **plotopts)

        return subplots


    def _collapse(self, overlay, pattern, fn, style_key):
        """
        Given an overlay object collapse the channels according to
        pattern using the supplied function. Any collapsed ViewableElement is
        then given the supplied style key.
        """
        pattern = [el.strip() for el in pattern.rsplit('*')]
        if len(pattern) > len(overlay): return overlay

        skip=0
        collapsed_overlay = overlay.clone(None)
        for i, key in enumerate(overlay.keys()):
            layer_labels = overlay.labels[i:len(pattern)+i]
            matching = all(l.endswith(p) for l, p in zip(layer_labels, pattern))
            if matching and len(layer_labels)==len(pattern):
                views = [el for el in overlay if el.label in layer_labels]
                if isinstance(overlay, Overlay):
                    views = np.product([Overlay.from_view(el) for el in overlay])
                overlay_slice = overlay.clone(views)
                collapsed_view = fn(overlay_slice)
                if isinstance(overlay, LayoutTree):
                    collapsed_overlay *= collapsed_view
                else:
                    collapsed_overlay[key] = collapsed_view
                skip = len(views)-1
            elif skip:
                skip = 0 if skip <= 0 else (skip - 1)
            else:
                if isinstance(overlay, LayoutTree):
                    collapsed_overlay *= overlay[key]
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


    def __call__(self, ranges=None):
        for plot in self.subplots.values():
            plot(ranges=ranges)
        self._adjust_legend()

        key = self._keys[-1]
        return self._finalize_axis(key, ranges=ranges, title=self._format_title(key))


    def update_frame(self, n, ranges=None):
        n = n if n < len(self) else len(self) - 1
        key = self._keys[n]
        if self.projection == '3d':
            self.ax.clear()

        for plot in self.subplots.values():
            plot.update_frame(n, ranges)
        self._finalize_axis(key, ranges)


Plot.defaults.update({AxisLayout: GridPlot,
                      NdLayout: LayoutPlot,
                      LayoutTree: LayoutPlot,
                      AdjointLayout: AdjointLayoutPlot,
                      NdOverlay: OverlayPlot,
                      Overlay: OverlayPlot})
