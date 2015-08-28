import numpy as np

import param

from bokeh.io import gridplot

from ...core import OrderedDict, CompositeOverlay, Element
from ...core import Store, Layout, AdjointLayout, NdLayout, Empty, GridSpace, HoloMap
from ...core.options import Compositor
from ...core import traversal
from ..plot import Plot, GenericCompositePlot, GenericLayoutPlot
from .renderer import BokehRenderer


class BokehPlot(Plot):
    """
    Plotting baseclass for the Bokeh backends, implementing the basic
    plotting interface for Bokeh based plots.
    """

    width = param.Integer(default=300, doc="""
        Width of the plot in pixels""")

    height = param.Integer(default=300, doc="""
        Height of the plot in pixels""")

    renderer = BokehRenderer

    @property
    def state(self):
        """
        The plotting state that gets updated via the update method and
        used by the renderer to generate output.
        """
        return self.handles['plot']



class GridPlot(BokehPlot, GenericCompositePlot):
    """
    Plot a group of elements in a grid layout based on a GridSpace element
    object.
    """

    def __init__(self, layout, ranges=None, keys=None, dimensions=None,
                 layout_num=1, **params):
        if not isinstance(layout, GridSpace):
            raise Exception("GridPlot only accepts GridSpace.")
        self.layout = layout
        self.rows, self.cols = layout.shape
        self.layout_num = layout_num
        extra_opts = self.lookup_options(layout, 'plot').options
        if not keys or not dimensions:
            dimensions, keys = traversal.unique_dimkeys(layout)
        if 'uniform' not in params:
            params['uniform'] = traversal.uniform(layout)

        super(GridPlot, self).__init__(keys=keys, dimensions=dimensions,
                                       **dict(extra_opts, **params))
        self.subplots, self.layout = self._create_subplots(layout, ranges)


    def _create_subplots(self, layout, ranges):
        layout = layout.map(Compositor.collapse_element, [CompositeOverlay],
                            clone=False)
        norm_opts = self._deep_options(layout, 'norm', ['axiswise'], [Element])
        axiswise = any(v.get('axiswise', False) for v in norm_opts.values())

        subplots = OrderedDict()
        frame_ranges = self.compute_ranges(layout, None, ranges)
        frame_ranges = OrderedDict([(key, self.compute_ranges(layout, key, frame_ranges))
                                    for key in self.keys])
        collapsed_layout = layout.clone(shared_data=False, id=layout.id)
        for i, coord in enumerate(layout.keys(full_grid=True)):
            r = i % self.cols
            c = i // self.cols

            if not isinstance(coord, tuple): coord = (coord,)
            view = layout.data.get(coord, None)
            # Create subplot
            if view is not None:
                vtype = view.type if isinstance(view, HoloMap) else view.__class__
                opts = self.lookup_options(view, 'plot').options
            else:
                vtype = None

            # Create axes
            kwargs = {}
            if c == 0 and r != 0:
                kwargs['xaxis'] = 'left-bare'
                kwargs['width'] = 175
            if c != 0 and r == 0 and not layout.ndims == 1:
                kwargs['yaxis'] = 'bottom-bare'
                kwargs['height'] = 175
            if c == 0 and r == 0:
                kwargs['width'] = 175
                kwargs['height'] = 175
            if r != 0 and c != 0:
                kwargs['xaxis'] = 'left-bare'
                kwargs['yaxis'] = 'bottom-bare'

            if 'width' not in kwargs:
                kwargs['width'] = 125
            if 'height' not in kwargs:
                kwargs['height'] = 125

            # Create subplot
            plotting_class = Store.registry[self.renderer.backend].get(vtype, None)
            if plotting_class is None:
                if view is not None:
                    self.warning("Bokeh plotting class for %s type not found, "
                                 "object will not be rendered." % vtype.__name__)
            else:
                subplot = plotting_class(view, dimensions=self.dimensions,
                                         show_title=False, subplot=True,
                                         ranges=frame_ranges, uniform=self.uniform,
                                         keys=self.keys, **dict(opts, **kwargs))
                collapsed_layout[coord] = (subplot.layout
                                           if isinstance(subplot, GenericCompositePlot)
                                           else subplot.map)
                subplots[coord] = subplot
        return subplots, collapsed_layout


    def initialize_plot(self, ranges=None):
        ranges = self.compute_ranges(self.layout, self.keys[-1], None)
        plots = [[] for r in range(self.cols)]
        passed_plots = []
        for i, coord in enumerate(self.layout.keys(full_grid=True)):
            r = i % self.cols
            subplot = self.subplots.get(coord, None)
            if subplot is not None:
                plot = subplot.initialize_plot(ranges=ranges, plots=passed_plots)
                plots[r].append(plot)
                passed_plots.append(plots[r][-1])
            else:
                plots[r].append(None)
                passed_plots.append(None)
        self.handles['plot'] = gridplot(plots[::-1])
        self.handles['plots'] = plots
        self.drawn = True

        return self.handles['plot']


    def update_frame(self, key, ranges=None):
        """
        Update the internal state of the Plot to represent the given
        key tuple (where integers represent frames). Returns this
        state.
        """
        ranges = self.compute_ranges(self.layout, key, ranges)
        plots = self.handles['plots']
        for i, coord in enumerate(self.layout.keys(full_grid=True)):
            r = i % self.cols
            c = i // self.cols
            subplot = self.subplots.get(coord, None)
            if subplot is not None:
                subplot.update_frame(key, ranges, plots[r][c])



class LayoutPlot(BokehPlot, GenericLayoutPlot):

    def __init__(self, layout, **params):
        super(LayoutPlot, self).__init__(layout, **params)
        self.layout, self.subplots = self._init_layout(layout)


    def _init_layout(self, layout):
        # Situate all the Layouts in the grid and compute the gridspec
        # indices for all the axes required by each LayoutPlot.
        gidx = 0
        layout_count = 0
        collapsed_layout = layout.clone(shared_data=False, id=layout.id)
        frame_ranges = self.compute_ranges(layout, None, None)
        frame_ranges = OrderedDict([(key, self.compute_ranges(layout, key, frame_ranges))
                                    for key in self.keys])
        layout_items = layout.grid_items()
        layout_dimensions = layout.kdims if isinstance(layout, NdLayout) else None
        layout_subplots, layouts = {}, {}
        for r, c in self.coords:
            # Get view at layout position and wrap in AdjointLayout
            _, view = layout_items.get((r, c), (None, None))
            view = view if isinstance(view, AdjointLayout) else AdjointLayout([view])
            layouts[(r, c)] = view

            # Compute the layout type from shape
            layout_type = 'Single'

            # Get the AdjoinLayout at the specified coordinate

            positions = AdjointLayoutPlot.layout_dict[layout_type]['positions']

            # Create temporary subplots to get projections types
            # to create the correct subaxes for all plots in the layout
            layout_key, _ = layout_items.get((r, c), (None, None))
            if isinstance(layout, NdLayout) and layout_key:
                layout_dimensions = OrderedDict(zip(layout_dimensions, layout_key))

            # Generate the axes and create the subplots with the appropriate
            # axis objects, handling any Empty objects.
            obj = layouts[(r, c)]
            empty = isinstance(obj.main, Empty)
            if empty:
                obj = AdjointLayout([])
            else:
                layout_count += 1
            subplot_data = self._create_subplots(obj, positions,
                                                 layout_dimensions, frame_ranges,
                                                 num=0 if empty else layout_count)
            subplots, adjoint_layout = subplot_data

            # Generate the AdjointLayoutsPlot which will coordinate
            # plotting of AdjointLayouts in the larger grid
            plotopts = self.lookup_options(view, 'plot').options
            layout_plot = AdjointLayoutPlot(adjoint_layout, layout_type, subplots, **plotopts)
            layout_subplots[(r, c)] = layout_plot
            if layout_key:
                collapsed_layout[layout_key] = adjoint_layout
        return collapsed_layout, layout_subplots


    def _create_subplots(self, layout, positions, layout_dimensions, ranges, num=0):
        """
        Plot all the views contained in the AdjointLayout Object using axes
        appropriate to the layout configuration. All the axes are
        supplied by LayoutPlot - the purpose of the call is to
        invoke subplots with correct options and styles and hide any
        empty axes as necessary.
        """
        subplots = {}
        projections = []
        adjoint_clone = layout.clone(shared_data=False, id=layout.id)
        subplot_opts = dict(show_title=False, adjoined=layout)
        for pos in positions:
            # Pos will be one of 'main', 'top' or 'right' or None
            element = layout.get(pos, None)
            if element is None:
                continue

            # Customize plotopts depending on position.
            plotopts = self.lookup_options(element, 'plot').options

            # Options common for any subplot
            override_opts = {}
            sublabel_opts = {}
            if pos == 'main':
                own_params = self.get_param_values(onlychanged=True)
                sublabel_opts = {k: v for k, v in own_params
                                 if 'sublabel_' in k}
            else:
                continue

            # Override the plotopts as required
            plotopts = dict(sublabel_opts, **plotopts)
            plotopts.update(override_opts)
            vtype = element.type if isinstance(element, HoloMap) else element.__class__
            plot_type = Store.registry[self.renderer.backend].get(vtype, None)
            if plot_type is None:
                self.warning("Bokeh plotting class for %s type not found, object will "
                             "not be rendered." % vtype.__name__)
                continue
            plot_type = Store.registry[self.renderer.backend][vtype]
            num = num if len(self.coords) > 1 else 0
            subplots[pos] = plot_type(element, keys=self.keys,
                                      dimensions=self.dimensions,
                                      layout_dimensions=layout_dimensions,
                                      ranges=ranges, subplot=True,
                                      uniform=self.uniform, layout_num=num,
                                      **plotopts)
            if issubclass(plot_type, GenericCompositePlot):
                adjoint_clone[pos] = subplots[pos].layout
            else:
                adjoint_clone[pos] = subplots[pos].map
        return subplots, adjoint_clone


    def initialize_plot(self, ranges=None):
        ranges = self.compute_ranges(self.layout, self.keys[-1], None)
        plots = [[] for i in range(self.rows)]
        passed_plots = []
        for r, c in self.coords:
            subplot = self.subplots.get((r, c), None)
            if subplot is not None:
                plots[r].append(subplot.initialize_plot(ranges=ranges,
                                                        plots=passed_plots))
                passed_plots.append(plots[r][-1])

        self.handles['plot'] = gridplot(plots)
        self.handles['plots'] = plots
        self.drawn = True

        return self.handles['plot']


    def update_frame(self, key, ranges=None):
        """
        Update the internal state of the Plot to represent the given
        key tuple (where integers represent frames). Returns this
        state.
        """
        ranges = self.compute_ranges(self.layout, key, ranges)
        plots = self.handles['plots']
        for r, c in self.coords:
            subplot = self.subplots.get((r, c), None)
            if subplot is not None:
                subplot.update_frame(key, ranges, plots[r][c])


class AdjointLayoutPlot(BokehPlot, GenericCompositePlot):

    layout_dict = {'Single': {'width_ratios': [4],
                              'height_ratios': [4],
                              'positions': ['main']}}

    def __init__(self, layout, layout_type, subplots, **params):
        # The AdjointLayout ViewableElement object
        self.layout = layout
        # Type may be set to 'Embedded Dual' by a call it grid_situate
        self.layout_type = layout_type
        self.view_positions = self.layout_dict[self.layout_type]['positions']

        # The supplied (axes, view) objects as indexed by position
        super(AdjointLayoutPlot, self).__init__(subplots=subplots, **params)


    def initialize_plot(self, ranges=None, plots=[]):
        """
        Plot all the views contained in the AdjointLayout Object using axes
        appropriate to the layout configuration. All the axes are
        supplied by LayoutPlot - the purpose of the call is to
        invoke subplots with correct options and styles and hide any
        empty axes as necessary.
        """
        plot = None
        for pos in ['main']:
            # Pos will be one of 'main', 'top' or 'right' or None
            subplot = self.subplots.get(pos, None)
            # If no view object or empty position, disable the axis
            if subplot:
                plot = subplot.initialize_plot(ranges=ranges, plots=plots)
        self.drawn = True
        return plot

    def update_frame(self, key, ranges=None, plot=None):
        plot = None
        for pos in ['main']:
            subplot = self.subplots.get(pos)
            if subplot is not None:
                plot = subplot.update_frame(key, ranges, plot)
        return plot
