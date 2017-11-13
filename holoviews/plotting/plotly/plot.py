import param
from plotly import tools

from ...core import (OrderedDict, NdLayout, AdjointLayout, Empty,
                     HoloMap, GridSpace, GridMatrix)
from ...element import Histogram
from ...core.options import Store
from ...core.util import wrap_tuple
from ..plot import DimensionedPlot, GenericLayoutPlot, GenericCompositePlot
from .util import add_figure

class PlotlyPlot(DimensionedPlot):

    backend = 'plotly'

    width = param.Integer(default=400)

    height = param.Integer(default=400)

    @property
    def state(self):
        """
        The plotting state that gets updated via the update method and
        used by the renderer to generate output.
        """
        return self.handles['fig']


    def initialize_plot(self, ranges=None):
        return self.generate_plot(self.keys[-1], ranges)


    def update_frame(self, key, ranges=None):
        return self.generate_plot(key, ranges)



class LayoutPlot(PlotlyPlot, GenericLayoutPlot):

    hspacing = param.Number(default=0.2, bounds=(0, 1))

    vspacing = param.Number(default=0.2, bounds=(0, 1))

    def __init__(self, layout, **params):
        super(LayoutPlot, self).__init__(layout, **params)
        self.layout, self.subplots, self.paths = self._init_layout(layout)

    def _get_size(self):
        rows, cols = self.layout.shape
        return cols*self.width*0.8, rows*self.height

    def _init_layout(self, layout):
        # Situate all the Layouts in the grid and compute the gridspec
        # indices for all the axes required by each LayoutPlot.
        layout_count = 0
        collapsed_layout = layout.clone(shared_data=False, id=layout.id)
        frame_ranges = self.compute_ranges(layout, None, None)
        frame_ranges = OrderedDict([(key, self.compute_ranges(layout, key, frame_ranges))
                                    for key in self.keys])
        layout_items = layout.grid_items()
        layout_dimensions = layout.kdims if isinstance(layout, NdLayout) else None
        layout_subplots, layouts, paths = {}, {}, {}
        for r, c in self.coords:
            # Get view at layout position and wrap in AdjointLayout
            key, view = layout_items.get((c, r) if self.transpose else (r, c), (None, None))
            view = view if isinstance(view, AdjointLayout) else AdjointLayout([view])
            layouts[(r, c)] = view
            paths[r, c] = key

            # Compute the layout type from shape
            layout_lens = {1:'Single', 2:'Dual', 3: 'Triple'}
            layout_type = layout_lens.get(len(view), 'Single')

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
        return collapsed_layout, layout_subplots, paths


    def _create_subplots(self, layout, positions, layout_dimensions, ranges, num=0):
        """
        Plot all the views contained in the AdjointLayout Object using axes
        appropriate to the layout configuration. All the axes are
        supplied by LayoutPlot - the purpose of the call is to
        invoke subplots with correct options and styles and hide any
        empty axes as necessary.
        """
        subplots = {}
        adjoint_clone = layout.clone(shared_data=False, id=layout.id)
        subplot_opts = dict(adjoined=layout)
        main_plot = None
        for pos in positions:
            # Pos will be one of 'main', 'top' or 'right' or None
            element = layout.get(pos, None)
            if element is None:
                continue

            # Options common for any subplot
            vtype = element.type if isinstance(element, HoloMap) else element.__class__
            plot_type = Store.registry[self.renderer.backend].get(vtype, None)
            plotopts = self.lookup_options(element, 'plot').options
            side_opts = {}
            if pos != 'main':
                plot_type = AdjointLayoutPlot.registry.get(vtype, plot_type)
                if pos == 'right':
                    side_opts = dict(height=main_plot.height, yaxis='right',
                                     invert_axes=True, width=120, labelled=['y'],
                                     xticks=2, show_title=False)
                else:
                    side_opts = dict(width=main_plot.width, xaxis='top',
                                     height=120, labelled=['x'], yticks=2,
                                     show_title=False)

            # Override the plotopts as required
            # Customize plotopts depending on position.
            plotopts = dict(side_opts, **plotopts)
            plotopts.update(subplot_opts)

            if plot_type is None:
                self.warning("Plotly plotting class for %s type not found, object will "
                             "not be rendered." % vtype.__name__)
                continue
            num = num if len(self.coords) > 1 else 0
            subplot = plot_type(element, keys=self.keys,
                                dimensions=self.dimensions,
                                layout_dimensions=layout_dimensions,
                                ranges=ranges, subplot=True,
                                uniform=self.uniform, layout_num=num,
                                **plotopts)
            subplots[pos] = subplot
            if isinstance(plot_type, type) and issubclass(plot_type, GenericCompositePlot):
                adjoint_clone[pos] = subplots[pos].layout
            else:
                adjoint_clone[pos] = subplots[pos].hmap
            if pos == 'main':
                main_plot = subplot

        return subplots, adjoint_clone


    def generate_plot(self, key, ranges=None):
        ranges = self.compute_ranges(self.layout, self.keys[-1], None)
        plots = [[] for i in range(self.rows)]
        insert_rows, insert_cols = [], []
        for r, c in self.coords:
            subplot = self.subplots.get((r, c), None)
            if subplot is not None:
                subplots = subplot.generate_plot(key, ranges=ranges)

                # Computes plotting offsets depending on
                # number of adjoined plots
                offset = sum(r >= ir for ir in insert_rows)
                if len(subplots) > 2:
                    # Add pad column in this position
                    insert_cols.append(c)
                    if r not in insert_rows:
                        # Insert and pad marginal row if none exists
                        plots.insert(r+offset, [None for _ in range(len(plots[r]))])
                        # Pad previous rows
                        for ir in range(r):
                            plots[ir].insert(c+1, None)
                        # Add to row offset
                        insert_rows.append(r)
                        offset += 1
                    # Add top marginal
                    plots[r+offset-1] += [subplots.pop(-1), None]
                elif len(subplots) > 1:
                    # Add pad column in this position
                    insert_cols.append(c)
                    # Pad previous rows
                    for ir in range(r):
                        plots[r].insert(c+1, None)
                    # Pad top marginal if one exists
                    if r in insert_rows:
                        plots[r+offset-1] += 2*[None]
                else:
                    # Pad top marginal if one exists
                    if r in insert_rows:
                        plots[r+offset-1] += [None] * (1+(c in insert_cols))
                plots[r+offset] += subplots
                if len(subplots) == 1 and c in insert_cols:
                    plots[r+offset].append(None)

        rows, cols = len(plots), len(plots[0])
        fig = tools.make_subplots(rows=rows, cols=cols, print_grid=False,
                                  horizontal_spacing=self.hspacing,
                                  vertical_spacing=self.vspacing)

        width, height = self._get_size()
        ax_idx = 0
        for r, row in enumerate(plots):
            for c, plot in enumerate(row):
                ax_idx += 1
                if plot:
                    add_figure(fig, plot, r, c, ax_idx)
        fig['layout'].update(height=height, width=width,
                             title=self._format_title(key))

        self.handles['fig'] = fig
        return self.handles['fig']



class AdjointLayoutPlot(PlotlyPlot):

    layout_dict = {'Single': {'positions': ['main']},
                   'Dual':   {'positions': ['main', 'right']},
                   'Triple': {'positions': ['main', 'right', 'top']}}

    registry = {}

    def __init__(self, layout, layout_type, subplots, **params):
        # The AdjointLayout ViewableElement object
        self.layout = layout
        # Type may be set to 'Embedded Dual' by a call it grid_situate
        self.layout_type = layout_type
        self.view_positions = self.layout_dict[self.layout_type]['positions']

        # The supplied (axes, view) objects as indexed by position
        super(AdjointLayoutPlot, self).__init__(subplots=subplots, **params)


    def initialize_plot(self, ranges=None):
        """
        Plot all the views contained in the AdjointLayout Object using axes
        appropriate to the layout configuration. All the axes are
        supplied by LayoutPlot - the purpose of the call is to
        invoke subplots with correct options and styles and hide any
        empty axes as necessary.
        """
        return self.generate_plot(self.keys[-1], ranges)


    def generate_plot(self, key, ranges=None):
        adjoined_plots = []
        for pos in ['main', 'right', 'top']:
            # Pos will be one of 'main', 'top' or 'right' or None
            subplot = self.subplots.get(pos, None)
            # If no view object or empty position, disable the axis
            if subplot:
                adjoined_plots.append(subplot.generate_plot(key, ranges=ranges))
        if not adjoined_plots: adjoined_plots = [None]
        return adjoined_plots



class GridPlot(PlotlyPlot, GenericCompositePlot):
    """
    Plot a group of elements in a grid layout based on a GridSpace element
    object.
    """

    hspacing = param.Number(default=0.05, bounds=(0, 1))

    vspacing = param.Number(default=0.05, bounds=(0, 1))

    def __init__(self, layout, ranges=None, layout_num=1, **params):
        if not isinstance(layout, GridSpace):
            raise Exception("GridPlot only accepts GridSpace.")
        super(GridPlot, self).__init__(layout=layout, layout_num=layout_num,
                                       ranges=ranges, **params)
        self.cols, self.rows = layout.shape
        self.subplots, self.layout = self._create_subplots(layout, ranges)

    def _create_subplots(self, layout, ranges):
        subplots = OrderedDict()
        frame_ranges = self.compute_ranges(layout, None, ranges)
        frame_ranges = OrderedDict([(key, self.compute_ranges(layout, key, frame_ranges))
                                    for key in self.keys])
        collapsed_layout = layout.clone(shared_data=False, id=layout.id)
        for i, coord in enumerate(layout.keys(full_grid=True)):
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
            if isinstance(layout, GridMatrix):
                if view.traverse(lambda x: x, [Histogram]):
                    kwargs['shared_axes'] = False

            # Create subplot
            plotting_class = Store.registry[self.renderer.backend].get(vtype, None)
            if plotting_class is None:
                if view is not None:
                    self.warning("Plotly plotting class for %s type not found, "
                                 "object will not be rendered." % vtype.__name__)
            else:
                subplot = plotting_class(view, dimensions=self.dimensions,
                                         show_title=False, subplot=True,
                                         ranges=frame_ranges, uniform=self.uniform,
                                         keys=self.keys, **dict(opts, **kwargs))
                collapsed_layout[coord] = (subplot.layout
                                           if isinstance(subplot, GenericCompositePlot)
                                           else subplot.hmap)
                subplots[coord] = subplot
        return subplots, collapsed_layout


    def generate_plot(self, key, ranges=None):
        ranges = self.compute_ranges(self.layout, self.keys[-1], None)
        plots = [[] for r in range(self.cols)]
        for i, coord in enumerate(self.layout.keys(full_grid=True)):
            r = i % self.cols
            subplot = self.subplots.get(wrap_tuple(coord), None)
            if subplot is not None:
                plot = subplot.initialize_plot(ranges=ranges)
                plots[r].append(plot)
            else:
                plots[r].append(None)

        rows, cols = len(plots), len(plots[0])
        fig = tools.make_subplots(rows=rows, cols=cols, print_grid=False,
                                  shared_xaxes=True, shared_yaxes=True,
                                  horizontal_spacing=self.hspacing,
                                  vertical_spacing=self.vspacing)
        ax_idx = 0
        for r, row in enumerate(plots):
            for c, plot in enumerate(row):
                ax_idx += 1
                if plot:
                    add_figure(fig, plot, r, c, ax_idx)
        w, h = self._get_size(subplot.width, subplot.height)
        fig['layout'].update(width=w, height=h,
                             title=self._format_title(key))

        self.handles['fig'] = fig
        return self.handles['fig']


    def _get_size(self, width, height):
        max_dim = max(self.layout.shape)
        # Reduce plot size as GridSpace gets larger
        shape_factor = 1. / max_dim
        # Expand small grids to a sensible viewing size
        expand_factor = 1 + (max_dim - 1) * 0.1
        scale_factor = expand_factor * shape_factor
        cols, rows = self.layout.shape
        return (scale_factor * cols * width,
                scale_factor * rows * height)
