import numpy as np

import param

from bokeh.io import gridplot, vplot, hplot
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Panel, Tabs

from ...core import OrderedDict, CompositeOverlay, Element
from ...core import Store, Layout, AdjointLayout, NdLayout, Empty, GridSpace, HoloMap
from ...core.options import Compositor
from ...core import traversal
from ...core.util import basestring
from ..plot import Plot, GenericCompositePlot, GenericLayoutPlot
from .renderer import BokehRenderer
from .util import layout_padding

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

    def get_data(self, element, ranges=None):
        """
        Returns the data from an element in the appropriate format for
        initializing or updating a ColumnDataSource and a dictionary
        which maps the expected keywords arguments of a glyph to
        the column in the datasource.
        """
        raise NotImplementedError


    def _init_datasource(self, data):
        """
        Initializes a data source to be passed into the bokeh glyph.
        """
        return ColumnDataSource(data=data)


    def _update_datasource(self, source, data):
        """
        Update datasource with data for a new frame.
        """
        for k, v in data.items():
            source.data[k] = v

    @property
    def state(self):
        """
        The plotting state that gets updated via the update method and
        used by the renderer to generate output.
        """
        return self.handles['plot']


    @property
    def current_handles(self):
        """
        Should return a list of plot objects that have changed and
        should be updated.
        """
        return []


    def _fontsize(self, key, label='fontsize', common=True):
        """
        Converts integer fontsizes to a string specifying
        fontsize in pt.
        """
        size = super(BokehPlot, self)._fontsize(key, label, common)
        return {k: v if isinstance(v, basestring) else '%spt' % v
                for k, v in size.items()}



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
                                           else subplot.hmap)
                subplots[coord] = subplot
        return subplots, collapsed_layout


    def initialize_plot(self, ranges=None, plots=[]):
        ranges = self.compute_ranges(self.layout, self.keys[-1], None)
        plots = [[] for r in range(self.cols)]
        passed_plots = list(plots)
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
                subplot.update_frame(key, ranges)



class LayoutPlot(BokehPlot, GenericLayoutPlot):

    shared_axes = param.Boolean(default=True, doc="""
        Whether axes should be shared across plots""")

    tabs = param.Boolean(default=False, doc="""
        Whether to display overlaid plots in separate panes""")

    def __init__(self, layout, **params):
        super(LayoutPlot, self).__init__(layout, **params)
        self.layout, self.subplots, self.paths = self._init_layout(layout)


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
        layout_subplots, layouts, paths = {}, {}, {}
        inserts_cols = []
        for r, c in self.coords:
            # Get view at layout position and wrap in AdjointLayout
            key, view = layout_items.get((r, c), (None, None))
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
        projections = []
        adjoint_clone = layout.clone(shared_data=False, id=layout.id)
        subplot_opts = dict(adjoined=layout)
        main, main_plot = None, None
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
                                     invert_axes=True, width=120, show_labels=['y'],
                                     xticks=2, show_title=False)
                else:
                    side_opts = dict(width=main_plot.width, xaxis='top',
                                     height=120, show_labels=['x'], yticks=2,
                                     show_title=False)

            # Override the plotopts as required
            # Customize plotopts depending on position.
            plotopts = dict(side_opts, **plotopts)
            plotopts.update(subplot_opts)

            if plot_type is None:
                self.warning("Bokeh plotting class for %s type not found, object will "
                             "not be rendered." % vtype.__name__)
                continue
            if plot_type in [GridPlot, LayoutPlot]:
                self.tabs = True
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
                main = element
                main_plot = subplot

        return subplots, adjoint_clone


    def initialize_plot(self, ranges=None):
        ranges = self.compute_ranges(self.layout, self.keys[-1], None)
        plots = [[] for i in range(self.rows)]
        passed_plots = []
        tab_titles = {}
        insert_rows, insert_cols = [], []
        adjoined = False
        for r, c in self.coords:
            subplot = self.subplots.get((r, c), None)
            if subplot is not None:
                shared_plots = passed_plots if self.shared_axes else None
                subplots = subplot.initialize_plot(ranges=ranges, plots=shared_plots)

                # Computes plotting offsets depending on
                # number of adjoined plots
                offset = sum(r >= ir for ir in insert_rows)
                if len(subplots) > 2:
                    adjoined = True
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
                    adjoined = True
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
                passed_plots.append(subplots[0])

            if self.tabs:
                if isinstance(self.layout, Layout):
                    tab_titles[r, c] = ' '.join(self.paths[r,c])
                else:
                    dim_vals = zip(self.layout.kdims, self.paths[r, c])
                    tab_titles[r, c] = ', '.join([d.pprint_value_string(k)
                                                  for d, k in dim_vals])

        # Replace None types with empty plots
        # to avoid bokeh bug
        if adjoined:
            plots = layout_padding(plots)

        # Determine the most appropriate composite plot type
        # If the object cannot be displayed in a single layout
        # it will be split into Tabs, for 1-row or 1-column
        # Layouts we use the vplot and hplots.
        # If there is a table and multiple rows and columns
        # everything will be forced to a vertical layout
        if self.tabs:
            panels = [Panel(child=child, title=tab_titles.get(r, c))
                      for r, row in enumerate(plots)
                      for c, child in enumerate(row)
                      if child is not None]
            layout_plot = Tabs(tabs=panels)
        elif len(plots) == 1 and not adjoined:
            layout_plot = vplot(hplot(*plots[0]))
        elif len(plots[0]) == 1:
            layout_plot = vplot(*[p[0] for p in plots])
        else:
            layout_plot = gridplot(plots)

        self.handles['plot'] = layout_plot
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
                subplot.update_frame(key, ranges)


class AdjointLayoutPlot(BokehPlot, GenericCompositePlot):

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


    def initialize_plot(self, ranges=None, plots=[]):
        """
        Plot all the views contained in the AdjointLayout Object using axes
        appropriate to the layout configuration. All the axes are
        supplied by LayoutPlot - the purpose of the call is to
        invoke subplots with correct options and styles and hide any
        empty axes as necessary.
        """
        adjoined_plots = []
        for pos in ['main', 'right', 'top']:
            # Pos will be one of 'main', 'top' or 'right' or None
            subplot = self.subplots.get(pos, None)
            # If no view object or empty position, disable the axis
            if subplot:
                adjoined_plots.append(subplot.initialize_plot(ranges=ranges, plots=plots))
        self.drawn = True
        if not adjoined_plots: adjoined_plots = [None]
        return adjoined_plots


    def update_frame(self, key, ranges=None):
        plot = None
        for pos in ['main', 'right', 'top']:
            subplot = self.subplots.get(pos)
            if subplot is not None:
                plot = subplot.update_frame(key, ranges)
        return plot
