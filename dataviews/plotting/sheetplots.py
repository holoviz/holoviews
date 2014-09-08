import copy
from itertools import groupby

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

import param

from .. import SheetStack, Points, View, SheetView, SheetOverlay, \
    CoordinateGrid, NdMapping, Contours, VectorField
from .dataplots import MatrixPlot
from .viewplots import OverlayPlot, Plot



class PointPlot(Plot):
    """
    Note that the 'cmap', 'vmin' and 'vmax' style arguments control
    how point magnitudes are rendered to different colors.
    """

    normalize_individually = param.Boolean(default=False, doc="""
      Whether to normalize the colors used to represent magnitude for
      each frame or across the stack (when color is applicable).""")

    style_opts = param.List(default=['alpha', 'color', 'edgecolors', 'facecolors',
                                     'linewidth', 'marker', 's', 'visible',
                                     'cmap', 'vmin', 'vmax'],
                            constant=True, doc="""
     The style options for PointPlot match those of matplotlib's
     scatter plot command.""")

    _stack_type = SheetStack
    _view_type = Points

    def __call__(self, axis=None, cyclic_index=0, lbrt=None):
        points = self._stack.last
        title = None if self.zorder > 0 else self._format_title(-1)
        ax = self._axis(axis, title, 'x', 'y', self._stack.bounds.lbrt())

        xs = points.data[:, 0] if len(points.data) else []
        ys = points.data[:, 1] if len(points.data) else []
        cs = points.data[:, 2] if points.data.shape[1]>=3 else None

        kwargs = View.options.style(points)[cyclic_index]
        scatterplot = ax.scatter(xs, ys, zorder=self.zorder,
                                 **({k:v for k,v in dict(kwargs, c=cs).items() if k!='color'}
                                    if cs is not None else kwargs))

        ax.add_collection(scatterplot)
        self.handles['scatter'] = scatterplot

        if cs is not None:
            clims = points.range if self.normalize_individually else self._stack.range
            scatterplot.set_clim(clims)

        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n if n < len(self) else len(self) - 1
        points = list(self._stack.values())[n]

        scatter = self.handles['scatter']
        scatter.set_offsets(points.data[:,0:2])
        if points.data.shape[1]==3:
            scatter.set_array(points.data[:,2])

        if self.normalize_individually:
            scatter.set_clim(points.range)
        self._update_title(n)
        plt.draw()


class VectorFieldPlot(Plot):
    """
    Renders vector fields in sheet coordinates. The vectors are
    expressed in polar coordinates and may be displayed according to
    angle alone (with some common, arbitrary arrow length) or may be
    true polar vectors.

    Optionally, the arrows may be colored but this dimension is
    redundant with either the specified angle or magnitudes. This
    choice is made by setting the color_dim parameter.

    Note that the 'cmap' style argument controls the color map used to
    color the arrows. The length of the arrows is controlled by the
    'scale' style option where a value of 1.0 is such that the largest
    arrow shown is no bigger than the smallest sampling distance.
    """

    style_opts = param.List(default=['alpha', 'color', 'edgecolors',
                                     'facecolors', 'linewidth',
                                     'marker', 's', 'visible', 'cmap',
                                     'scale', 'headlength',
                                     'headaxislength', 'pivot'], constant=True, doc="""
       The style options for PointPlot matching those of matplotlib's
       quiver plot command.""")


    color_dim = param.ObjectSelector(default=None,
                                     objects=['angle', 'magnitude', None], doc="""
       Which of the polar vector components is mapped to the color
       dimension (if any)""")

    normalize_individually = param.Boolean(default=False, doc="""
        Whether to normalize the colors used as an extra dimension
        per frame or across the stack (when color is applicable).""")

    arrow_heads = param.Boolean(default=True, doc="""
       Whether or not to draw arrow heads. If arrowheads are enabled,
       they may be customized with the 'headlength' and
       'headaxislength' style options.""")

    normalize_lengths = param.Boolean(default=True, doc="""
       Whether to normalize vector magnitudes automatically. If False,
       it will be assumed that the lengths have already been correctly
       normalized.""")

    _stack_type = SheetStack
    _view_type = VectorField

    def __init__(self, *args, **kwargs):
        super(VectorFieldPlot, self).__init__(*args, **kwargs)
        self._min_dist, self._max_magnitude = self._get_stack_info(self._stack)


    def _get_stack_info(self, stack):
        """
        Get the minimum sample distance and maximum magnitude
        """
        if self.normalize_individually:
            return None, None
        dists, magnitudes  = [], []
        for vfield in stack:
            dists.append(self._get_min_dist(vfield))

            if vfield.data.shape[1]>=4:
                magnitudes.append(max(vfield.data[:, 3]))
        return min(dists), max(magnitudes) if magnitudes else None


    def _get_info(self, vfield, input_scale):
        xs = vfield.data[:, 0] if len(vfield.data) else []
        ys = vfield.data[:, 1] if len(vfield.data) else []
        radians = vfield.data[:, 2] if len(vfield.data) else []
        magnitudes = vfield.data[:, 3] if vfield.data.shape[1]>=4 else np.array([1.0] * len(xs))
        colors = magnitudes if self.color_dim == 'magnitude' else radians

        max_magnitude = self._max_magnitude if self._max_magnitude else max(magnitudes)
        min_dist =      self._min_dist if self._min_dist else self._get_min_dist(vfield)

        if self.normalize_lengths:
            magnitudes =  magnitudes / max_magnitude

        return (xs, ys, list((radians / np.pi) * 180),
                magnitudes, colors, input_scale / min_dist)


    def _get_min_dist(self, vfield):
        "Get the minimum sampling distance."
        xys = np.array([complex(x,y) for x,y in zip(vfield.data[:,0],
                                                    vfield.data[:,1])])
        m, n = np.meshgrid(xys, xys)
        distances = abs(m-n)
        np.fill_diagonal(distances, np.inf)
        return  distances.min()


    def __call__(self, axis=None, cyclic_index=0, lbrt=None):
        vfield = self._stack.last
        title = None if self.zorder > 0 else self._format_title(-1)
        ax = self._axis(axis, title, 'x', 'y', self._stack.bounds.lbrt())

        colorized = self.color_dim is not None
        kwargs = View.options.style(vfield)[cyclic_index]
        input_scale = kwargs.pop('scale', 1.0)
        xs, ys, angles, lens, colors, scale = self._get_info(vfield, input_scale)

        args = (xs, ys, lens,  [0.0] * len(vfield.data))
        args = args + (colors,) if colorized else args

        if not self.arrow_heads:
            kwargs['headlength'] = kwargs['headaxislength'] = 0


        quiver = ax.quiver(*args, zorder=self.zorder,
                           units='x', scale_units='x',
                           scale = scale,
                           angles = angles ,
                           **({k:v for k,v in kwargs.items() if k!='color'}
                              if colorized else kwargs))

        if self.color_dim == 'angle':
            clims = vfield.value.range
            quiver.set_clim(clims)
        elif self.color_dim == 'magnitude':
            clims = vfield.range if self.normalize_individually else self._stack.range
            quiver.set_clim(clims)

        ax.add_collection(quiver)
        self.handles['quiver'] = quiver
        self.handles['input_scale'] = input_scale
        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n if n < len(self) else len(self) - 1
        vfield = list(self._stack.values())[n]
        self.handles['quiver'].set_offsets(vfield.data[:,0:2])
        input_scale = self.handles['input_scale']

        xs, ys, angles, lens, colors, scale = self._get_info(vfield, input_scale)

        # Set magnitudes, angles and colors if supplied.
        quiver = self.handles['quiver']
        quiver.U = lens
        quiver.angles = angles
        if self.color_dim is not None:
            quiver.set_array(colors)

        if self.normalize_individually and self.color_dim == 'magnitude':
            quiver.set_clim(vfield.range)

        self._update_title(n)
        plt.draw()



class ContourPlot(Plot):

    style_opts = param.List(default=['alpha', 'color', 'linestyle',
                                     'linewidth', 'visible'],
                            constant=True, doc="""
        The style options for ContourPlot match those of matplotlib's
        LineCollection class.""")

    _stack_type = SheetStack
    _view_type = Contours

    def __init__(self, *args, **kwargs):
        self.aspect = 'equal'
        super(ContourPlot, self).__init__(*args, **kwargs)


    def __call__(self, axis=None, cyclic_index=0, lbrt=None):
        lines = self._stack.last
        title = None if self.zorder > 0 else self._format_title(-1)
        ax = self._axis(axis, title, 'x', 'y', self._stack.bounds.lbrt())
        line_segments = LineCollection(lines.data, zorder=self.zorder,
                                       **View.options.style(lines)[cyclic_index])
        self.handles['line_segments'] = line_segments
        ax.add_collection(line_segments)
        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        contours = list(self._stack.values())[n]
        self.handles['line_segments'].set_paths(contours.data)
        self._update_title(n)
        plt.draw()



class SheetViewPlot(MatrixPlot):

    _stack_type = SheetStack
    _view_type = SheetView



class SheetPlot(OverlayPlot):
    """
    A generic plot that visualizes SheetOverlays which themselves may
    contain SheetLayers of type SheetView, Points or Contour objects.
    """


    style_opts = param.List(default=[], constant=True, doc="""
     SheetPlot renders overlay layers which individually have style
     options but SheetPlot itself does not.""")

    _stack_type = SheetStack
    _view_type = SheetOverlay


    def _check_stack(self, view):
        stack = super(SheetPlot, self)._check_stack(view)
        return self._collapse_channels(stack)


    def __call__(self, axis=None, lbrt=None):
        ax = self._axis(axis, None, 'x','y', self._stack.bounds.lbrt())
        stacks = self._stack.split_overlays()

        sorted_stacks = sorted(stacks, key=lambda x: x.style)
        style_groups = dict((k, enumerate(list(v))) for k,v
                            in groupby(sorted_stacks, lambda s: s.style))

        for zorder, stack in enumerate(stacks):
            cyclic_index, _ = next(style_groups[stack.style])
            plotype = Plot.defaults[stack.type]
            plot = plotype(stack, zorder=zorder, **View.options.plotting(stack).opts)

            plot(ax, cyclic_index=cyclic_index)
            self.plots.append(plot)

        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        for plot in self.plots:
            plot.update_frame(n)


class CoordinateGridPlot(OverlayPlot):
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

    num_ticks = param.Number(default=5)

    show_frame = param.Boolean(default=False)

    style_opts = param.List(default=['alpha', 'cmap', 'interpolation',
                                     'visible', 'filterrad', 'origin'],
                            constant=True, doc="""
       The style options for CoordinateGridPlot match those of
       matplotlib's imshow command.""")


    def __init__(self, grid, **kwargs):
        self.layout = kwargs.pop('layout', None)
        if not isinstance(grid, CoordinateGrid):
            raise Exception("CoordinateGridPlot only accepts ProjectionGrids.")
        self.grid = copy.deepcopy(grid)
        for k, stack in self.grid.items():
            self.grid[k] = self._collapse_channels(self.grid[k])
        Plot.__init__(self, **kwargs)


    def __call__(self, axis=None):
        grid_shape = [[v for (k, v) in col[1]]
                      for col in groupby(self.grid.items(), lambda item: item[0][0])]
        width, height, b_w, b_h = self._compute_borders(grid_shape)
        xticks, yticks = self._compute_ticks(width, height)

        ax = self._axis(axis, self._format_title(-1), xticks=xticks,
                        yticks=yticks, lbrt=(0, 0, width, height))

        self.handles['projs'] = []
        x, y = b_w, b_h
        for row in grid_shape:
            for view in row:
                w, h = self._get_dims(view)
                if view.type == SheetOverlay:
                    data = view.last[-1].data if self.situate else view.last[-1].roi.data
                    opts = View.options.style(view.last[-1]).opts
                else:
                    data = view.last.data if self.situate else view.last.roi.data
                    opts = View.options.style(view).opts

                self.handles['projs'].append(ax.imshow(data, extent=(x,x+w, y, y+h), **opts))
                y += h + b_h
            y = b_h
            x += w + b_w

        if not axis: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        for i, plot in enumerate(self.handles['projs']):
            key, view = list(self.grid.values())[i].items()[n]
            if isinstance(view, SheetOverlay):
                data = view[-1].data if self.situate else view[-1].roi.data
            else:
                data = view.data if self.situate else view.roi.data
            plot.set_data(data)
        self._update_title(n)
        plt.draw()


    def _format_title(self, n):
        stack = self.grid.values()[0]
        key, _ = stack.items()[n]
        title_format = stack.get_title(key if isinstance(key, tuple) else (key,), self.grid)
        if title_format is None:
            return None
        return title_format.format(label=self.grid.label, type=self.grid.__class__.__name__)


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


    def _compute_ticks(self, width, height):
        l, b, r, t = self.grid.lbrt

        xpositions = np.linspace(0, width, self.num_ticks)
        xlabels = np.linspace(l, r, self.num_ticks).round(3)
        ypositions = np.linspace(0, height, self.num_ticks)
        ylabels = np.linspace(b, t, self.num_ticks).round(3)
        return (xpositions, xlabels), (ypositions, ylabels)


    def __len__(self):
        return max([len(v) for v in self.grid if isinstance(v, NdMapping)]+[1])


Plot.defaults.update({SheetView: SheetViewPlot,
                      Points: PointPlot,
                      Contours: ContourPlot,
                      SheetOverlay: SheetPlot,
                      CoordinateGrid: CoordinateGridPlot,
                      VectorField:VectorFieldPlot})

Plot.sideplots.update({CoordinateGrid: CoordinateGridPlot})
