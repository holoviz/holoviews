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
    how point magnitudes are rendered.
    """

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
        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n if n < len(self) else len(self) - 1
        points = list(self._stack.values())[n]
        self.handles['scatter'].set_offsets(points.data[:,0:2])
        if points.data.shape[1]==3:
            self.handles['scatter'].set_array(points.data[:,2])
        self._update_title(n)
        plt.draw()


class VectorFieldPlot(Plot):
    """
    Renders 2D, 3D or 4D vector field. The 3rd and 4th dimension may
    be either the arrow colour or the relative arrow length. For 2D
    vector fields, all arrows have a relative length of 1.0.

    Note that the 'cmap', 'clim' style arguments control the colours
    of the vectors for 3D vector fields.
    """

    style_opts = param.List(default=['alpha', 'color', 'edgecolors', 'facecolors',
                                     'linewidth', 'marker', 's', 'visible',
                                     'cmap', 'clim', 'scale'],
                            constant=True, doc="""
     The style options for PointPlot match those of matplotlib's
     scatter plot command.""")

    extra_dimensions = param.List(default=['color', 'relative length'], doc="""
       How to represent the components that come after the angle
       component (in radians). Currently the components can be
       represented using relative arrow length or as with colour (via
       a colour map).

       For instance, the default list may be swapped so that colour is
       used before relative length for 3D and 4D VectorFields.""")

    _stack_type = SheetStack
    _view_type = VectorField


    def _get_info(self, vfield):
        xs = vfield.data[:, 0] if len(vfield.data) else []
        ys = vfield.data[:, 1] if len(vfield.data) else []
        radians = vfield.data[:, 2] if len(vfield.data) else []
        norm1 = vfield.data[:, 3] if vfield.data.shape[1]>=4 else None
        norm2 = vfield.data[:, 4] if vfield.data.shape[1]>=5 else None

        extra_dims = dict(zip(self.extra_dimensions,
                              [el for el in [norm1, norm2] if el is not None]))
        magnitudes = extra_dims.get('relative length', [1.0] * len(xs))

        return dict(extra_dims, xs=xs, ys=ys,
                    angles=list((radians / np.pi) * 180),
                    magnitudes=magnitudes)


    def __call__(self, axis=None, cyclic_index=0, lbrt=None):
        vfield = self._stack.last
        title = None if self.zorder > 0 else self._format_title(-1)
        ax = self._axis(axis, title, 'x', 'y', self._stack.bounds.lbrt())

        info = self._get_info(vfield)
        colorized = 'color' in info

        args = (info['xs'], info['ys'], info['magnitudes'],  [0.0] * len(vfield.data))
        args = args + (info['color'],) if colorized else args
        kwargs = View.options.style(vfield)[cyclic_index]

        quiver = ax.quiver(*args, zorder=self.zorder,
                            units='inches',
                            scale_units='inches',
                            angles= info['angles'] ,
                            **({k:v for k,v in kwargs.items() if k!='color'}
                               if colorized else kwargs))

        ax.add_collection(quiver)
        self.handles['quiver'] = quiver
        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n if n < len(self) else len(self) - 1
        vfield = list(self._stack.values())[n]
        self.handles['quiver'].set_offsets(vfield.data[:,0:2])

        info = self._get_info(vfield)

        # Set magnitudes, angles and colors if supplied.
        if 'magnitudes' in info:
            self.handles['quiver'].U = info['magnitudes']
        if 'angles' in info:
            self.handles['quiver'].angles = info['angles']
        if 'color' in info:
            self.handles['quiver'].set_array(info['color'])
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
