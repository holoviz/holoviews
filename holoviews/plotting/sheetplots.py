import copy
from itertools import groupby, product

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

import param

from ..core import NdMapping, View, Layer, Overlay
from ..views import Matrix, HeatMap, Points, SheetMatrix, Contours, VectorField
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

    _view_type = Points

    def __call__(self, axis=None, cyclic_index=0, lbrt=None):
        points = self._stack.last

        self.ax = self._init_axis(axis)

        xs = points.data[:, 0] if len(points.data) else []
        ys = points.data[:, 1] if len(points.data) else []
        cs = points.data[:, 2] if points.data.shape[1]>=3 else None

        kwargs = View.options.style(points)[cyclic_index]
        scatterplot = self.ax.scatter(xs, ys, zorder=self.zorder,
                                      **({k:v for k,v in dict(kwargs, c=cs).items() if k!='color'}
                                      if cs is not None else kwargs))

        self.ax.add_collection(scatterplot)
        self.handles['scatter'] = scatterplot

        if cs is not None:
            clims = points.range if self.normalize_individually else self._stack.range
            scatterplot.set_clim(clims)

        return self._finalize_axis(self._keys[-1])


    def update_handles(self, view, key, lbrt=None):
        scatter = self.handles['scatter']
        scatter.set_offsets(view.data[:,0:2])
        if view.data.shape[1]==3:
            scatter.set_array(view.data[:,2])

        if self.normalize_individually:
            scatter.set_clim(view.range)


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

        if self.normalize_lengths and max_magnitude != 0:
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
        self.ax = self._init_axis(axis)

        colorized = self.color_dim is not None
        kwargs = View.options.style(vfield)[cyclic_index]
        input_scale = kwargs.pop('scale', 1.0)
        xs, ys, angles, lens, colors, scale = self._get_info(vfield, input_scale)

        args = (xs, ys, lens,  [0.0] * len(vfield.data))
        args = args + (colors,) if colorized else args

        if not self.arrow_heads:
            kwargs['headlength'] = kwargs['headaxislength'] = 0

        if 'pivot' not in kwargs: kwargs['pivot'] = 'mid'

        quiver = self.ax.quiver(*args, zorder=self.zorder,
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

        self.ax.add_collection(quiver)
        self.handles['quiver'] = quiver
        self.handles['input_scale'] = input_scale

        return self._finalize_axis(self._keys[-1], lbrt=lbrt)



    def update_handles(self, view, key, lbrt=None):
        self.handles['quiver'].set_offsets(view.data[:,0:2])
        input_scale = self.handles['input_scale']

        xs, ys, angles, lens, colors, scale = self._get_info(view, input_scale)

        # Set magnitudes, angles and colors if supplied.
        quiver = self.handles['quiver']
        quiver.U = lens
        quiver.angles = angles
        if self.color_dim is not None:
            quiver.set_array(colors)

        if self.normalize_individually and self.color_dim == 'magnitude':
            quiver.set_clim(view.range)


class ContourPlot(Plot):

    style_opts = param.List(default=['alpha', 'color', 'linestyle',
                                     'linewidth', 'visible'],
                            constant=True, doc="""
        The style options for ContourPlot match those of matplotlib's
        LineCollection class.""")

    _view_type = Contours

    def __init__(self, *args, **kwargs):
        self.aspect = 'equal'
        super(ContourPlot, self).__init__(*args, **kwargs)


    def __call__(self, axis=None, cyclic_index=0, lbrt=None):
        lines = self._stack.last
        self.ax = self._init_axis(axis)

        line_segments = LineCollection(lines.data, zorder=self.zorder,
                                       **View.options.style(lines)[cyclic_index])
        self.handles['line_segments'] = line_segments
        self.ax.add_collection(line_segments)

        return self._finalize_axis(self._keys[-1], lbrt=lbrt)


    def update_handles(self, view, key, lbrt=None):
        self.handles['line_segments'].set_paths(view.data)


class MatrixPlot(Plot):

    normalize_individually = param.Boolean(default=False)

    show_values = param.Boolean(default=True, doc="""
        Whether to annotate the values when displaying a HeatMap.""")

    style_opts = param.List(default=['alpha', 'cmap', 'interpolation',
                                     'visible', 'filterrad', 'origin', 'clims'],
                            constant=True, doc="""
        The style options for MatrixPlot are a subset of those used
        by matplotlib's imshow command. If supplied, the clim option
        will be ignored as it is computed from the input View.""")

    _view_type = (Matrix, Layer)

    def __call__(self, axis=None, cyclic_index=0, lbrt=None):

        self.ax = self._init_axis(axis)
        view = self._stack.last

        (l, b, r, t) = (0, 0, 1, 1) if isinstance(view, HeatMap)\
            else self._stack.last.lbrt
        xticks, yticks = self._compute_ticks(view)

        opts = View.options.style(view)[cyclic_index]
        data = view.data
        clims = opts.pop('clims', None)
        if view.depth != 1:
            opts.pop('cmap', None)
        elif isinstance(view, HeatMap):
            data = view.data
            data = np.ma.array(data, mask=np.isnan(data))
            cmap_name = opts.pop('cmap', None)
            cmap = copy.copy(plt.cm.get_cmap('gray' if cmap_name is None else cmap_name))
            cmap.set_bad('w', 1.)
            opts['cmap'] = cmap

        im = self.ax.imshow(data, extent=[l, r, b, t], zorder=self.zorder, **opts)
        if clims is None:
            clims = view.range if self.normalize_individually else self._stack.range
        im.set_clim(clims)
        self.handles['im'] = im

        if isinstance(view, HeatMap):
            self.ax.set_aspect(float(r - l)/(t-b))
            self.handles['annotations'] = {}
            if self.show_values:
                self._annotate_values(view)

        return self._finalize_axis(self._keys[-1], lbrt=(l, b, r, t),
                                   xticks=xticks, yticks=yticks)


    def _compute_ticks(self, view):
        if isinstance(view, HeatMap):
            dim1_keys, dim2_keys = view.dense_keys()
            num_x, num_y = len(dim1_keys), len(dim2_keys)
            xstep, ystep = 1.0/num_x, 1.0/num_y
            xpos = np.linspace(xstep/2., 1.0-xstep/2., num_x)
            ypos = np.linspace(ystep/2., 1.0-ystep/2., num_y)
            return (xpos, dim1_keys), (ypos, dim2_keys)
        else:
            return None, None


    def _annotate_values(self, view):
        dim1_keys, dim2_keys = view.dense_keys()
        num_x, num_y = len(dim1_keys), len(dim2_keys)
        xstep, ystep = 1.0/num_x, 1.0/num_y
        xpos = np.linspace(xstep/2., 1.0-xstep/2., num_x)
        ypos = np.linspace(ystep/2., 1.0-ystep/2., num_y)
        coords = product(dim1_keys, dim2_keys)
        plot_coords = product(xpos, ypos)
        for plot_coord, coord in zip(plot_coords, coords):
            text = round(view._data.get(coord, np.NaN), 3)
            if plot_coord not in self.handles['annotations']:
                annotation = self.ax.annotate(text, xy=plot_coord,
                                              xycoords='axes fraction',
                                              horizontalalignment='center',
                                              verticalalignment='center')
                self.handles['annotations'][plot_coord] = annotation
            else:
                self.handles['annotations'][plot_coord].set_text(text)
        old_coords = set(self.handles['annotations'].keys()) - set(product(xpos, ypos))
        for plot_coord in old_coords:
            annotation = self.handles['annotations'].pop(plot_coord)
            annotation.remove()



    def update_handles(self, view, key, lbrt=None):
        im = self.handles.get('im', None)
        im.set_data(view.data)

        if isinstance(view, HeatMap) and self.show_values:
           self._annotate_values(view)

        if self.normalize_individually:
            im.set_clim(view.range)



class MatrixGridPlot(OverlayPlot):
    """
    MatrixGridPlot evenly spaces out plots of individual projections on
    a grid, even when they differ in size. Since this class uses a single
    axis to generate all the individual plots it is much faster than the
    equivalent using subplots.
    """

    border = param.Number(default=10, doc="""
        Aggregate border as a fraction of total plot size.""")

    num_ticks = param.Number(default=5)

    show_frame = param.Boolean(default=False)

    style_opts = param.List(default=['alpha', 'cmap', 'interpolation',
                                     'visible', 'filterrad', 'origin'],
                            constant=True, doc="""
       The style options for MatrixGridPlot match those of
       matplotlib's imshow command.""")


    def __init__(self, grid, **kwargs):
        self.layout = kwargs.pop('layout', None)
        self.grid = copy.deepcopy(grid)
        for k, stack in self.grid.items():
            self.grid[k] = self._check_stack(self.grid[k])
        Plot.__init__(self, **kwargs)
        self._keys = self.grid.all_keys


    def __call__(self, axis=None):
        grid_shape = [[v for (k, v) in col[1]]
                      for col in groupby(self.grid.items(), lambda item: item[0][0])]
        width, height, b_w, b_h = self._compute_borders(grid_shape)
        xticks, yticks = self._compute_ticks(width, height)

        self.ax = self._init_axis(axis)

        self.handles['projs'] = []
        x, y = b_w, b_h
        for row in grid_shape:
            for view in row:
                w, h = self._get_dims(view)
                if view.type == Overlay:
                    data = view.last.last.data
                    opts = View.options.style(view.last.last).opts
                else:
                    data = view.last.data
                    opts = View.options.style(view).opts

                plot = self.ax.imshow(data, extent=(x,x+w, y, y+h), **opts)
                self.handles['projs'].append(plot)
                y += h + b_h
            y = b_h
            x += w + b_w

        key = self._keys[-1]
        return self._finalize_axis(key, lbrt=(0, 0, width, height),
                                   title=self._format_title(key),
                                   xticks=xticks, yticks=yticks)


    def update_frame(self, n):
        n = n  if n < len(self) else len(self) - 1
        key = self._keys[n]
        for i, plot in enumerate(self.handles['projs']):
            view = self.grid.values()[i][key]
            data = view[-1].data if isinstance(view, Overlay) else view.data
            plot.set_data(data)

        grid_shape = [[v for (k, v) in col[1]]
                      for col in groupby(self.grid.items(), lambda item: item[0][0])]
        width, height, b_w, b_h = self._compute_borders(grid_shape)

        self._finalize_axis(key, lbrt=(0, 0, width, height))


    def _format_title(self, key):
        stack = self.grid.values()[0]
        view = stack.get(key, None)
        if view is None: return ""
        title_format = stack.get_title(key if isinstance(key, tuple) else (key,), self.grid)
        if title_format is None:
            return None
        return title_format.format(label=self.grid.label, type=self.grid.__class__.__name__)


    def _get_dims(self, view):
        l, b, r, t = view.lbrt
        return (r - l, t - b)


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
        l, b, r, t = self.grid.grid_lbrt

        xpositions = np.linspace(0, width, self.num_ticks)
        xlabels = np.linspace(l, r, self.num_ticks).round(3)
        ypositions = np.linspace(0, height, self.num_ticks)
        ylabels = np.linspace(b, t, self.num_ticks).round(3)
        return (xpositions, xlabels), (ypositions, ylabels)


    def __len__(self):
        return max([len(v) for v in self.grid if isinstance(v, NdMapping)]+[1])


Plot.defaults.update({Matrix: MatrixPlot,
                      HeatMap: MatrixPlot,
                      SheetMatrix: MatrixPlot,
                      Points: PointPlot,
                      Contours: ContourPlot,
                      VectorField: VectorFieldPlot})

