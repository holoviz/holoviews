import numpy as np
from matplotlib import ticker

import param

from ..core.options import Store
from ..core.util import match_spec
from ..element.chart3d import Scatter3D, Surface
from .element import ElementPlot
from .chart import PointPlot


class Plot3D(ElementPlot):
    """
    Plot3D provides a common baseclass for mplot3d based
    plots.
    """

    azimuth = param.Integer(default=-60, bounds=(-90, 90), doc="""
        Azimuth angle in the x,y plane.""")

    elevation = param.Integer(default=30, bounds=(0, 180), doc="""
        Elevation angle in the z-axis.""")

    distance = param.Integer(default=10, bounds=(7, 15), doc="""
        Distance from the plotted object.""")

    projection = param.ObjectSelector(default='3d', doc="""
        The projection of the matplotlib axis.""")

    show_frame = param.Boolean(default=False, doc="""
        Whether to draw a frame around the figure.""")

    show_grid = param.Boolean(default=False, doc="""
        Whether to draw a grid in the figure.""")

    show_xaxis = param.ObjectSelector(default='fixed',
                                      objects=['fixed'], doc="""
        Whether and where to display the xaxis.""")

    show_yaxis = param.ObjectSelector(default='fixed',
                                      objects=['fixed'], doc="""
        Whether and where to display the yaxis.""")

    def _finalize_axis(self, key, zlabel=None, zticks=None, **kwargs):
        """
        Extends the ElementPlot _finalize_axis method to set appropriate
        labels, and axes options for 3D Plots.
        """
        axis = self.handles['axis']

        if self.zorder == 0 and axis is not None and key is not None:
            view = self.map.get(key, None) if hasattr(self, 'map') else None
            if view is not None:
                if hasattr(view, 'zlabel') and zlabel is None:
                    zlabel = view.zlabel
                axis.set_zlabel(zlabel)

            if zticks:
                axis.set_yticks(zticks[0])
                axis.set_yticklabels(zticks[1])
            else:
                axis.yaxis.set_major_locator(ticker.MaxNLocator(self.yticks))
        self.handles['fig'].set_frameon(False)
        axis.grid(True)
        axis.view_init(elev=self.elevation, azim=self.azimuth)
        axis.dist = self.distance
        axis.set_axis_bgcolor('white')
        return super(Plot3D, self)._finalize_axis(key, **kwargs)


    def get_extents(self, element, ranges):
        extents = super(Plot3D, self).get_extents(element, ranges)
        if len(extents) == 4:
            l, b, r, t = extents
            zmin, zmax = self.zlim if self.rescale_individually else self.map.zlim
        else:
            l, b, zmin, r, t, zmax = extents
        zdim = element.get_dimension(2).name
        if ranges is not None:
            zrange = ranges.get(zdim)
            if not xrange is None:
                zmin, zmax = (np.min([zrange[0], zmin]) if zmin else zrange[0],
                              np.max([zrange[1], zmax]) if zmax else zrange[1])
        return l, b, zmin, r, t, zmax


    def update_frame(self, *args, **kwargs):
        """
        If on the bottom Layer, clear plot before drawing each frame.
        """
        if not self.subplot or self.zorder == 0:
            self.handles['axis'].cla()

        super(Plot3D, self).update_frame(*args, **kwargs)



class Scatter3DPlot(Plot3D, PointPlot):
    """
    Subclass of PointPlot allowing plotting of Points
    on a 3D axis, also allows mapping color and size
    onto a particular Dimension of the data.
    """

    color_index = param.Integer(default=4, doc="""
      Index of the dimension from which the color will the drawn""")

    size_index = param.Integer(default=3, doc="""
      Index of the dimension from which the sizes will the drawn.""")

    def __call__(self, ranges=None):
        axis = self.handles['axis']
        points = self.map.last

        key = self.keys[-1]
        self.update_handles(axis, points, key)

        return self._finalize_axis(key)

    def update_handles(self, axis, points, key, ranges=None):
        ndims = points.data.shape[1]
        xs = points.data[:, 0] if len(points.data) else []
        ys = points.data[:, 1] if len(points.data) else []
        zs = points.data[:, 2] if len(points.data) else []
        sz = points.data[:, self.size_index] if self.size_index < ndims else None
        cs = points.data[:, self.color_index] if self.color_index < ndims else None

        style = Store.lookup_options(points, 'style')[self.cyclic_index]
        if sz is not None and self.scaling_factor > 1:
            style['s'] = self._compute_size(sz, style)
        if cs is not None:
            style['c'] = cs
            style.pop('color', None)
        scatterplot = axis.scatter(xs, ys, zs, zorder=self.zorder, **style)

        self.handles['axis'].add_collection(scatterplot)
        self.handles['scatter'] = scatterplot
        self.handles['legend_handle'] = scatterplot

        if cs is not None:
            val_dim = points.dimensions(label=True)[self.color_index]
            ranges = self.compute_ranges(self.map, key, ranges)
            ranges = match_spec(points, ranges)
            scatterplot.set_clim(ranges[val_dim])



class SurfacePlot(Plot3D):
    """
    Plots surfaces wireframes and contours in 3D space.
    Provides options to switch the display type via the
    plot_type parameter has support for a number of
    styling options including strides and colors.
    """


    plot_type = param.ObjectSelector(default='surface',
                                     objects=['surface', 'wireframe',
                                              'contour'], doc="""
        Specifies the type of visualization for the Surface object.""")

    style_opts = ['cmap', 'color', 'shade', 'facecolors',
                  'rstride', 'cstride']

    def __call__(self, ranges=None):
        view = self.map.last
        key = self.keys[-1]
        self.update_handles(self.handles['axis'], view, key)

        return self._finalize_axis(key)


    def update_handles(self, axis, element, key, ranges=None):
        mat = element.data
        rn, cn = mat.shape
        l, b, zmin, r, t, zmax = self.get_extents(element, ranges)
        r, c = np.mgrid[l:r:(r-l)/float(rn), b:t:(t-b)/float(cn)]

        style_opts = Store.lookup_options(element, 'style')[self.cyclic_index]
        style_opts['vmin'] = zmin
        style_opts['vmax'] = zmax
        if self.plot_type == "wireframe":
            self.handles['surface'] = self.handles['axis'].plot_wireframe(r, c, mat, **style_opts)
        elif self.plot_type == "surface":
            self.handles['surface'] = self.handles['axis'].plot_surface(r, c, mat, **style_opts)
        elif self.plot_type == "contour":
            self.handles['surface'] = self.handles['axis'].contour3D(r, c, mat, **style_opts)
        self.handles['legend_handle'] = self.handles['surface']


Store.registry.update({Surface: SurfacePlot,
                       Scatter3D: Scatter3DPlot})
