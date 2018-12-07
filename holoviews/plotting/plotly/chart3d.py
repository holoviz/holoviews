from __future__ import absolute_import, division, unicode_literals

import param
import numpy as np

from matplotlib.cm import get_cmap
from plotly import colors
from plotly.figure_factory._trisurf import trisurf as trisurface

from ...core.options import SkipRendering
from .element import ElementPlot, ColorbarPlot
from .chart import ScatterPlot, CurvePlot


class Chart3DPlot(ElementPlot):

    aspect = param.Parameter(default='cube')

    camera_angle = param.NumericTuple(default=(0.2, 0.5, 0.1, 0.2))

    camera_position = param.NumericTuple(default=(0.1, 0, -0.1))

    camera_zoom = param.Integer(default=3)

    projection = param.String(default='3d')

    width = param.Integer(default=500)

    height = param.Integer(default=500)

    zticks = param.Parameter(default=None, doc="""
        Ticks along z-axis specified as an integer, explicit list of
        tick locations, list of tuples containing the locations.""")

    def init_layout(self, key, element, ranges):
        l, b, zmin, r, t, zmax = self.get_extents(element, ranges)

        xd, yd, zd = (element.get_dimension(i) for i in range(3))
        xaxis = dict(range=[l, r], title=xd.pprint_label)
        if self.logx:
            xaxis['type'] = 'log'
        self._get_ticks(xaxis, self.xticks)

        yaxis = dict(range=[b, t], title=yd.pprint_label)
        if self.logy:
            yaxis['type'] = 'log'
        self._get_ticks(yaxis, self.yticks)

        zaxis = dict(range=[zmin, zmax], title=zd.pprint_label)
        if self.logz:
            zaxis['type'] = 'log'
        self._get_ticks(zaxis, self.zticks)

        opts = {}
        if self.aspect == 'cube':
            opts['aspectmode'] = 'cube'
        else:
            opts['aspectmode'] = 'manual'
            opts['aspectratio'] = self.aspect
        scene = dict(xaxis=xaxis, yaxis=yaxis,
                     zaxis=zaxis, **opts)

        return dict(width=self.width, height=self.height,
                    title=self._format_title(key, separator=' '),
                    plot_bgcolor=self.bgcolor, scene=scene)

    def get_data(self, element, ranges, style):
        return [dict(x=element.dimension_values(0),
                     y=element.dimension_values(1),
                     z=element.dimension_values(2))]


class SurfacePlot(Chart3DPlot, ColorbarPlot):

    trace_kwargs = {'type': 'surface'}

    style_opts = ['opacity', 'lighting', 'lightposition', 'cmap']

    def graph_options(self, element, ranges, style):
        opts = super(SurfacePlot, self).graph_options(element, ranges, style)
        copts = self.get_color_opts(element.vdims[0], element, ranges, style)
        return dict(opts, **copts)

    def get_data(self, element, ranges, style):
        return [dict(x=element.dimension_values(0, False),
                     y=element.dimension_values(1, False),
                     z=element.dimension_values(2, flat=False))]


class Scatter3dPlot(Chart3DPlot, ScatterPlot):

    trace_kwargs = {'type': 'scatter3d', 'mode': 'markers'}


class Line3dPlot(Chart3DPlot, CurvePlot):

    trace_kwargs = {'type': 'scatter3d', 'mode': 'lines'}


class TriSurfacePlot(Chart3DPlot, ColorbarPlot):

    style_opts = ['cmap', 'edges_color', 'facecolor']

    def get_data(self, element, ranges, style):
        try:
            from scipy.spatial import Delaunay
        except:
            SkipRendering("SciPy not available, cannot plot TriSurface")
        x, y, z = (element.dimension_values(i) for i in range(3))
        points2D = np.vstack([x, y]).T
        tri = Delaunay(points2D)
        simplices = tri.simplices
        return [dict(x=x, y=y, z=z, simplices=simplices, edges_color='black',
                     scale=None)]

    def graph_options(self, element, ranges, style):
        if 'cmap' in style:
            cmap = style.pop('cmap')
            if cmap in colors.PLOTLY_SCALES:
                style['colormap'] = colors.PLOTLY_SCALES[cmap]
            else:
                cmap = get_cmap(cmap)
                style['colormap'] = [cmap(i) for i in np.linspace(0, 1)]
        style['show_colorbar'] = self.colorbar
        return style

    def init_graph(self, data, options):
        trace = super(TriSurfacePlot, self).init_graph(data, options)
        return trisurface(**trace)[0].to_plotly_json()
