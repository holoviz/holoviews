from __future__ import absolute_import, division, unicode_literals

import param
import numpy as np

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

    def get_data(self, element, ranges, style):
        return [dict(x=element.dimension_values(0),
                     y=element.dimension_values(1),
                     z=element.dimension_values(2))]


class SurfacePlot(Chart3DPlot, ColorbarPlot):

    trace_kwargs = {'type': 'surface'}

    style_opts = ['alpha', 'lighting', 'lightposition', 'cmap']

    def graph_options(self, element, ranges, style):
        opts = super(SurfacePlot, self).graph_options(element, ranges, style)
        copts = self.get_color_opts(element.vdims[0], element, ranges, style)
        copts['colorscale'] = style.get('cmap', 'Viridis')
        return dict(opts, **copts)

    def get_data(self, element, ranges, style):
        return [dict(x=element.dimension_values(0, False),
                     y=element.dimension_values(1, False),
                     z=element.dimension_values(2, flat=False))]


class Scatter3DPlot(Chart3DPlot, ScatterPlot):

    trace_kwargs = {'type': 'scatter3d', 'mode': 'markers'}


class Path3DPlot(Chart3DPlot, CurvePlot):

    trace_kwargs = {'type': 'scatter3d', 'mode': 'lines'}

    _per_trace = True

    _nonvectorized_styles = []

    def graph_options(self, element, ranges, style):
        opts = super(Path3DPlot, self).graph_options(element, ranges, style)
        opts['line'].pop('showscale', None)
        return opts

    def get_data(self, element, ranges, style):
        return [dict(x=el.dimension_values(0), y=el.dimension_values(1),
                     z=el.dimension_values(2))
                for el in element.split()]


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
        return [dict(x=x, y=y, z=z, simplices=simplices, edges_color='black')]

    def graph_options(self, element, ranges, style):
        opts = super(TriSurfacePlot, self).graph_options(element, ranges, style)
        copts = self.get_color_opts(element.dimensions()[2], element, ranges, style)
        opts['colormap'] = [tuple(v/255. for v in colors.hex_to_rgb(c))
                            for _, c in copts['colorscale']]
        opts['scale'] = [l for l, _ in copts['colorscale']]
        opts['show_colorbar'] = self.colorbar
        return {k: v for k, v in opts.items() if 'legend' not in k and k != 'name'}

    def init_graph(self, data, options, index=0):
        trace = super(TriSurfacePlot, self).init_graph(data, options, index)
        return trisurface(**trace)[0].to_plotly_json()
