import numpy as np
import plotly.graph_objs as go
from matplotlib.cm import get_cmap
from plotly import colors
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import Scene, XAxis, YAxis, ZAxis

try:
    from plotly.figure_factory._trisurf import trisurf as trisurface
except ImportError:
    pass

import param

from ...core.options import SkipRendering
from .element import ElementPlot, ColorbarPlot
from .chart import ScatterPlot

class Chart3DPlot(ElementPlot):

    aspect = param.Parameter(default='cube')

    camera_angle = param.NumericTuple(default=(0.2, 0.5, 0.1, 0.2))

    camera_position = param.NumericTuple(default=(0.1, 0, -0.1))

    camera_zoom = param.Integer(default=3)

    projection = param.String(default='3d')

    def init_layout(self, key, element, ranges):
        l, b, zmin, r, t, zmax = self.get_extents(element, ranges)

        xd, yd, zd = (element.get_dimension(i) for i in range(3))
        xaxis = dict(range=[l, r], title=xd.pprint_label)
        if self.logx:
            xaxis['type'] = 'log'

        yaxis = dict(range=[b, t], title=yd.pprint_label)
        if self.logy:
            yaxis['type'] = 'log'

        zaxis = dict(range=[zmin, zmax], title=zd.pprint_label)
        if self.logz:
            zaxis['type'] = 'log'

        opts = {}
        if self.aspect == 'cube':
            opts['aspectmode'] = 'cube'
        else:
            opts['aspectmode'] = 'manual'
            opts['aspectratio'] = self.aspect
        scene = Scene(xaxis=XAxis(xaxis), yaxis=YAxis(yaxis),
                      zaxis=ZAxis(zaxis), **opts)

        return dict(width=self.width, height=self.height,
                    title=self._format_title(key, separator=' '),
                    plot_bgcolor=self.bgcolor, scene=scene)


class SurfacePlot(ColorbarPlot, Chart3DPlot):

    graph_obj = go.Surface

    style_opts = ['opacity', 'lighting', 'lightposition', 'cmap']

    def graph_options(self, element, ranges):
        opts = super(SurfacePlot, self).graph_options(element, ranges)
        style = self.style[self.cyclic_index]
        copts = self.get_color_opts(element.vdims[0], element, ranges, style)
        return dict(opts, **copts)


    def get_data(self, element, ranges):
        return (), dict(x=element.dimension_values(0, False),
                        y=element.dimension_values(1, False),
                        z=element.dimension_values(2, flat=False))


class Scatter3dPlot(ScatterPlot, Chart3DPlot):

    graph_obj = go.Scatter3d

    def get_data(self, element, ranges):
        return (), dict(x=element.dimension_values(0),
                        y=element.dimension_values(1),
                        z=element.dimension_values(2))


class TrisurfacePlot(ColorbarPlot, Chart3DPlot):

    style_opts = ['cmap']

    def get_data(self, element, ranges):
        try:
            from scipy.spatial import Delaunay
        except:
            SkipRendering("SciPy not available, cannot plot Trisurface")
        x, y, z = (element.dimension_values(i) for i in range(3))
        points2D = np.vstack([x, y]).T
        tri = Delaunay(points2D)
        simplices = tri.simplices
        return (x, y, z, simplices, self.colorbar, 'black', None), {}

    def graph_options(self, element, ranges):
        opts = self.style[self.cyclic_index]
        if 'cmap' in opts:
            cmap = opts.pop('cmap')
            if cmap in colors.PLOTLY_SCALES:
                opts['colormap'] = colors.PLOTLY_SCALES[cmap]
            else:
                cmap = get_cmap(cmap)
                opts['colormap'] = [cmap(i) for i in np.linspace(0, 1)]
        return opts

    def init_graph(self, plot_args, plot_kwargs):
        if hasattr(FF, '_trisurf'):
            trisurf = FF._trisurf(*plot_args[:-1], **plot_kwargs)
        else:
            trisurf = trisurface(*plot_args, **plot_kwargs)
        return trisurf[0]
