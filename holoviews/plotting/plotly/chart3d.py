import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import Scene, XAxis, YAxis, ZAxis

import param

from ...core.spaces import DynamicMap
from .element import ElementPlot

class Chart3DPlot(ElementPlot):

    aspect = param.Parameter(default='cube')

    camera_angle = param.NumericTuple(default=(0.2, 0.5, 0.1, 0.2))

    camera_position = param.NumericTuple(default=(0.1, 0, -0.1))

    camera_zoom = param.Integer(default=3)

    projection = param.String(default='3d')

    def init_layout(self, key, element, ranges):
        l, b, zmin, r, t, zmax = self.get_extents(element, ranges)

        xd, yd = (element.get_dimension(i) for i in range(2))
        xaxis = dict(range=[l, r], title=str(xd))
        if self.logx:
            xaxis['type'] = 'log'

        yaxis = dict(range=[b, t], title=str(yd))
        if self.logy:
            yaxis['type'] = 'log'

        zaxis = dict(range=[zmin, zmax], title=str(yd))
        if self.logz:
            zaxis['type'] = 'log'

        scene = Scene(xaxis=XAxis(xaxis), yaxis=YAxis(yaxis),
                      zaxis=ZAxis(zaxis), aspectmode=self.aspect)

        return dict(width=self.width, height=self.height,
                    title=self._format_title(key, separator=' '),
                    plot_bgcolor=self.bgcolor, scene=scene)


class SurfacePlot(Chart3DPlot):

    def init_graph(self, element, ranges, **opts):
        data = go.Surface(
            x=element.dimension_values(0, True),
            y=element.dimension_values(1, True),
            z=element.data, **opts)
        return data


class Scatter3dPlot(Chart3DPlot):

    def init_graph(self, element, ranges, **opts):
        trace = go.Scatter3d(x=element.dimension_values(0),
                             y=element.dimension_values(1),
                             z=element.dimension_values(2),
                             mode = 'markers', **opts)
        return trace

