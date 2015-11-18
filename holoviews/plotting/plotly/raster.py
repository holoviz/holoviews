import numpy as np
import plotly.graph_objs as go

from .element import ElementPlot


class HeatMapPlot(ElementPlot):

    def init_graph(self, element, ranges):
        data = go.Heatmap(
        z=np.flipud(element.raster),
        x=element.dimension_values(0, True),
        y=element.dimension_values(0, True))
        return data

class RasterPlot(ElementPlot):

    def init_graph(self, element, ranges):
        data = go.Heatmap(
        z=element.data)
        return data

class SurfacePlot(ElementPlot):

    def init_graph(self, element, ranges):
        data = go.Surface(
        z=element.data)
        return data
