import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF

from .element import ElementPlot


class PointPlot(ElementPlot):
    
    def init_graph(self, element, ranges):
        trace = go.Scatter(x=element.dimension_values(0),
                           y=element.dimension_values(1),
                           mode = 'markers')
        return trace

class CurvePlot(ElementPlot):
    
    def init_graph(self, element, ranges):
        trace = go.Scatter(x=element.dimension_values(0),
                           y=element.dimension_values(1))
        return trace


class ErrorBarsPlot(ElementPlot):
    
    def init_graph(self, element, ranges):
        neg_error = element.dimension_values(2)
        pos_idx = 3 if len(element.dimensions()) > 3 else 2
        pos_error = element.dimension_values(pos_idx)

        trace = go.Scatter(x=element.dimension_values(0),
                           y=element.dimension_values(1),
                           error_y=dict(type='data',
                                        array=pos_error,
                                        arrayminus=neg_error,
                                        visible=True))
        return trace


class BivariatePlot(ElementPlot):
    
    def init_graph(self, element, ranges):
        trace = go.Histogram2dcontour(
            x=element.dimension_values(0), y=element.dimension_values(1),
            name='density', ncontours=20,
            colorscale='Hot', reversescale=True, showscale=False
        )
        return trace


class DistributionPlot(ElementPlot):
    
    def init_graph(self, element, ranges):
        trace = go.Histogram(x=element.dimension_values(0))
        return trace


class Scatter3dPlot(ElementPlot):
    
    def init_graph(self, element, ranges):
        trace = go.Scatter3d(x=element.dimension_values(0),
                             y=element.dimension_values(1),
                             z=element.dimension_values(2),
                             mode = 'markers')
        return trace


class VectorPlot(ElementPlot):

    def init_graph(self, element, ranges):
        args = [element.dimension_values(i) for i in range(4)]
        fig = FF.create_quiver(*args, scale=.25, arrow_scale=.4,
                               name='quiver', line=dict(width=1))
        return fig
