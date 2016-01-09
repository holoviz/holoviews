import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF

from .element import ElementPlot


class ScatterPlot(ElementPlot):

    style_opts = ['symbol', 'color']

    def graph_options(self, element, ranges):
        opts = super(ScatterPlot, self).graph_options(element, ranges)
        opts['mode'] = 'markers'
        opts['marker'] = self.style[self.cyclic_index]
        return opts

class PointPlot(ScatterPlot):

    def init_graph(self, element, ranges, **opts):
        trace = go.Scatter(x=element.dimension_values(0),
                           y=element.dimension_values(1),
                           **opts)
        return trace


class CurvePlot(ElementPlot):
    
    def init_graph(self, element, ranges, **opts):
        trace = go.Scatter(x=element.dimension_values(0),
                           y=element.dimension_values(1), **opts)
        return trace


class ErrorBarsPlot(ElementPlot):
    
    def init_graph(self, element, ranges, **opts):
        neg_error = element.dimension_values(2)
        pos_idx = 3 if len(element.dimensions()) > 3 else 2
        pos_error = element.dimension_values(pos_idx)

        trace = go.Scatter(x=element.dimension_values(0),
                           y=element.dimension_values(1),
                           error_y=dict(type='data',
                                        array=pos_error,
                                        arrayminus=neg_error,
                                        visible=True), **opts)
        return trace


class BivariatePlot(ElementPlot):
    
    def init_graph(self, element, ranges, **opts):
        trace = go.Histogram2dcontour(
            x=element.dimension_values(0), y=element.dimension_values(1),
            ncontours=20, colorscale='Hot', reversescale=True,
            showscale=False, **opts
        )
        return trace


class DistributionPlot(ElementPlot):
    
    def init_graph(self, element, ranges, **opts):
        trace = go.Histogram(x=element.dimension_values(0), **opts)
        return trace


class VectorPlot(ElementPlot):

    def init_graph(self, element, ranges, **opts):
        args = [element.dimension_values(i) for i in range(4)]
        fig = FF.create_quiver(*args, scale=.25, arrow_scale=.4,
                               line=dict(width=1), **opts)
        return fig
