import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import param

from ...core.options import Store
from .plot import PlotlyPlot
from ..plot import GenericElementPlot, GenericOverlayPlot
from .. import util


class ElementPlot(PlotlyPlot, GenericElementPlot):
    

    def initialize_plot(self, ranges=None):
        """
        Initializes a new plot object with the last available frame.
        """
        # Get element key and ranges for frame
        element = self.hmap.last
        key = self.keys[-1]
        return self.generate_plot(key, element, ranges)


    def generate_plot(self, key, element, ranges):
        self.current_frame = element
        self.current_key = key
        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = util.match_spec(element, ranges)

        graph = self.init_graph(element, ranges)
        self.handles['graph'] = graph
        
        layout = self.init_layout(key, element, ranges)
        
        if isinstance(graph, go.Figure):
            graph['layout'] = layout
            self.handles['fig'] = graph
        elif not (self.overlaid or self.subplot):
            fig = go.Figure(data=[graph], layout=layout)
            self.handles['fig'] = fig
            return fig
        return graph
        

    def init_graph(self, element, ranges):
        pass

    
    def init_layout(self, key, element, ranges):
        return dict(width=self.width, height=self.height,
                   title=self._format_title(key, separator=' '))


    def update_frame(self, key, ranges=None):
        """
        Updates an existing plot with data corresponding
        to the key.
        """
        element = self._get_frame(key)
        self.generate_plot(key, element, ranges)

        
    
class OverlayPlot(GenericOverlayPlot, ElementPlot):

    def initialize_plot(self, ranges=None):
        key = self.keys[-1]
        ranges = self.compute_ranges(self.hmap, key, ranges)
        graphs = []
        for key, subplot in self.subplots.items():
            graphs.append(subplot.initialize_plot(ranges))

        layout = self.init_layout(key, None, ranges)
        fig = go.Figure(data=graphs, layout=layout)
        self.handles['fig'] = fig
        return fig


class OverlayPlot(GenericOverlayPlot, ElementPlot):

    def initialize_plot(self, ranges=None):
        key = self.keys[-1]
        ranges = self.compute_ranges(self.hmap, key, ranges)
        graphs = []
        for key, subplot in self.subplots.items():
            graphs.append(subplot.initialize_plot(ranges))

        layout = self.init_layout(key, None, ranges)
        fig = go.Figure(data=graphs, layout=layout)
        self.handles['fig'] = fig
        return fig
