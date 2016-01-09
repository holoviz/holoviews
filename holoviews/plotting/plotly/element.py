import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import param

from ...core.options import Store
from .plot import PlotlyPlot
from ..plot import GenericElementPlot, GenericOverlayPlot
from .. import util


class ElementPlot(PlotlyPlot, GenericElementPlot):
    
    aspect = param.Parameter(default='cube', doc="""
        The aspect ratio mode of the plot. By default, a plot may
        select its own appropriate aspect ratio but sometimes it may
        be necessary to force a square aspect ratio (e.g. to display
        the plot as an element of a grid). The modes 'auto' and
        'equal' correspond to the axis modes of the same name in
        matplotlib, a numeric value may also be passed.""")

    def initialize_plot(self, ranges=None):
        """
        Initializes a new plot object with the last available frame.
        """
        # Get element key and ranges for frame
        return self.generate_plot(self.keys[-1], ranges)


    def generate_plot(self, key, ranges):
        element = self._get_frame(key) if len(self) > 1 else self.hmap.last
        self.current_frame = element
        self.current_key = key
        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = util.match_spec(element, ranges)

        opts = self.graph_options(element, ranges)
        graph = self.init_graph(element, ranges, **opts)
        self.handles['graph'] = graph
        
        layout = self.init_layout(key, element, ranges)
        self.handles['layout'] = layout
        
        if isinstance(graph, go.Figure):
            graph['layout'] = layout
            self.handles['fig'] = graph
        elif not (self.overlaid or self.subplot):
            fig = go.Figure(data=[graph], layout=layout)
            self.handles['fig'] = fig
            return fig
        return graph

    
    def graph_options(self, element, ranges):
        if self.overlay_dims:
            legend = ', '.join([d.pprint_value_string(v) for d, v in
                                self.overlay_dims.items()])
        else:
            legend = element.label

        opts = dict(showlegend=self.show_legend,
                    legendgroup=element.group,
                    name=legend)

        if self.layout_num:
            opts['xaxis'] = 'x' + str(self.layout_num)
            opts['yaxis'] = 'y' + str(self.layout_num)

        return opts


    def init_graph(self, element, ranges, **opts):
        pass

    
    def init_layout(self, key, element, ranges):
        l, b, r, t = self.get_extents(element, ranges)

        xd, yd = (element.get_dimension(i) for i in range(2))
        xaxis = dict(range=[l, r], title=str(xd))
        if self.logx:
            xaxis['type'] = 'log'

        yaxis = dict(range=[b, t], title=str(yd))
        if self.logy:
            yaxis['type'] = 'log'

        return dict(width=self.width, height=self.height,
                    title=self._format_title(key, separator=' '),
                    plot_bgcolor=self.bgcolor, xaxis=xaxis,
                    yaxis=yaxis)


    def update_frame(self, key, ranges=None):
        """
        Updates an existing plot with data corresponding
        to the key.
        """
        self.generate_plot(key, ranges)

        
    
class OverlayPlot(GenericOverlayPlot, ElementPlot):


    def initialize_plot(self, ranges=None):
        """
        Initializes a new plot object with the last available frame.
        """
        # Get element key and ranges for frame
        return self.generate_plot(self.hmap.keys()[0], ranges)

    
    def generate_plot(self, key, ranges):
        element = self._get_frame(key)

        ranges = self.compute_ranges(self.hmap, key, ranges)
        graphs = []
        for key, subplot in self.subplots.items():
            graphs.append(subplot.generate_plot(key, ranges))

        layout = self.init_layout(key, element, ranges)
        fig = go.Figure(data=graphs, layout=layout)
        self.handles['fig'] = fig
        return fig
