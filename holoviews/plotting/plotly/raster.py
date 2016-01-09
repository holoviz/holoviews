import numpy as np
import plotly.graph_objs as go

from .element import ElementPlot


class HeatMapPlot(ElementPlot):

    def init_graph(self, element, ranges, **opts):
        data = go.Heatmap(
        z=np.flipud(element.raster),
        x=element.dimension_values(0, True),
        y=element.dimension_values(1, True), **opts)
        return data

class RasterPlot(ElementPlot):

    style_opts = ['cmap']

    def graph_options(self, element, ranges):
        style = self.style[self.cyclic_index]
        opts = super(RasterPlot, self).graph_options(element, ranges)
        opts['zmin'], opts['zmax'] = ranges[element.get_dimension(2).name]
        opts['zauto'] = False
        if 'cmap' in style:
            opts['colorscale'] = style['cmap']
        return opts

    def init_graph(self, element, ranges, **opts):
        data = go.Heatmap(
            x=element.dimension_values(0, True),
            y=element.dimension_values(1, True),
            z=element.data, **opts)
        return data

