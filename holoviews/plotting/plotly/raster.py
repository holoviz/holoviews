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
        opts = super(RasterPlot, self).graph_options(element, ranges)
        opts['zmin'], opts['zmax'] = ranges[element.get_dimension(2).name]
        opts['zauto'] = False
        if 'cmap' in opts:
            opts['colorscale'] = opts.pop('cmap', None)
        return opts

    def init_graph(self, element, ranges, **opts):
        data = go.Heatmap(
            x=element.dimension_values(0, expanded=False),
            y=element.dimension_values(1, expanded=False),
            z=element.dimension_values(2, flat=False), **opts)
        return data

