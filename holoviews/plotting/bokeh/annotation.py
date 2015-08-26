import numpy as np

from ...element import HLine, VLine
from .element import ElementPlot, text_properties, line_properties


class TextPlot(ElementPlot):

    style_opts = text_properties
    _plot_method = 'text'

    def get_data(self, element, ranges=None):
        mapping = dict(x='x', y='y', text='text')
        return (dict(x=[element.x], y=[element.y],
                     text=[element.text]), mapping)

    def get_extents(self, element, ranges=None):
        return None, None, None, None


class LineAnnotationPlot(ElementPlot):

    style_opts = line_properties
    _plot_method = 'ray'

    def get_data(self, element, ranges=None):
        plot = self.handles['plot']
        if isinstance(element, HLine):
            low, high = plot.x_range.start, plot.x_range.end
            length = high-low
            angle = 0
            x, y = low, element.data
        elif isinstance(element, VLine):
            low, high = plot.y_range.start, plot.y_range.end
            length = high-low
            angle = np.pi/2
            x, y = element.data, low
        return (dict(x=[x], y=[y], angle=[angle], length=[length]),
                dict(x='x', y='y', angle='angle', length='length'))

    def get_extents(self, element, ranges=None):
        return None, None, None, None
