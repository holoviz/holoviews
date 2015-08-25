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
        if isinstance(element, HLine):
            angle = 0
            x, y = 0, element.data
        elif isinstance(element, VLine):
            angle = np.pi/2
            x, y = element.data, 0
        return (dict(x=[x], y=[y], angle=[angle], length=[100]),
                dict(x='x', y='y', angle='angle', length='length'))

    def get_extents(self, element, ranges=None):
        return None, None, None, None
