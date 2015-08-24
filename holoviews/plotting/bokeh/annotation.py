import numpy as np

from ...element import HLine, VLine
from .element import ElementPlot, text_properties, line_properties


class TextPlot(ElementPlot):

    style_opts = text_properties

    def get_data(self, element, ranges=None):
        return dict(x=[element.x], y=[element.y], text=[element.text])

    def _init_glyph(self, element, plot, source, ranges):
        self.handles['text'] = plot.text(x='x', y='y', text='text',
                                         source=source, **self.style)

    def get_extents(self, element, ranges=None):
        return None, None, None, None


class LineAnnotationPlot(ElementPlot):

    style_opts = line_properties

    def get_data(self, element, ranges=None):
        if isinstance(element, HLine):
            angle = 0
            x, y = 0, element.data
        elif isinstance(element, VLine):
            angle = np.pi/2
            x, y = element.data, 0
        return dict(x=[x], y=[y], angle=[angle], length=[100])


    def _init_glyph(self, element, plot, source, ranges):
        self.handles['line'] = plot.ray(x='x', y='y', length='length',
                                        angle='angle', source=source, **self.style)


    def get_extents(self, element, ranges=None):
        return None, None, None, None
