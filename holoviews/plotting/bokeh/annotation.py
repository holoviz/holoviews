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
    _plot_method = 'segment'

    def get_data(self, element, ranges=None):
        plot = self.handles['plot']
        if isinstance(element, HLine):
            x0 = plot.x_range.start
            y0 = element.data
            x1 = plot.x_range.end
            y1 = element.data
        elif isinstance(element, VLine):
            x0 = element.data
            y0 = plot.y_range.start
            x1 = element.data
            y1 = plot.y_range.end
        return (dict(x0=[x0], y0=[y0], x1=[x1], y1=[y1]),
                dict(x0='x0', y0='y0', x1='x1', y1='y1'))


    def get_extents(self, element, ranges=None):
        return None, None, None, None



class SplinePlot(ElementPlot):
    """
    Draw the supplied Spline annotation (see Spline docstring).
    Does not support matplotlib Path codes.
    """

    style_opts = line_properties
    _plot_method = 'bezier'

    def get_data(self, element, ranges=None):
        verts = np.array(element.data[0])
        xs, ys = verts[:, 0], verts[:, 1]
        return (dict(x0=[xs[0]], y0=[ys[0]], x1=[xs[-1]], y1=[ys[-1]],
                     cx0=[xs[1]], cy0=[ys[1]], cx1=[xs[2]], cy1=[ys[2]]),
                dict(x0='x0', y0='y0', x1='x1', y1='y1',
                     cx0='cx0', cx1='cx1', cy0='cy0', cy1='cy1'))
