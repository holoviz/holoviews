import numpy as np
from bokeh.models import BoxAnnotation

from ...element import HLine, VLine
from .element import ElementPlot, text_properties, line_properties


class TextPlot(ElementPlot):

    style_opts = text_properties
    _plot_method = 'text'

    def get_data(self, element, ranges=None, empty=False):
        mapping = dict(x='x', y='y', text='text')
        if empty:
            return dict(x=[], y=[], text=[]), mapping
        return (dict(x=[element.x], y=[element.y],
                     text=[element.text]), mapping)

    def get_extents(self, element, ranges=None):
        return None, None, None, None


class LineAnnotationPlot(ElementPlot):

    style_opts = line_properties

    _update_handles = ['glyph']

    def get_data(self, element, ranges=None, empty=False):
        data, mapping = {}, {}
        if isinstance(element, HLine):
            mapping['bottom'] = element.data
            mapping['top'] = element.data
        elif isinstance(element, VLine):
            mapping['left'] = element.data
            mapping['right'] = element.data
        return (data, mapping)


    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object.
        """
        properties.pop('source')
        properties.pop('legend')
        box = BoxAnnotation(plot=plot, level='overlay',
                            **dict(mapping, **properties))
        plot.renderers.append(box)
        return None, box


    def get_extents(self, element, ranges=None):
        return None, None, None, None



class SplinePlot(ElementPlot):
    """
    Draw the supplied Spline annotation (see Spline docstring).
    Does not support matplotlib Path codes.
    """

    style_opts = line_properties
    _plot_method = 'bezier'

    def get_data(self, element, ranges=None, empty=False):
        data_attrs = ['x0', 'y0', 'x1', 'y1',
                      'cx0', 'cx1', 'cy0', 'cy1']
        if empty:
            data = {attr: [] for attr in data_attrs}
        else:
            verts = np.array(element.data[0])
            xs, ys = verts[:, 0], verts[:, 1]
            data = dict(x0=[xs[0]], y0=[ys[0]], x1=[xs[-1]], y1=[ys[-1]],
                        cx0=[xs[1]], cy0=[ys[1]], cx1=[xs[2]], cy1=[ys[2]])

        return (data, dict(zip(data_attrs, data_attrs)))
