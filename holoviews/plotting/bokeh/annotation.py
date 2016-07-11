from collections import defaultdict

import numpy as np
from bokeh.models import BoxAnnotation

from ...element import HLine, VLine
from .element import ElementPlot, text_properties, line_properties


class TextPlot(ElementPlot):

    style_opts = text_properties
    _plot_methods = dict(single='text', batched='text')

    def get_data(self, element, ranges=None, empty=False):
        mapping = dict(x='x', y='y', text='text')
        if empty:
            return dict(x=[], y=[], text=[]), mapping
        if self.invert_axes:
            data = dict(x=[element.y], y=[element.x])
        else:
            data = dict(x=[element.x], y=[element.y])
        data['text'] = [element.text]
        return (data, mapping)


    def get_batched_data(self, element, ranges=None, empty=False):
        data = defaultdict(list)
        for key, el in element.data.items():
            eldata, elmapping = self.get_data(el, ranges, empty)
            for k, eld in eldata.items():
                data[k].extend(eld)
        return data, elmapping


    def get_extents(self, element, ranges=None):
        return None, None, None, None



class LineAnnotationPlot(ElementPlot):

    style_opts = line_properties

    _update_handles = ['glyph']

    def get_data(self, element, ranges=None, empty=False):
        data, mapping = {}, {}
        if (isinstance(element, HLine) or
            (isinstance(element, VLine) and self.invert_axes)):
            mapping['bottom'] = element.data
            mapping['top'] = element.data
        elif (isinstance(element, VLine) or
              (isinstance(element, HLine) and self.invert_axes)):
            mapping['left'] = element.data
            mapping['right'] = element.data
        return (data, mapping)


    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object.
        """
        properties.pop('source', None)
        properties.pop('legend', None)
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
    _plot_methods = dict(single='bezier')

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
