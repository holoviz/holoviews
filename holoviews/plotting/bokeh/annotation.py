from collections import defaultdict

import numpy as np
from bokeh.models import Span

from ...element import HLine
from .element import ElementPlot, text_properties, line_properties


class TextPlot(ElementPlot):

    style_opts = text_properties+['color']
    _plot_methods = dict(single='text', batched='text')

    def _glyph_properties(self, plot, element, source, ranges):
        props = super(TextPlot, self)._glyph_properties(plot, element, source, ranges)
        props['text_align'] = element.halign
        props['text_baseline'] = 'middle' if element.valign == 'center' else element.valign
        if 'color' in props:
            props['text_color'] = props.pop('color')
        return props

    def get_data(self, element, ranges=None, empty=False):
        mapping = dict(x='x', y='y', text='text')
        if empty:
            return dict(x=[], y=[], text=[]), mapping
        if self.invert_axes:
            data = dict(x=[element.y], y=[element.x])
        else:
            data = dict(x=[element.x], y=[element.y])
        self._categorize_data(data, ('x', 'y'), element.dimensions())
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

    _plot_methods = dict(single='Span')

    def get_data(self, element, ranges=None, empty=False):
        data, mapping = {}, {}
        dim = 'width' if isinstance(element, HLine) else 'height'
        if self.invert_axes:
            dim = 'width' if dim == 'height' else 'height'
        mapping['dimension'] = dim
        mapping['location'] = element.data
        return (data, mapping)

    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object.
        """
        box = Span(level='annotation', **mapping)
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
        data_attrs = ['x0', 'y0', 'cx0', 'cy0', 'cx1', 'cy1', 'x1', 'y1',]
        verts = np.array(element.data[0])
        inds = np.where(np.array(element.data[1])==1)[0]
        data = {da: [] for da in data_attrs}
        skipped = False
        for vs in np.split(verts, inds[1:]):
            if len(vs) != 4:
                skipped = len(vs) > 1
                continue
            for x, y, xl, yl in zip(vs[:, 0], vs[:, 1], data_attrs[::2], data_attrs[1::2]):
                data[xl].append(x)
                data[yl].append(y)
        if skipped:
            self.warning('Bokeh SplitPlot only support cubic splines, '
                         'unsupported splines were skipped during plotting.')
        data = {da: data[da] for da in data_attrs}
        return (data, dict(zip(data_attrs, data_attrs)))
