from collections import defaultdict

import numpy as np
from bokeh.models import Span, Arrow
try:
    from bokeh.models.arrow_heads import TeeHead, NormalHead
    arrow_start = {'<->': NormalHead, '<|-|>': NormalHead}
    arrow_end = {'->': NormalHead, '-[': TeeHead, '-|>': NormalHead, '-': None}
except:
    from bokeh.models.arrow_heads import OpenHead, NormalHead
    arrow_start = {'<->': NormalHead, '<|-|>': NormalHead}
    arrow_end = {'->': NormalHead, '-[': OpenHead, '-|>': NormalHead, '-': None}

from ...element import HLine
from ...core.util import datetime_types
from .element import ElementPlot, CompositeElementPlot, text_properties, line_properties
from .util import date_to_integer


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
        loc = element.data
        if isinstance(loc, datetime_types):
            loc = date_to_integer(loc)
        mapping['location'] = loc
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



class ArrowPlot(CompositeElementPlot):

    style_opts = (['arrow_%s' % p for p in line_properties+['size']] + text_properties)

    _style_groups = {'arrow': 'arrow', 'label': 'text'}

    _update_handles = ['glyph']

    _plot_methods = dict(single='text')

    def get_data(self, element, ranges=None, empty=False):
        plot = self.state
        label_mapping = dict(x='x', y='y', text='text')

        # Compute arrow
        x1, y1 = element.x, element.y
        axrange = plot.x_range if self.invert_axes else plot.y_range
        span = (axrange.end - axrange.start) / 6.
        if element.direction == '^':
            x2, y2 = x1, y1-span
            label_mapping['text_baseline'] = 'top'
        elif element.direction == '<':
            x2, y2 = x1+span, y1
            label_mapping['text_align'] = 'left'
            label_mapping['text_baseline'] = 'middle'
        elif element.direction == '>':
            x2, y2 = x1-span, y1
            label_mapping['text_align'] = 'right'
            label_mapping['text_baseline'] = 'middle'
        else:
            x2, y2 = x1, y1+span
            label_mapping['text_baseline'] = 'bottom'
        arrow_opts = {'x_end': x1, 'y_end': y1,
                      'x_start': x2, 'y_start': y2}

        # Define arrowhead
        arrow_opts['arrow_start'] = arrow_start.get(element.arrowstyle, None)
        arrow_opts['arrow_end'] = arrow_end.get(element.arrowstyle, NormalHead)

        # Compute label
        if self.invert_axes:
            label_data = dict(x=[y2], y=[x2])
        else:
            label_data = dict(x=[x2], y=[y2])
        label_data['text'] = [element.text]
        return ({'label': label_data},
                {'arrow': arrow_opts, 'label': label_mapping})

    def _init_glyph(self, plot, mapping, properties, key):
        """
        Returns a Bokeh glyph object.
        """
        properties.pop('legend')
        if key == 'arrow':
            properties.pop('source')
            arrow_end = mapping.pop('arrow_end')
            arrow_start = mapping.pop('arrow_start')
            start = arrow_start(**properties) if arrow_start else None
            end = arrow_end(**properties) if arrow_end else None
            glyph = Arrow(start=start, end=end, **dict(**mapping))
        else:
            properties = {p if p == 'source' else 'text_'+p: v
                          for p, v in properties.items()}
            glyph, _ = super(ArrowPlot, self)._init_glyph(plot, mapping, properties, 'text')
        plot.renderers.append(glyph)
        return None, glyph

    def get_extents(self, element, ranges=None):
        return None, None, None, None
