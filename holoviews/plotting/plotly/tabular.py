from __future__ import absolute_import, division, unicode_literals

import param

from .element import ElementPlot


class TablePlot(ElementPlot):

    height = param.Number(default=400)

    width = param.Number(default=400)

    trace_kwargs = {'type': 'table'}

    style_opts = ['line', 'fill', 'align', 'font', 'cell_height']

    _style_key = 'cells'

    def get_data(self, element, ranges, style):
        header = dict(values=[d.pprint_label for d in element.dimensions()])
        cells = dict(values=[[d.pprint_value(v) for v in element.dimension_values(d)]
                              for d in element.dimensions()])
        return [{'header': header, 'cells': cells}]

    def init_layout(self, key, element, ranges):
        return dict(width=self.width, height=self.height,
                    title=self._format_title(key, separator=' '),
                    plot_bgcolor=self.bgcolor)
