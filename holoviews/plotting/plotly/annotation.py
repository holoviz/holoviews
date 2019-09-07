from __future__ import absolute_import, division, unicode_literals

import param

from .chart import ScatterPlot


class LabelPlot(ScatterPlot):

    xoffset = param.Number(default=None, doc="""
      Amount of offset to apply to labels along x-axis.""")

    yoffset = param.Number(default=None, doc="""
      Amount of offset to apply to labels along x-axis.""")

    style_opts = ['visible', 'color', 'family', 'size']

    _nonvectorized_styles = []

    trace_kwargs = {'type': 'scatter', 'mode': 'text'}

    _style_key = 'textfont'

    def get_data(self, element, ranges, style):
        x, y = ('y', 'x') if self.invert_axes else ('x', 'y')
        text_dim = element.vdims[0]
        xs = element.dimension_values(0)
        if self.xoffset:
            xs = xs + self.xoffset
        ys = element.dimension_values(1)
        if self.yoffset:
            ys = ys + self.yoffset
        text = [text_dim.pprint_value(v) for v in element.dimension_values(2)]
        return [{x: xs, y: ys, 'text': text}]
