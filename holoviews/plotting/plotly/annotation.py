from __future__ import absolute_import, division, unicode_literals

import param

from .chart import ScatterPlot


class LabelPlot(ScatterPlot):

    style_opts = ['color', 'family', 'size']

    _nonvectorized_styles = []

    trace_kwargs = {'type': 'scatter', 'mode': 'text'}

    _style_key = 'textfont'

    def get_data(self, element, ranges, style):
        text_dim = element.vdims[0]
        return [dict(x=element.dimension_values(0),
                     y=element.dimension_values(1),
                     text=[text_dim.pprint_value(v)
                           for v in element.dimension_values(2)])]
