from __future__ import absolute_import, division, unicode_literals

import numpy as np

from ..mixins import GeomMixin
from .element import ColorbarPlot, LegendPlot
from .styles import line_properties, fill_properties


class SegmentPlot(GeomMixin, ColorbarPlot):
    """
    Segments are lines in 2D space where each two each dimensions specify a
    (x, y) node of the line.
    """
    style_opts = line_properties + ['cmap']

    _nonvectorized_styles = ['cmap']

    _plot_methods = dict(single='segment')

    def get_data(self, element, ranges, style):
        # Get [x0, y0, x1, y1]
        x0idx, y0idx, x1idx, y1idx = (
            (1, 0, 3, 2) if self.invert_axes else (0, 1, 2, 3)
        )

        # Compute segments
        x0s, y0s, x1s, y1s = (
            element.dimension_values(x0idx),
            element.dimension_values(y0idx),
            element.dimension_values(x1idx),
            element.dimension_values(y1idx)
        )

        data = {'x0': x0s, 'x1': x1s, 'y0': y0s, 'y1': y1s}
        mapping = dict(x0='x0', x1='x1', y0='y0', y1='y1')
        return (data, mapping, style)



class BoxesPlot(GeomMixin, LegendPlot, ColorbarPlot):

    style_opts = ['cmap', 'visible'] + line_properties + fill_properties
    _plot_methods = dict(single='rect')
    _batched_style_opts = line_properties + fill_properties
    _color_style = 'fill_color'

    def get_data(self, element, ranges, style):
        x0, y0, x1, y1 = (element.dimension_values(kd) for kd in element.kdims)
        x0, x1 = np.min([x0, x1], axis=0), np.max([x0, x1], axis=0)
        y0, y1 = np.min([y0, y1], axis=0), np.max([y0, y1], axis=0)
        data = {'x': (x1+x0)/2., 'y': (y1+y0)/2., 'width': x1-x0, 'height': y1-y0}
        if self.invert_axes:
            mapping = {'x': 'y', 'y': 'x', 'width': 'height', 'height': 'width'}
        else:
            mapping = {'x': 'x', 'y': 'y', 'width': 'width', 'height': 'height'}
        return data, mapping, style
