from __future__ import absolute_import, division, unicode_literals

import param

from ...element import HLine, VLine, HSpan, VSpan
from ..mixins import GeomMixin
from .element import ElementPlot


class ShapePlot(ElementPlot):

    # The plotly shape type ("line", "rect", etc.)
    _shape_type = None
    style_opts = ['opacity', 'fillcolor', 'line_color', 'line_width', 'line_dash']

    def init_graph(self, datum, options, index=0):
        shape = dict(type=self._shape_type, **dict(datum, **options))
        return dict(shapes=[shape])

    @staticmethod
    def build_path(xs, ys, closed=True):
        line_tos = ''.join(['L{x} {y}'.format(x=x, y=y)
                            for x, y in zip(xs[1:], ys[1:])])
        path = 'M{x0} {y0}{line_tos}'.format(
            x0=xs[0], y0=ys[0], line_tos=line_tos)

        if closed:
            path += 'Z'

        return path


class BoxShapePlot(GeomMixin, ShapePlot):
    _shape_type = 'rect'

    def get_data(self, element, ranges, style):
        inds = (1, 0, 3, 2) if self.invert_axes else (0, 1, 2, 3)
        x0s, y0s, x1s, y1s = (element.dimension_values(kd) for kd in inds)
        return [dict(x0=x0, x1=x1, y0=y0, y1=y1, xref='x', yref='y')
                for (x0, y0, x1, y1) in zip(x0s, y0s, x1s, y1s)]


class SegmentShapePlot(GeomMixin, ShapePlot):
    _shape_type = 'line'

    def get_data(self, element, ranges, style):
        inds = (1, 0, 3, 2) if self.invert_axes else (0, 1, 2, 3)
        x0s, y0s, x1s, y1s = (element.dimension_values(kd) for kd in inds)
        return [dict(x0=x0, x1=x1, y0=y0, y1=y1, xref='x', yref='y')
                for (x0, y0, x1, y1) in zip(x0s, y0s, x1s, y1s)]
    

class PathShapePlot(ShapePlot):
    _shape_type = 'path'

    def get_data(self, element, ranges, style):
        if self.invert_axes:
            ys = element.dimension_values(0)
            xs = element.dimension_values(1)
        else:
            xs = element.dimension_values(0)
            ys = element.dimension_values(1)

        path = ShapePlot.build_path(xs, ys)
        return [dict(path=path, xref='x', yref='y')]


class PathsPlot(ShapePlot):
    _shape_type = 'path'

    def get_data(self, element, ranges, style):
        paths = []
        for el in element.split():
            xdim, ydim = (1, 0) if self.invert_axes else (0, 1)
            xs = el.dimension_values(xdim)
            ys = el.dimension_values(ydim)
            path = ShapePlot.build_path(xs, ys)
            paths.append(dict(path=path, xref='x', yref='y'))
        return paths


class HVLinePlot(ShapePlot):

    apply_ranges = param.Boolean(default=False, doc="""
        Whether to include the annotation in axis range calculations.""")
    
    _shape_type = 'line'

    def get_data(self, element, ranges, style):
        if ((isinstance(element, HLine) and self.invert_axes) or
            (isinstance(element, VLine) and not self.invert_axes)):
            x = element.data
            visible = x is not None
            return [dict(
                x0=x, x1=x, y0=0, y1=1, xref='x', yref="paper", visible=visible
            )]
        else:
            y = element.data
            visible = y is not None
            return [dict(
                x0=0.0, x1=1.0, y0=y, y1=y, xref="paper", yref='y', visible=visible
            )]


class HVSpanPlot(ShapePlot):
    
    apply_ranges = param.Boolean(default=False, doc="""
        Whether to include the annotation in axis range calculations.""")

    _shape_type = 'rect'

    def get_data(self, element, ranges, style):
        
        if ((isinstance(element, HSpan) and self.invert_axes) or
            (isinstance(element, VSpan) and not self.invert_axes)):
            x0, x1 = element.data
            visible = not (x0 is None and x1 is None)
            return [dict(
                x0=x0, x1=x1, y0=0, y1=1, xref='x', yref="paper", visible=visible
            )]
        else:
            y0, y1 = element.data
            visible = not (y0 is None and y1 is None)
            return [dict(
                x0=0.0, x1=1.0, y0=y0, y1=y1, xref="paper", yref='y', visible=visible
            )]
