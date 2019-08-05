from __future__ import absolute_import, division, unicode_literals

from .element import ElementPlot


class ShapePlot(ElementPlot):

    # The plotly shape type ("line", "rect", etc.)
    _shape_type = None
    style_opts = ['opacity', 'fillcolor', 'line_color', 'line_width', 'line_dash']

    apply_ranges = False

    def init_graph(self, datum, options, index=0):
        shape = dict(type=self._shape_type, **datum, **options)
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


class PathShapePlot(ShapePlot):
    _shape_type = 'path'
    apply_ranges = True

    def get_data(self, element, ranges, style):
        if self.invert_axes:
            ys = element.dimension_values(0)
            xs = element.dimension_values(1)
        else:
            xs = element.dimension_values(0)
            ys = element.dimension_values(1)

        path = ShapePlot.build_path(xs, ys)
        return [dict(path=path, xref='x', yref='y')]


class HLinePlot(ShapePlot):
    _shape_type = 'line'

    def get_data(self, element, ranges, style):
        if self.invert_axes:
            x = element.y
            return [dict(x0=x, x1=x, y0=0, y1=1, xref='x', yref="paper")]
        else:
            y = element.y
            return [dict(x0=0.0, x1=1.0, y0=y, y1=y, xref="paper", yref='y')]


class VLinePlot(ShapePlot):
    _shape_type = 'line'

    def get_data(self, element, ranges, style):
        if self.invert_axes:
            y = element.x
            return [dict(x0=0.0, x1=1.0, y0=y, y1=y, xref="paper", yref='y')]
        else:
            x = element.x
            return [dict(x0=x, x1=x, y0=0, y1=1, xref='x', yref="paper")]
