from .element import ElementPlot, line_properties, fill_properties


class PathPlot(ElementPlot):

    style_opts = ['color'] + line_properties

    def get_data(self, element, ranges=None):
        xs = [path[:, 0] for path in element.data]
        ys = [path[:, 1] for path in element.data]
        return dict(xs=xs, ys=ys)

    def init_glyph(self, element, plot, source, ranges):
        paths = plot.multi_line(xs='xs', ys='ys', source=source,
                                legend=element.label, **self.style)
        self.handles['lines'] = paths


class PolygonPlot(PathPlot):

    style_opts = ['color'] + line_properties + fill_properties

    def init_glyph(self, element, plot, source, ranges):
        self.handles['patches'] = plot.patches(xs='xs', ys='ys', source=source,
                                               legend=element.label, **self.style)
