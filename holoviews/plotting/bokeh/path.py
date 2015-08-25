from .element import ElementPlot, line_properties, fill_properties


class PathPlot(ElementPlot):

    style_opts = ['color'] + line_properties

    def get_data(self, element, ranges=None):
        xs = [path[:, 0] for path in element.data]
        ys = [path[:, 1] for path in element.data]
        return dict(xs=xs, ys=ys)

    def _init_glyph(self, element, plot, source, properties):
        plot.multi_line(xs='xs', ys='ys', source=source,
                        legend=element.label, **properties)


class PolygonPlot(PathPlot):

    style_opts = ['color'] + line_properties + fill_properties

    def _init_glyph(self, element, plot, source, properties):
        plot.patches(xs='xs', ys='ys', source=source, legend=element.label,
                     **properties)
