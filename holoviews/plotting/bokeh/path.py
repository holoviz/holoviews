from .element import ElementPlot, line_properties, fill_properties


class PathPlot(ElementPlot):

    style_opts = ['color'] + line_properties
    _plot_method = 'multi_line'
    _mapping = dict(xs='xs', ys='ys')

    def get_data(self, element, ranges=None):
        xs = [path[:, 0] for path in element.data]
        ys = [path[:, 1] for path in element.data]
        return dict(xs=xs, ys=ys), self._mapping


class PolygonPlot(PathPlot):

    style_opts = ['color'] + line_properties + fill_properties
    _plot_method = 'patches'
