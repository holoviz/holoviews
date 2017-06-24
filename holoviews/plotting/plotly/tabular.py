import param
from plotly.figure_factory import create_table
from .element import ElementPlot


class TablePlot(ElementPlot):

    height = param.Number(default=400)

    width = param.Number(default=400)

    def get_data(self, element, ranges):
        headings = [[d.pprint_label for d in element.dimensions()]]
        data = list(zip(*((d.pprint_value(v) for v in element.dimension_values(d))
                        for d in element.dimensions())))
        return (headings+data,), {}

    def init_graph(self, plot_args, plot_kwargs):
        return create_table(*plot_args, **plot_kwargs)


    def graph_options(self, element, ranges):
        self.style = self.lookup_options(element, 'style')
        properties = self.style[self.cyclic_index]
        return properties


    def init_layout(self, key, element, ranges):
        return dict(width=self.width, height=self.height,
                    title=self._format_title(key, separator=' '),
                    plot_bgcolor=self.bgcolor)
