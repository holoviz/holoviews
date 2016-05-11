from bokeh.models.widgets import DataTable, TableColumn

import param

from .plot import BokehPlot
from ..plot import GenericElementPlot


class TablePlot(BokehPlot, GenericElementPlot):

    height = param.Number(default=None)

    width = param.Number(default=400)

    style_opts = ['row_headers', 'selectable', 'editable',
                  'sortable', 'fit_columns', 'width', 'height']

    def get_data(self, element, ranges=None, empty=False):
        dims = element.dimensions()
        return ({d.name: [] if empty else element.dimension_values(d) for d in dims},
                {d.name: d.name for d in dims})


    def initialize_plot(self, ranges=None, plot=None, plots=None, source=None):
        """
        Initializes a new plot object with the last available frame.
        """
        # Get element key and ranges for frame
        element = self.hmap.last
        key = self.keys[-1]
        self.current_frame = element
        self.current_key = key

        data, _ = self.get_data(element, ranges)
        if source is None:
            source = self._init_datasource(data)
        self.handles['source'] = source

        dims = element.dimensions()
        columns = [TableColumn(field=d.name, title=str(d)) for d in dims]
        properties = self.lookup_options(element, 'style')[self.cyclic_index]
        table = DataTable(source=source, columns=columns, height=self.height,
                          width=self.width, **properties)
        self.handles['plot'] = table
        self.handles['glyph_renderer'] = table
        self.drawn = True

        return table


    def update_frame(self, key, ranges=None, plot=None):
        """
        Updates an existing plot with data corresponding
        to the key.
        """
        element = self._get_frame(key)
        source = self.handles['source']
        data, _ = self.get_data(element, ranges)
        self._update_datasource(source, data)
