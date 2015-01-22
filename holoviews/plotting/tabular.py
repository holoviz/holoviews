from matplotlib.font_manager import FontProperties
from matplotlib.table import Table as mpl_Table

import param

from ..element import ItemTable, Table
from .plot import Plot


class TablePlot(Plot):
    """
    A TablePlot can plot both TableViews and ViewMaps which display
    as either a single static table or as an animated table
    respectively.
    """

    border = param.Number(default=0.05, bounds=(0.0, 0.5), doc="""
        The fraction of the plot that should be empty around the
        edges.""")

    float_precision = param.Integer(default=3, doc="""
        The floating point precision to use when printing float
        numeric data types.""")

    max_value_len = param.Integer(default=20, doc="""
        The maximum allowable string length of a value shown in any
        table cell. Any strings longer than this length will be
        truncated.""")

    max_font_size = param.Integer(default=12, doc="""
        The largest allowable font size for the text in each table
        cell.""")

    max_rows = param.Integer(default=15, doc="""
        The maximum number of Table rows before the table is
        summarized.""")

    font_types = param.Dict(default={'heading': FontProperties(weight='bold',
                                                               family='monospace')},
       doc="""The font style used for heading labels used for emphasis.""")

    style_opts = param.List(default=[], constant=True, doc="""
     TablePlot has specialized options which are controlled via plot
     options instead of matplotlib options.""")

    # Disable computing plot bounds from data.
    apply_databounds = False

    def pprint_value(self, value):
        """
        Generate the pretty printed representation of a value for
        inclusion in a table cell.
        """
        if isinstance(value, float):
            formatter = '{:.%df}' % self.float_precision
            formatted = formatter.format(value)
        else:
            formatted = str(value)

        if len(formatted) > self.max_value_len:
            return formatted[:(self.max_value_len-3)]+'...'
        else:
            return formatted


    def __call__(self, axis=None, cyclic_index=0, lbrt=None):
        tableview = self._map.last
        self.ax = self._init_axis(axis)

        self.ax.set_axis_off()
        size_factor = (1.0 - 2*self.border)
        table = mpl_Table(self.ax, bbox=[self.border, self.border,
                                         size_factor, size_factor])

        width = size_factor / tableview.cols
        height = size_factor / tableview.rows

        # Mapping from the cell coordinates to the dictionary key.
        summarize = tableview.rows > self.max_rows
        half_rows = self.max_rows/2
        rows = min([self.max_rows, tableview.rows])
        for row in range(rows):
            adjusted_row = row
            for col in range(tableview.cols):
                if summarize and row == half_rows:
                    cell_text = "..."
                else:
                    if summarize and row > half_rows:
                        adjusted_row = (tableview.rows - self.max_rows + row)
                    value = tableview.cell_value(adjusted_row, col)
                    cell_text = self.pprint_value(value)
                cellfont = self.font_types.get(tableview.cell_type(adjusted_row,col), None)
                font_kwargs = dict(fontproperties=cellfont) if cellfont else {}
                table.add_cell(row, col, width, height, text=cell_text,  loc='center',
                               **font_kwargs)

        table.set_fontsize(self.max_font_size)
        table.auto_set_font_size(True)
        self.ax.add_table(table)

        self.handles['table'] = table

        return self._finalize_axis(self._keys[-1])


    def update_handles(self, view, key, lbrt=None):
        table = self.handles['table']

        for coords, cell in table.get_celld().items():
            value = view.cell_value(*coords)
            cell.set_text_props(text=self.pprint_value(value))

        # Resize fonts across table as necessary
        table.set_fontsize(self.max_font_size)
        table.auto_set_font_size(True)

Plot.defaults.update({ItemTable: TablePlot,
                      Table: TablePlot})