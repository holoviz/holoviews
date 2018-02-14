from datetime import datetime as dt
from unittest import SkipTest

from holoviews.core.options import Store
from holoviews.element import Table
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import CDSStream

try:
    from bokeh.models.widgets import (
         NumberEditor, NumberFormatter, DateFormatter,
        DateEditor, StringFormatter, StringEditor, IntEditor
    )
    from holoviews.plotting.bokeh.callbacks import CDSCallback
    from holoviews.plotting.bokeh.renderer import BokehRenderer
    bokeh_renderer = BokehRenderer.instance(mode='server')
except:
    bokeh_renderer = None


class TestBokehTablePlot(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        if not bokeh_renderer:
            raise SkipTest("Bokeh required to test plot instantiation")
        Store.current_backend = 'bokeh'

    def tearDown(self):
        Store.current_backend = self.previous_backend
        bokeh_renderer.last_plot = None

    def test_table_plot(self):
        table = Table(([1, 2, 3], [1., 2., 3.], ['A', 'B', 'C']), ['x', 'y'], 'z')
        plot = bokeh_renderer.get_plot(table)
        dims = table.dimensions()
        formatters = (NumberFormatter, NumberFormatter, StringFormatter)
        editors = (IntEditor, NumberEditor, StringEditor)
        for dim, fmt, edit, column in zip(dims, formatters, editors, plot.state.columns):
            self.assertEqual(column.title, dim.pprint_label)
            self.assertIsInstance(column.formatter, fmt)
            self.assertIsInstance(column.editor, edit)

    def test_table_plot_escaped_dimension(self):
        table = Table([1, 2, 3], ['A Dimension'])
        plot = bokeh_renderer.get_plot(table)
        source = plot.handles['source']
        renderer = plot.handles['glyph_renderer']
        self.assertEqual(list(source.data.keys())[0], renderer.columns[0].field)

    def test_table_plot_datetimes(self):
        table = Table([dt.now(), dt.now()], 'Date')
        plot = bokeh_renderer.get_plot(table)
        column = plot.state.columns[0]
        self.assertEqual(column.title, 'Date')
        self.assertIsInstance(column.formatter, DateFormatter)
        self.assertIsInstance(column.editor, DateEditor)

    def test_table_plot_callback(self):
        table = Table(([1, 2, 3], [1., 2., 3.], ['A', 'B', 'C']), ['x', 'y'], 'z')
        CDSStream(source=table)
        plot = bokeh_renderer.get_plot(table)
        self.assertEqual(len(plot.callbacks), 1)
        self.assertIsInstance(plot.callbacks[0], CDSCallback)
