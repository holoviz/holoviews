from datetime import datetime as dt

from bokeh.models.widgets import (
        NumberEditor, NumberFormatter, DateFormatter,
    DateEditor, StringFormatter, StringEditor, IntEditor
)
from holoviews.core.options import Store
from holoviews.core.spaces import DynamicMap
from holoviews.element import Table
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.callbacks import CDSCallback
from holoviews.plotting.bokeh.renderer import BokehRenderer
from holoviews.streams import CDSStream, Stream

bokeh_renderer = BokehRenderer.instance(mode='server')


class TestBokehTablePlot(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
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

    def test_table_change_columns(self):
        lengths = {'a': 1, 'b': 2, 'c': 3}
        table = DynamicMap(lambda a: Table(range(lengths[a]), a), kdims=['a']).redim.values(a=['a', 'b', 'c'])
        plot = bokeh_renderer.get_plot(table)
        self.assertEqual(sorted(plot.handles['source'].data.keys()), ['a'])
        self.assertEqual(plot.handles['table'].columns[0].title, 'a')
        plot.update(('b',))
        self.assertEqual(sorted(plot.handles['source'].data.keys()), ['b'])
        self.assertEqual(plot.handles['table'].columns[0].title, 'b')

    def test_table_selected(self):
        table = Table([(0, 0), (1, 1), (2, 2)], ['x', 'y']).opts(selected=[0, 2])
        plot = bokeh_renderer.get_plot(table)
        cds = plot.handles['cds']
        self.assertEqual(cds.selected.indices, [0, 2])

    def test_table_update_selected(self):
        stream = Stream.define('Selected', selected=[])()
        table = Table([(0, 0), (1, 1), (2, 2)], ['x', 'y']).apply.opts(selected=stream.param.selected)
        plot = bokeh_renderer.get_plot(table)
        cds = plot.handles['cds']
        self.assertEqual(cds.selected.indices, [])
        stream.event(selected=[0, 2])
        self.assertEqual(cds.selected.indices, [0, 2])
