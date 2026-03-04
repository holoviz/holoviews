from datetime import datetime as dt

from bokeh.models.widgets import (
    DateEditor,
    DateFormatter,
    IntEditor,
    NumberEditor,
    NumberFormatter,
    StringEditor,
    StringFormatter,
)

import holoviews as hv
from holoviews.plotting.bokeh.callbacks import CDSCallback
from holoviews.plotting.bokeh.renderer import BokehRenderer
from holoviews.streams import CDSStream, Stream

bokeh_renderer = BokehRenderer.instance(mode='server')


class TestBokehTablePlot:

    def setup_method(self):
        self.previous_backend = hv.Store.current_backend
        hv.Store.current_backend = 'bokeh'

    def teardown_method(self):
        hv.Store.current_backend = self.previous_backend
        bokeh_renderer.last_plot = None

    def test_table_plot(self):
        table = hv.Table(([1, 2, 3], [1., 2., 3.], ['A', 'B', 'C']), ['x', 'y'], 'z')
        plot = bokeh_renderer.get_plot(table)
        dims = table.dimensions()
        formatters = (NumberFormatter, NumberFormatter, StringFormatter)
        editors = (IntEditor, NumberEditor, StringEditor)
        for dim, fmt, edit, column in zip(dims, formatters, editors, plot.state.columns, strict=True):
            assert column.title == dim.pprint_label
            assert isinstance(column.formatter, fmt)
            assert isinstance(column.editor, edit)

    def test_table_plot_escaped_dimension(self):
        table = hv.Table([1, 2, 3], ['A Dimension'])
        plot = bokeh_renderer.get_plot(table)
        source = plot.handles['source']
        renderer = plot.handles['glyph_renderer']
        assert next(iter(source.data.keys())) == renderer.columns[0].field

    def test_table_plot_datetimes(self):
        table = hv.Table([dt.now(), dt.now()], 'Date')
        plot = bokeh_renderer.get_plot(table)
        column = plot.state.columns[0]
        assert column.title == 'Date'
        assert isinstance(column.formatter, DateFormatter)
        assert isinstance(column.editor, DateEditor)

    def test_table_plot_callback(self):
        table = hv.Table(([1, 2, 3], [1., 2., 3.], ['A', 'B', 'C']), ['x', 'y'], 'z')
        CDSStream(source=table)
        plot = bokeh_renderer.get_plot(table)
        assert len(plot.callbacks) == 1
        assert isinstance(plot.callbacks[0], CDSCallback)

    def test_table_change_columns(self):
        lengths = {'a': 1, 'b': 2, 'c': 3}
        table = hv.DynamicMap(lambda a: hv.Table(range(lengths[a]), a), kdims=['a']).redim.values(a=['a', 'b', 'c'])
        plot = bokeh_renderer.get_plot(table)
        assert sorted(plot.handles['source'].data.keys()) == ['a']
        assert plot.handles['table'].columns[0].title == 'a'
        plot.update(('b',))
        assert sorted(plot.handles['source'].data.keys()) == ['b']
        assert plot.handles['table'].columns[0].title == 'b'

    def test_table_selected(self):
        table = hv.Table([(0, 0), (1, 1), (2, 2)], ['x', 'y']).opts(selected=[0, 2])
        plot = bokeh_renderer.get_plot(table)
        cds = plot.handles['cds']
        assert cds.selected.indices == [0, 2]

    def test_table_update_selected(self):
        stream = Stream.define('Selected', selected=[])()
        table = hv.Table([(0, 0), (1, 1), (2, 2)], ['x', 'y']).apply.opts(selected=stream.param.selected)
        plot = bokeh_renderer.get_plot(table)
        cds = plot.handles['cds']
        assert cds.selected.indices == []
        stream.event(selected=[0, 2])
        assert cds.selected.indices == [0, 2]
