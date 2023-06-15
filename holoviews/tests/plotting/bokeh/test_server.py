import time

import param
import pytest

from holoviews.core.spaces import DynamicMap
from holoviews.core.options import Store
from holoviews.element import Curve, Polygons, Path, HLine
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import Renderer
from holoviews.streams import Stream, RangeXY, PlotReset

from bokeh.client import pull_session
from bokeh.document import Document
from bokeh.io.doc import curdoc, set_curdoc
from bokeh.models import ColumnDataSource

from holoviews.plotting.bokeh.callbacks import (
    Callback, RangeXYCallback, ResetCallback
)
from holoviews.plotting.bokeh.renderer import BokehRenderer
from panel.widgets import DiscreteSlider, FloatSlider
from panel.io.state import state
from panel import serve

bokeh_renderer = BokehRenderer.instance(mode='server')


class TestBokehServerSetup(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        Store.current_backend = 'bokeh'
        self.doc = curdoc()
        set_curdoc(Document())
        self.nbcontext = Renderer.notebook_context
        with param.logging_level('ERROR'):
            Renderer.notebook_context = False

    def tearDown(self):
        Store.current_backend = self.previous_backend
        bokeh_renderer.last_plot = None
        Callback._callbacks = {}
        with param.logging_level('ERROR'):
            Renderer.notebook_context = self.nbcontext
        state.curdoc = None
        curdoc().clear()
        set_curdoc(self.doc)
        time.sleep(1)

    def test_render_server_doc_element(self):
        obj = Curve([])
        doc = bokeh_renderer.server_doc(obj)
        self.assertIs(doc, curdoc())
        self.assertIs(bokeh_renderer.last_plot.document, curdoc())

    def test_render_explicit_server_doc_element(self):
        obj = Curve([])
        doc = Document()
        server_doc = bokeh_renderer.server_doc(obj, doc)
        self.assertIs(server_doc, doc)
        self.assertIs(bokeh_renderer.last_plot.document, doc)

    def test_set_up_linked_change_stream_on_server_doc(self):
        obj = Curve([])
        stream = RangeXY(source=obj)
        server_doc = bokeh_renderer.server_doc(obj)
        self.assertIsInstance(server_doc, Document)
        self.assertEqual(len(bokeh_renderer.last_plot.callbacks), 1)
        cb = bokeh_renderer.last_plot.callbacks[0]
        self.assertIsInstance(cb, RangeXYCallback)
        self.assertEqual(cb.streams, [stream])
        assert 'rangesupdate' in bokeh_renderer.last_plot.state._event_callbacks

    def test_set_up_linked_event_stream_on_server_doc(self):
        obj = Curve([])
        stream = PlotReset(source=obj)
        server_doc = bokeh_renderer.server_doc(obj)
        self.assertIsInstance(server_doc, Document)
        cb = bokeh_renderer.last_plot.callbacks[0]
        self.assertIsInstance(cb, ResetCallback)
        self.assertEqual(cb.streams, [stream])



class TestBokehServer(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        Store.current_backend = 'bokeh'
        self._port = None

    def tearDown(self):
        Store.current_backend = self.previous_backend
        Callback._callbacks = {}
        state.kill_all_servers()
        time.sleep(1)

    def _launcher(self, obj, threaded=True, port=6001):
        self._port = port
        server = serve(obj, threaded=threaded, show=False, port=port)
        time.sleep(0.5)
        return server, self.session

    @property
    def session(self):
        url = "http://localhost:" + str(self._port) + "/"
        return pull_session(session_id='Test', url=url)

    def test_launch_simple_server(self):
        obj = Curve([])
        self._launcher(obj, port=6001)

    def test_launch_server_with_stream(self):
        el = Curve([])
        stream = RangeXY(source=el)

        obj, _ = bokeh_renderer._validate(el, None)
        server, _ = self._launcher(obj, port=6002)
        [(plot, _)] = obj._plots.values()

        cb = plot.callbacks[0]
        self.assertIsInstance(cb, RangeXYCallback)
        self.assertEqual(cb.streams, [stream])
        assert 'rangesupdate' in plot.state._event_callbacks

    @pytest.mark.flaky(max_runs=3)
    def test_launch_server_with_complex_plot(self):
        dmap = DynamicMap(lambda x_range, y_range: Curve([]), streams=[RangeXY()])
        overlay = dmap * HLine(0)
        static = Polygons([]) * Path([]) * Curve([])
        layout = overlay + static

        self._launcher(layout, port=6003)

    def test_server_dynamicmap_with_dims(self):
        dmap = DynamicMap(lambda y: Curve([1, 2, y]), kdims=['y']).redim.range(y=(0.1, 5))
        obj, _ = bokeh_renderer._validate(dmap, None)
        _, session = self._launcher(obj, port=6004)
        [(plot, _)] = obj._plots.values()
        [(doc, _)] = obj._documents.items()

        cds = session.document.roots[0].select_one({'type': ColumnDataSource})
        self.assertEqual(cds.data['y'][2], 0.1)
        slider = obj.layout.select(FloatSlider)[0]
        def run():
            slider.value = 3.1
        doc.add_next_tick_callback(run)
        time.sleep(1)
        cds = self.session.document.roots[0].select_one({'type': ColumnDataSource})
        self.assertEqual(cds.data['y'][2], 3.1)

    def test_server_dynamicmap_with_stream(self):
        stream = Stream.define('Custom', y=2)()
        dmap = DynamicMap(lambda y: Curve([1, 2, y]), kdims=['y'], streams=[stream])
        obj, _ = bokeh_renderer._validate(dmap, None)
        _, session = self._launcher(obj, port=6005)
        [(doc, _)] = obj._documents.items()

        cds = session.document.roots[0].select_one({'type': ColumnDataSource})
        self.assertEqual(cds.data['y'][2], 2)
        def run():
            stream.event(y=3)
        doc.add_next_tick_callback(run)
        time.sleep(1)
        cds = self.session.document.roots[0].select_one({'type': ColumnDataSource})
        self.assertEqual(cds.data['y'][2], 3)

    def test_server_dynamicmap_with_stream_dims(self):
        stream = Stream.define('Custom', y=2)()
        dmap = DynamicMap(lambda x, y: Curve([x, 1, y]), kdims=['x', 'y'],
                          streams=[stream]).redim.values(x=[1, 2, 3])
        obj, _ = bokeh_renderer._validate(dmap, None)
        _, session = self._launcher(obj, port=6006)
        [(doc, _)] = obj._documents.items()

        orig_cds = session.document.roots[0].select_one({'type': ColumnDataSource})
        self.assertEqual(orig_cds.data['y'][2], 2)
        def run():
            stream.event(y=3)
        doc.add_next_tick_callback(run)
        time.sleep(1)
        cds = self.session.document.roots[0].select_one({'type': ColumnDataSource})
        self.assertEqual(cds.data['y'][2], 3)

        self.assertEqual(orig_cds.data['y'][0], 1)
        slider = obj.layout.select(DiscreteSlider)[0]
        def run():
            slider.value = 3
        doc.add_next_tick_callback(run)
        time.sleep(1)
        cds = self.session.document.roots[0].select_one({'type': ColumnDataSource})
        self.assertEqual(cds.data['y'][0], 3)
