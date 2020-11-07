import time
import threading

from unittest import SkipTest
from threading import Event

import param

from holoviews.core.spaces import DynamicMap
from holoviews.core.options import Store
from holoviews.element import Curve, Polygons, Path, HLine
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import Renderer
from holoviews.streams import Stream, RangeXY, PlotReset

try:
    from bokeh.application.handlers import FunctionHandler
    from bokeh.application import Application
    from bokeh.client import pull_session
    from bokeh.document import Document
    from bokeh.io.doc import curdoc, set_curdoc
    from bokeh.models import ColumnDataSource
    from bokeh.server.server import Server

    from holoviews.plotting.bokeh.callbacks import (
        Callback, RangeXYCallback, ResetCallback
    )
    from holoviews.plotting.bokeh.renderer import BokehRenderer
    from panel.widgets import DiscreteSlider, FloatSlider
    from panel.io.server import StoppableThread
    from panel.io.state import state
    bokeh_renderer = BokehRenderer.instance(mode='server')
except:
    bokeh_renderer = None


class TestBokehServerSetup(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        if not bokeh_renderer:
            raise SkipTest("Bokeh required to test plot instantiation")
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
        x_range = bokeh_renderer.last_plot.handles['x_range']
        self.assertIn(cb.on_change, x_range._callbacks['start'])
        self.assertIn(cb.on_change, x_range._callbacks['end'])
        y_range = bokeh_renderer.last_plot.handles['y_range']
        self.assertIn(cb.on_change, y_range._callbacks['start'])
        self.assertIn(cb.on_change, y_range._callbacks['end'])

    def test_set_up_linked_event_stream_on_server_doc(self):
        obj = Curve([])
        stream = PlotReset(source=obj)
        server_doc = bokeh_renderer.server_doc(obj)
        self.assertIsInstance(server_doc, Document)
        cb = bokeh_renderer.last_plot.callbacks[0]
        self.assertIsInstance(cb, ResetCallback)
        self.assertEqual(cb.streams, [stream])
        plot = bokeh_renderer.last_plot.state
        self.assertIn(cb.on_event, plot._event_callbacks['reset'])



class TestBokehServer(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        if not bokeh_renderer:
            raise SkipTest("Bokeh required to test plot instantiation")
        Store.current_backend = 'bokeh'
        self._loaded = Event()
        self._port = None
        self._thread = None
        self._server = None

    def tearDown(self):
        Store.current_backend = self.previous_backend
        Callback._callbacks = {}
        if self._thread is not None:
            try:
                self._thread.stop()
            except:
                pass
        state._thread_id = None
        if self._server is not None:
            try:
                self._server.stop()
            except:
                pass
        time.sleep(1)

    def _launcher(self, obj, threaded=False, io_loop=None):
        if io_loop:
            io_loop.make_current()
        launched = []
        def modify_doc(doc):
            bokeh_renderer(obj, doc=doc)
            launched.append(True)
        handler = FunctionHandler(modify_doc)
        app = Application(handler)
        server = Server({'/': app}, port=0, io_loop=io_loop)
        server.start()
        self._port = server.port
        self._server = server
        if threaded:
            server.io_loop.add_callback(self._loaded.set)
            thread = threading.current_thread()
            state._thread_id = thread.ident if thread else None
            io_loop.start()
        else:
            url = "http://localhost:" + str(server.port) + "/"
            session = pull_session(session_id='Test', url=url, io_loop=server.io_loop)
            self.assertTrue(len(launched)==1)
            return session, server
        return None, server

    def _threaded_launcher(self, obj):
        from tornado.ioloop import IOLoop
        io_loop = IOLoop()
        thread = StoppableThread(target=self._launcher, io_loop=io_loop,
                                 args=(obj, True, io_loop))
        thread.setDaemon(True)
        thread.start()
        self._loaded.wait()
        self._thread = thread
        return self.session

    @property
    def session(self):
        url = "http://localhost:" + str(self._port) + "/"
        return pull_session(session_id='Test', url=url)

    def test_launch_simple_server(self):
        obj = Curve([])
        self._launcher(obj)

    def test_launch_server_with_stream(self):
        obj = Curve([])
        stream = RangeXY(source=obj)

        _, server = self._launcher(obj)
        cb = bokeh_renderer.last_plot.callbacks[0]
        self.assertIsInstance(cb, RangeXYCallback)
        self.assertEqual(cb.streams, [stream])
        x_range = bokeh_renderer.last_plot.handles['x_range']
        self.assertIn(cb.on_change, x_range._callbacks['start'])
        self.assertIn(cb.on_change, x_range._callbacks['end'])
        y_range = bokeh_renderer.last_plot.handles['y_range']
        self.assertIn(cb.on_change, y_range._callbacks['start'])
        self.assertIn(cb.on_change, y_range._callbacks['end'])
        server.stop()

    def test_launch_server_with_complex_plot(self):
        dmap = DynamicMap(lambda x_range, y_range: Curve([]), streams=[RangeXY()])
        overlay = dmap * HLine(0)
        static = Polygons([]) * Path([]) * Curve([])
        layout = overlay + static

        _, server = self._launcher(layout)
        server.stop()

    def test_server_dynamicmap_with_dims(self):
        dmap = DynamicMap(lambda y: Curve([1, 2, y]), kdims=['y']).redim.range(y=(0.1, 5))
        obj, _ = bokeh_renderer._validate(dmap, None)
        session = self._threaded_launcher(obj)
        [(plot, _)] = obj._plots.values()
        [(doc, _)] = obj.layout._documents.items()

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
        session = self._threaded_launcher(obj)
        [(doc, _)] = obj.layout._documents.items()

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
        session = self._threaded_launcher(obj)
        [(doc, _)] = obj.layout._documents.items()

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
