from unittest import SkipTest

from holoviews.core.spaces import DynamicMap
from holoviews.core.options import Store
from holoviews.element import Curve, Polygons, Path, HLine
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import Renderer
from holoviews.streams import RangeXY, PlotReset

try:
    from bokeh.application.handlers import FunctionHandler
    from bokeh.application import Application
    from bokeh.client import pull_session
    from bokeh.document import Document
    from bokeh.io import curdoc
    from bokeh.server.server import Server

    from holoviews.plotting.bokeh.callbacks import (
        Callback, RangeXYCallback, ResetCallback
    )
    from holoviews.plotting.bokeh.renderer import BokehRenderer
    bokeh_renderer = BokehRenderer.instance(mode='server')
except:
    bokeh_renderer = None


class TestBokehServerSetup(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        if not bokeh_renderer:
            raise SkipTest("Bokeh required to test plot instantiation")
        Store.current_backend = 'bokeh'
        self.nbcontext = Renderer.notebook_context 
        Renderer.notebook_context = False

    def tearDown(self):
        Store.current_backend = self.previous_backend
        bokeh_renderer.last_plot = None
        Callback._callbacks = {}
        Renderer.notebook_context = self.nbcontext

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



class TestBokehServerRun(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        if not bokeh_renderer:
            raise SkipTest("Bokeh required to test plot instantiation")
        Store.current_backend = 'bokeh'

    def tearDown(self):
        Store.current_backend = self.previous_backend
        Callback._callbacks = {}

    def test_launch_simple_server(self):
        obj = Curve([])
        launched = []
        def modify_doc(doc):
            bokeh_renderer(obj, doc=doc)
            launched.append(True)
            server.stop()
        handler = FunctionHandler(modify_doc)
        app = Application(handler)
        server = Server({'/': app}, port=0)
        server.start()
        url = "http://localhost:" + str(server.port) + "/"
        pull_session(session_id='Test', url=url, io_loop=server.io_loop)
        self.assertTrue(len(launched)==1)

    def test_launch_server_with_stream(self):
        obj = Curve([])
        stream = RangeXY(source=obj)

        launched = []
        def modify_doc(doc):
            bokeh_renderer(obj, doc=doc)
            launched.append(True)
            server.stop()
        handler = FunctionHandler(modify_doc)
        app = Application(handler)
        server = Server({'/': app}, port=0)
        server.start()
        url = "http://localhost:" + str(server.port) + "/"
        pull_session(session_id='Test', url=url, io_loop=server.io_loop)

        self.assertTrue(len(launched)==1)
        cb = bokeh_renderer.last_plot.callbacks[0]
        self.assertIsInstance(cb, RangeXYCallback)
        self.assertEqual(cb.streams, [stream])
        x_range = bokeh_renderer.last_plot.handles['x_range']
        self.assertIn(cb.on_change, x_range._callbacks['start'])
        self.assertIn(cb.on_change, x_range._callbacks['end'])
        y_range = bokeh_renderer.last_plot.handles['y_range']
        self.assertIn(cb.on_change, y_range._callbacks['start'])
        self.assertIn(cb.on_change, y_range._callbacks['end'])


    def test_launch_server_with_complex_plot(self):
        dmap = DynamicMap(lambda x_range, y_range: Curve([]), streams=[RangeXY()])
        overlay = dmap * HLine(0)
        static = Polygons([]) * Path([]) * Curve([])
        layout = overlay + static

        launched = []
        def modify_doc(doc):
            bokeh_renderer(layout, doc=doc)
            launched.append(True)
            server.stop()
        handler = FunctionHandler(modify_doc)
        app = Application(handler)
        server = Server({'/': app}, port=0)
        server.start()
        url = "http://localhost:" + str(server.port) + "/"
        pull_session(session_id='Test', url=url, io_loop=server.io_loop)
        self.assertTrue(len(launched)==1)
