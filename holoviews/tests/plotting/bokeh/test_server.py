import asyncio
import socket
import time

import param
from bokeh.client import pull_session
from bokeh.document import Document
from bokeh.io.doc import curdoc, set_curdoc
from bokeh.models import ColumnDataSource
from panel import serve
from panel.io.state import state
from panel.widgets import DiscreteSlider, FloatSlider

import holoviews as hv
from holoviews.plotting import Renderer
from holoviews.plotting.bokeh.callbacks import Callback, RangeXYCallback, ResetCallback
from holoviews.plotting.bokeh.renderer import BokehRenderer
from holoviews.plotting.bokeh.util import BOKEH_GE_3_8_0
from holoviews.streams import PlotReset, RangeXY, Stream

bokeh_renderer = BokehRenderer.instance(mode="server")


def _wait_for_port(port, *, connected, timeout=10):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.1)
            is_connected = sock.connect_ex(("localhost", port)) == 0
        if is_connected == connected:
            return
        time.sleep(0.05)
    state = "open" if connected else "closed"
    raise TimeoutError(f"Port {port} never became {state}")


class TestBokehServerSetup:
    def setup_method(self):
        self.previous_backend = hv.Store.current_backend
        hv.Store.current_backend = "bokeh"
        self.doc = curdoc()
        set_curdoc(Document())
        self.nbcontext = Renderer.notebook_context
        with param.logging_level("ERROR"):
            Renderer.notebook_context = False

    def teardown_method(self):
        hv.Store.current_backend = self.previous_backend
        bokeh_renderer.last_plot = None
        Callback._callbacks = {}
        with param.logging_level("ERROR"):
            Renderer.notebook_context = self.nbcontext
        state.curdoc = None
        curdoc().clear()
        set_curdoc(self.doc)

    def test_render_server_doc_element(self):
        obj = hv.Curve([])
        doc = bokeh_renderer.server_doc(obj)
        if not BOKEH_GE_3_8_0:
            # Updating the config which is introduced in Bokeh 3.8 changes the curdoc()
            # Something like this is done in Panel:
            # curdoc().config.update(notifications=None)
            assert doc == curdoc()
        assert bokeh_renderer.last_plot.document == doc

    def test_render_explicit_server_doc_element(self):
        obj = hv.Curve([])
        doc = Document()
        server_doc = bokeh_renderer.server_doc(obj, doc)
        assert server_doc is doc
        assert bokeh_renderer.last_plot.document is doc

    def test_set_up_linked_change_stream_on_server_doc(self):
        obj = hv.Curve([])
        stream = RangeXY(source=obj)
        server_doc = bokeh_renderer.server_doc(obj)
        assert isinstance(server_doc, Document)
        assert len(bokeh_renderer.last_plot.callbacks) == 1
        cb = bokeh_renderer.last_plot.callbacks[0]
        assert isinstance(cb, RangeXYCallback)
        assert cb.streams == [stream]
        assert "rangesupdate" in bokeh_renderer.last_plot.state._event_callbacks

    def test_set_up_linked_event_stream_on_server_doc(self):
        obj = hv.Curve([])
        stream = PlotReset(source=obj)
        server_doc = bokeh_renderer.server_doc(obj)
        assert isinstance(server_doc, Document)
        cb = bokeh_renderer.last_plot.callbacks[0]
        assert isinstance(cb, ResetCallback)
        assert cb.streams == [stream]


class TestBokehServer:
    def setup_method(self):
        self.previous_backend = hv.Store.current_backend
        hv.Store.current_backend = "bokeh"
        self._port = None

    def teardown_method(self):
        hv.Store.current_backend = self.previous_backend
        Callback._callbacks = {}
        state.kill_all_servers()
        if self._port is not None:
            _wait_for_port(self._port, connected=False)

    def _wait_and_assert_cds_value(self, *, key, index, expected, timeout=10):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            cds = self.session.document.roots[0].select_one({"type": ColumnDataSource})
            if cds.data[key][index] == expected:
                return
            time.sleep(0.05)
        cds = self.session.document.roots[0].select_one({"type": ColumnDataSource})
        assert cds.data[key][index] == expected

    def _launcher(self, obj, threaded=True, port=6001):
        try:
            # In Python 3.12 this will raise a:
            # `DeprecationWarning: There is no current event loop`
            asyncio.get_event_loop()
        except Exception:
            asyncio.set_event_loop(asyncio.new_event_loop())

        self._port = port
        server = serve(obj, threaded=threaded, show=False, port=port)
        _wait_for_port(port, connected=True)
        return server, self.session

    @property
    def session(self):
        url = "http://localhost:" + str(self._port) + "/"
        return pull_session(session_id="Test", url=url)

    def test_launch_simple_server(self):
        obj = hv.Curve([])
        self._launcher(obj, port=6001)

    def test_launch_server_with_stream(self):
        el = hv.Curve([])
        stream = RangeXY(source=el)

        obj, _ = bokeh_renderer._validate(el, None)
        _server, _ = self._launcher(obj, port=6002)
        [(plot, _)] = obj._plots.values()

        cb = plot.callbacks[0]
        assert isinstance(cb, RangeXYCallback)
        assert cb.streams == [stream]
        assert "rangesupdate" in plot.state._event_callbacks

    def test_launch_server_with_complex_plot(self):
        dmap = hv.DynamicMap(lambda x_range, y_range: hv.Curve([]), streams=[RangeXY()])
        overlay = dmap * hv.HLine(0)
        static = hv.Polygons([]) * hv.Path([]) * hv.Curve([])
        layout = overlay + static

        self._launcher(layout, port=6003)

    def test_server_dynamicmap_with_dims(self):
        dmap = hv.DynamicMap(lambda y: hv.Curve([1, 2, y]), kdims=["y"]).redim.range(y=(0.1, 5))
        obj, _ = bokeh_renderer._validate(dmap, None)
        _, session = self._launcher(obj, port=6004)
        [(_plot, _)] = obj._plots.values()
        [(doc, _)] = obj._documents.items()

        cds = session.document.roots[0].select_one({"type": ColumnDataSource})
        assert cds.data["y"][2] == 0.1
        slider = obj.layout.select(FloatSlider)[0]

        def run():
            slider.value = 3.1

        doc.add_next_tick_callback(run)
        self._wait_and_assert_cds_value(key="y", index=2, expected=3.1)

    def test_server_dynamicmap_with_stream(self):
        stream = Stream.define("Custom", y=2)()
        dmap = hv.DynamicMap(lambda y: hv.Curve([1, 2, y]), kdims=["y"], streams=[stream])
        obj, _ = bokeh_renderer._validate(dmap, None)
        _, session = self._launcher(obj, port=6005)
        [(doc, _)] = obj._documents.items()

        cds = session.document.roots[0].select_one({"type": ColumnDataSource})
        assert cds.data["y"][2] == 2

        def loaded():
            state._schedule_on_load(doc, None)

        doc.add_next_tick_callback(loaded)

        def run():
            stream.event(y=3)

        doc.add_next_tick_callback(run)
        self._wait_and_assert_cds_value(key="y", index=2, expected=3)

    def test_server_dynamicmap_with_stream_dims(self):
        stream = Stream.define("Custom", y=2)()
        dmap = hv.DynamicMap(
            lambda x, y: hv.Curve([x, 1, y]), kdims=["x", "y"], streams=[stream]
        ).redim.values(x=[1, 2, 3])
        obj, _ = bokeh_renderer._validate(dmap, None)
        _, session = self._launcher(obj, port=6006)
        [(doc, _)] = obj._documents.items()

        orig_cds = session.document.roots[0].select_one({"type": ColumnDataSource})
        assert orig_cds.data["y"][2] == 2

        def loaded():
            state._schedule_on_load(doc, None)

        doc.add_next_tick_callback(loaded)

        def run():
            stream.event(y=3)

        doc.add_next_tick_callback(run)
        self._wait_and_assert_cds_value(key="y", index=2, expected=3)

        assert orig_cds.data["y"][0] == 1
        slider = obj.layout.select(DiscreteSlider)[0]

        def run():
            slider.value = 3

        doc.add_next_tick_callback(run)
        self._wait_and_assert_cds_value(key="y", index=0, expected=3)
