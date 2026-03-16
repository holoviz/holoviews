import holoviews as hv

from .utils import IPythonCase


class TestDisplayHooks(IPythonCase):
    def setup_method(self):
        super().setup_method()
        from holoviews.ipython import notebook_extension

        if not notebook_extension._loaded:
            notebook_extension("matplotlib", ip=self.ip)
        self.backup = hv.Store.display_formats
        hv.Store.display_formats = self.format

    def teardown_method(self):
        from holoviews.ipython import notebook_extension

        self.ip.run_line_magic("unload_ext", "holoviews.ipython")
        hv.Store.display_hooks = self.backup
        notebook_extension._loaded = False
        super().teardown_method()


class TestHTMLDisplay(TestDisplayHooks):
    format = ["html"]

    def test_store_render_html(self):
        curve = hv.Curve([1, 2, 3])
        data, _metadata = hv.Store.render(curve)
        mime_types = {"text/html"}
        assert set(data) == mime_types


class TestPNGDisplay(TestDisplayHooks):
    format = ["png"]

    def test_store_render_png(self):
        curve = hv.Curve([1, 2, 3])
        data, _metadata = hv.Store.render(curve)
        mime_types = {"image/png"}
        assert set(data) == mime_types


class TestSVGDisplay(TestDisplayHooks):
    format = ["svg"]

    def test_store_render_svg(self):
        curve = hv.Curve([1, 2, 3])
        data, _metadata = hv.Store.render(curve)
        mime_types = {"image/svg+xml"}
        assert set(data) == mime_types


class TestCombinedDisplay(TestDisplayHooks):
    format = ["html", "svg", "png"]

    def test_store_render_combined(self):
        curve = hv.Curve([1, 2, 3])
        data, _metadata = hv.Store.render(curve)
        mime_types = {"text/html", "image/svg+xml", "image/png"}
        assert set(data) == mime_types
