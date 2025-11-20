from holoviews import Curve, Store
from holoviews.ipython import notebook_extension

from .utils import IPythonCase


class TestDisplayHooks(IPythonCase):

    def setup_method(self):
        super().setup_method()
        if not notebook_extension._loaded:
            notebook_extension('matplotlib', ip=self.ip)
        self.backup = Store.display_formats
        Store.display_formats = self.format

    def teardown_method(self):
        self.ip.run_line_magic("unload_ext", "holoviews.ipython")
        Store.display_hooks = self.backup
        notebook_extension._loaded = False
        super().teardown_method()


class TestHTMLDisplay(TestDisplayHooks):

    def setup_method(self):
        self.format = ['html']
        super().setup_method()

    def test_store_render_html(self):
        curve = Curve([1, 2, 3])
        data, _metadata = Store.render(curve)
        mime_types = {'text/html'}
        assert set(data) == mime_types


class TestPNGDisplay(TestDisplayHooks):

    def setup_method(self):
        self.format = ['png']
        super().setup_method()

    def test_store_render_png(self):
        curve = Curve([1, 2, 3])
        data, _metadata = Store.render(curve)
        mime_types = {'image/png'}
        assert set(data) == mime_types


class TestSVGDisplay(TestDisplayHooks):

    def setup_method(self):
        self.format = ['svg']
        super().setup_method()

    def test_store_render_svg(self):
        curve = Curve([1, 2, 3])
        data, _metadata = Store.render(curve)
        mime_types = {'image/svg+xml'}
        assert set(data) == mime_types


class TestCombinedDisplay(TestDisplayHooks):

    def setup_method(self):
        self.format = ['html', 'svg', 'png']
        super().setup_method()

    def test_store_render_combined(self):
        curve = Curve([1, 2, 3])
        data, _metadata = Store.render(curve)
        mime_types = {'text/html', 'image/svg+xml', 'image/png'}
        assert set(data) == mime_types
