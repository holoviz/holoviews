from holoviews import Store, Curve
from holoviews.ipython import notebook_extension, IPTestCase


class TestDisplayHooks(IPTestCase):

    def setUp(self):
        super(TestDisplayHooks, self).setUp()
        if not notebook_extension._loaded:
            notebook_extension('matplotlib', ip=self.ip)
        self.backup = Store.display_formats
        Store.display_formats = self.format

    def tearDown(self):
        Store._custom_options = {k:{} for k in Store._custom_options.keys()}
        self.ip.run_line_magic("unload_ext", "holoviews.ipython")
        del self.ip
        Store.display_hooks = self.backup
        notebook_extension._loaded = False
        super(TestDisplayHooks, self).tearDown()


class TestHTMLDisplay(TestDisplayHooks):

    def setUp(self):
        self.format = ['html']
        super(TestHTMLDisplay, self).setUp()

    def test_store_render_html(self):
        curve = Curve([1, 2, 3])
        data, metadata = Store.render(curve)
        mime_types = {'text/html'}
        self.assertEqual(set(data), mime_types)


class TestPNGDisplay(TestDisplayHooks):

    def setUp(self):
        self.format = ['png']
        super(TestPNGDisplay, self).setUp()

    def test_store_render_png(self):
        curve = Curve([1, 2, 3])
        data, metadata = Store.render(curve)
        mime_types = {'image/png'}
        self.assertEqual(set(data), mime_types)


class TestSVGDisplay(TestDisplayHooks):

    def setUp(self):
        self.format = ['svg']
        super(TestSVGDisplay, self).setUp()

    def test_store_render_svg(self):
        curve = Curve([1, 2, 3])
        data, metadata = Store.render(curve)
        mime_types = {'image/svg+xml'}
        self.assertEqual(set(data), mime_types)


class TestCombinedDisplay(TestDisplayHooks):

    def setUp(self):
        self.format = ['html', 'svg', 'png']
        super(TestCombinedDisplay, self).setUp()

    def test_store_render_combined(self):
        curve = Curve([1, 2, 3])
        data, metadata = Store.render(curve)
        mime_types = {'text/html', 'image/svg+xml', 'image/png'}
        self.assertEqual(set(data), mime_types)
