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
        mime_types = {'text/html', 'application/javascript',
                      'application/vnd.holoviews_exec.v0+json'}
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
        mime_types = {'text/html', 'application/javascript',
                      'application/vnd.holoviews_exec.v0+json',
                      'image/svg+xml', 'image/png'}
        self.assertEqual(set(data), mime_types)


class TestBokehTheming(TestDisplayHooks):

    def setUp(self):
        self.format = ['png']
        super(TestBokehTheming, self).setUp()

    def testfail(self):
        from bokeh.themes.theme import Theme
        from holoviews.ipython import display
        import holoviews as hv
        hv.extension('bokeh')

        theme = Theme(
            json={
        'attrs' : {
            'Figure' : {
                'background_fill_color': '#2F2F2F',
                'border_fill_color': '#2F2F2F',
                'outline_line_color': '#444444',
            },
            'Grid': {
                'grid_line_dash': [6, 4],
                'grid_line_alpha': .3,
            },

            'Axis': {
                'major_label_text_color': 'white',
                'axis_label_text_color': 'white',
                'major_tick_line_color': 'white',
                'minor_tick_line_color': 'white',
                'axis_line_color': "white"
            }
          }
        })

        renderer = hv.renderer('bokeh')
        renderer.theme = theme

        curve = hv.Curve([1,2,3])
        display(curve)
        self.assertEqual(renderer.last_plot.state.outline_line_color, '#444444')

