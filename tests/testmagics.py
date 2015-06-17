from unittest import SkipTest

from holoviews.core.options import Store
try:
    from holoviews import ipython
    from holoviews.ipython import IPTestCase
except ImportError:
    raise SkipTest("Required dependencies not satisfied for testing magics")

from holoviews.operation import Compositor

class ExtensionTestCase(IPTestCase):

    def setUp(self):
        super(ExtensionTestCase, self).setUp()
        self.ip.run_line_magic("load_ext", "holoviews.ipython")

    def tearDown(self):
        self.ip.run_line_magic("unload_ext", "holoviews.ipython")
        del self.ip
        super(ExtensionTestCase, self).tearDown()



class TestOptsMagic(ExtensionTestCase):

    def setUp(self):
        super(TestOptsMagic, self).setUp()
        self.cell("import numpy as np")
        self.cell("from holoviews.element import Image")

    def tearDown(self):
        Store.custom_options(val = {})
        super(TestOptsMagic, self).tearDown()


    def test_cell_opts_style(self):

        self.cell("mat1 = Image(np.random.rand(5,5), name='mat1')")

        self.assertEqual(self.get_object('mat1').id, None)
        self.cell_magic('opts', " Image (cmap='hot')", 'mat1')
        self.assertEqual(self.get_object('mat1').id, 0)

        assert 0 in Store.custom_options(), "Custom OptionTree creation failed"
        self.assertEqual(
            Store.lookup_options('matplotlib',
                                 self.get_object('mat1'), 'style').options.get('cmap',None),'hot')


    def test_cell_opts_plot(self):

        self.cell("mat1 = Image(np.random.rand(5,5), name='mat1')")

        self.assertEqual(self.get_object('mat1').id, None)
        self.cell_magic('opts', " Image [show_title=False]", 'mat1')
        self.assertEqual(self.get_object('mat1').id, 0)

        assert 0 in Store.custom_options(), "Custom OptionTree creation failed"
        self.assertEqual(
            Store.lookup_options('matplotlib',
                                 self.get_object('mat1'), 'plot').options.get('show_title',True),False)


    def test_cell_opts_norm(self):

        self.cell("mat1 = Image(np.random.rand(5,5), name='mat1')")

        self.assertEqual(self.get_object('mat1').id, None)
        self.cell_magic('opts', " Image {+axiswise}", 'mat1')
        self.assertEqual(self.get_object('mat1').id, 0)

        assert 0 in Store.custom_options(), "Custom OptionTree creation failed"
        self.assertEqual(
            Store.lookup_options('matplotlib',
                                 self.get_object('mat1'), 'norm').options.get('axiswise',True), True)




class TestOutputMagic(ExtensionTestCase):

    def tearDown(self):
        ipython.OutputMagic.options = ipython.OutputMagic.defaults
        super(TestOutputMagic, self).tearDown()

    def test_output_svg(self):
        self.line_magic('output', "fig='svg'")
        self.assertEqual(ipython.OutputMagic.options.get('fig', None), 'svg')

    def test_output_holomap_scrubber(self):
        self.line_magic('output', "holomap='scrubber'")
        self.assertEqual(ipython.OutputMagic.options.get('holomap', None), 'scrubber')

    def test_output_holomap_widgets(self):
        self.line_magic('output', "holomap='widgets'")
        self.assertEqual(ipython.OutputMagic.options.get('holomap', None), 'widgets')

    def test_output_widgets_live(self):
        self.line_magic('output', "widgets='live'")
        self.assertEqual(ipython.OutputMagic.options.get('widgets', None), 'live')


    def test_output_fps(self):
        self.line_magic('output', "fps=100")
        self.assertEqual(ipython.OutputMagic.options.get('fps', None), 100)

    def test_output_size(self):
        self.line_magic('output', "size=50")
        self.assertEqual(ipython.OutputMagic.options.get('size', None), 50)


    def test_output_invalid_size(self):
        self.line_magic('output', "size=-50")
        self.assertEqual(ipython.OutputMagic.options.get('size', None), 100)


class TestCompositorMagic(ExtensionTestCase):

    def setUp(self):
        super(TestCompositorMagic, self).setUp()
        self.cell("import numpy as np")
        self.cell("from holoviews.element import Image")


    def tearDown(self):
        Compositor.definitions = []
        super(TestCompositorMagic, self).tearDown()

    def test_display_compositor_definition(self):
        definition = " display factory(Image * Image * Image) RGBTEST"
        self.line_magic('compositor', definition)

        assert len(Compositor.definitions) == 1, "Compositor definition not created"
        self.assertEqual(Compositor.definitions[0].group, 'RGBTEST')
        self.assertEqual(Compositor.definitions[0].mode, 'display')


    def test_data_compositor_definition(self):
        definition = " data transform(Image * Image) HCSTEST"
        self.line_magic('compositor', definition)
        assert len(Compositor.definitions) == 1, "Compositor definition not created"
        self.assertEqual(Compositor.definitions[0].group, 'HCSTEST')
        self.assertEqual(Compositor.definitions[0].mode, 'data')

