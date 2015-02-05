from holoviews.core.options import Store # Options

from holoviews import ipython
from holoviews.ipython import IPTestCase

from holoviews.operation import Channel, toRGB, toHCS

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
        self.cell("from holoviews.element import Matrix")

    def tearDown(self):
        Store.custom_options = {}
        super(TestOptsMagic, self).tearDown()


    def test_cell_opts_style(self):

        self.cell("mat1 = Matrix(np.random.rand(5,5), name='mat1')")

        self.assertEqual(self.get_object('mat1').id, None)
        self.cell_magic('opts', " Matrix (cmap='hot')", 'mat1')
        self.assertEqual(self.get_object('mat1').id, 0)

        assert 0 in Store.custom_options, "Custom OptionTree creation failed"
        self.assertEqual(
            Store.lookup_options(self.get_object('mat1'), 'style').options.get('cmap',None),'hot')


    def test_cell_opts_plot(self):

        self.cell("mat1 = Matrix(np.random.rand(5,5), name='mat1')")

        self.assertEqual(self.get_object('mat1').id, None)
        self.cell_magic('opts', " Matrix [show_title=False]", 'mat1')
        self.assertEqual(self.get_object('mat1').id, 0)

        assert 0 in Store.custom_options, "Custom OptionTree creation failed"
        self.assertEqual(
            Store.lookup_options(self.get_object('mat1'), 'plot').options.get('show_title',True),False)


    def test_cell_opts_norm(self):

        self.cell("mat1 = Matrix(np.random.rand(5,5), name='mat1')")

        self.assertEqual(self.get_object('mat1').id, None)
        self.cell_magic('opts', " Matrix {-groupwise}", 'mat1')
        self.assertEqual(self.get_object('mat1').id, 0)

        assert 0 in Store.custom_options, "Custom OptionTree creation failed"
        self.assertEqual(
            Store.lookup_options(self.get_object('mat1'), 'norm').options.get('groupwise',True), False)




class TestViewMagic(ExtensionTestCase):

    def tearDown(self):
        ipython.ViewMagic.options = ipython.ViewMagic.defaults
        super(TestViewMagic, self).tearDown()

    def test_view_svg(self):
        self.line_magic('view', "fig='svg'")
        self.assertEqual(ipython.ViewMagic.options.get('fig', None), 'svg')

    def test_view_holomap_scrubber(self):
        self.line_magic('view', "holomap='scrubber'")
        self.assertEqual(ipython.ViewMagic.options.get('holomap', None), 'scrubber')

    def test_view_holomap_widgets(self):
        self.line_magic('view', "holomap='widgets'")
        self.assertEqual(ipython.ViewMagic.options.get('holomap', None), 'widgets')

    def test_view_widgets_live(self):
        self.line_magic('view', "widgets='live'")
        self.assertEqual(ipython.ViewMagic.options.get('widgets', None), 'live')


    def test_view_fps(self):
        self.line_magic('view', "fps=100")
        self.assertEqual(ipython.ViewMagic.options.get('fps', None), 100)

    def test_view_size(self):
        self.line_magic('view', "size=50")
        self.assertEqual(ipython.ViewMagic.options.get('size', None), 50)


    def test_view_invalid_size(self):
        self.line_magic('view', "size=-50")
        self.assertEqual(ipython.ViewMagic.options.get('size', None), 100)


class TestChannelMagic(ExtensionTestCase):

    def setUp(self):
        super(TestChannelMagic, self).setUp()
        self.cell("import numpy as np")
        self.cell("from holoviews.element import Matrix")


    def tearDown(self):
        Channel.definitions = []
        super(TestChannelMagic, self).tearDown()

    def test_RGB_channel_definition(self):
        self.cell("R = Matrix(np.random.rand(5,5), value='R')")
        self.cell("G = Matrix(np.random.rand(5,5), value='G')")
        self.cell("B = Matrix(np.random.rand(5,5), value='B')")
        self.cell("overlay = R * G * B")

        definition = " toRGB(Matrix * Matrix * Matrix) RGBTEST"
        self.line_magic('channels', definition)

        assert len(Channel.definitions) == 1, "Channel definition not created"
        self.assertEqual(Channel.definitions[0].value, 'RGBTEST')


    def test_HCS_channel_definition(self):
        self.cell("H = Matrix(np.random.rand(5,5), value='H')")
        self.cell("C = Matrix(np.random.rand(5,5), value='C')")
        self.cell("S = Matrix(np.random.rand(5,5), value='S')")

        self.cell("overlay = H * C * S")

        definition = " toHCS(Matrix * Matrix * Matrix) HCSTEST"
        self.line_magic('channels', definition)
        assert len(Channel.definitions) == 1, "Channel definition not created"
        self.assertEqual(Channel.definitions[0].value, 'HCSTEST')


if __name__ == "__main__":
    import sys
    import nose
    nose.runmodule(argv=[sys.argv[0], "--logging-level", "ERROR"])
