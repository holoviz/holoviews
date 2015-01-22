from holoviews.core import ViewableElement
from holoviews.testing import IPTestCase

from holoviews import ipython, CompositeOverlay, ViewableElement
from holoviews.core.options import OptionsGroup, Options
from holoviews.core.options import PlotOpts, StyleOpts, ChannelOpts


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
        self.cell("from holoviews.core import BoundingBox, CompositeOverlay, Overlay")
        self.cell("from holoviews.element import Matrix, Points")

        # Clear the options map
        self.options = OptionsGroup([Options('plotting', PlotOpts),
                                     Options('style', StyleOpts)])
        ViewableElement.options = self.options
        self.options.Matrix = StyleOpts()

    def tearDown(self):
        del self.options
        ViewableElement.options  = None
        super(TestOptsMagic, self).tearDown()

    #============================#
    # Cell Magic StyleOpts tests #
    #============================#

    def test_cell_opts_style_name1(self):
        self.cell("mat1 = Matrix(np.random.rand(5,5), name='mat1')")
        self.cell_magic('opts', " Matrix cmap='hot'", 'mat1')
        self.assertEqual(self.get_object('mat1').style, 'Custom[<mat1>]_Matrix')


    def test_cell_opts_style_name2(self):
        self.cell("mat2 = Matrix(np.random.rand(5,5), name='mat2')")
        self.cell_magic('opts', " Matrix cmap='cool'", 'mat2')
        self.assertEqual(self.get_object('mat2').style, 'Custom[<mat2>]_Matrix')


    def test_cell_opts_style1(self):
        self.cell("mat1 = Matrix(np.random.rand(5,5), name='mat1')")
        self.cell_magic('opts', " Matrix cmap='hot'", 'mat1')
        self.assertEqual(self.options.style['Custom[<mat1>]_Matrix'].opts,
                         {'cmap':'hot'})

    def test_cell_opts_style2(self):
        self.cell("mat2 = Matrix(np.random.rand(5,5), name='mat2')")
        self.cell_magic('opts', " Matrix cmap='cool' interpolation='bilinear'", 'mat2')
        self.assertEqual(self.options.style['Custom[<mat2>]_Matrix'].opts,
                         {'cmap':'cool', 'interpolation':'bilinear'})

    #============================#
    # Line Magic StyleOpts tests #
    #============================#

    def test_line_opts_nostyle(self):
        self.assertEqual(self.options.style['Matrix'].opts, {})

    def test_line_opts_style1(self):
        self.line_magic('opts', " Matrix cmap='hot'")
        self.assertEqual(self.options.style['Matrix'].opts, {'cmap':'hot'})

    def test_line_opts_style2(self):
        self.line_magic('opts', " Matrix cmap='cool' interpolation='bilinear'")
        self.assertEqual(self.options.style['Matrix'].opts,
                         {'cmap':'cool', 'interpolation':'bilinear'})


    #===========================#
    # Cell Magic PlotOpts tests #
    #===========================#


    def test_cell_magic_plotopts1(self):
        self.cell("mat1 = Matrix(np.random.rand(5,5), name='mat1')")
        self.cell_magic('opts', " Matrix [show_title=True]", 'mat1')
        self.assertEqual(self.options.plotting['Custom[<mat1>]_Matrix'].opts,
                         {'show_title':True})

    def test_cell_magic_plotopts_and_styleopts(self):
        self.cell("mat2 = Matrix(np.random.rand(5,5), name='mat2')")
        self.cell_magic('opts', " Matrix [show_grid=True] cmap='jet'", 'mat2')
        self.assertEqual(self.options.plotting['Custom[<mat2>]_Matrix'].opts,
                         {'show_grid':True})
        self.assertEqual(self.options.style['Custom[<mat2>]_Matrix'].opts,
                         {'cmap':'jet'})

    def test_cell_magic_complex_example(self):
        self.cell("""o = Overlay([Matrix(np.random.rand(5,5)),
                           Points(np.random.rand(2,5))], name='complex_view')""")
        opts = " Matrix [show_grid=True] cmap='hsv' Points [show_title=False] color='r'"
        self.cell_magic('opts', opts, 'o')
        self.assertEqual(self.options.style['Custom[<complex_view>]_Matrix'].opts,
                         {'cmap':'hsv'})
        self.assertEqual(self.options.plotting['Custom[<complex_view>]_Matrix'].opts,
                         {'show_grid':True})
        self.assertEqual(self.options.style['Custom[<complex_view>]_Points'].opts,
                         {'color':'r'})
        self.assertEqual(self.options.plotting['Custom[<complex_view>]_Points'].opts,
                         {'show_title':False})


    def test_cell_magic_syntaxerror(self):
        self.cell("mat1 = Matrix(np.random.rand(5,5), name='mat1')")

        try:
            self.cell_magic('opts', " Matrix [show_title='True]", 'mat1')
            raise AssertionError
        except SyntaxError:
            pass


class ViewsMagic(ExtensionTestCase):

    def setUp(self):
        ipython.ViewMagic.PERCENTAGE_SIZE = 100
        ipython.ViewMagic.FIGURE_FORMAT = None
        ipython.ViewMagic.VIDEO_FORMAT = 'webm'
        ipython.ViewMagic.FPS = None
        super(ViewsMagic, self).setUp()

    def test_view_svg(self):
        self.line_magic('view', 'svg')
        self.assertEqual(ipython.ViewMagic.FIGURE_FORMAT, 'svg')

    def test_view_png(self):
        self.line_magic('view', 'png')
        self.assertEqual(ipython.ViewMagic.FIGURE_FORMAT, 'png')

    def test_view_svg_gif(self):
        self.line_magic('view', 'svg gif')
        self.assertEqual(ipython.ViewMagic.FIGURE_FORMAT, 'svg')
        self.assertEqual(ipython.ViewMagic.VIDEO_FORMAT, 'gif')


    def test_view_svg_gif_10fps(self):
        self.line_magic('view', 'svg gif:10')
        self.assertEqual(ipython.ViewMagic.FIGURE_FORMAT, 'svg')
        self.assertEqual(ipython.ViewMagic.VIDEO_FORMAT, 'gif')
        self.assertEqual(ipython.ViewMagic.FPS, 10)


    def test_view_png_h264_20fps_half_size(self):
        self.line_magic('view', 'png h264:20 50')
        self.assertEqual(ipython.ViewMagic.FIGURE_FORMAT, 'png')
        self.assertEqual(ipython.ViewMagic.VIDEO_FORMAT, 'h264')
        self.assertEqual(ipython.ViewMagic.FPS, 20)
        self.assertEqual(ipython.ViewMagic.PERCENTAGE_SIZE, 50)

    def tearDown(self):
        super(ViewsMagic, self).tearDown()


class TestChannelMagic(ExtensionTestCase):

    def setUp(self):
        super(TestChannelMagic, self).setUp()
        self.cell("import numpy as np")
        self.cell("from holoviews.core import CompositeOverlay")
        self.cell("from holoviews.element import Matrix")
        self.channels = OptionsGroup([Options('definitions',
                                              ChannelOpts)])
        CompositeOverlay.channels = self.channels


    def tearDown(self):
        del self.channels
        super(TestChannelMagic, self).tearDown()

    def test_RGBA_channeldef(self):
        self.cell("R = Matrix(np.random.rand(5,5), label='R_Channel')")
        self.cell("G = Matrix(np.random.rand(5,5), label='G_Channel')")
        self.cell("B = Matrix(np.random.rand(5,5), label='B_Channel')")
        self.cell("overlay = Overlay([R, G, B], name='RGBTest')")
        definition = " R_Channel * G_Channel * B_Channel => RGBA []"
        self.cell_magic('channels', definition, 'overlay')

        expected_key = 'Custom[<RGBTest>]_RGBA'
        self.assertEqual(CompositeOverlay.channels.keys(), [expected_key])
        self.assertEqual(CompositeOverlay.channels[expected_key].pattern, 'R_Channel * G_Channel * B_Channel')
        self.assertEqual(CompositeOverlay.channels[expected_key].mode, 'RGBA')


    def test_HCS_channeldef(self):
        self.cell("H = Matrix(np.random.rand(5,5), label='H_Channel')")
        self.cell("C = Matrix(np.random.rand(5,5), label='C_Channel')")
        self.cell("S = Matrix(np.random.rand(5,5), label='S_Channel')")
        self.cell("overlay = Overlay([H, C, S], name='HCSTest')")
        definition = " H_Channel * C_Channel *S_Channel => HCS [S_multiplier=2.5]"
        self.cell_magic('channels', definition, 'overlay')

        expected_key = 'Custom[<HCSTest>]_HCS'
        self.assertEqual(CompositeOverlay.channels.keys(), [expected_key])
        self.assertEqual(CompositeOverlay.channels[expected_key].pattern, 'H_Channel * C_Channel * S_Channel')
        self.assertEqual(CompositeOverlay.channels[expected_key].mode, 'HCS')
        self.assertEqual(CompositeOverlay.channels[expected_key].opts, {'S_multiplier':2.5})

if __name__ == "__main__":
    import sys
    import nose
    nose.runmodule(argv=[sys.argv[0], "--logging-level", "ERROR"])
