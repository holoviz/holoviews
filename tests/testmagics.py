from .utils import IPTestCase

from dataviews import ipython
from dataviews import SheetOverlay
from dataviews.options import OptionsGroup, Options
from dataviews.options import PlotOpts, StyleOpts, ChannelOpts
from dataviews.views import View


class ExtensionTestCase(IPTestCase):

    def setUp(self):
        super(ExtensionTestCase, self).setUp()
        self.ip.run_line_magic("load_ext", "dataviews.ipython")

    def tearDown(self):
        self.ip.run_line_magic("unload_ext", "dataviews.ipython")
        del self.ip
        super(ExtensionTestCase, self).tearDown()



class TestOptsMagic(ExtensionTestCase):

    def setUp(self):
        super(TestOptsMagic, self).setUp()
        self.cell("import numpy as np")
        self.cell("from dataviews.sheetviews import BoundingBox")
        self.cell("from dataviews import SheetView, Points, SheetOverlay")

        # Clear the options map
        self.options = OptionsGroup([Options('plotting', PlotOpts),
                                     Options('style', StyleOpts)])
        View.options = self.options
        self.options.SheetView = StyleOpts()

    def tearDown(self):
        del self.options
        View.options  = None
        super(TestOptsMagic, self).tearDown()

    #============================#
    # Cell Magic StyleOpts tests #
    #============================#

    def test_cell_opts_style_name1(self):
        self.cell("sv1 = SheetView(np.random.rand(5,5), name='sv1')")
        self.cell_magic('opts', " SheetView cmap='hot'", 'sv1')
        self.assertEqual(self.get_object('sv1').style, 'Custom[<sv1>]_SheetView')


    def test_cell_opts_style_name2(self):
        self.cell("sv2 = SheetView(np.random.rand(5,5), name='sv2')")
        self.cell_magic('opts', " SheetView cmap='cool'", 'sv2')
        self.assertEqual(self.get_object('sv2').style, 'Custom[<sv2>]_SheetView')


    def test_cell_opts_style1(self):
        self.cell("sv1 = SheetView(np.random.rand(5,5), name='sv1')")
        self.cell_magic('opts', " SheetView cmap='hot'", 'sv1')
        self.assertEqual(self.options.style['Custom[<sv1>]_SheetView'].opts,
                         {'cmap':'hot'})

    def test_cell_opts_style2(self):
        self.cell("sv2 = SheetView(np.random.rand(5,5), name='sv2')")
        self.cell_magic('opts', " SheetView cmap='cool' interpolation='bilinear'", 'sv2')
        self.assertEqual(self.options.style['Custom[<sv2>]_SheetView'].opts,
                         {'cmap':'cool', 'interpolation':'bilinear'})

    #============================#
    # Line Magic StyleOpts tests #
    #============================#

    def test_line_opts_nostyle(self):
        self.assertEqual(self.options.style['SheetView'].opts, {})

    def test_line_opts_style1(self):
        self.line_magic('opts', " SheetView cmap='hot'")
        self.assertEqual(self.options.style['SheetView'].opts, {'cmap':'hot'})

    def test_line_opts_style2(self):
        self.line_magic('opts', " SheetView cmap='cool' interpolation='bilinear'")
        self.assertEqual(self.options.style['SheetView'].opts,
                         {'cmap':'cool', 'interpolation':'bilinear'})


    #===========================#
    # Cell Magic PlotOpts tests #
    #===========================#


    def test_cell_magic_plotopts1(self):
        self.cell("sv1 = SheetView(np.random.rand(5,5), name='sv1')")
        self.cell_magic('opts', " SheetView [show_title=True]", 'sv1')
        self.assertEqual(self.options.plotting['Custom[<sv1>]_SheetView'].opts,
                         {'show_title':True})

    def test_cell_magic_plotopts_and_styleopts(self):
        self.cell("sv2 = SheetView(np.random.rand(5,5), name='sv2')")
        self.cell_magic('opts', " SheetView [show_grid=True] cmap='jet'", 'sv2')
        self.assertEqual(self.options.plotting['Custom[<sv2>]_SheetView'].opts,
                         {'show_grid':True})
        self.assertEqual(self.options.style['Custom[<sv2>]_SheetView'].opts,
                         {'cmap':'jet'})

    def test_cell_magic_complex_example(self):
        self.cell("""o = SheetOverlay([SheetView(np.random.rand(5,5)),
                           Points(np.random.rand(2,5))], BoundingBox(), name='complex_view')""")
        opts = " SheetView [show_grid=True] cmap='hsv' Points [show_title=False] color='r'"
        self.cell_magic('opts', opts, 'o')
        self.assertEqual(self.options.style['Custom[<complex_view>]_SheetView'].opts,
                         {'cmap':'hsv'})
        self.assertEqual(self.options.plotting['Custom[<complex_view>]_SheetView'].opts,
                         {'show_grid':True})
        self.assertEqual(self.options.style['Custom[<complex_view>]_Points'].opts,
                         {'color':'r'})
        self.assertEqual(self.options.plotting['Custom[<complex_view>]_Points'].opts,
                         {'show_title':False})


    def test_cell_magic_syntaxerror(self):
        self.cell("sv1 = SheetView(np.random.rand(5,5), name='sv1')")

        try:
            self.cell_magic('opts', " SheetView [show_title='True]", 'sv1')
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
        self.cell("from dataviews.sheetviews import BoundingBox")
        self.cell("from dataviews import SheetView, SheetOverlay")
        self.channels = OptionsGroup([Options('definitions',
                                              ChannelOpts)])
        SheetOverlay.channels = self.channels


    def tearDown(self):
        del self.channels
        super(TestChannelMagic, self).tearDown()

    def test_RGBA_channeldef(self):
        self.cell("R = SheetView(np.random.rand(5,5), label='R_Channel')")
        self.cell("G = SheetView(np.random.rand(5,5), label='G_Channel')")
        self.cell("B = SheetView(np.random.rand(5,5), label='B_Channel')")
        self.cell("overlay = SheetOverlay([R, G, B], BoundingBox(), name='RGBTest')")
        definition = " R_Channel * G_Channel * B_Channel => RGBA []"
        self.cell_magic('channels', definition, 'overlay')

        expected_key = 'Custom[<RGBTest>]_RGBA'
        self.assertEqual(SheetOverlay.channels.keys(), [expected_key])
        self.assertEqual(SheetOverlay.channels[expected_key].pattern, 'R_Channel * G_Channel * B_Channel')
        self.assertEqual(SheetOverlay.channels[expected_key].mode, 'RGBA')


    def test_HCS_channeldef(self):
        self.cell("H = SheetView(np.random.rand(5,5), label='H_Channel')")
        self.cell("C = SheetView(np.random.rand(5,5), label='C_Channel')")
        self.cell("S = SheetView(np.random.rand(5,5), label='S_Channel')")
        self.cell("overlay = SheetOverlay([H, C, S], BoundingBox(), name='HCSTest')")
        definition = " H_Channel * C_Channel *S_Channel => HCS [S_multiplier=2.5]"
        self.cell_magic('channels', definition, 'overlay')

        expected_key = 'Custom[<HCSTest>]_HCS'
        self.assertEqual(SheetOverlay.channels.keys(), [expected_key])
        self.assertEqual(SheetOverlay.channels[expected_key].pattern, 'H_Channel * C_Channel * S_Channel')
        self.assertEqual(SheetOverlay.channels[expected_key].mode, 'HCS')
        self.assertEqual(SheetOverlay.channels[expected_key].opts, {'S_multiplier':2.5})

if __name__ == "__main__":
    import sys
    import nose
    nose.runmodule(argv=[sys.argv[0], "--logging-level", "ERROR"])
