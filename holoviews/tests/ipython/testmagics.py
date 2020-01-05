from unittest import SkipTest

import holoviews as hv
from holoviews.core.options import Store
try:
    from holoviews import ipython            # noqa (Import test)
    from holoviews.ipython import IPTestCase
except ImportError:
    raise SkipTest("Required dependencies not satisfied for testing magics")

from holoviews.operation import Compositor
from pyviz_comms import CommManager

class ExtensionTestCase(IPTestCase):

    def setUp(self):
        super(ExtensionTestCase, self).setUp()
        self.ip.run_line_magic("load_ext", "holoviews.ipython")
        for renderer in Store.renderers.values():
            renderer.comm_manager = CommManager

    def tearDown(self):
        Store._custom_options = {k:{} for k in Store._custom_options.keys()}
        self.ip.run_line_magic("unload_ext", "holoviews.ipython")
        del self.ip
        super(ExtensionTestCase, self).tearDown()



class TestOptsMagic(ExtensionTestCase):

    def setUp(self):
        super(TestOptsMagic, self).setUp()
        self.cell("import numpy as np")
        self.cell("from holoviews import DynamicMap, Curve, Image")

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

    def test_cell_opts_style_dynamic(self):

        self.cell("dmap = DynamicMap(lambda X: Curve(np.random.rand(5,2), name='dmap'), kdims=['x'])"
                  ".redim.range(x=(0, 10)).opts(style={'Curve': dict(linewidth=2, color='black')})")

        self.assertEqual(self.get_object('dmap').id, None)
        self.cell_magic('opts', " Curve (linewidth=3 alpha=0.5)", 'dmap')
        self.assertEqual(self.get_object('dmap').id, 0)

        assert 0 in Store.custom_options(), "Custom OptionTree creation failed"
        opts = Store.lookup_options('matplotlib', self.get_object('dmap')[0], 'style').options
        self.assertEqual(opts, {'linewidth': 3, 'alpha': 0.5, 'color': 'black'})


    def test_cell_opts_plot_float_division(self):

        self.cell("mat1 = Image(np.random.rand(5,5), name='mat1')")

        self.assertEqual(self.get_object('mat1').id, None)
        self.cell_magic('opts', " Image [aspect=3/4]", 'mat1')
        self.assertEqual(self.get_object('mat1').id, 0)

        assert 0 in Store.custom_options(), "Custom OptionTree creation failed"
        self.assertEqual(
            Store.lookup_options('matplotlib',
                                 self.get_object('mat1'), 'plot').options.get('aspect',False), 3/4.0)


    def test_cell_opts_plot(self):

        self.cell("mat1 = Image(np.random.rand(5,5), name='mat1')")

        self.assertEqual(self.get_object('mat1').id, None)
        self.cell_magic('opts', " Image [show_title=False]", 'mat1')
        self.assertEqual(self.get_object('mat1').id, 0)

        assert 0 in Store.custom_options(), "Custom OptionTree creation failed"
        self.assertEqual(
            Store.lookup_options('matplotlib',
                                 self.get_object('mat1'), 'plot').options.get('show_title',True),False)


    def test_cell_opts_plot_dynamic(self):

        self.cell("dmap = DynamicMap(lambda X: Image(np.random.rand(5,5), name='dmap'), kdims=['x'])"
                  ".redim.range(x=(0, 10)).opts(plot={'Image': dict(xaxis='top', xticks=3)})")

        self.assertEqual(self.get_object('dmap').id, None)
        self.cell_magic('opts', " Image [xaxis=None yaxis='right']", 'dmap')
        self.assertEqual(self.get_object('dmap').id, 0)

        assert 0 in Store.custom_options(), "Custom OptionTree creation failed"
        opts = Store.lookup_options('matplotlib', self.get_object('dmap')[0], 'plot').options
        self.assertEqual(opts, {'xticks': 3, 'xaxis': None, 'yaxis': 'right'})


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
        super(TestOutputMagic, self).tearDown()

    def test_output_svg(self):
        self.line_magic('output', "fig='svg'")
        self.assertEqual(hv.util.OutputSettings.options.get('fig', None), 'svg')

    def test_output_holomap_scrubber(self):
        self.line_magic('output', "holomap='scrubber'")
        self.assertEqual(hv.util.OutputSettings.options.get('holomap', None), 'scrubber')

    def test_output_holomap_widgets(self):
        self.line_magic('output', "holomap='widgets'")
        self.assertEqual(hv.util.OutputSettings.options.get('holomap', None), 'widgets')

    def test_output_widgets_live(self):
        self.line_magic('output', "widgets='live'")
        self.assertEqual(hv.util.OutputSettings.options.get('widgets', None), 'live')


    def test_output_fps(self):
        self.line_magic('output', "fps=100")
        self.assertEqual(hv.util.OutputSettings.options.get('fps', None), 100)

    def test_output_size(self):
        self.line_magic('output', "size=50")
        self.assertEqual(hv.util.OutputSettings.options.get('size', None), 50)

    def test_output_invalid_size(self):
        self.line_magic('output', "size=-50")
        self.assertEqual(hv.util.OutputSettings.options.get('size', None), None)


class TestCompositorMagic(ExtensionTestCase):

    def setUp(self):
        super(TestCompositorMagic, self).setUp()
        self.cell("import numpy as np")
        self.cell("from holoviews.element import Image")
        self.definitions = list(Compositor.definitions)
        Compositor.definitions[:] = []

    def tearDown(self):
        Compositor.definitions[:] = self.definitions
        super(TestCompositorMagic, self).tearDown()

    def test_display_compositor_definition(self):
        definition = " display factory(Image * Image * Image) RGBTEST"
        self.line_magic('compositor', definition)

        compositors = [c for c in Compositor.definitions if c.group=='RGBTEST']
        self.assertEqual(len(compositors), 1)
        self.assertEqual(compositors[0].group, 'RGBTEST')
        self.assertEqual(compositors[0].mode, 'display')


    def test_data_compositor_definition(self):
        definition = " data transform(Image * Image) HCSTEST"
        self.line_magic('compositor', definition)

        compositors = [c for c in Compositor.definitions if c.group=='HCSTEST']
        self.assertEqual(len(compositors), 1)
        self.assertEqual(compositors[0].group, 'HCSTEST')
        self.assertEqual(compositors[0].mode, 'data')

