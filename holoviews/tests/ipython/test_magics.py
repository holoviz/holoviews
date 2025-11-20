import warnings

import pytest
from pyviz_comms import CommManager

import holoviews as hv
from holoviews.core.options import Store
from holoviews.operation import Compositor

try:
    import holoviews.ipython
    from holoviews.element.comparison import IPTestCase
except ImportError:
    pytest.skip("IPython required to test IPython magics", allow_module_level=True)


class ExtensionTestCase(IPTestCase):

    def setUp(self):
        super().setUp()
        self.ip.run_line_magic("load_ext", "holoviews.ipython")
        for renderer in Store.renderers.values():
            renderer.comm_manager = CommManager  # TODO: Should set it back

    def tearDown(self):
        self.ip.run_line_magic("unload_ext", "holoviews.ipython")
        super().tearDown()

    def cell_magic(self, *args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = super().cell_magic(*args, **kwargs)
        assert str(w[0].message).startswith("IPython magic is deprecated")
        return output

    def line_magic(self, *args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = super().line_magic(*args, **kwargs)
        assert str(w[0].message).startswith("IPython magic is deprecated")
        return output


class TestOptsMagic(ExtensionTestCase):

    def setUp(self):
        super().setUp()
        self.cell("import numpy as np")
        self.cell("from holoviews import DynamicMap, Curve, Image")

    def tearDown(self):
        Store.custom_options(val = {})
        super().tearDown()

    def test_cell_opts_style(self):

        self.cell("mat1 = Image(np.random.rand(5,5), name='mat1')")

        assert self.get_object('mat1').id is None
        self.cell_magic('opts', " Image (cmap='hot')", 'mat1')
        assert self.get_object('mat1').id == 0

        assert 0 in Store.custom_options(), "Custom OptionTree creation failed"
        assert Store.lookup_options('matplotlib',
                                 self.get_object('mat1'), 'style').options.get('cmap',None) == 'hot'

    @pytest.mark.flaky(reruns=3)
    def test_cell_opts_style_dynamic(self):

        self.cell("dmap = DynamicMap(lambda X: Curve(np.random.rand(5,2), name='dmap'), kdims=['x'])"
                  ".redim.range(x=(0, 10)).opts({'Curve': dict(linewidth=2, color='black')})")

        assert self.get_object('dmap').id is None
        self.cell_magic('opts', " Curve (linewidth=3 alpha=0.5)", 'dmap')
        assert self.get_object('dmap').id == 0

        assert 0 in Store.custom_options(), "Custom OptionTree creation failed"
        opts = Store.lookup_options('matplotlib', self.get_object('dmap')[0], 'style').options
        assert opts == {'linewidth': 3, 'alpha': 0.5, 'color': 'black'}


    def test_cell_opts_plot_float_division(self):

        self.cell("mat1 = Image(np.random.rand(5,5), name='mat1')")

        assert self.get_object('mat1').id is None
        self.cell_magic('opts', " Image [aspect=3/4]", 'mat1')
        assert self.get_object('mat1').id == 0

        assert 0 in Store.custom_options(), "Custom OptionTree creation failed"
        output = Store.lookup_options('matplotlib', self.get_object('mat1'), 'plot').options.get('aspect', False)
        assert output == 3/4


    def test_cell_opts_plot(self):

        self.cell("mat1 = Image(np.random.rand(5,5), name='mat1')")

        assert self.get_object('mat1').id is None
        self.cell_magic('opts', " Image [show_title=False]", 'mat1')
        assert self.get_object('mat1').id == 0

        assert 0 in Store.custom_options(), "Custom OptionTree creation failed"
        assert Store.lookup_options('matplotlib',
                                 self.get_object('mat1'), 'plot').options.get('show_title',True) is False


    @pytest.mark.flaky(reruns=3)
    def test_cell_opts_plot_dynamic(self):

        self.cell("dmap = DynamicMap(lambda X: Image(np.random.rand(5,5), name='dmap'), kdims=['x'])"
                  ".redim.range(x=(0, 10)).opts({'Image': dict(xaxis='top', xticks=3)})")

        assert self.get_object('dmap').id is None
        self.cell_magic('opts', " Image [xaxis=None yaxis='right']", 'dmap')
        assert self.get_object('dmap').id == 0

        assert 0 in Store.custom_options(), "Custom OptionTree creation failed"
        opts = Store.lookup_options('matplotlib', self.get_object('dmap')[0], 'plot').options
        assert opts == {'xticks': 3, 'xaxis': None, 'yaxis': 'right'}


    def test_cell_opts_norm(self):

        self.cell("mat1 = Image(np.random.rand(5,5), name='mat1')")

        assert self.get_object('mat1').id is None
        self.cell_magic('opts', " Image {+axiswise}", 'mat1')
        assert self.get_object('mat1').id == 0

        assert 0 in Store.custom_options(), "Custom OptionTree creation failed"
        assert Store.lookup_options('matplotlib',
                                 self.get_object('mat1'), 'norm').options.get('axiswise',True) is True



class TestOutputMagic(ExtensionTestCase):

    def tearDown(self):
        super().tearDown()

    def test_output_svg(self):
        self.line_magic('output', "fig='svg'")
        assert hv.util.OutputSettings.options.get('fig', None) == 'svg'

    def test_output_holomap_scrubber(self):
        self.line_magic('output', "holomap='scrubber'")
        assert hv.util.OutputSettings.options.get('holomap', None) == 'scrubber'

    def test_output_holomap_widgets(self):
        self.line_magic('output', "holomap='widgets'")
        assert hv.util.OutputSettings.options.get('holomap', None) == 'widgets'

    def test_output_widgets_live(self):
        self.line_magic('output', "widgets='live'")
        assert hv.util.OutputSettings.options.get('widgets', None) == 'live'


    def test_output_fps(self):
        self.line_magic('output', "fps=100")
        assert hv.util.OutputSettings.options.get('fps', None) == 100

    def test_output_size(self):
        self.line_magic('output', "size=50")
        assert hv.util.OutputSettings.options.get('size', None) == 50

    def test_output_invalid_size(self):
        self.line_magic('output', "size=-50")
        assert hv.util.OutputSettings.options.get('size', None) is None


class TestCompositorMagic(ExtensionTestCase):

    def setUp(self):
        super().setUp()
        self.cell("import numpy as np")
        self.cell("from holoviews.element import Image")
        self.definitions = list(Compositor.definitions)
        Compositor.definitions[:] = []

    def tearDown(self):
        Compositor.definitions[:] = self.definitions
        super().tearDown()

    def test_display_compositor_definition(self):
        definition = " display factory(Image * Image * Image) RGBTEST"
        self.line_magic('compositor', definition)

        compositors = [c for c in Compositor.definitions if c.group=='RGBTEST']
        assert len(compositors) == 1
        assert compositors[0].group == 'RGBTEST'
        assert compositors[0].mode == 'display'


    def test_data_compositor_definition(self):
        definition = " data transform(Image * Image) HCSTEST"
        self.line_magic('compositor', definition)

        compositors = [c for c in Compositor.definitions if c.group=='HCSTEST']
        assert len(compositors) == 1
        assert compositors[0].group == 'HCSTEST'
        assert compositors[0].mode == 'data'
