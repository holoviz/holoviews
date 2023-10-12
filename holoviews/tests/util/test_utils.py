"""
Unit tests of the helper functions in utils
"""
from unittest import SkipTest

from holoviews import notebook_extension
from holoviews.element.comparison import ComparisonTestCase
from holoviews import Store
from holoviews.util import output, opts, OutputSettings, Options

from holoviews.core.options import OptionTree
from pyviz_comms import CommManager

from holoviews.plotting import mpl
from holoviews.plotting import bokeh

BACKENDS = ['matplotlib', 'bokeh']

from ..utils import LoggingComparisonTestCase

try:
    import notebook
except ImportError:
    notebook = None


class TestOutputUtil(ComparisonTestCase):

    def setUp(self):
        if notebook is None:
            raise SkipTest("Jupyter Notebook not available")
        notebook_extension(*BACKENDS)
        Store.current_backend = 'matplotlib'
        Store.renderers['matplotlib'] = mpl.MPLRenderer.instance()
        Store.renderers['bokeh'] = bokeh.BokehRenderer.instance()
        OutputSettings.options =  dict(OutputSettings.defaults.items())

        super().setUp()

    def tearDown(self):
        Store.renderers['matplotlib'] = mpl.MPLRenderer.instance()
        Store.renderers['bokeh'] = bokeh.BokehRenderer.instance()
        OutputSettings.options =  dict(OutputSettings.defaults.items())
        for renderer in Store.renderers.values():
            renderer.comm_manager = CommManager
        super().tearDown()

    def test_output_util_svg_string(self):
        self.assertEqual(OutputSettings.options.get('fig', None), None)
        output("fig='svg'")
        self.assertEqual(OutputSettings.options.get('fig', None), 'svg')

    def test_output_util_png_kwargs(self):
        self.assertEqual(OutputSettings.options.get('fig', None), None)
        output(fig='png')
        self.assertEqual(OutputSettings.options.get('fig', None), 'png')

    def test_output_util_backend_string(self):
        self.assertEqual(OutputSettings.options.get('backend', None), None)
        output("backend='bokeh'")
        self.assertEqual(OutputSettings.options.get('backend', None), 'bokeh')

    def test_output_util_backend_kwargs(self):
        self.assertEqual(OutputSettings.options.get('backend', None), None)
        output(backend='bokeh')
        self.assertEqual(OutputSettings.options.get('backend', None), 'bokeh')

    def test_output_util_object_noop(self):
        self.assertEqual(output("fig='svg'",3), 3)


class TestOptsUtil(LoggingComparisonTestCase):
    """
    Mirrors the magic tests in TestOptsMagic
    """

    def setUp(self):
        self.backend = Store.current_backend
        Store.current_backend = 'matplotlib'
        self.store_copy = OptionTree(sorted(Store.options().items()),
                                     groups=Options._option_groups)
        super().setUp()

    def tearDown(self):
        Store.current_backend = self.backend
        Store.options(val=self.store_copy)
        Store._custom_options = {k:{} for k in Store._custom_options.keys()}
        super().tearDown()

    def test_opts_builder_repr(self):
        magic= "Bivariate [bandwidth=0.5] (cmap='jet') Points [logx=True] (size=2)"
        expected= ["opts.Bivariate(bandwidth=0.5, cmap='jet')",
                   "opts.Points(logx=True, size=2)"]
        reprs = opts._builder_reprs(magic)
        self.assertEqual(reprs, expected)

    def test_opts_builder_repr_line_magic(self):
        magic= "%opts Bivariate [bandwidth=0.5] (cmap='jet') Points [logx=True] (size=2)"
        expected= ["opts.Bivariate(bandwidth=0.5, cmap='jet')",
                   "opts.Points(logx=True, size=2)"]
        reprs = opts._builder_reprs(magic)
        self.assertEqual(reprs, expected)

    def test_opts_builder_repr_cell_magic(self):
        magic= "%%opts Bivariate [bandwidth=0.5] (cmap='jet') Points [logx=True] (size=2)"
        expected= ["opts.Bivariate(bandwidth=0.5, cmap='jet')",
                   "opts.Points(logx=True, size=2)"]
        reprs = opts._builder_reprs(magic)
        self.assertEqual(reprs, expected)

    def test_opts_builder_repr_options_dotted(self):
        options = [Options('Bivariate.Test.Example', bandwidth=0.5, cmap='Blues'),
                   Options('Points', size=2, logx=True)]
        expected= ["opts.Bivariate('Test.Example', bandwidth=0.5, cmap='Blues')",
                   "opts.Points(logx=True, size=2)"]
        reprs = opts._builder_reprs(options)
        self.assertEqual(reprs, expected)
