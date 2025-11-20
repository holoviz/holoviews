"""
Unit tests of the helper functions in utils
"""
import pytest
from pyviz_comms import CommManager

from holoviews import Store
from holoviews.core.options import OptionTree
from holoviews.plotting import bokeh
from holoviews.util import Options, OutputSettings, opts, output

BACKENDS = ['matplotlib', 'bokeh']

from ..utils import LoggingComparisonTestCase

try:
    import notebook
except ImportError:
    notebook = None

try:
    from holoviews.plotting import mpl
except ImportError:
    mpl = None



class TestOutputUtil:

    def setup_method(self):
        if notebook is None:
            pytest.skip("Jupyter Notebook not available")
        if mpl is None:
            pytest.skip("Matplotlib not available")
        from holoviews.ipython import notebook_extension

        notebook_extension(*BACKENDS)
        Store.current_backend = 'matplotlib'
        Store.renderers['matplotlib'] = mpl.MPLRenderer.instance()
        Store.renderers['bokeh'] = bokeh.BokehRenderer.instance()
        OutputSettings.options =  dict(OutputSettings.defaults.items())


    def teardown_method(self):
        Store.renderers['matplotlib'] = mpl.MPLRenderer.instance()
        Store.renderers['bokeh'] = bokeh.BokehRenderer.instance()
        OutputSettings.options =  dict(OutputSettings.defaults.items())
        for renderer in Store.renderers.values():
            renderer.comm_manager = CommManager

    def test_output_util_svg_string(self):
        assert OutputSettings.options.get('fig', None) is None
        output("fig='svg'")
        assert OutputSettings.options.get('fig', None) == 'svg'

    def test_output_util_png_kwargs(self):
        assert OutputSettings.options.get('fig', None) is None
        output(fig='png')
        assert OutputSettings.options.get('fig', None) == 'png'

    def test_output_util_backend_string(self):
        assert OutputSettings.options.get('backend', None) is None
        output("backend='bokeh'")
        assert OutputSettings.options.get('backend', None) == 'bokeh'

    def test_output_util_backend_kwargs(self):
        assert OutputSettings.options.get('backend', None) is None
        output(backend='bokeh')
        assert OutputSettings.options.get('backend', None) == 'bokeh'

    def test_output_util_object_noop(self):
        assert output("fig='svg'",3) == 3


class TestOptsUtil(LoggingComparisonTestCase):
    """
    Mirrors the magic tests in TestOptsMagic
    """

    def setup_method(self):
        if mpl is None:
            pytest.skip("Matplotlib not available")
        self.backend = Store.current_backend
        Store.current_backend = 'matplotlib'
        self.store_copy = OptionTree(sorted(Store.options().items()),
                                     groups=Options._option_groups)
        super().setup_method()

    def teardown_method(self):
        Store.current_backend = self.backend
        Store.options(val=self.store_copy)
        super().teardown_method()

    def test_opts_builder_repr(self):
        magic= "Bivariate [bandwidth=0.5] (cmap='jet') Points [logx=True] (size=2)"
        expected= ["opts.Bivariate(bandwidth=0.5, cmap='jet')",
                   "opts.Points(logx=True, size=2)"]
        reprs = opts._builder_reprs(magic)
        assert reprs == expected

    def test_opts_builder_repr_line_magic(self):
        magic= "%opts Bivariate [bandwidth=0.5] (cmap='jet') Points [logx=True] (size=2)"
        expected= ["opts.Bivariate(bandwidth=0.5, cmap='jet')",
                   "opts.Points(logx=True, size=2)"]
        reprs = opts._builder_reprs(magic)
        assert reprs == expected

    def test_opts_builder_repr_cell_magic(self):
        magic= "%%opts Bivariate [bandwidth=0.5] (cmap='jet') Points [logx=True] (size=2)"
        expected= ["opts.Bivariate(bandwidth=0.5, cmap='jet')",
                   "opts.Points(logx=True, size=2)"]
        reprs = opts._builder_reprs(magic)
        assert reprs == expected

    def test_opts_builder_repr_options_dotted(self):
        options = [Options('Bivariate.Test.Example', bandwidth=0.5, cmap='Blues'),
                   Options('Points', size=2, logx=True)]
        expected= ["opts.Bivariate('Test.Example', bandwidth=0.5, cmap='Blues')",
                   "opts.Points(logx=True, size=2)"]
        reprs = opts._builder_reprs(options)
        assert reprs == expected
