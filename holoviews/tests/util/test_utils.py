"""
Unit tests of the helper functions in utils
"""
from pyviz_comms import CommManager

from holoviews import Store
from holoviews.core.options import OptionTree
from holoviews.plotting import bokeh
from holoviews.util import Options, OutputSettings, opts, output

from ..utils import LoggingComparison, optional_dependencies

BACKENDS = ['matplotlib', 'bokeh']

_, notebook_skip = optional_dependencies("notebook")
_, mpl_skip = optional_dependencies("matplotlib")


@mpl_skip
@notebook_skip
class TestOutputUtil:

    def setup_method(self):
        from holoviews.ipython import notebook_extension
        from holoviews.plotting import mpl

        notebook_extension(*BACKENDS)
        Store.current_backend = 'matplotlib'
        Store.renderers['matplotlib'] = mpl.MPLRenderer.instance()
        Store.renderers['bokeh'] = bokeh.BokehRenderer.instance()
        OutputSettings.options =  dict(OutputSettings.defaults.items())

    def teardown_method(self):
        from holoviews.plotting import mpl

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


@mpl_skip
class TestOptsUtil(LoggingComparison):
    """
    Mirrors the magic tests in TestOptsMagic
    """

    def setup_method(self):
        from holoviews.plotting import mpl  # noqa: F401
        self.backend = Store.current_backend
        Store.current_backend = 'matplotlib'
        self.store_copy = OptionTree(sorted(Store.options().items()),
                                     groups=Options._option_groups)

    def teardown_method(self):
        Store.current_backend = self.backend
        Store.options(val=self.store_copy)

    def test_opts_builder_repr_options_dotted(self):
        options = [Options('Bivariate.Test.Example', bandwidth=0.5, cmap='Blues'),
                   Options('Points', size=2, logx=True)]
        expected= ["opts.Bivariate('Test.Example', bandwidth=0.5, cmap='Blues')",
                   "opts.Points(logx=True, size=2)"]
        reprs = opts._builder_reprs(options)
        assert reprs == expected
