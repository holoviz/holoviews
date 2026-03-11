"""
Unit tests of the helper functions in utils
"""

from pyviz_comms import CommManager

import holoviews as hv
from holoviews.core.options import OptionTree
from holoviews.plotting import bokeh
from holoviews.util import OutputSettings

from ..utils import LoggingComparison, optional_dependencies

BACKENDS = ["matplotlib", "bokeh"]

_, notebook_skip = optional_dependencies("notebook")
_, mpl_skip = optional_dependencies("matplotlib")


@mpl_skip
@notebook_skip
class TestOutputUtil:
    def setup_method(self):
        from holoviews.ipython import notebook_extension
        from holoviews.plotting import mpl

        notebook_extension(*BACKENDS)
        hv.Store.current_backend = "matplotlib"
        hv.Store.renderers["matplotlib"] = mpl.MPLRenderer.instance()
        hv.Store.renderers["bokeh"] = bokeh.BokehRenderer.instance()
        OutputSettings.options = dict(OutputSettings.defaults.items())

    def teardown_method(self):
        from holoviews.plotting import mpl

        hv.Store.renderers["matplotlib"] = mpl.MPLRenderer.instance()
        hv.Store.renderers["bokeh"] = bokeh.BokehRenderer.instance()
        OutputSettings.options = dict(OutputSettings.defaults.items())
        for renderer in hv.Store.renderers.values():
            renderer.comm_manager = CommManager

    def test_output_util_svg_string(self):
        assert OutputSettings.options.get("fig", None) is None
        hv.output("fig='svg'")
        assert OutputSettings.options.get("fig", None) == "svg"

    def test_output_util_png_kwargs(self):
        assert OutputSettings.options.get("fig", None) is None
        hv.output(fig="png")
        assert OutputSettings.options.get("fig", None) == "png"

    def test_output_util_backend_string(self):
        assert OutputSettings.options.get("backend", None) is None
        hv.output("backend='bokeh'")
        assert OutputSettings.options.get("backend", None) == "bokeh"

    def test_output_util_backend_kwargs(self):
        assert OutputSettings.options.get("backend", None) is None
        hv.output(backend="bokeh")
        assert OutputSettings.options.get("backend", None) == "bokeh"

    def test_output_util_object_noop(self):
        assert hv.output("fig='svg'", 3) == 3


@mpl_skip
class TestOptsUtil(LoggingComparison):
    """
    Mirrors the magic tests in TestOptsMagic
    """

    def setup_method(self):
        from holoviews.plotting import mpl  # noqa: F401

        self.backend = hv.Store.current_backend
        hv.Store.current_backend = "matplotlib"
        self.store_copy = OptionTree(
            sorted(hv.Store.options().items()), groups=hv.Options._option_groups
        )

    def teardown_method(self):
        hv.Store.current_backend = self.backend
        hv.Store.options(val=self.store_copy)

    def test_opts_builder_repr_options_dotted(self):
        options = [
            hv.Options("Bivariate.Test.Example", bandwidth=0.5, cmap="Blues"),
            hv.Options("Points", size=2, logx=True),
        ]
        expected = [
            "opts.Bivariate('Test.Example', bandwidth=0.5, cmap='Blues')",
            "opts.Points(logx=True, size=2)",
        ]
        reprs = hv.opts._builder_reprs(options)
        assert reprs == expected
