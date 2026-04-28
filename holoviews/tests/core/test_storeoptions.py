"""
Unit tests of the StoreOptions class used to control custom options on
Store as used by the %opts magic.
"""

from __future__ import annotations

import numpy as np

import holoviews as hv
from holoviews.plotting import bokeh  # noqa: F401

from ..utils import mpl, mpl_skip

if mpl:
    import holoviews.plotting.mpl


@mpl_skip
class TestStoreOptionsMerge:
    def setup_method(self):
        hv.Store.current_backend = "matplotlib"
        self.expected = {"Image": {"plot": {"fig_size": 150}, "style": {"cmap": "Blues"}}}

    def test_full_spec_format(self):
        out = hv.StoreOptions.merge_options(
            ["plot", "style"],
            options={"Image": {"plot": dict(fig_size=150), "style": dict(cmap="Blues")}},
        )
        assert out == self.expected

    def test_options_partitioned_format(self):
        out = hv.StoreOptions.merge_options(
            ["plot", "style"],
            options=dict(plot={"Image": dict(fig_size=150)}, style={"Image": dict(cmap="Blues")}),
        )
        assert out == self.expected

    def test_partitioned_format(self):
        out = hv.StoreOptions.merge_options(
            ["plot", "style"],
            plot={"Image": dict(fig_size=150)},
            style={"Image": dict(cmap="Blues")},
        )
        assert out == self.expected


@mpl_skip
class TestStoreOptsMethod:
    """
    The .opts method makes use of most of the functionality in
    StoreOptions.
    """

    def setup_method(self):
        hv.Store.current_backend = "matplotlib"

    def test_overlay_options_partitioned(self):
        """
        The new style introduced in #73
        """
        data = [zip(range(10), range(10), strict=True), zip(range(5), range(5), strict=True)]
        o = hv.Overlay([hv.Curve(c) for c in data]).opts(
            {"Curve.Curve": {"show_grid": False, "color": "k"}}
        )

        assert not hv.Store.lookup_options("matplotlib", o.Curve.I, "plot").kwargs["show_grid"]
        assert not hv.Store.lookup_options("matplotlib", o.Curve.II, "plot").kwargs["show_grid"]
        assert hv.Store.lookup_options("matplotlib", o.Curve.I, "style").kwargs["color"] == "k"
        assert hv.Store.lookup_options("matplotlib", o.Curve.II, "style").kwargs["color"] == "k"

    def test_overlay_options_complete(self):
        """
        Complete specification style.
        """
        data = [zip(range(10), range(10), strict=True), zip(range(5), range(5), strict=True)]
        o = hv.Overlay([hv.Curve(c) for c in data]).opts(
            {"Curve.Curve": {"show_grid": True, "color": "b"}}
        )

        assert hv.Store.lookup_options("matplotlib", o.Curve.I, "plot").kwargs["show_grid"]
        assert hv.Store.lookup_options("matplotlib", o.Curve.II, "plot").kwargs["show_grid"]
        assert hv.Store.lookup_options("matplotlib", o.Curve.I, "style").kwargs["color"] == "b"
        assert hv.Store.lookup_options("matplotlib", o.Curve.II, "style").kwargs["color"] == "b"

    def test_layout_options_short_style(self):
        """
        Short __call__ syntax.
        """
        im = hv.Image(np.random.rand(10, 10))
        layout = (im + im).opts({"Layout": dict({"hspace": 5})})
        assert hv.Store.lookup_options("matplotlib", layout, "plot").kwargs["hspace"] == 5

    def test_layout_options_long_style(self):
        """
        The old (longer) syntax in __call__
        """
        im = hv.Image(np.random.rand(10, 10))
        layout = (im + im).opts({"Layout": dict({"hspace": 10})})
        assert hv.Store.lookup_options("matplotlib", layout, "plot").kwargs["hspace"] == 10

    def test_holomap_opts(self):
        hmap = hv.HoloMap({0: hv.Image(np.random.rand(10, 10))}).opts(xaxis=None)
        opts = hv.Store.lookup_options("matplotlib", hmap.last, "plot")
        assert opts.kwargs["xaxis"] is None

    def test_holomap_options(self):
        hmap = hv.HoloMap({0: hv.Image(np.random.rand(10, 10))}).options(xaxis=None)
        opts = hv.Store.lookup_options("matplotlib", hmap.last, "plot")
        assert opts.kwargs["xaxis"] is None

    def test_holomap_options_empty_no_exception(self):
        hv.HoloMap({0: hv.Image(np.random.rand(10, 10))}).options()
