"""
Unit tests of the StoreOptions class used to control custom options on
Store as used by the %opts magic.
"""

from __future__ import annotations

import itertools
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import holoviews as hv
from holoviews.plotting import bokeh  # noqa: F401

from .._deps import mpl, mpl_skip

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
        layout = (im + im).opts({"Layout": {"hspace": 5}})
        assert hv.Store.lookup_options("matplotlib", layout, "plot").kwargs["hspace"] == 5

    def test_layout_options_long_style(self):
        """
        The old (longer) syntax in __call__
        """
        im = hv.Image(np.random.rand(10, 10))
        layout = (im + im).opts({"Layout": {"hspace": 10}})
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


@mpl_skip
class TestStoreOptionsCustomIdGrowth:
    """
    Re-customizing retained, already-customized objects must keep custom-option
    ids bounded and must not let concurrent customizations collide on an id.
    """

    def setup_method(self):
        self._backend = hv.Store.current_backend
        hv.Store.set_current_backend("matplotlib")

    def teardown_method(self):
        hv.Store.current_backend = self._backend

    @staticmethod
    def _max_custom_id():
        keys = list(hv.Store._custom_options.get("matplotlib", {}).keys())
        return max(keys) if keys else 0

    def test_recustomization_id_growth_is_not_exponential(self):
        el = hv.Curve([1, 2, 3])
        max_ids = []
        for _ in range(20):
            el = el.opts(color="red")
            max_ids.append(self._max_custom_id())

        growth = [b - a for a, b in itertools.pairwise(max_ids)]
        assert max(growth) <= 4, f"id growth per call is not bounded: {max_ids}"

    def test_cross_backend_recustomization_keeps_children_distinct(self):
        c1 = hv.Curve([1, 2, 3]).opts(color="red", backend="bokeh")
        c2 = hv.Curve([3, 2, 1]).opts(color="green", backend="bokeh")
        ov = (c1 * c2).opts({"Curve": {"linewidth": 3}}, backend="matplotlib")

        first, second = ov.Curve.I, ov.Curve.II
        assert first.id != second.id
        assert hv.Store.lookup_options("bokeh", first, "style").kwargs["color"] == "red"
        assert hv.Store.lookup_options("bokeh", second, "style").kwargs["color"] == "green"

    def test_overlay_subset_update_keeps_ids_bounded(self):
        names = list("abc")
        curves = {k: hv.Curve([(0, 0)], label=k) for k in names}
        base = self._max_custom_id()
        max_ids = []
        for cycle in range(15):
            k = names[cycle % len(names)]  # only one curve gets fresh data
            curves[k] = hv.Curve([(x, x + cycle) for x in range(cycle + 2)], label=k)
            hv.Overlay(list(curves.values())).opts({"Curve": {"linewidth": 2}})
            max_ids.append(self._max_custom_id() - base)

        assert max_ids[-1] < 500, f"id growth is not polynomial: {max_ids}"

    def test_concurrent_recustomization_keeps_styles_distinct(self):
        n_workers = 8
        barrier = threading.Barrier(n_workers)

        def worker(w):
            barrier.wait()
            mismatches = 0
            for _ in range(120):
                el = hv.Curve([1, 2, 3], label=f"w{w}").opts(linewidth=w + 1)
                got = hv.Store.lookup_options("matplotlib", el, "style").kwargs.get("linewidth")
                mismatches += got != w + 1
            return mismatches

        # A normal opts() runs within one thread-scheduling quantum, so the id
        # race stays hidden; a tiny switch interval forces interleaving.
        old = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                mismatches = sum(executor.map(worker, range(n_workers)))
        finally:
            sys.setswitchinterval(old)

        assert mismatches == 0, f"concurrent customizations collided on ids: {mismatches}"

    def test_concurrent_id_reservation_yields_disjoint_blocks(self):
        n_workers = 8
        barrier = threading.Barrier(n_workers)

        def worker(_):
            barrier.wait()
            ids = []
            for n in range(1, 120):
                start = hv.StoreOptions.reserve_ids(n)
                ids.extend(range(start, start + n))
            return ids

        old = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)  # force interleaving of the read-modify-write
        try:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                reserved = [i for block in executor.map(worker, range(n_workers)) for i in block]
        finally:
            sys.setswitchinterval(old)

        assert len(reserved) == len(set(reserved)), "reserved id blocks overlap"
