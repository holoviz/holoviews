"""
Unit tests of the StoreOptions class used to control custom options on
Store as used by the %opts magic.
"""

from __future__ import annotations

import itertools

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
    Re-customizing a retained, already-customized object must not grow the
    custom-option ids exponentially. The buggy behaviour minted a new id of
    ``old_id + max(all_ids) + 2``, which doubles the maximum id on every call
    when the object already holds the store's largest id.
    """

    def setup_method(self):
        hv.Store.current_backend = "matplotlib"

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

        # The object carries a single custom-option id, so a correct
        # relocation grows the maximum by a small constant per call (linear).
        # The exponential bug instead grows it by roughly the current maximum
        # each call, so per-step growth quickly exceeds any small bound.
        growth = [b - a for a, b in itertools.pairwise(max_ids)]
        assert max(growth) <= 4, f"id growth per call is not bounded: {max_ids}"

    def test_cross_backend_recustomization_keeps_children_distinct(self):
        # Children customized in bokeh hold distinct ids that are absent from
        # the matplotlib store. Re-customizing the overlay in matplotlib with a
        # spec targeting the children must relocate each child id to a distinct
        # new id; collapsing them onto a single id aliases the children and
        # loses their distinct bokeh styles.
        c1 = hv.Curve([1, 2, 3]).opts(color="red", backend="bokeh")
        c2 = hv.Curve([3, 2, 1]).opts(color="green", backend="bokeh")
        ov = (c1 * c2).opts({"Curve": {"linewidth": 3}}, backend="matplotlib")

        first, second = ov.Curve.I, ov.Curve.II
        assert first.id != second.id
        assert hv.Store.lookup_options("bokeh", first, "style").kwargs["color"] == "red"
        assert hv.Store.lookup_options("bokeh", second, "style").kwargs["color"] == "green"

    def test_overlay_subset_update_keeps_ids_bounded(self):
        # A dashboard overlay where only a subset of curves receives fresh data
        # each restyle cycle (e.g. jittered streaming updates where some curves
        # have no new message). Curves that skip an update are the same retained
        # objects carrying their previous ids; restyling the overlay relocates
        # them, and deriving the new id from the old value doubled them every
        # cycle. No static/reference curve is required -- the retention comes
        # purely from the partial updates.
        names = list("abc")
        curves = {k: hv.Curve([(0, 0)], label=k) for k in names}
        base = self._max_custom_id()
        max_ids = []
        for cycle in range(15):
            k = names[cycle % len(names)]  # only one curve gets fresh data
            curves[k] = hv.Curve([(x, x + cycle) for x in range(cycle + 2)], label=k)
            hv.Overlay(list(curves.values())).opts({"Curve": {"linewidth": 2}})
            max_ids.append(self._max_custom_id() - base)

        # Polynomial growth after the fix; exponential (doubling per cycle)
        # before it, so the final value blows far past any polynomial bound.
        assert max_ids[-1] < 500, f"id growth is not polynomial: {max_ids}"

    def test_concurrent_recustomization_is_thread_safe(self):
        # Customizing from multiple threads must not collide on relocated ids.
        # Allocating ids from a recomputed max(store)+1 offset gives concurrent
        # calls the same block, so one thread's freshly stored tree is reused
        # by another and the id is gone by the time it is propagated, raising
        # "New option id ... does not match any option trees". future.result()
        # re-raises any such error from the worker threads, failing the test.
        import threading
        from concurrent.futures import ThreadPoolExecutor

        n_workers = 8
        barrier = threading.Barrier(n_workers)

        def worker(w):
            barrier.wait()
            for i in range(60):
                el = hv.Curve([1, 2, 3], label=f"w{w}")
                for _ in range(4):
                    el = el.opts(linewidth=(i % 5) + 1)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(worker, w) for w in range(n_workers)]
            for future in futures:
                future.result()
