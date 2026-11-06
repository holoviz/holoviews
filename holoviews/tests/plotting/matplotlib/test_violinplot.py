from __future__ import annotations

import holoviews as hv
from holoviews.plotting.mpl.util import MPL_GE_3_9_0
from holoviews.testing import assert_data_equal

from .test_plot import TestMPLPlot, mpl_renderer


class TestMPLViolinPlot(TestMPLPlot):
    def test_violin_simple(self, rng):
        values = rng.random(100)
        violin = hv.Violin(values)
        plot = mpl_renderer.get_plot(violin)
        data, style, _axis_opts = plot.get_data(violin, {}, {})
        assert_data_equal(data[0][0], values)
        assert style["positions"] == [0]
        if MPL_GE_3_9_0:
            assert style["tick_labels"] == [""]
        else:
            assert style["labels"] == [""]

    def test_violin_simple_overlay(self, rng):
        values = rng.random(100)
        violin = hv.Violin(values) * hv.Violin(values)
        plot = mpl_renderer.get_plot(violin)
        p1, p2 = plot.subplots.values()
        assert_data_equal(
            p1.handles["boxes"][0].get_path().vertices, p2.handles["boxes"][0].get_path().vertices
        )
        for b1, b2 in zip(
            p1.handles["bodies"][0].get_paths(), p2.handles["bodies"][0].get_paths(), strict=True
        ):
            assert_data_equal(b1.vertices, b2.vertices)

    def test_violin_multi(self, rng):
        violin = hv.Violin((rng.integers(0, 2, 100), rng.random(100)), kdims=["A"]).sort()
        plot = mpl_renderer.get_plot(violin)
        data, style, _axis_opts = plot.get_data(violin, {}, {})
        assert_data_equal(data[0][0], violin.select(A=0).dimension_values(1))
        assert_data_equal(data[0][1], violin.select(A=1).dimension_values(1))
        assert style["positions"] == [0, 1]
        if MPL_GE_3_9_0:
            assert style["tick_labels"] == ["0", "1"]
        else:
            assert style["labels"] == ["0", "1"]
