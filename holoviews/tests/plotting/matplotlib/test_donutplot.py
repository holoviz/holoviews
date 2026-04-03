import numpy as np
import pandas as pd
import pytest

import holoviews as hv

from .test_plot import TestMPLPlot, mpl_renderer


class TestDonutPlot(TestMPLPlot):
    def test_donut_simple(self):
        d = hv.Donut([("A", 30), ("B", 70)])
        plot = mpl_renderer.get_plot(d)
        assert len(plot.handles["axis"].patches) == 2

    @pytest.mark.parametrize(
        ("values", "expected_patches"),
        [
            ([("A", 50), ("B", 50)], 2),
            ([("A", 10), ("B", 20), ("C", 30)], 3),
            ([("Only", 42)], 1),
        ],
    )
    def test_donut_num_patches(self, values, expected_patches):
        d = hv.Donut(values)
        plot = mpl_renderer.get_plot(d)
        assert len(plot.handles["axis"].patches) == expected_patches

    def test_donut_dataframe_input(self):
        df = pd.DataFrame({"Category": ["A", "B", "C"], "Value": [10, 20, 30]})
        d = hv.Donut(df, kdims="Category", vdims="Value")
        plot = mpl_renderer.get_plot(d)
        assert len(plot.handles["axis"].patches) == 3

    def test_donut_inner_radius_zero(self):
        """inner_radius=0 should produce full pie wedges."""
        d = hv.Donut([("A", 30), ("B", 70)]).opts(inner_radius=0)
        plot = mpl_renderer.get_plot(d)
        patches = plot.handles["axis"].patches
        assert len(patches) == 2
        assert patches[0].width == 1.0

    def test_donut_wedge_angles(self):
        """Wedge angles should span the full 360 degrees."""
        d = hv.Donut([("A", 25), ("B", 75)])
        plot = mpl_renderer.get_plot(d)
        patches = plot.handles["axis"].patches
        total_angle = sum(p.theta2 - p.theta1 for p in patches)
        np.testing.assert_almost_equal(total_angle, 360.0)

    def test_donut_legend_present(self):
        d = hv.Donut([("A", 30), ("B", 70)]).opts(show_legend=True)
        plot = mpl_renderer.get_plot(d)
        legend = plot.handles["axis"].get_legend()
        assert legend is not None
        texts = [t.get_text() for t in legend.get_texts()]
        assert "A" in texts
        assert "B" in texts

    def test_donut_legend_disabled(self):
        d = hv.Donut([("A", 30), ("B", 70)]).opts(show_legend=False)
        plot = mpl_renderer.get_plot(d)
        legend = plot.handles["axis"].get_legend()
        assert legend is None

    def test_donut_empty(self):
        d = hv.Donut([], kdims=["x"], vdims=["y"])
        plot = mpl_renderer.get_plot(d)
        assert len(plot.handles["axis"].patches) == 0

    def test_donut_equal_aspect(self):
        d = hv.Donut([("A", 30), ("B", 70)])
        plot = mpl_renderer.get_plot(d)
        assert plot.handles["axis"].get_aspect() == "equal"

    def test_donut_all_zero_values(self):
        d = hv.Donut([("A", 0), ("B", 0)])
        plot = mpl_renderer.get_plot(d)
        assert len(plot.handles["axis"].patches) == 2

    def test_donut_show_labels(self):
        d = hv.Donut([("A", 30), ("B", 70)]).opts(show_labels=True)
        plot = mpl_renderer.get_plot(d)
        texts = [t.get_text() for t in plot.handles["axis"].texts]
        assert "A" in texts
        assert "B" in texts

    def test_donut_total_label_total(self):
        d = hv.Donut([("A", 300), ("B", 700)]).opts(total_label="total")
        plot = mpl_renderer.get_plot(d)
        texts = [t.get_text() for t in plot.handles["axis"].texts]
        assert "$1.0K" in texts

    def test_donut_total_label_custom(self):
        d = hv.Donut([("A", 30), ("B", 70)]).opts(total_label="Budget")
        plot = mpl_renderer.get_plot(d)
        texts = [t.get_text() for t in plot.handles["axis"].texts]
        assert "Budget" in texts

    def test_donut_total_label_disabled_for_pie(self):
        """total_label should not appear when inner_radius=0."""
        d = hv.Donut([("A", 30), ("B", 70)]).opts(total_label="total", inner_radius=0)
        plot = mpl_renderer.get_plot(d)
        texts = [t.get_text() for t in plot.handles["axis"].texts]
        assert not any("$" in t for t in texts)
