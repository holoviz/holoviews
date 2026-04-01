import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

import holoviews as hv

from .test_plot import TestMPLPlot, mpl_renderer


class TestWaterfallPlot(TestMPLPlot):
    def test_waterfall_simple(self):
        w = hv.Waterfall([("A", 10), ("B", -3), ("C", 5)])
        plot = mpl_renderer.get_plot(w)
        assert len(plot.handles["axis"].patches) == 4

    @pytest.mark.parametrize(
        ("patch_idx", "expected_y", "expected_height"),
        [
            (0, 0, 10),
            (1, 7, 3),
            (2, 7, 5),
            (3, 0, 12),
        ],
    )
    def test_waterfall_cumulation(self, patch_idx, expected_y, expected_height):
        w = hv.Waterfall([("A", 10), ("B", -3), ("C", 5)])
        plot = mpl_renderer.get_plot(w)
        patch = plot.handles["axis"].patches[patch_idx]
        np.testing.assert_almost_equal(patch.get_y(), expected_y)
        np.testing.assert_almost_equal(patch.get_height(), expected_height)

    @pytest.mark.parametrize(
        ("values", "patch_idx", "expected_y", "expected_height"),
        [
            ([("A", 5), ("B", 10)], 0, 0, 5),
            ([("A", 5), ("B", 10)], 1, 5, 10),
            ([("A", 5), ("B", 10)], 2, 0, 15),
            ([("A", -5), ("B", -10)], 0, -5, 5),
            ([("A", -5), ("B", -10)], 1, -15, 10),
            ([("A", -5), ("B", -10)], 2, -15, 15),
        ],
    )
    def test_waterfall_patch_geometry(self, values, patch_idx, expected_y, expected_height):
        w = hv.Waterfall(values)
        plot = mpl_renderer.get_plot(w)
        patch = plot.handles["axis"].patches[patch_idx]
        np.testing.assert_almost_equal(patch.get_y(), expected_y)
        np.testing.assert_almost_equal(patch.get_height(), expected_height)

    def test_waterfall_dataframe_input(self):
        df = pd.DataFrame({"Category": ["Revenue", "COGS", "Opex"], "Amount": [100, -40, -30]})
        w = hv.Waterfall(df, kdims="Category", vdims="Amount")
        plot = mpl_renderer.get_plot(w)
        assert len(plot.handles["axis"].patches) == 4

    @pytest.mark.parametrize(("show_total", "expected_patches"), [(True, 3), (False, 2)])
    def test_waterfall_show_total(self, show_total, expected_patches):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(show_total=show_total)
        plot = mpl_renderer.get_plot(w)
        assert len(plot.handles["axis"].patches) == expected_patches

    def test_waterfall_nan_user_data(self):
        """Regression: a genuine NaN in user data must not be treated as the total sentinel."""
        w = hv.Waterfall([("Revenue", 100.0), ("Missing", np.nan), ("Cost", -40.0)])
        plot = mpl_renderer.get_plot(w)
        assert len(plot.handles["axis"].patches) == 4

    def test_waterfall_total_label_collision(self):
        """Regression: a user bar named 'Total' must raise ValueError."""
        w = hv.Waterfall([("Revenue", 100), ("Total", 50)])
        with pytest.raises(ValueError, match="total label 'Total' conflicts"):
            mpl_renderer.get_plot(w)

    def test_waterfall_total_label_collision_custom_label(self):
        """Setting a different total_label resolves the collision."""
        w = hv.Waterfall([("Revenue", 100), ("Total", 50)]).opts(total_label="Net")
        plot = mpl_renderer.get_plot(w)
        assert len(plot.handles["axis"].patches) == 3

    def test_waterfall_all_negative_total_bar_orientation(self):
        """Regression: all-negative waterfall total bar must have height >= 0."""
        w = hv.Waterfall([("A", -5), ("B", -10)])
        plot = mpl_renderer.get_plot(w)
        for patch in plot.handles["axis"].patches:
            assert patch.get_height() >= 0, f"Negative bar height found: {patch.get_height()}"

    def test_waterfall_colors_default(self):
        w = hv.Waterfall([("A", 10), ("B", -3)])
        plot = mpl_renderer.get_plot(w)
        patches = plot.handles["axis"].patches
        assert patches[0].get_facecolor() == to_rgba("steelblue")
        assert patches[1].get_facecolor() == to_rgba("crimson")
        assert patches[2].get_facecolor() == to_rgba("steelblue")

    def test_waterfall_colors_custom(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(
            start_color="blue", positive_color="blue", negative_color="orange", total_color="gray"
        )
        plot = mpl_renderer.get_plot(w)
        patches = plot.handles["axis"].patches
        assert patches[0].get_facecolor() == to_rgba("blue")
        assert patches[1].get_facecolor() == to_rgba("orange")
        assert patches[2].get_facecolor() == to_rgba("gray")

    def test_waterfall_connectors_present(self):
        w = hv.Waterfall([("A", 10), ("B", -3), ("C", 5)])
        plot = mpl_renderer.get_plot(w)
        assert len(plot.handles["axis"].lines) == 3

    def test_waterfall_connectors_disabled(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(show_connectors=False)
        plot = mpl_renderer.get_plot(w)
        assert len(plot.handles["axis"].lines) == 0

    @pytest.mark.parametrize(
        ("show_total", "label", "present"),
        [
            (True, "Total", True),
            (False, "Total", False),
        ],
    )
    def test_waterfall_tick_labels(self, show_total, label, present):
        w = hv.Waterfall([("Revenue", 100), ("COGS", -40)]).opts(show_total=show_total)
        plot = mpl_renderer.get_plot(w)
        tick_labels = [t.get_text() for t in plot.handles["axis"].get_xticklabels()]
        assert "Revenue" in tick_labels
        assert "COGS" in tick_labels
        assert (label in tick_labels) == present

    def test_waterfall_legend_present(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(show_legend=True)
        plot = mpl_renderer.get_plot(w)
        legend = plot.handles["axis"].get_legend()
        assert legend is not None
        texts = [t.get_text() for t in legend.get_texts()]
        assert "Start" in texts
        assert "Negative" in texts
        assert "Total" in texts

    def test_waterfall_legend_only_present_kinds(self):
        w = hv.Waterfall([("A", 10), ("B", 20)]).opts(show_total=False, show_legend=True)
        plot = mpl_renderer.get_plot(w)
        texts = [t.get_text() for t in plot.handles["axis"].get_legend().get_texts()]
        assert "Start" in texts
        assert "Negative" not in texts

    def test_waterfall_empty(self):
        w = hv.Waterfall([], kdims=["x"], vdims=["y"])
        plot = mpl_renderer.get_plot(w)
        assert len(plot.handles["axis"].patches) == 0

    def test_waterfall_single_bar(self):
        w = hv.Waterfall([("Only", 42)])
        plot = mpl_renderer.get_plot(w)
        assert len(plot.handles["axis"].patches) == 2

    def test_waterfall_invert_axes(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(invert_axes=True)
        plot = mpl_renderer.get_plot(w)
        assert plot.handles["axis"].patches[0].get_width() == 10

    @pytest.mark.parametrize(("show_total", "expected_patches"), [(True, 5), (False, 2)])
    def test_waterfall_datetime_kdim(self, show_total, expected_patches):
        """Regression: datetime kdim must not raise DTypePromotionError."""
        dates = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"])
        amounts = [100, 50, -30, -10]
        if not show_total:
            dates = dates[:2]
            amounts = amounts[:2]
        df = pd.DataFrame({"Category": dates, "Amount": amounts})
        w = hv.Waterfall(df, kdims="Category", vdims="Amount").opts(show_total=show_total)
        plot = mpl_renderer.get_plot(w)
        assert len(plot.handles["axis"].patches) == expected_patches

    def test_waterfall_datetime_kdim_total_label_in_ticks(self):
        """Total bar label appears in x-axis tick labels even with datetime kdim."""
        df = pd.DataFrame(
            {
                "Category": pd.to_datetime(["2021-01-01", "2021-01-02"]),
                "Amount": [100, -30],
            }
        )
        w = hv.Waterfall(df, kdims="Category", vdims="Amount").opts(total_label="Net")
        plot = mpl_renderer.get_plot(w)
        tick_labels = [t.get_text() for t in plot.handles["axis"].get_xticklabels()]
        assert "Net" in tick_labels
