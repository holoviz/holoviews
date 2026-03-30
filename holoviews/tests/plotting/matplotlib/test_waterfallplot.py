import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

import holoviews as hv

from .test_plot import TestMPLPlot, mpl_renderer


class TestWaterfallPlot(TestMPLPlot):
    # ----------------------------------------------------------------
    # Basic rendering
    # ----------------------------------------------------------------

    def test_waterfall_simple(self):
        w = hv.Waterfall([("A", 10), ("B", -3), ("C", 5)])
        plot = mpl_renderer.get_plot(w)
        ax = plot.handles["axis"]
        # 3 data bars + 1 total bar = 4 patches
        assert len(ax.patches) == 4

    def test_waterfall_cumulation(self):
        w = hv.Waterfall([("A", 10), ("B", -3), ("C", 5)])
        plot = mpl_renderer.get_plot(w)
        ax = plot.handles["axis"]
        patches = ax.patches
        # A: bottom=0, height=10
        np.testing.assert_almost_equal(patches[0].get_y(), 0)
        np.testing.assert_almost_equal(patches[0].get_height(), 10)
        # B: bottom=7, height=3 (negative bar from 7→10)
        np.testing.assert_almost_equal(patches[1].get_y(), 7)
        np.testing.assert_almost_equal(patches[1].get_height(), 3)
        # C: bottom=7, height=5
        np.testing.assert_almost_equal(patches[2].get_y(), 7)
        np.testing.assert_almost_equal(patches[2].get_height(), 5)
        # Total: bottom=0, height=12
        np.testing.assert_almost_equal(patches[3].get_y(), 0)
        np.testing.assert_almost_equal(patches[3].get_height(), 12)

    def test_waterfall_all_positive(self):
        w = hv.Waterfall([("A", 5), ("B", 10)])
        plot = mpl_renderer.get_plot(w)
        ax = plot.handles["axis"]
        patches = ax.patches
        # A: bottom=0, top=5; B: bottom=5, top=15; Total: bottom=0, top=15
        np.testing.assert_almost_equal(patches[0].get_y(), 0)
        np.testing.assert_almost_equal(patches[0].get_height(), 5)
        np.testing.assert_almost_equal(patches[1].get_y(), 5)
        np.testing.assert_almost_equal(patches[1].get_height(), 10)
        np.testing.assert_almost_equal(patches[2].get_y(), 0)
        np.testing.assert_almost_equal(patches[2].get_height(), 15)

    def test_waterfall_all_negative(self):
        w = hv.Waterfall([("A", -5), ("B", -10)])
        plot = mpl_renderer.get_plot(w)
        ax = plot.handles["axis"]
        patches = ax.patches
        # A: bottom=-5, top=0 -> height=5
        np.testing.assert_almost_equal(patches[0].get_y(), -5)
        np.testing.assert_almost_equal(patches[0].get_height(), 5)
        # B: bottom=-15, top=-5 -> height=10
        np.testing.assert_almost_equal(patches[1].get_y(), -15)
        np.testing.assert_almost_equal(patches[1].get_height(), 10)

    def test_waterfall_dataframe_input(self):
        df = pd.DataFrame(
            {
                "Category": ["Revenue", "COGS", "Opex"],
                "Amount": [100, -40, -30],
            }
        )
        w = hv.Waterfall(df, kdims="Category", vdims="Amount")
        plot = mpl_renderer.get_plot(w)
        ax = plot.handles["axis"]
        assert len(ax.patches) == 4  # 3 + total

    # ----------------------------------------------------------------
    # show_total parameter
    # ----------------------------------------------------------------

    def test_waterfall_show_total_true(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(show_total=True)
        plot = mpl_renderer.get_plot(w)
        ax = plot.handles["axis"]
        assert len(ax.patches) == 3  # A, B, Total

    def test_waterfall_show_total_false(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(show_total=False)
        plot = mpl_renderer.get_plot(w)
        ax = plot.handles["axis"]
        assert len(ax.patches) == 2  # A, B only

    # ----------------------------------------------------------------
    # Colors
    # ----------------------------------------------------------------

    def test_waterfall_colors_default(self):
        w = hv.Waterfall([("A", 10), ("B", -3)])
        plot = mpl_renderer.get_plot(w)
        ax = plot.handles["axis"]
        patches = ax.patches
        assert patches[0].get_facecolor() == to_rgba(
            "steelblue"
        )  # start (first bar uses start_color)
        assert patches[1].get_facecolor() == to_rgba("crimson")  # negative
        assert patches[2].get_facecolor() == to_rgba("steelblue")  # total (inherits start_color)

    def test_waterfall_colors_custom(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(
            start_color="blue", positive_color="blue", negative_color="orange", total_color="gray"
        )
        plot = mpl_renderer.get_plot(w)
        ax = plot.handles["axis"]
        patches = ax.patches
        assert patches[0].get_facecolor() == to_rgba("blue")  # start
        assert patches[1].get_facecolor() == to_rgba("orange")  # negative
        assert patches[2].get_facecolor() == to_rgba("gray")  # total

    # ----------------------------------------------------------------
    # Connectors
    # ----------------------------------------------------------------

    def test_waterfall_connectors_present(self):
        w = hv.Waterfall([("A", 10), ("B", -3), ("C", 5)])
        plot = mpl_renderer.get_plot(w)
        ax = plot.handles["axis"]
        # 3 connectors between 4 bars
        assert len(ax.lines) == 3

    def test_waterfall_connectors_disabled(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(show_connectors=False)
        plot = mpl_renderer.get_plot(w)
        ax = plot.handles["axis"]
        assert len(ax.lines) == 0

    # ----------------------------------------------------------------
    # Tick labels
    # ----------------------------------------------------------------

    def test_waterfall_tick_labels(self):
        w = hv.Waterfall([("Revenue", 100), ("COGS", -40)])
        plot = mpl_renderer.get_plot(w)
        ax = plot.handles["axis"]
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "Revenue" in tick_labels
        assert "COGS" in tick_labels
        assert "Total" in tick_labels

    def test_waterfall_tick_labels_no_total(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(show_total=False)
        plot = mpl_renderer.get_plot(w)
        ax = plot.handles["axis"]
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "Total" not in tick_labels

    # ----------------------------------------------------------------
    # Legend
    # ----------------------------------------------------------------

    def test_waterfall_legend_present(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(show_legend=True)
        plot = mpl_renderer.get_plot(w)
        ax = plot.handles["axis"]
        legend = ax.get_legend()
        assert legend is not None
        texts = [t.get_text() for t in legend.get_texts()]
        assert "Start" in texts
        assert "Negative" in texts
        assert "Total" in texts

    def test_waterfall_legend_only_present_kinds(self):
        w = hv.Waterfall([("A", 10), ("B", 20)]).opts(show_total=False, show_legend=True)
        plot = mpl_renderer.get_plot(w)
        ax = plot.handles["axis"]
        legend = ax.get_legend()
        assert legend is not None
        texts = [t.get_text() for t in legend.get_texts()]
        assert "Start" in texts
        assert "Negative" not in texts

    # ----------------------------------------------------------------
    # Empty / edge cases
    # ----------------------------------------------------------------

    def test_waterfall_empty(self):
        w = hv.Waterfall([], kdims=["x"], vdims=["y"])
        plot = mpl_renderer.get_plot(w)
        ax = plot.handles["axis"]
        assert len(ax.patches) == 0

    def test_waterfall_single_bar(self):
        w = hv.Waterfall([("Only", 42)])
        plot = mpl_renderer.get_plot(w)
        ax = plot.handles["axis"]
        assert len(ax.patches) == 2  # Only + Total

    # ----------------------------------------------------------------
    # Invert axes
    # ----------------------------------------------------------------

    def test_waterfall_invert_axes(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(invert_axes=True)
        plot = mpl_renderer.get_plot(w)
        ax = plot.handles["axis"]
        patches = ax.patches
        # For horizontal bars, get_width() gives the value extent
        assert patches[0].get_width() == 10
