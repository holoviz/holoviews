import numpy as np
import pandas as pd

import holoviews as hv
from holoviews.testing import assert_data_equal

from .test_plot import TestBokehPlot, bokeh_renderer


class TestWaterfallPlot(TestBokehPlot):
    # ----------------------------------------------------------------
    # Basic rendering
    # ----------------------------------------------------------------

    def test_waterfall_simple(self):
        w = hv.Waterfall([("A", 10), ("B", -3), ("C", 5)])
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        # 3 data bars + 1 total bar = 4
        assert len(source.data["top"]) == 4
        assert len(source.data["bottom"]) == 4
        assert len(source.data["x"]) == 4

    def test_waterfall_cumulation(self):
        w = hv.Waterfall([("A", 10), ("B", -3), ("C", 5)])
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        # A: bottom=0, top=10; B: bottom=7, top=10; C: bottom=7, top=12
        # Total: bottom=0, top=12
        assert_data_equal(source.data["bottom"], np.array([0, 7, 7, 0]))
        assert_data_equal(source.data["top"], np.array([10, 10, 12, 12]))

    def test_waterfall_all_positive(self):
        w = hv.Waterfall([("A", 5), ("B", 10), ("C", 3)])
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        assert_data_equal(source.data["bottom"], np.array([0, 5, 15, 0]))
        assert_data_equal(source.data["top"], np.array([5, 15, 18, 18]))

    def test_waterfall_all_negative(self):
        w = hv.Waterfall([("A", -5), ("B", -10)])
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        # A: bottom=-5, top=0; B: bottom=-15, top=-5; Total: bottom=0, top=-15
        assert_data_equal(source.data["bottom"], np.array([-5, -15, 0]))
        assert_data_equal(source.data["top"], np.array([0, -5, -15]))

    def test_waterfall_dataframe_input(self):
        df = pd.DataFrame(
            {
                "Category": ["Revenue", "COGS", "Opex"],
                "Amount": [100, -40, -30],
            }
        )
        w = hv.Waterfall(df, kdims="Category", vdims="Amount")
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        assert len(source.data["top"]) == 4  # 3 + total

    # ----------------------------------------------------------------
    # show_total parameter
    # ----------------------------------------------------------------

    def test_waterfall_show_total_true(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(show_total=True)
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        assert len(source.data["x"]) == 3  # A, B, Total

    def test_waterfall_show_total_false(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(show_total=False)
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        assert len(source.data["x"]) == 2  # A, B only

    def test_waterfall_custom_total_label(self):
        w = hv.Waterfall([("A", 10)]).opts(total_label="Sum")
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        assert "Sum" in list(source.data["x"])

    # ----------------------------------------------------------------
    # Colors
    # ----------------------------------------------------------------

    def test_waterfall_colors_default(self):
        w = hv.Waterfall([("A", 10), ("B", -3)])
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        colors = source.data["fill_color"]
        assert colors[0] == "steelblue"  # start (first bar uses start_color)
        assert colors[1] == "crimson"  # negative
        assert colors[2] == "steelblue"  # total (inherits start_color)

    def test_waterfall_colors_custom(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(
            start_color="blue", positive_color="blue", negative_color="orange", total_color="gray"
        )
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        colors = source.data["fill_color"]
        assert colors[0] == "blue"  # start
        assert colors[1] == "orange"  # negative
        assert colors[2] == "gray"  # total

    # ----------------------------------------------------------------
    # Connectors
    # ----------------------------------------------------------------

    def test_waterfall_connectors_present(self):
        w = hv.Waterfall([("A", 10), ("B", -3), ("C", 5)])
        plot = bokeh_renderer.get_plot(w)
        assert "connectors" in plot.handles
        assert "connector_source" in plot.handles

    def test_waterfall_connectors_disabled(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(show_connectors=False)
        plot = bokeh_renderer.get_plot(w)
        assert "connectors" not in plot.handles

    def test_waterfall_connector_data(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(show_total=False)
        plot = bokeh_renderer.get_plot(w)
        src = plot.handles["connector_source"]
        # One connector from A→B at y=10 (cumulative after A)
        assert_data_equal(np.array(src.data["y0"]), np.array([10.0]))
        assert_data_equal(np.array(src.data["y1"]), np.array([10.0]))

    # ----------------------------------------------------------------
    # Categorical axis / factors
    # ----------------------------------------------------------------

    def test_waterfall_x_range_factors(self):
        w = hv.Waterfall([("A", 10), ("B", -3)])
        plot = bokeh_renderer.get_plot(w)
        x_range = plot.handles["x_range"]
        assert list(x_range.factors) == ["A", "B", "Total"]

    def test_waterfall_x_range_no_total(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(show_total=False)
        plot = bokeh_renderer.get_plot(w)
        x_range = plot.handles["x_range"]
        assert list(x_range.factors) == ["A", "B"]

    # ----------------------------------------------------------------
    # Empty / edge cases
    # ----------------------------------------------------------------

    def test_waterfall_empty(self):
        w = hv.Waterfall([], kdims=["x"], vdims=["y"])
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        for v in source.data.values():
            assert len(v) == 0

    def test_waterfall_single_bar(self):
        w = hv.Waterfall([("Only", 42)])
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        assert len(source.data["top"]) == 2  # Only + Total
        assert_data_equal(source.data["bottom"], np.array([0, 0]))
        assert_data_equal(source.data["top"], np.array([42, 42]))

    # ----------------------------------------------------------------
    # Hover
    # ----------------------------------------------------------------

    def test_waterfall_hover_column_lengths(self):
        """Regression: hover must not cause mismatched column lengths."""
        w = hv.Waterfall([("Revenue", 100), ("COGS", -40), ("Opex", -30), ("Tax", -10)]).opts(
            tools=["hover"]
        )
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        lengths = {k: len(v) for k, v in source.data.items()}
        expected_len = 5  # 4 data bars + 1 total
        for col, length in lengths.items():
            assert length == expected_len, (
                f"Column {col!r} has length {length}, expected {expected_len}"
            )

    def test_waterfall_hover_with_dataframe(self):
        """Regression: hover with DataFrame input and custom dim names."""
        df = pd.DataFrame(
            {
                "Category": ["Revenue", "COGS", "Opex", "Tax"],
                "Amount": [100, -40, -30, -10],
            }
        )
        w = hv.Waterfall(df, kdims="Category", vdims="Amount").opts(tools=["hover"])
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        lengths = {k: len(v) for k, v in source.data.items()}
        expected_len = 5
        for col, length in lengths.items():
            assert length == expected_len, (
                f"Column {col!r} has length {length}, expected {expected_len}"
            )

    def test_waterfall_hover_no_total(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(show_total=False, tools=["hover"])
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        lengths = {k: len(v) for k, v in source.data.items()}
        expected_len = 2
        for col, length in lengths.items():
            assert length == expected_len, (
                f"Column {col!r} has length {length}, expected {expected_len}"
            )

    def test_waterfall_hover_data_values(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(show_total=True, tools=["hover"])
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        assert "kind" in source.data
        assert list(source.data["kind"]) == [
            np.str_("start"),
            np.str_("negative"),
            np.str_("total"),
        ]

    # ----------------------------------------------------------------
    # Invert axes
    # ----------------------------------------------------------------

    def test_waterfall_invert_axes(self):
        from bokeh.models.glyphs import HBar

        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(w)
        glyph = plot.handles["glyph"]
        assert isinstance(glyph, HBar)

    # ----------------------------------------------------------------
    # Bar width
    # ----------------------------------------------------------------

    def test_waterfall_bar_width(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(bar_width=0.5)
        plot = bokeh_renderer.get_plot(w)
        glyph = plot.handles["glyph"]
        assert glyph.width == 0.5
