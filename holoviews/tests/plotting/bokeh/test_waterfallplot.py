import numpy as np
import pandas as pd
import pytest
from bokeh.models.glyphs import HBar

import holoviews as hv

from .test_plot import TestBokehPlot, bokeh_renderer


class TestWaterfallPlot(TestBokehPlot):
    def test_waterfall_simple(self):
        w = hv.Waterfall([("A", 10), ("B", -3), ("C", 5)])
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        assert len(source.data["top"]) == 4
        assert len(source.data["bottom"]) == 4
        assert len(source.data["x"]) == 4

    def test_waterfall_cumulation(self):
        w = hv.Waterfall([("A", 10), ("B", -3), ("C", 5)])
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        np.testing.assert_equal(source.data["bottom"], np.array([0, 7, 7, 0]))
        np.testing.assert_equal(source.data["top"], np.array([10, 10, 12, 12]))

    @pytest.mark.parametrize(
        ("values", "expected_bottom", "expected_top"),
        [
            ([("A", 5), ("B", 10), ("C", 3)], [0, 5, 15, 0], [5, 15, 18, 18]),
            ([("A", -5), ("B", -10)], [-5, -15, -15], [0, -5, 0]),
        ],
    )
    def test_waterfall_cumulation_parametrized(self, values, expected_bottom, expected_top):
        w = hv.Waterfall(values)
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        np.testing.assert_equal(source.data["bottom"], np.array(expected_bottom))
        np.testing.assert_equal(source.data["top"], np.array(expected_top))

    def test_waterfall_dataframe_input(self):
        df = pd.DataFrame({"Category": ["Revenue", "COGS", "Opex"], "Amount": [100, -40, -30]})
        w = hv.Waterfall(df, kdims="Category", vdims="Amount")
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        assert len(source.data["top"]) == 4

    @pytest.mark.parametrize(("show_total", "expected_len"), [(True, 3), (False, 2)])
    def test_waterfall_show_total(self, show_total, expected_len):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(show_total=show_total)
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        assert len(source.data["x"]) == expected_len

    def test_waterfall_custom_total_label(self):
        w = hv.Waterfall([("A", 10)]).opts(total_label="Sum")
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        assert "Sum" in list(source.data["x"])

    def test_waterfall_nan_user_data(self):
        w = hv.Waterfall([("Revenue", 100.0), ("Missing", np.nan), ("Cost", -40.0)]).opts(
            tools=["hover"]
        )
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        assert len(source.data["top"]) == 4
        kinds = list(source.data["kind"])
        assert kinds.count("total") == 1
        assert kinds[-1] == "total"

    def test_waterfall_total_label_collision(self):
        """Regression: a user bar named 'Total' must raise ValueError."""
        w = hv.Waterfall([("Revenue", 100), ("Total", 50)])
        with pytest.raises(ValueError, match="total_label 'Total' conflicts"):
            bokeh_renderer.get_plot(w)

    def test_waterfall_total_label_collision_custom_label(self):
        """Setting a different total_label resolves the collision."""
        w = hv.Waterfall([("Revenue", 100), ("Total", 50)]).opts(total_label="Net")
        plot = bokeh_renderer.get_plot(w)
        assert len(plot.handles["source"].data["x"]) == 3

    def test_waterfall_all_negative_total_bar_orientation(self):
        """Regression: all-negative waterfall total bar must satisfy top >= bottom."""
        w = hv.Waterfall([("A", -5), ("B", -10)])
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        bottoms = source.data["bottom"]
        tops = source.data["top"]
        assert all(t >= b for t, b in zip(tops, bottoms, strict=False)), (
            f"top < bottom found: tops={tops}, bottoms={bottoms}"
        )

    def test_waterfall_colors_default(self):
        w = hv.Waterfall([("A", 10), ("B", -3)])
        plot = bokeh_renderer.get_plot(w)
        colors = plot.handles["source"].data["fill_color"]
        assert colors[0] == "steelblue"
        assert colors[1] == "crimson"
        assert colors[2] == "steelblue"

    def test_waterfall_colors_custom(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(
            start_color="blue", positive_color="blue", negative_color="orange", total_color="gray"
        )
        plot = bokeh_renderer.get_plot(w)
        colors = plot.handles["source"].data["fill_color"]
        assert colors[0] == "blue"
        assert colors[1] == "orange"
        assert colors[2] == "gray"

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
        np.testing.assert_equal(np.array(src.data["y0"]), np.array([10.0]))
        np.testing.assert_equal(np.array(src.data["y1"]), np.array([10.0]))

    @pytest.mark.parametrize(
        ("show_total", "expected_factors"),
        [
            (True, ["A", "B", "Total"]),
            (False, ["A", "B"]),
        ],
    )
    def test_waterfall_x_range_factors(self, show_total, expected_factors):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(show_total=show_total)
        plot = bokeh_renderer.get_plot(w)
        assert list(plot.handles["x_range"].factors) == expected_factors

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
        assert len(source.data["top"]) == 2
        np.testing.assert_equal(source.data["bottom"], np.array([0, 0]))
        np.testing.assert_equal(source.data["top"], np.array([42, 42]))

    @pytest.mark.parametrize(
        ("data", "kdims", "vdims", "expected_len"),
        [
            ([("A", 100), ("B", -40), ("C", -30), ("D", -10)], None, None, 5),
            (pd.DataFrame({"x": list("ABCD"), "y": [100, -40, -30, -10]}), "x", "y", 5),
            ([("A", 10), ("B", -3)], None, None, 2),
        ],
    )
    def test_waterfall_hover_column_lengths(self, data, kdims, vdims, expected_len):
        """Regression: hover must not cause mismatched column lengths."""
        kwargs = {}
        if kdims:
            kwargs["kdims"] = kdims
        if vdims:
            kwargs["vdims"] = vdims
        opts = {"tools": ["hover"]}
        if expected_len == 2:
            opts["show_total"] = False
        w = hv.Waterfall(data, **kwargs).opts(**opts)
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        for col, vals in source.data.items():
            assert len(vals) == expected_len, (
                f"Column {col!r} has length {len(vals)}, expected {expected_len}"
            )

    def test_waterfall_hover_data_values(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(show_total=True, tools=["hover"])
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        assert "kind" in source.data
        assert list(source.data["kind"]) == ["start", "negative", "total"]

    def test_waterfall_invert_axes(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(w)
        assert isinstance(plot.handles["glyph"], HBar)

    def test_waterfall_bar_width(self):
        w = hv.Waterfall([("A", 10), ("B", -3)]).opts(bar_width=0.5)
        plot = bokeh_renderer.get_plot(w)
        assert plot.handles["glyph"].width == 0.5

    @pytest.mark.parametrize(("show_total", "expected_len"), [(True, 5), (False, 2)])
    def test_waterfall_datetime_kdim(self, show_total, expected_len):
        """Regression: datetime kdim must not raise DTypePromotionError."""
        dates = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"])
        amounts = [100, 50, -30, -10]
        if not show_total:
            dates = dates[:2]
            amounts = amounts[:2]
        df = pd.DataFrame({"Category": dates, "Amount": amounts})
        w = hv.Waterfall(df, kdims="Category", vdims="Amount").opts(show_total=show_total)
        plot = bokeh_renderer.get_plot(w)
        source = plot.handles["source"]
        assert len(source.data["top"]) == expected_len
        assert len(source.data["bottom"]) == expected_len

    def test_waterfall_datetime_kdim_total_label_in_factors(self):
        """Total bar label appears in x_range factors even with datetime kdim."""
        df = pd.DataFrame(
            {
                "Category": pd.to_datetime(["2021-01-01", "2021-01-02"]),
                "Amount": [100, -30],
            }
        )
        w = hv.Waterfall(df, kdims="Category", vdims="Amount").opts(total_label="Net")
        plot = bokeh_renderer.get_plot(w)
        assert "Net" in list(plot.handles["x_range"].factors)
