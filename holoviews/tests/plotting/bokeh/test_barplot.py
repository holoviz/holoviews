from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from bokeh.models import (
    CategoricalColorMapper,
    DatetimeAxis,
    LinearAxis,
    LinearColorMapper,
)

import holoviews as hv
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.testing import assert_data_equal

from ...plotting.utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer


class TestBarPlot(TestBokehPlot):
    def test_bars_hover_ensure_kdims_sanitized(self):
        obj = hv.Bars(np.random.rand(10, 2), kdims=["Dim with spaces"])
        obj = obj.opts(tools=["hover"])
        self._test_hover_info(obj, [("Dim with spaces", "@{Dim_with_spaces}"), ("y", "@{y}")])

    def test_bars_hover_ensure_vdims_sanitized(self):
        obj = hv.Bars(np.random.rand(10, 2), vdims=["Dim with spaces"])
        obj = obj.opts(tools=["hover"])
        self._test_hover_info(obj, [("x", "@{x}"), ("Dim with spaces", "@{Dim_with_spaces}")])

    def test_bars_suppress_legend(self):
        bars = hv.Bars([("A", 1), ("B", 2)]).opts(show_legend=False)
        plot = bokeh_renderer.get_plot(bars)
        plot.initialize_plot()
        fig = plot.state
        assert len(fig.legend) == 0

    def test_empty_bars(self):
        bars = hv.Bars([], kdims=["x", "y"], vdims=["z"])
        plot = bokeh_renderer.get_plot(bars)
        plot.initialize_plot()
        source = plot.handles["source"]
        for v in source.data.values():
            assert len(v) == 0

    def test_bars_single_value(self):
        df = pd.DataFrame({"time": [1], "value": [-1]})
        bars = hv.Bars(df)
        plot = bokeh_renderer.get_plot(bars)
        source = plot.handles["source"]
        assert source.data["time"] == np.array([1])
        assert source.data["value"] == np.array([-1])

    def test_bars_grouped_categories(self):
        bars = hv.Bars(
            [("A", 0, 1), ("A", 1, -1), ("B", 0, 2)], kdims=["Index", "Category"], vdims=["Value"]
        )
        plot = bokeh_renderer.get_plot(bars)
        source = plot.handles["source"]
        assert [tuple(x) for x in source.data["xoffsets"]] == [("A", "0"), ("B", "0"), ("A", "1")]
        assert list(source.data["Category"]) == ["0", "0", "1"]
        assert_data_equal(source.data["Value"], np.array([1, 2, -1]))
        x_range = plot.handles["x_range"]
        assert x_range.factors == [("A", "0"), ("A", "1"), ("B", "0"), ("B", "1")]

    def test_bars_multi_level_sorted(self):
        box = hv.Bars(
            (["A", "B"] * 15, [3, 10, 1] * 10, np.random.randn(30)), ["Group", "Category"], "Value"
        ).aggregate(function=np.mean)
        plot = bokeh_renderer.get_plot(box)
        x_range = plot.handles["x_range"]
        assert x_range.factors == [
            ("A", "1"),
            ("A", "3"),
            ("A", "10"),
            ("B", "1"),
            ("B", "3"),
            ("B", "10"),
        ]

    def test_box_whisker_multi_level_sorted_alphanumerically(self):
        box = hv.Bars(
            ([3, 10, 1] * 10, ["A", "B"] * 15, np.random.randn(30)), ["Group", "Category"], "Value"
        ).aggregate(function=np.mean)
        plot = bokeh_renderer.get_plot(box)
        x_range = plot.handles["x_range"]
        assert x_range.factors == [
            ("1", "A"),
            ("1", "B"),
            ("3", "A"),
            ("3", "B"),
            ("10", "A"),
            ("10", "B"),
        ]

    def test_bars_multi_level_two_factors_in_overlay(self):
        # See: https://github.com/holoviz/holoviews/pull/5850
        box = hv.Bars(
            (["1", "2", "3"] * 10, ["A", "B"] * 15, np.random.randn(30)),
            ["Group", "Category"],
            "Value",
        ).aggregate(function=np.mean)
        overlay = hv.Overlay([box])
        plot = bokeh_renderer.get_plot(overlay)
        left_axis = plot.handles["plot"].left[0]
        assert isinstance(left_axis, LinearAxis)

    def test_bars_positive_negative_mixed(self):
        bars = hv.Bars(
            [("A", 0, 1), ("A", 1, -1), ("B", 0, 2)], kdims=["Index", "Category"], vdims=["Value"]
        )
        plot = bokeh_renderer.get_plot(bars.opts(stacked=True))
        source = plot.handles["source"]
        assert list(source.data["Category"]) == ["1", "0", "0"]
        assert list(source.data["Index"]) == ["A", "A", "B"]
        assert_data_equal(source.data["top"], np.array([0, 1, 2]))
        assert_data_equal(source.data["bottom"], np.array([-1, 0, 0]))

    @pytest.mark.parametrize(
        ("df", "baseline", "expected_bottom"),
        [
            (
                pd.DataFrame({"x": ["a", "b", "c"], "high": [3, 5, 4], "low": [1, 2, 1.5]}),
                "low",
                [1, 2, 1.5],
            ),
            # baseline by dimension index; 'low' is dimension 2 (x, high, low)
            (pd.DataFrame({"x": ["a", "b"], "high": [3, 5], "low": [1, 2]}), 2, [1, 2]),
            (
                pd.DataFrame({"x": ["a", "b"], "high": [3.0, 5.0], "low": [1.0, np.nan]}),
                "low",
                [1.0, np.nan],
            ),
            (
                pd.DataFrame(
                    {
                        "x": pd.date_range("2024-01-01", periods=3),
                        "high": [3.0, 5.0, 4.0],
                        "low": [1.0, 2.0, 1.5],
                    }
                ),
                "low",
                [1.0, 2.0, 1.5],
            ),
        ],
        ids=["by_name", "by_index", "nan", "datetime_x"],
    )
    def test_bars_baseline_floating_source_data(self, df, baseline, expected_bottom):
        bars = hv.Bars(df, "x", ["high", "low"]).opts(baseline=baseline)
        plot = bokeh_renderer.get_plot(bars)
        source = plot.handles["source"]
        glyph = plot.handles["glyph"]
        assert_data_equal(source.data["bottom"], np.array(expected_bottom))
        # Both ends are column references (floating), not the scalar 0 baseline.
        assert property_to_dict(glyph.top) == "high"
        assert property_to_dict(glyph.bottom) == "bottom"

    def test_bars_baseline_floating_inverted(self):
        # invert_axes draws hbars: bottom/top map to the left/right ends.
        df = pd.DataFrame({"x": ["a", "b", "c"], "high": [3, 5, 4], "low": [1, 2, 1.5]})
        bars = hv.Bars(df, "x", ["high", "low"]).opts(baseline="low", invert_axes=True)
        plot = bokeh_renderer.get_plot(bars)
        source = plot.handles["source"]
        glyph = plot.handles["glyph"]
        assert_data_equal(source.data["bottom"], np.array([1, 2, 1.5]))
        assert property_to_dict(glyph.left) == "bottom"
        assert property_to_dict(glyph.right) == "high"

    def test_bars_baseline_floating_timedelta(self):
        # Gantt-style: timedelta value dimensions float between Start and End.
        df = pd.DataFrame(
            {
                "Task": ["Build", "Test", "Deploy"],
                "Start": pd.to_timedelta(["1h", "3h", "5h30m"]),
                "End": pd.to_timedelta(["3h", "5h30m", "7h"]),
            }
        )
        bars = hv.Bars(df, "Task", ["End", "Start"]).opts(baseline="Start", invert_axes=True)
        plot = bokeh_renderer.get_plot(bars)
        source = plot.handles["source"]
        glyph = plot.handles["glyph"]
        # Normalize timedelta units before comparing (assert_data_equal uses
        # almost-equal, which cannot promote timedelta64 to float).
        np.testing.assert_array_equal(
            np.asarray(source.data["bottom"]).astype("timedelta64[ns]"),
            np.asarray(bars.dimension_values("Start")).astype("timedelta64[ns]"),
        )
        np.testing.assert_array_equal(
            np.asarray(source.data["End"]).astype("timedelta64[ns]"),
            np.asarray(bars.dimension_values("End")).astype("timedelta64[ns]"),
        )
        assert property_to_dict(glyph.left) == "bottom"
        assert property_to_dict(glyph.right) == "End"

    def test_bars_baseline_floating_range_excludes_zero(self):
        # Floating bars span [low, high]; 0 must not be forced into the range.
        df = pd.DataFrame({"x": ["a", "b"], "high": [30.0, 40.0], "low": [10.0, 20.0]})
        bars = hv.Bars(df, "x", ["high", "low"]).opts(baseline="low", padding=0)
        plot = bokeh_renderer.get_plot(bars)
        y_range = plot.handles["y_range"]
        assert y_range.start == 10.0
        assert y_range.end == 40.0

    def test_bars_baseline_range_includes_zero_without_baseline(self):
        # Without baseline the same data is anchored at 0 (regression guard).
        df = pd.DataFrame({"x": ["a", "b"], "high": [30.0, 40.0], "low": [10.0, 20.0]})
        bars = hv.Bars(df, "x", ["high", "low"]).opts(padding=0)
        plot = bokeh_renderer.get_plot(bars)
        y_range = plot.handles["y_range"]
        assert y_range.start == 0

    def test_bars_baseline_range_uses_top_dim_when_baseline_is_vdims0(self):
        low_dim = hv.Dimension("low", soft_range=(0, None))
        df = pd.DataFrame({"x": ["a", "b"], "high": [30.0, 40.0], "low": [10.0, 20.0]})
        bars = hv.Bars(df, "x", [low_dim, "high"]).opts(baseline="low", padding=0)
        plot = bokeh_renderer.get_plot(bars)
        y_range = plot.handles["y_range"]
        assert y_range.start == 10.0
        assert y_range.end == 40.0

    @pytest.mark.parametrize("low", [[6.0, 8.0], [1.0, 8.0]], ids=["all_exceed", "one_exceeds"])
    def test_bars_baseline_exceeds_errors(self, low):
        # The baseline must be the lower end of every bar; an inverted range
        # (low > high) is a usage error, even for a single bar.
        df = pd.DataFrame({"x": ["a", "b"], "high": [3.0, 5.0], "low": low})
        bars = hv.Bars(df, "x", ["high", "low"]).opts(baseline="low")
        with pytest.raises(ValueError, match="exceed"):
            bokeh_renderer.get_plot(bars)

    def test_bars_baseline_low_first(self):
        # Order-flexible: baseline names the lower dim and the remaining value
        # dimension is the upper end, so ['Low', 'High'] + baseline='Low' works.
        df = pd.DataFrame({"x": ["a", "b", "c"], "low": [1, 2, 1.5], "high": [3, 5, 4]})
        bars = hv.Bars(df, "x", ["low", "high"]).opts(baseline="low")
        plot = bokeh_renderer.get_plot(bars)
        source = plot.handles["source"]
        glyph = plot.handles["glyph"]
        assert_data_equal(source.data["high"], np.array([3, 5, 4]))
        assert_data_equal(source.data["bottom"], np.array([1, 2, 1.5]))
        assert property_to_dict(glyph.top) == "high"
        assert property_to_dict(glyph.bottom) == "bottom"

    @pytest.mark.parametrize(
        ("vdims", "baseline"),
        [(["high"], "nope"), (["high"], "high")],
        ids=["unresolved", "only_value_dim"],
    )
    def test_bars_baseline_unusable_warns(self, vdims, baseline):
        # An unresolved baseline, or one that leaves no other value dimension
        # as the upper end, falls back to a zero baseline.
        df = pd.DataFrame({"x": ["a", "b"], "high": [3, 5], "low": [1, 2]})
        bars = hv.Bars(df, "x", vdims).opts(baseline=baseline)
        with ParamLogStream() as log:
            plot = bokeh_renderer.get_plot(bars)
        log_msg = log.stream.read()
        assert f"Could not use baseline dimension {baseline!r}" in log_msg
        assert "bottom" not in plot.handles["source"].data

    def test_bars_baseline_grouped(self):
        # Each grouped bar floats from its baseline (Low) up to vdims[0] (High).
        bars = hv.Bars(
            [("Q1", "E", 10, 2), ("Q1", "W", 7, 1), ("Q2", "E", 12, 3), ("Q2", "W", 9, 4)],
            kdims=["Quarter", "Region"],
            vdims=["High", "Low"],
        ).opts(baseline="Low")
        plot = bokeh_renderer.get_plot(bars)
        source = plot.handles["source"]
        # Order depends on the group iteration, so compare order-independently.
        assert property_to_dict(plot.handles["glyph"].bottom) == "bottom"
        assert sorted(source.data["bottom"]) == [1, 2, 3, 4]
        assert sorted(source.data["High"]) == [7, 9, 10, 12]

    def test_bars_baseline_stacked_errors(self):
        # Stacking defines the segment baselines, so a baseline is a usage error.
        bars = hv.Bars(
            [("A", 0, 1), ("A", 1, -1), ("B", 0, 2)], kdims=["Index", "Category"], vdims=["Value"]
        ).opts(stacked=True, baseline="Value")
        with pytest.raises(ValueError, match="stacked"):
            bokeh_renderer.get_plot(bars)

    def test_bars_logy(self):
        bars = hv.Bars([("A", 1), ("B", 2), ("C", 3)], kdims=["Index"], vdims=["Value"])
        plot = bokeh_renderer.get_plot(bars.opts(logy=True))
        source = plot.handles["source"]
        glyph = plot.handles["glyph"]
        y_range = plot.handles["y_range"]
        assert list(source.data["Index"]) == ["A", "B", "C"]
        assert_data_equal(source.data["Value"], np.array([1, 2, 3]))
        assert glyph.bottom == 0.01
        assert y_range.start == 0.01
        assert y_range.end == 3.348369522101713

    def test_bars_logy_explicit_range(self):
        bars = hv.Bars(
            [("A", 1), ("B", 2), ("C", 3)], kdims=["Index"], vdims=["Value"]
        ).redim.range(Value=(0.001, 3))
        plot = bokeh_renderer.get_plot(bars.opts(logy=True))
        source = plot.handles["source"]
        glyph = plot.handles["glyph"]
        y_range = plot.handles["y_range"]
        assert list(source.data["Index"]) == ["A", "B", "C"]
        assert_data_equal(source.data["Value"], np.array([1, 2, 3]))
        assert glyph.bottom == 0.001
        assert y_range.start == 0.001
        assert y_range.end == 3

    def test_bars_ylim(self):
        bars = hv.Bars([1, 2, 3]).opts(ylim=(0, 200))
        plot = bokeh_renderer.get_plot(bars)
        y_range = plot.handles["y_range"]
        assert y_range.start == 0
        assert y_range.end == 200

    def test_bars_padding_square(self):
        points = hv.Bars([(1, 2), (2, -1), (3, 3)]).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(points)
        y_range = plot.handles["y_range"]
        assert y_range.start == -1.4
        assert y_range.end == 3.4

    def test_bars_padding_square_positive(self):
        points = hv.Bars([(1, 2), (2, 1), (3, 3)]).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(points)
        y_range = plot.handles["y_range"]
        assert y_range.start == 0
        assert y_range.end == 3.2

    def test_bars_padding_square_negative(self):
        points = hv.Bars([(1, -2), (2, -1), (3, -3)]).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(points)
        y_range = plot.handles["y_range"]
        assert y_range.start == -3.2
        assert y_range.end == 0

    def test_bars_padding_nonsquare(self):
        bars = hv.Bars([(1, 2), (2, 1), (3, 3)]).opts(padding=0.1, width=600)
        plot = bokeh_renderer.get_plot(bars)
        y_range = plot.handles["y_range"]
        assert y_range.start == 0
        assert y_range.end == 3.2

    def test_bars_padding_logx(self):
        bars = hv.Bars([(1, 1), (2, 2), (3, 3)]).opts(padding=0.1, logx=True)
        plot = bokeh_renderer.get_plot(bars)
        y_range = plot.handles["y_range"]
        assert y_range.start == 0
        assert y_range.end == 3.2

    def test_bars_padding_logy(self):
        bars = hv.Bars([(1, 2), (2, 1), (3, 3)]).opts(padding=0.1, logy=True)
        plot = bokeh_renderer.get_plot(bars)
        y_range = plot.handles["y_range"]
        assert y_range.start == 0.01
        assert y_range.end == 3.3483695221017129

    def test_bars_boolean_kdims(self):
        data = pd.DataFrame(
            {"x1": [1, 1, 2, 2], "x2": [False, True, False, True], "y": [3, 1, 2, 2]}
        )
        bars = hv.Bars(data, kdims=["x1", "x2"])
        plot = bokeh_renderer.get_plot(bars)
        x_range = plot.handles["x_range"]
        assert x_range.factors == [("1", "False"), ("1", "True"), ("2", "False"), ("2", "True")]

    ###########################
    #    Styling mapping      #
    ###########################

    def test_bars_color_op(self):
        bars = hv.Bars(
            [(0, 0, "#000"), (0, 1, "#F00"), (0, 2, "#0F0")], vdims=["y", "color"]
        ).opts(color="color")
        plot = bokeh_renderer.get_plot(bars)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert cds.data["color"] == ["#000", "#F00", "#0F0"]
        assert property_to_dict(glyph.fill_color) == {"field": "color"}
        assert property_to_dict(glyph.line_color) == "black"

    def test_bars_linear_color_op(self):
        bars = hv.Bars([(0, 0, 0), (0, 1, 1), (0, 2, 2)], vdims=["y", "color"]).opts(color="color")
        plot = bokeh_renderer.get_plot(bars)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        cmapper = plot.handles["color_color_mapper"]
        assert isinstance(cmapper, LinearColorMapper)
        assert cmapper.low == 0
        assert cmapper.high == 2
        assert_data_equal(cds.data["color"], np.array([0, 1, 2]))
        assert property_to_dict(glyph.fill_color) == {"field": "color", "transform": cmapper}
        assert property_to_dict(glyph.line_color) == "black"

    def test_bars_categorical_color_op(self):
        bars = hv.Bars([(0, 0, "A"), (0, 1, "B"), (0, 2, "C")], vdims=["y", "color"]).opts(
            color="color"
        )
        plot = bokeh_renderer.get_plot(bars)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        cmapper = plot.handles["color_color_mapper"]
        assert isinstance(cmapper, CategoricalColorMapper)
        assert cmapper.factors == ["A", "B", "C"]
        assert cds.data["color"] == ["A", "B", "C"]
        assert property_to_dict(glyph.fill_color) == {"field": "color", "transform": cmapper}
        assert property_to_dict(glyph.line_color) == "black"

    def test_bars_line_color_op(self):
        bars = hv.Bars(
            [(0, 0, "#000"), (0, 1, "#F00"), (0, 2, "#0F0")], vdims=["y", "color"]
        ).opts(line_color="color")
        plot = bokeh_renderer.get_plot(bars)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert cds.data["line_color"] == ["#000", "#F00", "#0F0"]
        assert property_to_dict(glyph.fill_color) != {"field": "line_color"}
        assert property_to_dict(glyph.line_color) == {"field": "line_color"}

    def test_bars_fill_color_op(self):
        bars = hv.Bars(
            [(0, 0, "#000"), (0, 1, "#F00"), (0, 2, "#0F0")], vdims=["y", "color"]
        ).opts(fill_color="color")
        plot = bokeh_renderer.get_plot(bars)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert cds.data["fill_color"] == ["#000", "#F00", "#0F0"]
        assert property_to_dict(glyph.fill_color) == {"field": "fill_color"}
        assert property_to_dict(glyph.line_color) != {"field": "fill_color"}

    def test_bars_alpha_op(self):
        bars = hv.Bars([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)], vdims=["y", "alpha"]).opts(
            alpha="alpha"
        )
        plot = bokeh_renderer.get_plot(bars)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert_data_equal(cds.data["alpha"], np.array([0, 0.2, 0.7]))
        assert property_to_dict(glyph.fill_alpha) == {"field": "alpha"}

    def test_bars_line_alpha_op(self):
        bars = hv.Bars([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)], vdims=["y", "alpha"]).opts(
            line_alpha="alpha"
        )
        plot = bokeh_renderer.get_plot(bars)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert_data_equal(cds.data["line_alpha"], np.array([0, 0.2, 0.7]))
        assert property_to_dict(glyph.line_alpha) == {"field": "line_alpha"}
        assert property_to_dict(glyph.fill_alpha) != {"field": "line_alpha"}

    def test_bars_fill_alpha_op(self):
        bars = hv.Bars([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)], vdims=["y", "alpha"]).opts(
            fill_alpha="alpha"
        )
        plot = bokeh_renderer.get_plot(bars)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert_data_equal(cds.data["fill_alpha"], np.array([0, 0.2, 0.7]))
        assert property_to_dict(glyph.line_alpha) != {"field": "fill_alpha"}
        assert property_to_dict(glyph.fill_alpha) == {"field": "fill_alpha"}

    def test_bars_line_width_op(self):
        bars = hv.Bars([(0, 0, 1), (0, 1, 4), (0, 2, 8)], vdims=["y", "line_width"]).opts(
            line_width="line_width"
        )
        plot = bokeh_renderer.get_plot(bars)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert_data_equal(cds.data["line_width"], np.array([1, 4, 8]))
        assert property_to_dict(glyph.line_width) == {"field": "line_width"}

    def test_op_ndoverlay_value(self):
        colors = ["blue", "red"]
        overlay = hv.NdOverlay(
            {color: hv.Bars(np.arange(i + 2)) for i, color in enumerate(colors)}, "Color"
        ).opts("Bars", fill_color="Color")
        plot = bokeh_renderer.get_plot(overlay)
        for subplot, color in zip(plot.subplots.values(), colors, strict=True):
            assert subplot.handles["glyph"].fill_color == color

    def test_bars_continuous_data_list_same_interval(self):
        bars = hv.Bars(([0, 1, 2], [10, 20, 30]))
        plot = bokeh_renderer.get_plot(bars)
        np.testing.assert_almost_equal(plot.handles["glyph"].width, 0.8)

    def test_bars_continuous_data_list_same_interval_custom_width(self):
        bars = hv.Bars(([0, 1, 2], [10, 20, 30])).opts(bar_width=0.5)
        plot = bokeh_renderer.get_plot(bars)
        assert plot.handles["glyph"].width == 0.5

    def test_bars_continuous_data_list_diff_interval(self):
        bars = hv.Bars(([0, 3, 10], [10, 20, 30]))
        plot = bokeh_renderer.get_plot(bars)
        np.testing.assert_almost_equal(plot.handles["glyph"].width, 0.11428571)

    def test_bars_continuous_datetime(self):
        bars = hv.Bars((pd.date_range("1/1/2000", periods=10), np.random.rand(10)))
        plot = bokeh_renderer.get_plot(bars)
        np.testing.assert_almost_equal(plot.handles["glyph"].width, 69120000.0)

    def test_bars_continuous_datetime_single(self):
        bars = hv.Bars([(pd.Timestamp("2024-01-01"), 5)])
        plot = bokeh_renderer.get_plot(bars)
        np.testing.assert_almost_equal(plot.handles["glyph"].width, 69120000.0)

    def test_bars_continuous_datetime_duplicates(self):
        bars = hv.Bars([(pd.Timestamp("2024-01-01"), 5), (pd.Timestamp("2024-01-01"), 3)])
        plot = bokeh_renderer.get_plot(bars)
        np.testing.assert_almost_equal(plot.handles["glyph"].width, 69120000.0)

    def test_bars_continuous_datetime_timezone_in_overlay(self):
        # See: https://github.com/holoviz/holoviews/issues/6364
        bars = hv.Bars((pd.date_range("1/1/2000", periods=10, tz="UTC"), np.random.rand(10)))
        overlay = hv.Overlay([bars])
        plot = bokeh_renderer.get_plot(overlay)
        assert isinstance(plot.handles["xaxis"], DatetimeAxis)

    def test_bars_continuous_datetime_stacked(self):
        # See: https://github.com/holoviz/holoviews/issues/6288
        data = pd.DataFrame(
            {
                "x": pd.to_datetime(
                    [
                        "2017-01-01T00:00:00",
                        "2017-01-01T00:00:00",
                        "2017-01-01T01:00:00",
                        "2017-01-01T01:00:00",
                    ]
                ),
                "cat": ["A", "B", "A", "B"],
                "y": [1, 2, 3, 4],
            }
        )
        bars = hv.Bars(data, ["x", "cat"], ["y"]).opts(stacked=True)
        plot = bokeh_renderer.get_plot(bars)
        assert isinstance(plot.handles["xaxis"], DatetimeAxis)

    def test_bars_not_continuous_data_list(self):
        bars = hv.Bars([("A", 1), ("B", 2), ("C", 3)])
        plot = bokeh_renderer.get_plot(bars)
        assert plot.handles["glyph"].width == 0.8

    def test_bars_not_continuous_data_list_custom_width(self):
        bars = hv.Bars([("A", 1), ("B", 2), ("C", 3)]).opts(bar_width=1)
        plot = bokeh_renderer.get_plot(bars)
        assert plot.handles["glyph"].width == 1

    def test_bars_categorical_order(self):
        cells_dtype = pd.CategoricalDtype(
            pd.array(["~1M", "~10M", "~100M"], dtype="string"),
            ordered=True,
        )
        df = pd.DataFrame(
            dict(
                cells=cells_dtype.categories.astype(cells_dtype),
                time=pd.array([2.99, 18.5, 835.2]),
                function=pd.array(["read", "read", "read"]),
            )
        )

        bars = hv.Bars(df, ["function", "cells"], ["time"])
        plot = bokeh_renderer.get_plot(bars)
        x_factors = plot.handles["x_range"].factors

        np.testing.assert_equal(
            x_factors,
            [
                ("read", "~1M"),
                ("read", "~10M"),
                ("read", "~100M"),
            ],
        )

    def test_bars_group(self):
        samples = 100

        pets = ["Cat", "Dog", "Hamster", "Rabbit"]
        genders = ["Female", "Male", "N/A"]

        np.random.seed(100)
        pets_sample = np.random.choice(pets, samples)
        gender_sample = np.random.choice(genders, samples)

        bars = hv.Bars(
            (pets_sample, gender_sample, np.ones(samples)), ["Pets", "Gender"]
        ).aggregate(function=np.sum)
        plot = bokeh_renderer.get_plot(bars)
        assert plot.handles["glyph"].width == 0.8

    def test_bar_group_stacked(self):
        samples = 100

        pets = ["Cat", "Dog", "Hamster", "Rabbit"]
        genders = ["Female", "Male", "N/A"]

        np.random.seed(100)
        pets_sample = np.random.choice(pets, samples)
        gender_sample = np.random.choice(genders, samples)

        bars = (
            hv.Bars((pets_sample, gender_sample, np.ones(samples)), ["Pets", "Gender"])
            .aggregate(function=np.sum)
            .opts(stacked=True)
        )
        plot = bokeh_renderer.get_plot(bars)
        assert plot.handles["glyph"].width == 0.8

    def test_bar_stacked_stack_variable_sorted(self):
        # Check that if the stack dim is ordered
        df = pd.DataFrame({"a": [*range(50), *range(50)], "b": sorted("ab" * 50), "c": range(100)})
        bars = hv.Bars(df, kdims=["a", "b"], vdims=["c"]).opts(stacked=True)
        plot = bokeh_renderer.get_plot(bars)
        assert plot.handles["glyph"].width == 0.8

    def test_bar_narrow_non_monotonous_xvals(self):
        # Tests regression: https://github.com/holoviz/hvplot/issues/1450
        dic = {"ratio": [0.82, 1.11, 3, 6], "count": [1, 2, 1, 3]}
        bars = hv.Bars(dic, kdims=["ratio"], vdims=["count"])
        plot = bokeh_renderer.get_plot(bars)
        assert np.isclose(plot.handles["glyph"].width, 0.232)


@pytest.mark.parametrize("stacked", [True, False])
def test_grouped_bars_color(stacked):
    # Test for https://github.com/holoviz/holoviews/issues/6580
    data = {
        "A": ["A1", "A1", "A1", "A1", "A2", "A2", "A2", "A2", "A3", "A3", "A3", "A3"],
        "B": ["B1", "B2", "B3", "B4", "B1", "B2", "B3", "B4", "B1", "B2", "B3", "B4"],
        "count": [100, 50, 25, 10, 80, 60, 30, 15, 90, 45, 20, 5],
    }

    df = pd.DataFrame(data)
    bar = hv.Bars(df, kdims=["A", "B"], vdims=["count"]).opts(
        stacked=stacked,
        cmap="Category10",
        color="B",
    )

    output = bokeh_renderer.get_plot(bar).handles["cds"].data["color"]
    expected = sorted(data["B"])
    assert output == expected
