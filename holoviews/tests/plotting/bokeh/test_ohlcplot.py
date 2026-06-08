from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import holoviews as hv
from holoviews.plotting.bokeh.util import property_to_dict

from .test_plot import TestBokehPlot, bokeh_renderer


def _field(prop):
    """Return the column a glyph property maps to.

    Bokeh may store a vectorized property either as a ``Field``/``Value``
    object or as a bare field-name string, depending on version.
    """
    prop = property_to_dict(prop)
    return prop["field"] if isinstance(prop, dict) else prop


class TestOHLCPlot(TestBokehPlot):
    def _sources(self, ohlc):
        plot = bokeh_renderer.get_plot(ohlc)
        quad = plot.handles["quad_1_source"].data
        seg = plot.handles["segment_1_source"].data
        return plot, quad, seg

    def test_body_spans_open_close(self):
        # tuple order is (x, open, high, low, close)
        ohlc = hv.OHLC([(0, 10, 12, 9, 11)])
        _, quad, _ = self._sources(ohlc)
        np.testing.assert_equal(np.asarray(quad["bottom"]), np.array([10]))
        np.testing.assert_equal(np.asarray(quad["top"]), np.array([11]))

    @pytest.mark.parametrize(
        ("open_", "close", "up"),
        [(10, 11, True), (11, 10.5, False), (10, 10, True)],
        ids=["up", "down", "equal_is_up"],
    )
    def test_direction_color(self, open_, close, up):
        ohlc = hv.OHLC([(0, open_, 12, 9, close)])
        plot, quad, _ = self._sources(ohlc)
        expected = plot.pos_color if up else plot.neg_color
        assert quad["fill_color"][0] == expected

    def test_body_outline_matches_fill(self):
        ohlc = hv.OHLC([(0, 10, 12, 9, 11), (1, 11, 13, 10, 10.5)])
        plot, _, _ = self._sources(ohlc)
        quad_glyph = plot.handles["quad_1_glyph"]
        assert _field(quad_glyph.line_color) == _field(quad_glyph.fill_color) == "fill_color"

    def test_body_base_color_not_a_style_option(self):
        plot = bokeh_renderer.get_plot(hv.OHLC([(0, 10, 12, 9, 11)]))
        for opt in ("body_color", "body_fill_color", "body_line_color"):
            assert opt not in plot.style_opts
        assert "body_fill_alpha" in plot.style_opts
        assert "wick_line_color" in plot.style_opts

    def test_wick_spans_low_high(self):
        ohlc = hv.OHLC([(0, 10, 12, 9, 11)])
        _, _, seg = self._sources(ohlc)
        np.testing.assert_equal(np.asarray(seg["low"]), np.array([9]))
        np.testing.assert_equal(np.asarray(seg["high"]), np.array([12]))
        np.testing.assert_equal(np.asarray(seg["x"]), np.array([0]))

    def test_body_width_from_min_spacing(self):
        ohlc = hv.OHLC([(0, 1, 2, 0, 1.5), (1, 1.5, 2.5, 1, 2), (2, 2, 3, 1.5, 2.5)])
        _, quad, _ = self._sources(ohlc)
        # default bar_width=0.5, min spacing=1 -> half-width 0.25
        np.testing.assert_allclose(np.asarray(quad["left"]), np.array([0, 1, 2]) - 0.25)
        np.testing.assert_allclose(np.asarray(quad["right"]), np.array([0, 1, 2]) + 0.25)

    def test_bar_width_option(self):
        ohlc = hv.OHLC([(0, 1, 2, 0, 1.5), (2, 1.5, 2.5, 1, 2)]).opts(bar_width=1.0)
        _, quad, _ = self._sources(ohlc)
        # min spacing=2, bar_width=1.0 -> half-width 1.0
        np.testing.assert_allclose(np.asarray(quad["left"]), np.array([0, 2]) - 1.0)
        np.testing.assert_allclose(np.asarray(quad["right"]), np.array([0, 2]) + 1.0)

    def test_irregular_spacing_uses_minimum_gap(self):
        # gaps are 1 and 3; the smallest (1) sets the width
        ohlc = hv.OHLC([(0, 1, 2, 0, 1.5), (1, 1.5, 2.5, 1, 2), (4, 2, 3, 1.5, 2.5)])
        _, quad, _ = self._sources(ohlc)
        np.testing.assert_allclose(np.asarray(quad["left"]), np.array([0, 1, 4]) - 0.25)

    def test_unsorted_x_aligns_body_to_original_order(self):
        # spacing is 1; bodies must align to each row's own x, not the sorted x
        ohlc = hv.OHLC([(2, 1, 2, 0, 1.5), (0, 1.5, 2.5, 1, 2), (1, 2, 3, 1.5, 2.5)])
        _, quad, _ = self._sources(ohlc)
        np.testing.assert_allclose(np.asarray(quad["left"]), np.array([2, 0, 1]) - 0.25)
        np.testing.assert_allclose(np.asarray(quad["right"]), np.array([2, 0, 1]) + 0.25)

    def test_duplicate_x_does_not_collapse_width(self):
        # a zero gap (duplicate x) must be ignored when picking the min spacing
        ohlc = hv.OHLC([(0, 1, 2, 0, 1.5), (0, 1.2, 2.2, 0.2, 1.6), (1, 2, 3, 1.5, 2.5)])
        _, quad, _ = self._sources(ohlc)
        widths = np.asarray(quad["right"]) - np.asarray(quad["left"])
        np.testing.assert_allclose(widths, np.repeat(0.5, 3))

    def test_single_point_zero_width(self):
        ohlc = hv.OHLC([(5, 10, 12, 9, 11)])
        _, quad, _ = self._sources(ohlc)
        np.testing.assert_equal(np.asarray(quad["left"]), np.array([5]))
        np.testing.assert_equal(np.asarray(quad["right"]), np.array([5]))

    def test_empty_element_builds(self):
        ohlc = hv.OHLC([], vdims=["open", "high", "low", "close"])
        # should construct a plot without raising
        bokeh_renderer.get_plot(ohlc)

    def test_extra_vdim_renders(self):
        df = pd.DataFrame(
            {
                "x": [0, 1],
                "open": [10, 11],
                "high": [12, 13],
                "low": [9, 10],
                "close": [11, 10.5],
                "volume": [100, 200],
            }
        )
        ohlc = hv.OHLC(df, "x", ["open", "high", "low", "close", "volume"])
        _, quad, _ = self._sources(ohlc)
        np.testing.assert_equal(np.asarray(quad["bottom"]), np.array([10, 10.5]))

    def test_datetime_axis_width_pandas(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "open": [1, 2, 3],
                "high": [2, 3, 4],
                "low": [0.5, 1.5, 2.5],
                "close": [1.8, 2.5, 3.2],
            }
        )
        ohlc = hv.OHLC(df, "date", ["open", "high", "low", "close"])
        _, quad, _ = self._sources(ohlc)
        left = np.asarray(quad["left"])
        right = np.asarray(quad["right"])
        assert left.dtype.kind == "M"
        # default bar_width=0.5 over a 1-day spacing -> 12h total body width
        np.testing.assert_array_equal(right - left, np.repeat(np.timedelta64(12, "h"), 3))

    def test_datetime_axis_width_numpy_day_resolution(self):
        # numpy datetime64[D] has a coarse unit; the width must not truncate
        # to zero from the fractional bar_width (regression for the ns guard).
        dates = np.arange("2024-01-01", "2024-01-04", dtype="datetime64[D]")
        ohlc = hv.OHLC(
            (dates, [1, 2, 3], [2, 3, 4], [0.5, 1.5, 2.5], [1.8, 2.5, 3.2]),
            "date",
            ["open", "high", "low", "close"],
        )
        _, quad, _ = self._sources(ohlc)
        widths = np.asarray(quad["right"]) - np.asarray(quad["left"])
        assert np.all(widths.astype("timedelta64[h]") == np.timedelta64(12, "h"))

    def test_yrange_envelopes_low_high(self):
        # 'open' alone would clip the wicks; the value axis must span low..high
        ohlc = hv.OHLC([(0, 10, 20, 5, 11), (1, 11, 18, 6, 12)])
        plot = bokeh_renderer.get_plot(ohlc)
        y_range = plot.handles["y_range"]
        assert y_range.start <= 5
        assert y_range.end >= 20

    def test_yrange_respects_hard_range(self):
        ohlc = hv.OHLC([(0, 10, 20, 5, 11), (1, 11, 18, 6, 12)]).redim.range(high=(0, 50))
        plot = bokeh_renderer.get_plot(ohlc)
        y_range = plot.handles["y_range"]
        assert y_range.start == 0
        assert y_range.end == 50

    def test_default_axes_mapping(self):
        ohlc = hv.OHLC([(0, 10, 12, 9, 11)])
        plot, _, _ = self._sources(ohlc)
        quad_glyph = plot.handles["quad_1_glyph"]
        # body bottom/top come from the price-derived columns
        assert _field(quad_glyph.bottom) == "bottom"
        assert _field(quad_glyph.top) == "top"

    def test_invert_axes(self):
        ohlc = hv.OHLC([(0, 10, 12, 9, 11)]).opts(invert_axes=True)
        plot, _, _ = self._sources(ohlc)
        quad_glyph = plot.handles["quad_1_glyph"]
        # prices now map onto the x-axis: glyph left/right read the price columns
        assert _field(quad_glyph.left) == "bottom"
        assert _field(quad_glyph.right) == "top"

    def test_nan_close_falls_to_up_color_and_nan_body(self):
        ohlc = hv.OHLC([(0, 10, 12, 9, 11), (1, np.nan, 13, 10, np.nan)])
        plot, quad, _ = self._sources(ohlc)
        # NaN comparisons are False, so a NaN candle is treated as "up"
        assert quad["fill_color"][1] == plot.pos_color
        # and its body bounds are NaN, so Bokeh simply skips drawing it
        assert np.isnan(np.asarray(quad["top"], dtype=float)[1])
        assert np.isnan(np.asarray(quad["bottom"], dtype=float)[1])
