from __future__ import annotations

import numpy as np
import pandas as pd

import holoviews as hv
from holoviews.testing import assert_data_equal

from .test_plot import TestPlotlyPlot


class TestOHLCPlot(TestPlotlyPlot):
    def test_candlestick_trace(self):
        # tuple order (x, open, high, low, close)
        ohlc = hv.OHLC([(0, 10, 12, 9, 11), (1, 11, 13, 10, 10.5)])
        state = self._get_plot_state(ohlc)
        trace = state["data"][0]
        assert trace["type"] == "candlestick"
        assert_data_equal(trace["open"], np.array([10, 11]))
        assert_data_equal(trace["high"], np.array([12, 13]))
        assert_data_equal(trace["low"], np.array([9, 10]))
        assert_data_equal(trace["close"], np.array([11, 10.5]))

    def test_direction_colors(self):
        ohlc = hv.OHLC([(0, 10, 12, 9, 11)]).opts(pos_color="#123456", neg_color="#654321")
        trace = self._get_plot_state(ohlc)["data"][0]
        assert trace["increasing"]["fillcolor"] == "#123456"
        assert trace["decreasing"]["fillcolor"] == "#654321"

    def test_wick_color_default_and_override(self):
        trace = self._get_plot_state(hv.OHLC([(0, 10, 12, 9, 11)]))["data"][0]
        assert trace["increasing"]["line"]["color"] == "black"
        assert trace["decreasing"]["line"]["color"] == "black"
        trace = self._get_plot_state(hv.OHLC([(0, 10, 12, 9, 11)]).opts(wick_color="gray"))[
            "data"
        ][0]
        assert trace["increasing"]["line"]["color"] == "gray"
        assert trace["decreasing"]["line"]["color"] == "gray"

    def test_rangeslider_hidden(self):
        ohlc = hv.OHLC([(0, 10, 12, 9, 11)])
        state = self._get_plot_state(ohlc)
        assert state["layout"]["xaxis"]["rangeslider"]["visible"] is False

    def test_yrange_envelopes_low_high(self):
        ohlc = hv.OHLC([(0, 10, 20, 5, 11), (1, 11, 18, 6, 12)]).opts(padding=0)
        state = self._get_plot_state(ohlc)
        b, t = state["layout"]["yaxis"]["range"]
        assert b <= 5
        assert t >= 20

    def test_datetime_axis(self):
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
        trace = self._get_plot_state(ohlc)["data"][0]
        assert trace["type"] == "candlestick"
        assert len(trace["x"]) == 3

    def test_nan_values(self):
        ohlc = hv.OHLC([(0, 10, 12, 9, 11), (1, np.nan, 13, 10, np.nan)])
        trace = self._get_plot_state(ohlc)["data"][0]
        assert np.isnan(np.asarray(trace["open"], dtype=float)[1])
        assert np.isnan(np.asarray(trace["close"], dtype=float)[1])

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
        trace = self._get_plot_state(ohlc)["data"][0]
        assert trace["type"] == "candlestick"
        assert_data_equal(trace["close"], np.array([11, 10.5]))
