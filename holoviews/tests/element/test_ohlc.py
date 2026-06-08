from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import holoviews as hv


class TestOHLCElement:
    def test_default_dimensions(self):
        ohlc = hv.OHLC([(0, 1, 2, 0.5, 1.5)])
        assert ohlc.kdims[0].name == "x"
        assert [vd.name for vd in ohlc.vdims[:4]] == ["open", "high", "low", "close"]

    def test_construction_from_dataframe(self):
        df = pd.DataFrame(
            {
                "x": [0, 1, 2],
                "open": [10, 11, 12],
                "high": [12, 12, 13],
                "low": [9, 10, 11],
                "close": [11, 10, 12.5],
            }
        )
        ohlc = hv.OHLC(df, "x", ["open", "high", "low", "close"])
        assert len(ohlc) == 3
        np.testing.assert_equal(ohlc.dimension_values("close"), np.array([11, 10, 12.5]))

    def test_construction_with_custom_dims(self):
        df = pd.DataFrame({"t": [0, 1], "o": [1, 2], "h": [2, 3], "l": [0, 1], "c": [1.5, 2.5]})
        ohlc = hv.OHLC(df, "t", ["o", "h", "l", "c"])
        assert ohlc.kdims[0].name == "t"
        assert [vd.name for vd in ohlc.vdims] == ["o", "h", "l", "c"]

    def test_requires_four_value_dimensions(self):
        df = pd.DataFrame({"x": [0, 1], "open": [1, 2], "high": [2, 3], "low": [0, 1]})
        with pytest.raises(ValueError, match="length must be at least 4, not 3"):
            hv.OHLC(df, "x", ["open", "high", "low"])

    def test_extra_vdim_retained(self):
        df = pd.DataFrame(
            {
                "x": [0, 1],
                "open": [1, 2],
                "high": [2, 3],
                "low": [0, 1],
                "close": [1.5, 2.5],
                "volume": [100, 200],
            }
        )
        ohlc = hv.OHLC(df, "x", ["open", "high", "low", "close", "volume"])
        assert "volume" in [vd.name for vd in ohlc.vdims]
        np.testing.assert_equal(ohlc.dimension_values("volume"), np.array([100, 200]))

    def test_datetime_index_pandas(self):
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
        assert ohlc.dimension_values("date").dtype.kind == "M"

    def test_datetime_index_numpy(self):
        dates = np.arange("2024-01-01", "2024-01-04", dtype="datetime64[D]")
        ohlc = hv.OHLC(
            (dates, [1, 2, 3], [2, 3, 4], [0.5, 1.5, 2.5], [1.8, 2.5, 3.2]),
            "date",
            ["open", "high", "low", "close"],
        )
        assert ohlc.dimension_values("date").dtype.kind == "M"
        assert len(ohlc) == 3

    def test_nan_values_retained(self):
        ohlc = hv.OHLC([(0, 10, 12, 9, 11), (1, np.nan, 13, 10, np.nan)])
        assert len(ohlc) == 2
        close = np.asarray(ohlc.dimension_values("close"), dtype=float)
        assert np.isnan(close[1])
