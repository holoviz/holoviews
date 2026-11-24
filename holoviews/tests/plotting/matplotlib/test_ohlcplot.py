from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import holoviews as hv

from .test_plot import TestMPLPlot, mpl_renderer


class TestOHLCPlot(TestMPLPlot):
    def _data(self, ohlc):
        # get_data is pure; ranges/style are only consumed by _apply_transforms,
        # which is a no-op for an empty style dict.
        plot = mpl_renderer.get_plot(ohlc)
        (bodies, wicks, colors), _style, axis_kwargs = plot.get_data(ohlc, {}, {})
        return plot, bodies, wicks, colors, axis_kwargs

    def test_bar_edgecolor_default_and_option(self):
        from matplotlib.colors import to_rgba

        plot = mpl_renderer.get_plot(hv.OHLC([(0, 10, 12, 9, 11)]))
        np.testing.assert_allclose(plot.handles["artist"].get_edgecolor()[0], to_rgba("black"))
        plot = mpl_renderer.get_plot(hv.OHLC([(0, 10, 12, 9, 11)]).opts(bar_edgecolor="navy"))
        np.testing.assert_allclose(plot.handles["artist"].get_edgecolor()[0], to_rgba("navy"))

    def test_body_geometry(self):
        # tuple order (x, open, high, low, close)
        ohlc = hv.OHLC([(0, 10, 12, 9, 11), (1, 11, 13, 10, 10.5)])
        _, bodies, _, _, _ = self._data(ohlc)
        # up candle spans open(10)->close(11)
        np.testing.assert_almost_equal(bodies[0].get_y(), 10)
        np.testing.assert_almost_equal(bodies[0].get_height(), 1)
        # down candle spans close(10.5)->open(11)
        np.testing.assert_almost_equal(bodies[1].get_y(), 10.5)
        np.testing.assert_almost_equal(bodies[1].get_height(), 0.5)

    @pytest.mark.parametrize(
        ("open_", "close", "up"),
        [(10, 11, True), (11, 10.5, False), (10, 10, True)],
        ids=["up", "down", "equal_is_up"],
    )
    def test_direction_color(self, open_, close, up):
        ohlc = hv.OHLC([(0, open_, 12, 9, close)])
        plot, _, _, colors, _ = self._data(ohlc)
        assert colors[0] == (plot.pos_color if up else plot.neg_color)

    def test_wick_segments(self):
        ohlc = hv.OHLC([(0, 10, 12, 9, 11)])
        _, _, wicks, _, _ = self._data(ohlc)
        np.testing.assert_almost_equal(np.asarray(wicks[0]), [(0, 9), (0, 12)])

    def test_body_width_from_min_spacing(self):
        ohlc = hv.OHLC([(0, 1, 2, 0, 1.5), (1, 1.5, 2.5, 1, 2), (2, 2, 3, 1.5, 2.5)])
        _, bodies, _, _, _ = self._data(ohlc)
        # default bar_width=0.5 over spacing 1 -> body width 0.5
        for i, x in enumerate([0, 1, 2]):
            np.testing.assert_almost_equal(bodies[i].get_width(), 0.5)
            np.testing.assert_almost_equal(bodies[i].get_x(), x - 0.25)

    def test_duplicate_x_does_not_collapse_width(self):
        ohlc = hv.OHLC([(0, 1, 2, 0, 1.5), (0, 1.2, 2.2, 0.2, 1.6), (1, 2, 3, 1.5, 2.5)])
        _, bodies, _, _, _ = self._data(ohlc)
        for body in bodies:
            np.testing.assert_almost_equal(body.get_width(), 0.5)

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
        _, bodies, _, _, axis_kwargs = self._data(ohlc)
        # one-day spacing in matplotlib date units -> body width 0.5
        np.testing.assert_almost_equal(bodies[0].get_width(), 0.5)
        # a date formatter is attached to the x dimension
        assert axis_kwargs["dimensions"][0].value_format is not None

    def test_invert_axes(self):
        ohlc = hv.OHLC([(0, 10, 12, 9, 11), (1, 11, 13, 10, 10.5)]).opts(invert_axes=True)
        _, bodies, wicks, _, _ = self._data(ohlc)
        # prices now run along x: rectangle x spans open..close
        np.testing.assert_almost_equal(bodies[0].get_x(), 10)
        np.testing.assert_almost_equal(bodies[0].get_width(), 1)
        np.testing.assert_almost_equal(np.asarray(wicks[0]), [(9, 0), (12, 0)])

    def test_nan_close(self):
        ohlc = hv.OHLC([(0, 10, 12, 9, 11), (1, np.nan, 13, 10, np.nan)])
        plot, bodies, _, colors, _ = self._data(ohlc)
        assert colors[1] == plot.pos_color
        assert np.isnan(bodies[1].get_height())

    def test_yrange_envelopes_low_high(self):
        # zero padding so the y-limits match low..high exactly
        ohlc = hv.OHLC([(0, 10, 20, 5, 11), (1, 11, 18, 6, 12)]).opts(padding=0)
        plot = mpl_renderer.get_plot(ohlc)
        b, t = plot.handles["axis"].get_ylim()
        np.testing.assert_almost_equal(b, 5)
        np.testing.assert_almost_equal(t, 20)

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
        _, bodies, _, _, _ = self._data(ohlc)
        assert len(bodies) == 2
        np.testing.assert_almost_equal(bodies[0].get_y(), 10)
