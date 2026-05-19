from __future__ import annotations

import numpy as np

import holoviews as hv
from holoviews.streams import Buffer

from .test_plot import TestBokehPlot, bokeh_renderer


class TestSpreadPlot(TestBokehPlot):
    def test_spread_stream_data(self):
        buffer = Buffer({"y": np.array([]), "yerror": np.array([]), "x": np.array([])})
        dmap = hv.DynamicMap(hv.Spread, streams=[buffer])
        plot = bokeh_renderer.get_plot(dmap)
        buffer.send({"y": [1, 2, 1, 4], "yerror": [0.5, 0.2, 0.1, 0.5], "x": [0, 1, 2, 3]})
        cds = plot.handles["cds"]
        np.testing.assert_array_equal(
            cds.data["x"], np.array([0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0])
        )
        np.testing.assert_array_equal(
            cds.data["y"], np.array([0.5, 1.8, 0.9, 3.5, 4.5, 1.1, 2.2, 1.5])
        )

    def test_spread_with_nans(self):
        spread = hv.Spread(
            [
                (0, 0, 0, 1),
                (1, 0, 0, 2),
                (2, 0, 0, 3),
                (3, np.nan, np.nan, np.nan),
                (4, 0, 0, 5),
                (5, 0, 0, 6),
                (6, 0, 0, 7),
            ],
            vdims=["y", "neg", "pos"],
        )
        plot = bokeh_renderer.get_plot(spread)
        cds = plot.handles["cds"]
        np.testing.assert_array_equal(
            cds.data["x"],
            np.array([0.0, 1.0, 2.0, 2.0, 1.0, 0.0, np.nan, 4.0, 5.0, 6.0, 6.0, 5.0, 4.0]),
        )
        np.testing.assert_array_equal(
            cds.data["y"],
            np.array([0.0, 0.0, 0.0, 3.0, 2.0, 1.0, np.nan, 0.0, 0.0, 0.0, 7.0, 6.0, 5.0]),
        )

    def test_spread_empty(self):
        spread = hv.Spread([])
        plot = bokeh_renderer.get_plot(spread)
        cds = plot.handles["cds"]
        assert cds.data["x"] == []
        assert cds.data["y"] == []

    def test_spread_padding_square(self):
        spread = hv.Spread([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(spread)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == 0.8
        assert x_range.end == 3.2
        assert y_range.start == 0.19999999999999996
        assert y_range.end == 3.8

    def test_spread_padding_hard_range(self):
        spread = (
            hv.Spread([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)])
            .redim.range(y=(0, 4))
            .opts(padding=0.1)
        )
        plot = bokeh_renderer.get_plot(spread)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == 0.8
        assert x_range.end == 3.2
        assert y_range.start == 0
        assert y_range.end == 4

    def test_spread_padding_soft_range(self):
        spread = (
            hv.Spread([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)])
            .redim.soft_range(y=(0, 3.5))
            .opts(padding=0.1)
        )
        plot = bokeh_renderer.get_plot(spread)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == 0.8
        assert x_range.end == 3.2
        assert y_range.start == 0
        assert y_range.end == 3.5

    def test_spread_padding_nonsquare(self):
        spread = hv.Spread([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).opts(padding=0.1, width=600)
        plot = bokeh_renderer.get_plot(spread)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == 0.9
        assert x_range.end == 3.1
        assert y_range.start == 0.19999999999999996
        assert y_range.end == 3.8

    def test_spread_padding_logx(self):
        spread = hv.Spread([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).opts(padding=0.1, logx=True)
        plot = bokeh_renderer.get_plot(spread)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert np.isclose(x_range.start, 0.89595845984076228)
        assert np.isclose(x_range.end, 3.3483695221017129)
        assert np.isclose(y_range.start, 0.2)
        assert np.isclose(y_range.end, 3.8)

    def test_spread_padding_logy(self):
        spread = hv.Spread([(1, 1, 0.5), (2, 2, 0.5), (3, 3, 0.5)]).opts(padding=0.1, logy=True)
        plot = bokeh_renderer.get_plot(spread)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert np.isclose(x_range.start, 0.8)
        assert np.isclose(x_range.end, 3.2)
        assert np.isclose(y_range.start, 0.41158562699652224)
        assert np.isclose(y_range.end, 4.2518491541367327)

    def test_spread_datetime_x(self):
        dates = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")
        spread = hv.Spread((dates, [1.0, 2.0, 3.0], [0.5, 0.5, 0.5]))
        plot = bokeh_renderer.get_plot(spread)
        cds = plot.handles["cds"]
        expected_x = np.array(
            ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-03", "2020-01-02", "2020-01-01"],
            dtype="datetime64[ns]",
        )
        expected_y = np.array([0.5, 1.5, 2.5, 3.5, 2.5, 1.5])
        np.testing.assert_array_equal(cds.data["x"], expected_x)
        np.testing.assert_array_equal(cds.data["y"], expected_y)

    def test_spread_datetime_x_with_nat(self):
        dates = np.array(
            ["2020-01-01", "2020-01-02", "NaT", "2020-01-04", "2020-01-05"], dtype="datetime64[ns]"
        )
        spread = hv.Spread((dates, [1.0, 2.0, np.nan, 4.0, 5.0], [0.5, 0.5, np.nan, 0.5, 0.5]))
        plot = bokeh_renderer.get_plot(spread)
        cds = plot.handles["cds"]
        expected_x = np.array(
            [
                "2020-01-01",
                "2020-01-02",
                "2020-01-02",
                "2020-01-01",
                "NaT",
                "2020-01-04",
                "2020-01-05",
                "2020-01-05",
                "2020-01-04",
            ],
            dtype="datetime64[ns]",
        )
        expected_y = np.array([0.5, 1.5, 2.5, 1.5, np.nan, 3.5, 4.5, 5.5, 4.5])
        np.testing.assert_array_equal(cds.data["x"], expected_x)
        np.testing.assert_array_equal(cds.data["y"], expected_y)

    def test_spread_timedelta_x(self):
        tds = np.array([0, 1, 2], dtype="timedelta64[D]")
        spread = hv.Spread((tds, [1.0, 2.0, 3.0], [0.5, 0.5, 0.5]))
        plot = bokeh_renderer.get_plot(spread)
        cds = plot.handles["cds"]
        expected_x = np.array([0, 1, 2, 2, 1, 0], dtype="timedelta64[D]")
        expected_y = np.array([0.5, 1.5, 2.5, 3.5, 2.5, 1.5])
        np.testing.assert_array_equal(cds.data["x"], expected_x)
        np.testing.assert_array_equal(cds.data["y"], expected_y)

    def test_spread_timedelta_x_with_nat(self):
        tds = np.array([0, 1, "NaT", 3, 4], dtype="timedelta64[D]")
        spread = hv.Spread((tds, [1.0, 2.0, np.nan, 4.0, 5.0], [0.5, 0.5, np.nan, 0.5, 0.5]))
        plot = bokeh_renderer.get_plot(spread)
        cds = plot.handles["cds"]
        expected_x = np.array([0, 1, 1, 0, "NaT", 3, 4, 4, 3], dtype="timedelta64[D]")
        expected_y = np.array([0.5, 1.5, 2.5, 1.5, np.nan, 3.5, 4.5, 5.5, 4.5])
        np.testing.assert_array_equal(cds.data["x"], expected_x)
        np.testing.assert_array_equal(cds.data["y"], expected_y)
