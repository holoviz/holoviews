from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytest
from matplotlib.text import Text

import holoviews as hv

from ...utils import LoggingComparison
from .test_plot import TestMPLPlot, mpl_renderer


class TestBarPlot(LoggingComparison, TestMPLPlot):
    def test_bars_continuous_data_list_same_interval(self):
        bars = hv.Bars(([0, 1, 2], [10, 20, 30]))
        plot = mpl_renderer.get_plot(bars)
        ax = plot.handles["axis"]
        np.testing.assert_almost_equal(ax.get_xlim(), (-0.4, 2.4))
        assert ax.patches[0].get_width() == 0.8

    def test_bars_continuous_data_list_diff_interval(self):
        bars = hv.Bars(([0, 3, 10], [10, 20, 30]))
        plot = mpl_renderer.get_plot(bars)
        ax = plot.handles["axis"]
        np.testing.assert_almost_equal(ax.get_xlim(), (-1.2, 11.2))
        np.testing.assert_almost_equal(ax.patches[0].get_width(), 2.4)
        assert len(ax.get_xticks()) > 3

    def test_bars_continuous_datetime(self):
        bars = hv.Bars((pd.date_range("1/1/2000", periods=10), np.random.rand(10)))
        plot = mpl_renderer.get_plot(bars)
        ax = plot.handles["axis"]
        assert ax.get_xticklabels()[0].get_text() == "2000-01-01"
        assert ax.get_xticklabels()[-1].get_text() == "2000-01-10"
        assert ax.patches[0].get_width() == 0.8
        assert len(ax.get_xticks()) == 10

        bars.opts(xformatter=mdates.DateFormatter("%d"))
        plot = mpl_renderer.get_plot(bars)
        ax = plot.handles["axis"]
        assert ax.get_xticklabels()[0].get_text() == "01"
        assert ax.get_xticklabels()[-1].get_text() == "10"

    def test_bars_continuous_datetime_single(self):
        bars = hv.Bars([(pd.Timestamp("2024-01-01"), 5)])
        plot = mpl_renderer.get_plot(bars)
        ax = plot.handles["axis"]
        assert ax.patches[0].get_width() == 0.8

    def test_bars_continuous_datetime_duplicates(self):
        bars = hv.Bars([(pd.Timestamp("2024-01-01"), 5), (pd.Timestamp("2024-01-01"), 3)])
        plot = mpl_renderer.get_plot(bars)
        ax = plot.handles["axis"]
        assert ax.patches[0].get_width() == 0.8

    def test_bars_not_continuous_data_list(self):
        bars = hv.Bars([("A", 1), ("B", 2), ("C", 3)])
        plot = mpl_renderer.get_plot(bars)
        ax = plot.handles["axis"]
        np.testing.assert_almost_equal(ax.get_xlim(), (-0.4, 2.4))
        assert ax.patches[0].get_width() == 0.8
        np.testing.assert_equal(ax.get_xticks(), [0, 1, 2])
        np.testing.assert_equal(
            [xticklabel.get_text() for xticklabel in ax.get_xticklabels()],
            ["A", "B", "C"],
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
        plot = mpl_renderer.get_plot(bars)
        ax = plot.handles["axis"]

        np.testing.assert_almost_equal(ax.get_xlim(), (-0.1333333, 3.6666667))
        assert ax.patches[0].get_width() == 0.26666666666666666
        ticklabels = ax.get_xticklabels()
        expected = [
            Text(0.0, 0, "Female"),
            Text(0.26666666666666666, 0, "N/A"),
            Text(0.26693333333333336, -0.04, "Cat"),
            Text(0.5333333333333333, 0, "Male"),
            Text(1.0, 0, "Female"),
            Text(1.2666666666666666, 0, "N/A"),
            Text(1.2669333333333332, -0.04, "Rabbit"),
            Text(1.5333333333333332, 0, "Male"),
            Text(2.0, 0, "Female"),
            Text(2.2666666666666666, 0, "N/A"),
            Text(2.2669333333333332, -0.04, "Hamster"),
            Text(2.533333333333333, 0, "Male"),
            Text(3.0, 0, "Female"),
            Text(3.2666666666666666, 0, "N/A"),
            Text(3.2669333333333332, -0.04, "Dog"),
            Text(3.533333333333333, 0, "Male"),
        ]

        for i, ticklabel in enumerate(ticklabels):
            assert ticklabel.get_text() == expected[i].get_text()
            assert ticklabel.get_position() == expected[i].get_position()

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
        plot = mpl_renderer.get_plot(bars)
        ax = plot.handles["axis"]

        np.testing.assert_almost_equal(ax.get_xlim(), (-0.4, 3.4))
        assert ax.patches[0].get_width() == 0.8
        ticklabels = ax.get_xticklabels()
        expected = [
            Text(0.0, 0, "Cat"),
            Text(1.0, 0, "Rabbit"),
            Text(2.0, 0, "Hamster"),
            Text(3.0, 0, "Dog"),
        ]

        for i, ticklabel in enumerate(ticklabels):
            assert ticklabel.get_text() == expected[i].get_text()
            assert ticklabel.get_position() == expected[i].get_position()

    def test_group_dim(self):
        bars = hv.Bars(
            ([3, 10, 1] * 10, ["A", "B"] * 15, np.random.randn(30)),
            ["Group", "Category"],
            "Value",
        ).aggregate(function=np.mean)
        plot = mpl_renderer.get_plot(bars)
        ax = plot.handles["axis"]

        np.testing.assert_almost_equal(ax.get_xlim(), (-0.2, 2.6))
        assert ax.patches[0].get_width() == 0.4
        assert len(ax.get_xticks()) > 3

        xticklabels = ["A", "1", "B", "A", "3", "B", "A", "10", "B"]
        for i, tick in enumerate(ax.get_xticklabels()):
            assert tick.get_text() == xticklabels[i]

    @pytest.mark.parametrize(
        ("df", "baseline", "bottoms", "heights"),
        [
            (
                pd.DataFrame(
                    {"x": ["a", "b", "c"], "high": [3.0, 5.0, 4.0], "low": [1.0, 2.0, 1.5]}
                ),
                "low",
                [1.0, 2.0, 1.5],
                [2.0, 3.0, 2.5],
            ),
            # baseline by dimension index; 'low' is dimension 2 (x, high, low)
            (
                pd.DataFrame({"x": ["a", "b"], "high": [3.0, 5.0], "low": [1.0, 2.0]}),
                2,
                [1.0, 2.0],
                [2.0, 3.0],
            ),
            # A NaN baseline propagates rather than silently snapping to zero.
            (
                pd.DataFrame({"x": ["a", "b"], "high": [3.0, 5.0], "low": [1.0, np.nan]}),
                "low",
                [1.0, np.nan],
                [2.0, np.nan],
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
                [2.0, 3.0, 2.5],
            ),
        ],
        ids=["by_name", "by_index", "nan", "datetime_x"],
    )
    def test_bars_baseline_floating(self, df, baseline, bottoms, heights):
        # Bottom of each bar is the baseline; height spans up to the upper dim.
        bars = hv.Bars(df, "x", ["high", "low"]).opts(baseline=baseline)
        ax = mpl_renderer.get_plot(bars).handles["axis"]
        np.testing.assert_allclose([p.get_y() for p in ax.patches], bottoms)
        np.testing.assert_allclose([p.get_height() for p in ax.patches], heights)

    def test_bars_baseline_floating_inverted(self):
        df = pd.DataFrame({"x": ["a", "b"], "high": [3.0, 5.0], "low": [1.0, 2.0]})
        bars = hv.Bars(df, "x", ["high", "low"]).opts(baseline="low", invert_axes=True)
        plot = mpl_renderer.get_plot(bars)
        ax = plot.handles["axis"]
        np.testing.assert_allclose([p.get_x() for p in ax.patches], [1.0, 2.0])
        np.testing.assert_allclose([p.get_width() for p in ax.patches], [2.0, 3.0])

    def test_bars_baseline_floating_ylim_excludes_zero(self):
        df = pd.DataFrame({"x": ["a", "b"], "high": [30.0, 40.0], "low": [10.0, 20.0]})
        bars = hv.Bars(df, "x", ["high", "low"]).opts(baseline="low", padding=0)
        plot = mpl_renderer.get_plot(bars)
        y0, y1 = plot.handles["axis"].get_ylim()
        assert y0 == 10.0
        assert y1 == 40.0

    def test_bars_baseline_exceeds_errors(self):
        # The baseline must be the lower end of every bar; an inverted range
        # (low > high) is a usage error.
        df = pd.DataFrame({"x": ["a", "b"], "high": [3.0, 5.0], "low": [6.0, 8.0]})
        bars = hv.Bars(df, "x", ["high", "low"]).opts(baseline="low")
        with pytest.raises(ValueError, match="exceed"):
            mpl_renderer.get_plot(bars)

    def test_bars_baseline_low_first(self):
        # Order-flexible: ['low', 'high'] + baseline='low' spans low -> high.
        df = pd.DataFrame({"x": ["a", "b", "c"], "low": [1.0, 2.0, 1.5], "high": [3.0, 5.0, 4.0]})
        bars = hv.Bars(df, "x", ["low", "high"]).opts(baseline="low")
        plot = mpl_renderer.get_plot(bars)
        ax = plot.handles["axis"]
        np.testing.assert_allclose([p.get_y() for p in ax.patches], [1.0, 2.0, 1.5])
        np.testing.assert_allclose([p.get_height() for p in ax.patches], [2.0, 3.0, 2.5])

    def test_bars_baseline_grouped(self):
        # Each grouped bar floats from its baseline (Low) up to the upper dim (High).
        bars = hv.Bars(
            [("Q1", "E", 10, 2), ("Q1", "W", 7, 1), ("Q2", "E", 12, 3), ("Q2", "W", 9, 4)],
            kdims=["Quarter", "Region"],
            vdims=["High", "Low"],
        ).opts(baseline="Low")
        plot = mpl_renderer.get_plot(bars)
        ax = plot.handles["axis"]
        assert sorted(p.get_y() for p in ax.patches) == [1.0, 2.0, 3.0, 4.0]

    def test_bars_baseline_stacked_errors(self):
        bars = hv.Bars(
            [("A", 0, 1), ("A", 1, -1), ("B", 0, 2)], kdims=["Index", "Category"], vdims=["Value"]
        ).opts(stacked=True, baseline="Value")
        with pytest.raises(ValueError, match="stacked"):
            mpl_renderer.get_plot(bars)

    def test_bars_baseline_unresolved_warns(self):
        df = pd.DataFrame({"x": ["a", "b"], "high": [3, 5]})
        bars = hv.Bars(df, "x", ["high"]).opts(baseline="nope")
        mpl_renderer.get_plot(bars)
        self.log_handler.assert_contains("WARNING", "Could not use baseline dimension 'nope'")
