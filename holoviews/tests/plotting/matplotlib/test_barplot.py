import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.text import Text

from holoviews.element import Bars

from ...utils import LoggingComparisonTestCase
from .test_plot import TestMPLPlot, mpl_renderer


class TestBarPlot(LoggingComparisonTestCase, TestMPLPlot):

    def test_bars_continuous_data_list_same_interval(self):
        bars = Bars(([0, 1, 2], [10, 20, 30]))
        plot = mpl_renderer.get_plot(bars)
        ax = plot.handles["axis"]
        np.testing.assert_almost_equal(ax.get_xlim(), (-0.4, 2.4))
        assert ax.patches[0].get_width() == 0.8

    def test_bars_continuous_data_list_diff_interval(self):
        bars = Bars(([0, 3, 10], [10, 20, 30]))
        plot = mpl_renderer.get_plot(bars)
        ax = plot.handles["axis"]
        np.testing.assert_almost_equal(ax.get_xlim(), (-1.2, 11.2))
        np.testing.assert_almost_equal(ax.patches[0].get_width(), 2.4)
        assert len(ax.get_xticks()) > 3

    def test_bars_continuous_datetime(self):
        bars = Bars((pd.date_range("1/1/2000", periods=10), np.random.rand(10)))
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

    def test_bars_not_continuous_data_list(self):
        bars = Bars([("A", 1), ("B", 2), ("C", 3)])
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

        bars = Bars(
            (pets_sample, gender_sample, np.ones(samples)), ["Pets", "Gender"]
        ).aggregate(function=np.sum)
        plot = mpl_renderer.get_plot(bars)
        ax = plot.handles["axis"]

        np.testing.assert_almost_equal(ax.get_xlim(), (-0.1333333,  3.6666667))
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
            Bars((pets_sample, gender_sample, np.ones(samples)), ["Pets", "Gender"])
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
        bars = Bars(
            ([3, 10, 1] * 10, ["A", "B"] * 15, np.random.randn(30)),
            ["Group", "Category"],
            "Value",
        ).aggregate(function=np.mean)
        plot = mpl_renderer.get_plot(bars)
        ax = plot.handles["axis"]

        np.testing.assert_almost_equal(ax.get_xlim(), (-0.2,  2.6))
        assert ax.patches[0].get_width() == 0.4
        assert len(ax.get_xticks()) > 3

        xticklabels = ['A', '1', 'B', 'A', '3', 'B', 'A', '10', 'B']
        for i, tick in enumerate(ax.get_xticklabels()):
            assert tick.get_text() == xticklabels[i]
