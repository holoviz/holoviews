import datetime as dt

import numpy as np
import pytest

import holoviews as hv
from holoviews.core.options import AbbreviatedException
from holoviews.operation import histogram
from holoviews.plotting.util import hex2rgb

from ...utils import LoggingComparison
from .test_plot import TestMPLPlot, mpl_renderer


class TestHistogramPlot(LoggingComparison, TestMPLPlot):
    def test_histogram_datetime64_plot(self):
        dates = np.array([dt.datetime(2017, 1, i) for i in range(1, 5)])
        hist = histogram(hv.Dataset(dates, "Date"), num_bins=4)
        plot = mpl_renderer.get_plot(hist)
        artist = plot.handles["artist"]
        ax = plot.handles["axis"]
        assert ax.get_xlim() == (17167.0, 17170.0)
        bounds = [17167.0, 17167.75, 17168.5, 17169.25]
        assert [p.get_x() for p in artist.patches] == bounds

    def test_histogram_padding_square(self):
        points = hv.Histogram([(1, 2), (2, -1), (3, 3)]).opts(padding=0.1)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles["axis"].get_xlim(), plot.handles["axis"].get_ylim()
        assert x_range[0] == 0.19999999999999996
        assert x_range[1] == 3.8
        assert y_range[0] == -1.4
        assert y_range[1] == 3.4

    def test_histogram_padding_square_positive(self):
        points = hv.Histogram([(1, 2), (2, 1), (3, 3)]).opts(padding=0.1)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles["axis"].get_xlim(), plot.handles["axis"].get_ylim()
        assert x_range[0] == 0.19999999999999996
        assert x_range[1] == 3.8
        assert y_range[0] == 0
        assert y_range[1] == 3.2

    def test_histogram_padding_square_negative(self):
        points = hv.Histogram([(1, -2), (2, -1), (3, -3)]).opts(padding=0.1)
        plot = mpl_renderer.get_plot(points)
        x_range, y_range = plot.handles["axis"].get_xlim(), plot.handles["axis"].get_ylim()
        assert x_range[0] == 0.19999999999999996
        assert x_range[1] == 3.8
        assert y_range[0] == -3.2
        assert y_range[1] == 0

    def test_histogram_padding_nonsquare(self):
        histogram = hv.Histogram([(1, 2), (2, 1), (3, 3)]).opts(padding=0.1, aspect=2)
        plot = mpl_renderer.get_plot(histogram)
        x_range, y_range = plot.handles["axis"].get_xlim(), plot.handles["axis"].get_ylim()
        assert x_range[0] == 0.35
        assert x_range[1] == 3.65
        assert y_range[0] == 0
        assert y_range[1] == 3.2

    def test_histogram_padding_logx(self):
        histogram = hv.Histogram([(1, 1), (2, 2), (3, 3)]).opts(padding=0.1, logx=True)
        plot = mpl_renderer.get_plot(histogram)
        x_range, y_range = plot.handles["axis"].get_xlim(), plot.handles["axis"].get_ylim()
        assert x_range[0] == 0.41158562699652224
        assert x_range[1] == 4.2518491541367327
        assert y_range[0] == 0
        assert y_range[1] == 3.2

    def test_histogram_padding_logy(self):
        histogram = hv.Histogram([(1, 2), (2, 1), (3, 3)]).opts(padding=0.1, logy=True)
        plot = mpl_renderer.get_plot(histogram)
        x_range, y_range = plot.handles["axis"].get_xlim(), plot.handles["axis"].get_ylim()
        assert x_range[0] == 0.19999999999999996
        assert x_range[1] == 3.8
        assert y_range[0] == 0.01
        assert y_range[1] == 3.3483695221017129
        self.log_handler.assert_contains(
            "WARNING", "Logarithmic axis range encountered value less than"
        )

    def test_histogram_padding_datetime_square(self):
        histogram = hv.Histogram(
            [(np.datetime64(f"2016-04-0{i}", "ns"), i) for i in range(1, 4)]
        ).opts(padding=0.1)
        plot = mpl_renderer.get_plot(histogram)
        x_range, y_range = plot.handles["axis"].get_xlim(), plot.handles["axis"].get_ylim()
        assert x_range[0] == 16891.2
        assert x_range[1] == 16894.8
        assert y_range[0] == 0
        assert y_range[1] == 3.2

    def test_histogram_padding_datetime_nonsquare(self):
        histogram = hv.Histogram(
            [(np.datetime64(f"2016-04-0{i}", "ns"), i) for i in range(1, 4)]
        ).opts(padding=0.1, aspect=2)
        plot = mpl_renderer.get_plot(histogram)
        x_range, y_range = plot.handles["axis"].get_xlim(), plot.handles["axis"].get_ylim()
        assert x_range[0] == 16891.35
        assert x_range[1] == 16894.65
        assert y_range[0] == 0
        assert y_range[1] == 3.2

    ###########################
    #    Styling mapping      #
    ###########################

    def test_histogram_color_op(self):
        histogram = hv.Histogram(
            [(0, 0, "#000000"), (0, 1, "#FF0000"), (0, 2, "#00FF00")], vdims=["y", "color"]
        ).opts(color="color")
        plot = mpl_renderer.get_plot(histogram)
        artist = plot.handles["artist"]
        children = artist.get_children()
        for c, w in zip(children, ["#000000", "#FF0000", "#00FF00"], strict=True):
            assert c.get_facecolor() == (*(c / 255.0 for c in hex2rgb(w)), 1)

    def test_histogram_linear_color_op(self):
        histogram = hv.Histogram([(0, 0, 0), (0, 1, 1), (0, 2, 2)], vdims=["y", "color"]).opts(
            color="color"
        )
        msg = "ValueError: Mapping a continuous dimension to a color"
        with pytest.raises(AbbreviatedException, match=msg):
            mpl_renderer.get_plot(histogram)

    def test_histogram_categorical_color_op(self):
        histogram = hv.Histogram(
            [(0, 0, "A"), (0, 1, "B"), (0, 2, "C")], vdims=["y", "color"]
        ).opts(color="color")
        msg = "ValueError: Mapping a continuous dimension to a color"
        with pytest.raises(AbbreviatedException, match=msg):
            mpl_renderer.get_plot(histogram)

    def test_histogram_line_color_op(self):
        histogram = hv.Histogram(
            [(0, 0, "#000"), (0, 1, "#F00"), (0, 2, "#0F0")], vdims=["y", "color"]
        ).opts(edgecolor="color")
        plot = mpl_renderer.get_plot(histogram)
        artist = plot.handles["artist"]
        children = artist.get_children()
        assert children[0].get_edgecolor() == (0, 0, 0, 1)
        assert children[1].get_edgecolor() == (1, 0, 0, 1)
        assert children[2].get_edgecolor() == (0, 1, 0, 1)

    def test_histogram_alpha_op(self):
        histogram = hv.Histogram([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)], vdims=["y", "alpha"]).opts(
            alpha="alpha"
        )
        msg = 'ValueError: Mapping a dimension to the "alpha" style'
        with pytest.raises(AbbreviatedException, match=msg):
            mpl_renderer.get_plot(histogram)

    def test_histogram_line_width_op(self):
        histogram = hv.Histogram(
            [(0, 0, 1), (0, 1, 4), (0, 2, 8)], vdims=["y", "line_width"]
        ).opts(linewidth="line_width")
        plot = mpl_renderer.get_plot(histogram)
        artist = plot.handles["artist"]
        children = artist.get_children()
        for c, w in zip(children, [1, 4, 8], strict=True):
            assert c.get_linewidth() == w

    def test_op_ndoverlay_value(self):
        colors = ["blue", "red"]
        overlay = hv.NdOverlay(
            {color: hv.Histogram(np.arange(i + 2)) for i, color in enumerate(colors)}, "Color"
        ).opts("Histogram", facecolor="Color")
        plot = mpl_renderer.get_plot(overlay)
        colors = [(0, 0, 1, 1), (1, 0, 0, 1)]
        for subplot, color in zip(plot.subplots.values(), colors, strict=True):
            children = subplot.handles["artist"].get_children()
            for c in children:
                assert c.get_facecolor() == color

    def test_histogram_stack_ndoverlay(self):
        edges = np.array([0, 1, 2, 3])
        h1 = hv.Histogram((edges, np.array([1, 2, 3])))
        h2 = hv.Histogram((edges, np.array([6, 4, 2])))
        h3 = hv.Histogram((edges, np.array([8, 1, 2])))
        overlay = hv.NdOverlay({0: h1, 1: h2, 2: h3})
        stacked = hv.Histogram.stack(overlay)
        plot = mpl_renderer.get_plot(stacked)
        expected_baselines = [
            np.array([0, 0, 0]),
            np.array([1, 2, 3]),
            np.array([7, 6, 5]),
        ]
        expected_heights = [
            np.array([1, 2, 3]),
            np.array([6, 4, 2]),
            np.array([8, 1, 2]),
        ]
        for (_, subplot), baseline, heights in zip(
            plot.subplots.items(), expected_baselines, expected_heights, strict=True
        ):
            bars = subplot.handles["artist"]
            for i, bar in enumerate(bars):
                assert bar.get_y() == baseline[i]
                assert bar.get_height() == heights[i]

    def test_histogram_stack_overlay(self):
        edges = np.array([0, 1, 2, 3])
        h1 = hv.Histogram((edges, np.array([1, 2, 3])), label="A")
        h2 = hv.Histogram((edges, np.array([6, 4, 2])), label="B")
        stacked = hv.Histogram.stack(h1 * h2)
        plot = mpl_renderer.get_plot(stacked)
        subplots = list(plot.subplots.values())
        # First histogram: baseline=0
        bars0 = subplots[0].handles["artist"]
        for bar in bars0:
            assert bar.get_y() == 0
        np.testing.assert_array_equal([bar.get_height() for bar in bars0], [1, 2, 3])
        # Second histogram: baseline=first's top
        bars1 = subplots[1].handles["artist"]
        np.testing.assert_array_equal([bar.get_y() for bar in bars1], [1, 2, 3])
        np.testing.assert_array_equal([bar.get_height() for bar in bars1], [6, 4, 2])
