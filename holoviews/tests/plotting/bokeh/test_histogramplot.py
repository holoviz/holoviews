import datetime as dt

import numpy as np
from bokeh.models import CategoricalColorMapper, DatetimeAxis, LinearColorMapper

import holoviews as hv
from holoviews.operation import histogram
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.testing import assert_data_equal, assert_element_equal

from ...utils import LoggingComparison
from .test_plot import TestBokehPlot, bokeh_renderer


class TestSideHistogramPlot(LoggingComparison, TestBokehPlot):
    def test_side_histogram_no_cmapper(self):
        points = hv.Points(np.random.rand(100, 2))
        plot = bokeh_renderer.get_plot(points.hist())
        plot.initialize_plot()
        adjoint_plot = next(iter(plot.subplots.values()))
        main_plot = adjoint_plot.subplots["main"]
        right_plot = adjoint_plot.subplots["right"]
        assert "color_mapper" not in main_plot.handles
        assert "color_mapper" not in right_plot.handles

    def test_side_histogram_cmapper(self):
        """Assert histogram shares colormapper"""
        x, y = np.mgrid[-50:51, -50:51] * 0.1
        img = hv.Image(np.sin(x**2 + y**2), bounds=(-1, -1, 1, 1))
        plot = bokeh_renderer.get_plot(img.hist())
        plot.initialize_plot()
        adjoint_plot = next(iter(plot.subplots.values()))
        main_plot = adjoint_plot.subplots["main"]
        right_plot = adjoint_plot.subplots["right"]
        assert main_plot.handles["color_mapper"] is right_plot.handles["color_mapper"]
        assert_element_equal(main_plot.handles["color_dim"], img.vdims[0])

    def test_side_histogram_cmapper_weighted(self):
        """Assert weighted histograms share colormapper"""
        x, y = np.mgrid[-50:51, -50:51] * 0.1
        img = hv.Image(np.sin(x**2 + y**2), bounds=(-1, -1, 1, 1))
        adjoint = img.hist(dimension=["x", "y"], weight_dimension="z", mean_weighted=True)
        plot = bokeh_renderer.get_plot(adjoint)
        plot.initialize_plot()
        adjoint_plot = next(iter(plot.subplots.values()))
        main_plot = adjoint_plot.subplots["main"]
        right_plot = adjoint_plot.subplots["right"]
        top_plot = adjoint_plot.subplots["top"]
        assert main_plot.handles["color_mapper"] is right_plot.handles["color_mapper"]
        assert main_plot.handles["color_mapper"] is top_plot.handles["color_mapper"]
        assert_element_equal(main_plot.handles["color_dim"], img.vdims[0])

    def test_histogram_datetime64_plot(self):
        dates = np.array([dt.datetime(2017, 1, i) for i in range(1, 5)])
        hist = histogram(hv.Dataset(dates, "Date"), num_bins=4)
        plot = bokeh_renderer.get_plot(hist)
        source = plot.handles["source"]
        data = {
            "top": np.array([1, 1, 1, 1]),
            "left": np.array(
                [
                    "2017-01-01T00:00:00.000000",
                    "2017-01-01T18:00:00.000000",
                    "2017-01-02T12:00:00.000000",
                    "2017-01-03T06:00:00.000000",
                ],
                dtype="datetime64[us]",
            ),
            "right": np.array(
                [
                    "2017-01-01T18:00:00.000000",
                    "2017-01-02T12:00:00.000000",
                    "2017-01-03T06:00:00.000000",
                    "2017-01-04T00:00:00.000000",
                ],
                dtype="datetime64[us]",
            ),
        }
        for k, v in data.items():
            assert_data_equal(source.data[k], v)
        xaxis = plot.handles["xaxis"]
        range_x = plot.handles["x_range"]
        assert isinstance(xaxis, DatetimeAxis)
        assert range_x.start == np.datetime64("2017-01-01T00:00:00.000000", "us")
        assert range_x.end == np.datetime64("2017-01-04T00:00:00.000000", "us")

    def test_histogram_padding_square(self):
        points = hv.Histogram([(1, 2), (2, -1), (3, 3)]).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == 0.19999999999999996
        assert x_range.end == 3.8
        assert y_range.start == -1.4
        assert y_range.end == 3.4

    def test_histogram_padding_square_positive(self):
        points = hv.Histogram([(1, 2), (2, 1), (3, 3)]).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == 0.19999999999999996
        assert x_range.end == 3.8
        assert y_range.start == 0
        assert y_range.end == 3.2

    def test_histogram_padding_square_negative(self):
        points = hv.Histogram([(1, -2), (2, -1), (3, -3)]).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == 0.19999999999999996
        assert x_range.end == 3.8
        assert y_range.start == -3.2
        assert y_range.end == 0

    def test_histogram_padding_nonsquare(self):
        histogram = hv.Histogram([(1, 2), (2, 1), (3, 3)]).opts(padding=0.1, width=600)
        plot = bokeh_renderer.get_plot(histogram)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == 0.35
        assert x_range.end == 3.65
        assert y_range.start == 0
        assert y_range.end == 3.2

    def test_histogram_padding_logx(self):
        histogram = hv.Histogram([(1, 1), (2, 2), (3, 3)]).opts(padding=0.1, logx=True)
        plot = bokeh_renderer.get_plot(histogram)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == 0.41158562699652224
        assert x_range.end == 4.2518491541367327
        assert y_range.start == 0
        assert y_range.end == 3.2

    def test_histogram_padding_logy(self):
        histogram = hv.Histogram([(1, 2), (2, 1), (3, 3)]).opts(padding=0.1, logy=True)
        plot = bokeh_renderer.get_plot(histogram)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == 0.19999999999999996
        assert x_range.end == 3.8
        assert y_range.start == 0.01
        assert y_range.end == 3.3483695221017129
        # We should not have logged 'Logarithmic axis range encountered value less than'
        last_line = self.log_handler.tail("WARNING", n=1)
        assert last_line == []

    def test_histogram_padding_datetime_square(self):
        histogram = hv.Histogram(
            [(np.datetime64(f"2016-04-0{i}", "ns"), i) for i in range(1, 4)]
        ).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(histogram)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == np.datetime64("2016-03-31T04:48:00.000000000")
        assert x_range.end == np.datetime64("2016-04-03T19:12:00.000000000")
        assert y_range.start == 0
        assert y_range.end == 3.2

    def test_histogram_padding_datetime_nonsquare(self):
        histogram = hv.Histogram(
            [(np.datetime64(f"2016-04-0{i}", "ns"), i) for i in range(1, 4)]
        ).opts(padding=0.1, width=600)
        plot = bokeh_renderer.get_plot(histogram)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == np.datetime64("2016-03-31T08:24:00.000000000")
        assert x_range.end == np.datetime64("2016-04-03T15:36:00.000000000")
        assert y_range.start == 0
        assert y_range.end == 3.2

    ###########################
    #    Styling mapping      #
    ###########################

    def test_histogram_color_op(self):
        histogram = hv.Histogram(
            [(0, 0, "#000"), (0, 1, "#F00"), (0, 2, "#0F0")], vdims=["y", "color"]
        ).opts(color="color")
        plot = bokeh_renderer.get_plot(histogram)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert cds.data["color"] == ["#000", "#F00", "#0F0"]
        assert property_to_dict(glyph.fill_color) == {"field": "color"}
        assert glyph.line_color == "black"

    def test_histogram_linear_color_op(self):
        histogram = hv.Histogram([(0, 0, 0), (0, 1, 1), (0, 2, 2)], vdims=["y", "color"]).opts(
            color="color"
        )
        plot = bokeh_renderer.get_plot(histogram)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        cmapper = plot.handles["color_color_mapper"]
        assert isinstance(cmapper, LinearColorMapper)
        assert cmapper.low == 0
        assert cmapper.high == 2
        np.testing.assert_array_equal(cds.data["color"], [0, 1, 2])
        assert property_to_dict(glyph.fill_color) == {"field": "color", "transform": cmapper}
        assert glyph.line_color == "black"

    def test_histogram_categorical_color_op(self):
        histogram = hv.Histogram(
            [(0, 0, "A"), (0, 1, "B"), (0, 2, "C")], vdims=["y", "color"]
        ).opts(color="color")
        plot = bokeh_renderer.get_plot(histogram)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        cmapper = plot.handles["color_color_mapper"]
        assert isinstance(cmapper, CategoricalColorMapper)
        assert cmapper.factors == ["A", "B", "C"]
        assert cds.data["color"] == ["A", "B", "C"]
        assert property_to_dict(glyph.fill_color) == {"field": "color", "transform": cmapper}
        assert glyph.line_color == "black"

    def test_histogram_line_color_op(self):
        histogram = hv.Histogram(
            [(0, 0, "#000"), (0, 1, "#F00"), (0, 2, "#0F0")], vdims=["y", "color"]
        ).opts(line_color="color")
        plot = bokeh_renderer.get_plot(histogram)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert cds.data["line_color"] == ["#000", "#F00", "#0F0"]
        assert property_to_dict(glyph.fill_color) != {"field": "line_color"}
        assert property_to_dict(glyph.line_color) == {"field": "line_color"}

    def test_histogram_fill_color_op(self):
        histogram = hv.Histogram(
            [(0, 0, "#000"), (0, 1, "#F00"), (0, 2, "#0F0")], vdims=["y", "color"]
        ).opts(fill_color="color")
        plot = bokeh_renderer.get_plot(histogram)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert cds.data["fill_color"] == ["#000", "#F00", "#0F0"]
        assert property_to_dict(glyph.fill_color) == {"field": "fill_color"}
        assert property_to_dict(glyph.line_color) != {"field": "fill_color"}

    def test_histogram_alpha_op(self):
        histogram = hv.Histogram([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)], vdims=["y", "alpha"]).opts(
            alpha="alpha"
        )
        plot = bokeh_renderer.get_plot(histogram)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert_data_equal(cds.data["alpha"], np.array([0, 0.2, 0.7]))
        assert property_to_dict(glyph.fill_alpha) == {"field": "alpha"}

    def test_histogram_line_alpha_op(self):
        histogram = hv.Histogram([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)], vdims=["y", "alpha"]).opts(
            line_alpha="alpha"
        )
        plot = bokeh_renderer.get_plot(histogram)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert_data_equal(cds.data["line_alpha"], np.array([0, 0.2, 0.7]))
        assert property_to_dict(glyph.line_alpha) == {"field": "line_alpha"}
        assert property_to_dict(glyph.fill_alpha) != {"field": "line_alpha"}

    def test_histogram_fill_alpha_op(self):
        histogram = hv.Histogram([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)], vdims=["y", "alpha"]).opts(
            fill_alpha="alpha"
        )
        plot = bokeh_renderer.get_plot(histogram)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert_data_equal(cds.data["fill_alpha"], np.array([0, 0.2, 0.7]))
        assert property_to_dict(glyph.line_alpha) != {"field": "fill_alpha"}
        assert property_to_dict(glyph.fill_alpha) == {"field": "fill_alpha"}

    def test_histogram_line_width_op(self):
        histogram = hv.Histogram(
            [(0, 0, 1), (0, 1, 4), (0, 2, 8)], vdims=["y", "line_width"]
        ).opts(line_width="line_width")
        plot = bokeh_renderer.get_plot(histogram)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert_data_equal(cds.data["line_width"], np.array([1, 4, 8]))
        assert property_to_dict(glyph.line_width) == {"field": "line_width"}

    def test_op_ndoverlay_value(self):
        colors = ["blue", "red"]
        overlay = hv.NdOverlay(
            {color: hv.Histogram(np.arange(i + 2)) for i, color in enumerate(colors)}, "Color"
        ).opts("Histogram", fill_color="Color")
        plot = bokeh_renderer.get_plot(overlay)
        for subplot, color in zip(plot.subplots.values(), colors, strict=True):
            assert subplot.handles["glyph"].fill_color == color

    def test_histogram_stack_ndoverlay(self):
        edges = np.array([0, 1, 2, 3])
        h1 = hv.Histogram((edges, np.array([1, 2, 3])))
        h2 = hv.Histogram((edges, np.array([6, 4, 2])))
        h3 = hv.Histogram((edges, np.array([8, 1, 2])))
        overlay = hv.NdOverlay({0: h1, 1: h2, 2: h3})
        stacked = hv.Histogram.stack(overlay)
        plot = bokeh_renderer.get_plot(stacked)
        expected_baselines = [
            np.array([0, 0, 0]),
            np.array([1, 2, 3]),
            np.array([7, 6, 5]),
        ]
        expected_tops = [
            np.array([1, 2, 3]),
            np.array([7, 6, 5]),
            np.array([15, 7, 7]),
        ]
        for (_, subplot), baseline, top in zip(
            plot.subplots.items(), expected_baselines, expected_tops, strict=True
        ):
            source = subplot.handles["source"]
            np.testing.assert_array_equal(source.data["bottom"], baseline)
            np.testing.assert_array_equal(source.data["top"], top)

    def test_histogram_stack_overlay(self):
        edges = np.array([0, 1, 2, 3])
        h1 = hv.Histogram((edges, np.array([1, 2, 3])), label="A")
        h2 = hv.Histogram((edges, np.array([6, 4, 2])), label="B")
        stacked = hv.Histogram.stack(h1 * h2)
        plot = bokeh_renderer.get_plot(stacked)
        subplots = list(plot.subplots.values())
        # First histogram: baseline=0
        source0 = subplots[0].handles["source"]
        np.testing.assert_array_equal(source0.data["bottom"], np.array([0, 0, 0]))
        np.testing.assert_array_equal(source0.data["top"], np.array([1, 2, 3]))
        # Second histogram: baseline=first's top
        source1 = subplots[1].handles["source"]
        np.testing.assert_array_equal(source1.data["bottom"], np.array([1, 2, 3]))
        np.testing.assert_array_equal(source1.data["top"], np.array([7, 6, 5]))
