import unittest

import numpy as np

import holoviews as hv
from holoviews.element import (
    HLine, VLine, Text, Labels, Arrow, HSpan, VSpan, Slope,
    HLines, VLines, HSpans, VSpans,
)
from holoviews.plotting.bokeh.util import bokeh32

from .test_plot import TestBokehPlot, bokeh_renderer

if bokeh32:
    from bokeh.models import VSpan as BkVSpan, HSpan as BkHSpan, VStrip as BkVStrip, HStrip as BkHStrip


class TestHVLinePlot(TestBokehPlot):

    def test_hline_invert_axes(self):
        hline = HLine(1.1).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(hline)
        span = plot.handles['glyph']
        self.assertEqual(span.dimension, 'height')
        self.assertEqual(span.location, 1.1)

    def test_hline_plot(self):
        hline = HLine(1.1)
        plot = bokeh_renderer.get_plot(hline)
        span = plot.handles['glyph']
        self.assertEqual(span.dimension, 'width')
        self.assertEqual(span.location, 1.1)

    def test_vline_invert_axes(self):
        vline = VLine(1.1).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(vline)
        span = plot.handles['glyph']
        self.assertEqual(span.dimension, 'width')
        self.assertEqual(span.location, 1.1)

    def test_vline_plot(self):
        vline = VLine(1.1)
        plot = bokeh_renderer.get_plot(vline)
        span = plot.handles['glyph']
        self.assertEqual(span.dimension, 'height')
        self.assertEqual(span.location, 1.1)


class TestHVSpanPlot(TestBokehPlot):

    def test_hspan_invert_axes(self):
        hspan = HSpan(1.1, 1.5).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(hspan)
        span = plot.handles['glyph']

        self.assertEqual(span.left, 1.1)
        self.assertEqual(span.right, 1.5)
        self.assertEqual(span.bottom, None)
        self.assertEqual(span.top, None)
        self.assertEqual(span.visible, True)

    def test_hspan_plot(self):
        hspan = HSpan(1.1, 1.5)
        plot = bokeh_renderer.get_plot(hspan)
        span = plot.handles['glyph']
        self.assertEqual(span.left, None)
        self.assertEqual(span.right, None)
        self.assertEqual(span.bottom, 1.1)
        self.assertEqual(span.top, 1.5)
        self.assertEqual(span.visible, True)

    def test_hspan_empty(self):
        vline = HSpan(None)
        plot = bokeh_renderer.get_plot(vline)
        span = plot.handles['glyph']
        self.assertEqual(span.visible, False)

    def test_vspan_invert_axes(self):
        vspan = VSpan(1.1, 1.5).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(vspan)
        span = plot.handles['glyph']
        self.assertEqual(span.left, None)
        self.assertEqual(span.right, None)
        self.assertEqual(span.bottom, 1.1)
        self.assertEqual(span.top, 1.5)
        self.assertEqual(span.visible, True)

    def test_vspan_plot(self):
        vspan = VSpan(1.1, 1.5)
        plot = bokeh_renderer.get_plot(vspan)
        span = plot.handles['glyph']
        self.assertEqual(span.left, 1.1)
        self.assertEqual(span.right, 1.5)
        self.assertEqual(span.bottom, None)
        self.assertEqual(span.top, None)
        self.assertEqual(span.visible, True)

    def test_vspan_empty(self):
        vline = VSpan(None)
        plot = bokeh_renderer.get_plot(vline)
        span = plot.handles['glyph']
        self.assertEqual(span.visible, False)


class TestSlopePlot(TestBokehPlot):

    def test_slope(self):
        hspan = Slope(2, 10)
        plot = bokeh_renderer.get_plot(hspan)
        slope = plot.handles['glyph']
        self.assertEqual(slope.gradient, 2)
        self.assertEqual(slope.y_intercept, 10)

    def test_slope_invert_axes(self):
        hspan = Slope(2, 10).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(hspan)
        slope = plot.handles['glyph']
        self.assertEqual(slope.gradient, 0.5)
        self.assertEqual(slope.y_intercept, -5)



class TestTextPlot(TestBokehPlot):

    def test_text_plot(self):
        text = Text(0, 0, 'Test')
        plot = bokeh_renderer.get_plot(text)
        source = plot.handles['source']
        self.assertEqual(source.data, {'x': [0], 'y': [0], 'text': ['Test']})

    def test_text_plot_fontsize(self):
        text = Text(0, 0, 'Test', fontsize=18)
        plot = bokeh_renderer.get_plot(text)
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.text_font_size, '18Pt')

    def test_text_plot_rotation(self):
        text = Text(0, 0, 'Test', rotation=90)
        plot = bokeh_renderer.get_plot(text)
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.angle, np.pi/2.)

    def test_text_plot_rotation_style(self):
        text = Text(0, 0, 'Test').opts(angle=90)
        plot = bokeh_renderer.get_plot(text)
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.angle, np.pi/2.)


class TestArrowPlot(TestBokehPlot):

    def _compare_arrow_plot(self, plot, start, end):
        print(plot.handles)
        arrow_glyph = plot.handles['arrow_1_glyph']
        arrow_cds = plot.handles['arrow_1_source']
        label_glyph = plot.handles['text_1_glyph']

        label_cds = plot.handles['text_1_source']
        x0, y0 = start
        x1, y1 = end
        self.assertEqual(label_glyph.x, 'x')
        self.assertEqual(label_glyph.y, 'y')
        self.assertEqual(label_cds.data, {'x': [x0], 'y': [y0], 'text': ['Test']})
        self.assertEqual(arrow_glyph.x_start, 'x_start')
        self.assertEqual(arrow_glyph.y_start, 'y_start')
        self.assertEqual(arrow_glyph.x_end, 'x_end')
        self.assertEqual(arrow_glyph.y_end, 'y_end')
        self.assertEqual(arrow_cds.data, {'x_start': [x0], 'x_end': [x1],
                                          'y_start': [y0], 'y_end': [y1]})

    def test_arrow_plot_left(self):
        arrow = Arrow(0, 0, 'Test')
        plot = bokeh_renderer.get_plot(arrow)
        self._compare_arrow_plot(plot, (1/6., 0), (0, 0))

    def test_arrow_plot_up(self):
        arrow = Arrow(0, 0, 'Test', '^')
        plot = bokeh_renderer.get_plot(arrow)
        self._compare_arrow_plot(plot, (0, -1/6.), (0, 0))

    def test_arrow_plot_right(self):
        arrow = Arrow(0, 0, 'Test', '>')
        plot = bokeh_renderer.get_plot(arrow)
        self._compare_arrow_plot(plot, (-1/6., 0), (0, 0))

    def test_arrow_plot_down(self):
        arrow = Arrow(0, 0, 'Test', 'v')
        plot = bokeh_renderer.get_plot(arrow)
        self._compare_arrow_plot(plot, (0, 1/6.), (0, 0))


class TestLabelsPlot(TestBokehPlot):

    def test_labels_plot(self):
        text = Labels([(0, 0, 'Test')])
        plot = bokeh_renderer.get_plot(text)
        source = plot.handles['source']
        data = {'x': np.array([0]), 'y': np.array([0]), 'Label': ['Test']}
        for c, col in source.data.items():
            self.assertEqual(col, data[c])

    def test_labels_plot_rotation_style(self):
        text = Labels([(0, 0, 'Test')]).opts(angle=90)
        plot = bokeh_renderer.get_plot(text)
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.angle, np.pi/2.)


class TestHVLinesPlot(TestBokehPlot):

    def setUp(self):
        if not bokeh32:
            raise unittest.SkipTest("Bokeh 3.2 added H/VLines")
        super().setUp()

    def test_hlines_plot(self):
        hlines = HLines(
            {"y": [0, 1, 2, 5.5], "extra": [-1, -2, -3, -44]}, vdims=["extra"]
        )
        plot = bokeh_renderer.get_plot(hlines)
        assert isinstance(plot.handles["glyph"], BkHSpan)
        assert plot.handles["xaxis"].axis_label == "x"
        assert plot.handles["yaxis"].axis_label == "y"

        assert plot.handles["x_range"].start == 0
        assert plot.handles["x_range"].end == 1
        assert plot.handles["y_range"].start == 0
        assert plot.handles["y_range"].end == 5.5

        source = plot.handles["source"]
        assert list(source.data) == ["y", "extra"]
        assert (source.data["y"] == [0, 1, 2, 5.5]).all()
        assert (source.data["extra"] == [-1, -2, -3, -44]).all()

    def test_hlines_xlabel_ylabel(self):
        hlines = HLines(
            {"y": [0, 1, 2, 5.5], "extra": [-1, -2, -3, -44]}, vdims=["extra"]
        ).opts(xlabel="xlabel", ylabel="xlabel")
        plot = bokeh_renderer.get_plot(hlines)
        assert isinstance(plot.handles["glyph"], BkHSpan)
        assert plot.handles["xaxis"].axis_label == "xlabel"
        assert plot.handles["yaxis"].axis_label == "xlabel"

    def test_hlines_array(self):
        hlines = HLines(np.array([0, 1, 2, 5.5]))
        plot = bokeh_renderer.get_plot(hlines)
        assert isinstance(plot.handles["glyph"], BkHSpan)
        assert plot.handles["xaxis"].axis_label == "x"
        assert plot.handles["yaxis"].axis_label == "y"

        assert plot.handles["x_range"].start == 0
        assert plot.handles["x_range"].end == 1
        assert plot.handles["y_range"].start == 0
        assert plot.handles["y_range"].end == 5.5

        source = plot.handles["source"]
        assert list(source.data) == ["y"]
        assert (source.data["y"] == [0, 1, 2, 5.5]).all()

    def test_hlines_plot_invert_axes(self):
        hlines = HLines(
            {"y": [0, 1, 2, 5.5], "extra": [-1, -2, -3, -44]}, vdims=["extra"]
        ).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(hlines)
        assert isinstance(plot.handles["glyph"], BkVSpan)
        assert plot.handles["xaxis"].axis_label == "y"
        assert plot.handles["yaxis"].axis_label == "x"

        assert plot.handles["x_range"].start == 0
        assert plot.handles["x_range"].end == 5.5
        assert plot.handles["y_range"].start == 0
        assert plot.handles["y_range"].end == 1

        source = plot.handles["source"]
        assert list(source.data) == ["y", "extra"]
        assert (source.data["y"] == [0, 1, 2, 5.5]).all()
        assert (source.data["extra"] == [-1, -2, -3, -44]).all()

    def test_hlines_nondefault_kdim(self):
        hlines = HLines(
            {"extra": [0, 1, 2, 5.5]}, kdims=["extra"]
        )
        plot = bokeh_renderer.get_plot(hlines)
        assert isinstance(plot.handles["glyph"], BkHSpan)
        assert plot.handles["xaxis"].axis_label == "x"
        assert plot.handles["yaxis"].axis_label == "y"

        assert plot.handles["x_range"].start == 0
        assert plot.handles["x_range"].end == 1
        assert plot.handles["y_range"].start == 0
        assert plot.handles["y_range"].end == 5.5

        source = plot.handles["source"]
        assert list(source.data) == ["extra"]
        assert (source.data["extra"] == [0, 1, 2, 5.5]).all()

    def test_vlines_plot(self):
        vlines = VLines(
            {"x": [0, 1, 2, 5.5], "extra": [-1, -2, -3, -44]}, vdims=["extra"]
        )
        plot = bokeh_renderer.get_plot(vlines)
        assert isinstance(plot.handles["glyph"], BkVSpan)
        assert plot.handles["xaxis"].axis_label == "x"
        assert plot.handles["yaxis"].axis_label == "y"

        assert plot.handles["x_range"].start == 0
        assert plot.handles["x_range"].end == 5.5
        assert plot.handles["y_range"].start == 0
        assert plot.handles["y_range"].end == 1

        source = plot.handles["source"]
        assert list(source.data) == ["x", "extra"]
        assert (source.data["x"] == [0, 1, 2, 5.5]).all()
        assert (source.data["extra"] == [-1, -2, -3, -44]).all()

    def test_vlines_plot_invert_axes(self):
        vlines = VLines(
            {"x": [0, 1, 2, 5.5], "extra": [-1, -2, -3, -44]}, vdims=["extra"]
        ).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(vlines)
        assert isinstance(plot.handles["glyph"], BkHSpan)
        assert plot.handles["xaxis"].axis_label == "y"
        assert plot.handles["yaxis"].axis_label == "x"

        assert plot.handles["x_range"].start == 0
        assert plot.handles["x_range"].end == 1
        assert plot.handles["y_range"].start == 0
        assert plot.handles["y_range"].end == 5.5

        source = plot.handles["source"]
        assert list(source.data) == ["x", "extra"]
        assert (source.data["x"] == [0, 1, 2, 5.5]).all()
        assert (source.data["extra"] == [-1, -2, -3, -44]).all()

    def test_vlines_nondefault_kdim(self):
        vlines = VLines(
            {"extra": [0, 1, 2, 5.5]}, kdims=["extra"]
        )
        plot = bokeh_renderer.get_plot(vlines)
        assert isinstance(plot.handles["glyph"], BkVSpan)
        assert plot.handles["xaxis"].axis_label == "x"
        assert plot.handles["yaxis"].axis_label == "y"

        assert plot.handles["x_range"].start == 0
        assert plot.handles["x_range"].end == 5.5
        assert plot.handles["y_range"].start == 0
        assert plot.handles["y_range"].end == 1

        source = plot.handles["source"]
        assert list(source.data) == ["extra"]
        assert (source.data["extra"] == [0, 1, 2, 5.5]).all()

    def test_vlines_hlines_overlay(self):
        hlines = HLines(
            {"y": [0, 1, 2, 5.5], "extra": [-1, -2, -3, -44]}, vdims=["extra"]
        )
        vlines = VLines(
            {"x": [0, 1, 2, 5.5], "extra": [-1, -2, -3, -44]}, vdims=["extra"]
        )
        plot = bokeh_renderer.get_plot(hlines * vlines)
        assert plot.handles["xaxis"].axis_label == "x"
        assert plot.handles["yaxis"].axis_label == "y"

        assert plot.handles["x_range"].start == 0
        assert plot.handles["x_range"].end == 5.5
        assert plot.handles["y_range"].start == 0
        assert plot.handles["y_range"].end == 5.5

    def test_vlines_hlines_overlay_non_annotation(self):
        non_annotation = hv.Curve([], kdims=["time"])
        hlines = HLines(
            {"y": [0, 1, 2, 5.5], "extra": [-1, -2, -3, -44]}, vdims=["extra"]
        )
        vlines = VLines(
            {"x": [0, 1, 2, 5.5], "extra": [-1, -2, -3, -44]}, vdims=["extra"]
        )
        plot = bokeh_renderer.get_plot(non_annotation * hlines * vlines)
        assert plot.handles["xaxis"].axis_label == "time"
        assert plot.handles["yaxis"].axis_label == "y"

    def test_coloring_hline(self):
        hlines = HLines({"y": [1, 2, 3]})
        hlines = hlines.opts(
            alpha=hv.dim("y").norm(),
            line_color="red",
            line_dash=hv.dim("y").bin([0, 1.5, 3], ["dashed", "solid"]),
        )

        plot = hv.renderer("bokeh").get_plot(hlines)
        assert plot.handles["glyph"].line_color == "red"

        data = plot.handles["glyph_renderer"].data_source.data
        np.testing.assert_allclose(data["alpha"], [0, 0.5, 1])
        assert data["line_dash"] == ["dashed", "solid", "solid"]


class TestHVSpansPlot(TestBokehPlot):

    def setUp(self):
        if not bokeh32:
            raise unittest.SkipTest("Bokeh 3.2 added H/VSpans")
        super().setUp()

    def test_hspans_plot(self):
        hspans = HSpans(
            {"y0": [0, 3, 5.5], "y1": [1, 4, 6.5], "extra": [-1, -2, -3]}, vdims=["extra"]
        )
        plot = bokeh_renderer.get_plot(hspans)
        assert isinstance(plot.handles["glyph"], BkHStrip)
        assert plot.handles["xaxis"].axis_label == "x"
        assert plot.handles["yaxis"].axis_label == "y"

        assert plot.handles["x_range"].start == 0
        assert plot.handles["x_range"].end == 1
        assert plot.handles["y_range"].start == 0
        assert plot.handles["y_range"].end == 6.5

        source = plot.handles["source"]
        assert list(source.data) == ["y0", "y1", "extra"]
        assert (source.data["y0"] == [0, 3, 5.5]).all()
        assert (source.data["y1"] == [1, 4, 6.5]).all()
        assert (source.data["extra"] == [-1, -2, -3]).all()

    def test_hspans_plot_xlabel_ylabel(self):
        hspans = HSpans(
            {"y0": [0, 3, 5.5], "y1": [1, 4, 6.5], "extra": [-1, -2, -3]}, vdims=["extra"]
        ).opts(xlabel="xlabel", ylabel="xlabel")
        plot = bokeh_renderer.get_plot(hspans)
        assert isinstance(plot.handles["glyph"], BkHStrip)
        assert plot.handles["xaxis"].axis_label == "xlabel"
        assert plot.handles["yaxis"].axis_label == "xlabel"

    def test_hspans_plot_invert_axes(self):
        hspans = HSpans(
            {"y0": [0, 3, 5.5], "y1": [1, 4, 6.5], "extra": [-1, -2, -3]}, vdims=["extra"]
        ).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(hspans)
        assert isinstance(plot.handles["glyph"], BkVStrip)
        assert plot.handles["xaxis"].axis_label == "y"
        assert plot.handles["yaxis"].axis_label == "x"

        assert plot.handles["x_range"].start == 0
        assert plot.handles["x_range"].end == 6.5
        assert plot.handles["y_range"].start == 0
        assert plot.handles["y_range"].end == 1

        source = plot.handles["source"]
        assert list(source.data) == ["y0", "y1", "extra"]
        assert (source.data["y0"] == [0, 3, 5.5]).all()
        assert (source.data["y1"] == [1, 4, 6.5]).all()
        assert (source.data["extra"] == [-1, -2, -3]).all()

    def test_hspans_nondefault_kdims(self):
        hspans = HSpans(
            {"other0": [0, 3, 5.5], "other1": [1, 4, 6.5]}, kdims=["other0", "other1"]
        )
        plot = bokeh_renderer.get_plot(hspans)
        assert isinstance(plot.handles["glyph"], BkHStrip)
        assert plot.handles["xaxis"].axis_label == "x"
        assert plot.handles["yaxis"].axis_label == "y"

        assert plot.handles["x_range"].start == 0
        assert plot.handles["x_range"].end == 1
        assert plot.handles["y_range"].start == 0
        assert plot.handles["y_range"].end == 6.5

        source = plot.handles["source"]
        assert list(source.data) == ["other0", "other1"]
        assert (source.data["other0"] == [0, 3, 5.5]).all()
        assert (source.data["other1"] == [1, 4, 6.5]).all()

    def test_vspans_plot(self):
        vspans = VSpans(
            {"x0": [0, 3, 5.5], "x1": [1, 4, 6.5], "extra": [-1, -2, -3]}, vdims=["extra"]
        )
        plot = bokeh_renderer.get_plot(vspans)
        assert isinstance(plot.handles["glyph"], BkVStrip)
        assert plot.handles["xaxis"].axis_label == "x"
        assert plot.handles["yaxis"].axis_label == "y"

        assert plot.handles["x_range"].start == 0
        assert plot.handles["x_range"].end == 6.5
        assert plot.handles["y_range"].start == 0
        assert plot.handles["y_range"].end == 1

        source = plot.handles["source"]
        assert list(source.data) == ["x0", "x1", "extra"]
        assert (source.data["x0"] == [0, 3, 5.5]).all()
        assert (source.data["x1"] == [1, 4, 6.5]).all()
        assert (source.data["extra"] == [-1, -2, -3]).all()

    def test_vspans_plot_invert_axes(self):
        vspans = VSpans(
            {"x0": [0, 3, 5.5], "x1": [1, 4, 6.5], "extra": [-1, -2, -3]}, vdims=["extra"]
        ).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(vspans)
        assert isinstance(plot.handles["glyph"], BkHStrip)
        assert plot.handles["xaxis"].axis_label == "y"
        assert plot.handles["yaxis"].axis_label == "x"

        assert plot.handles["x_range"].start == 0
        assert plot.handles["x_range"].end == 1
        assert plot.handles["y_range"].start == 0
        assert plot.handles["y_range"].end == 6.5

        source = plot.handles["source"]
        assert list(source.data) == ["x0", "x1", "extra"]
        assert (source.data["x0"] == [0, 3, 5.5]).all()
        assert (source.data["x1"] == [1, 4, 6.5]).all()
        assert (source.data["extra"] == [-1, -2, -3]).all()

    def test_vspans_nondefault_kdims(self):
        vspans = VSpans(
            {"other0": [0, 3, 5.5], "other1": [1, 4, 6.5]}, kdims=["other0", "other1"]
        )
        plot = bokeh_renderer.get_plot(vspans)
        assert isinstance(plot.handles["glyph"], BkVStrip)
        assert plot.handles["xaxis"].axis_label == "x"
        assert plot.handles["yaxis"].axis_label == "y"

        assert plot.handles["x_range"].start == 0
        assert plot.handles["x_range"].end == 6.5
        assert plot.handles["y_range"].start == 0
        assert plot.handles["y_range"].end == 1

        source = plot.handles["source"]
        assert list(source.data) == ["other0", "other1"]
        assert (source.data["other0"] == [0, 3, 5.5]).all()
        assert (source.data["other1"] == [1, 4, 6.5]).all()

    def test_vspans_hspans_overlay(self):
        hspans = HSpans(
            {"y0": [0, 3, 5.5], "y1": [1, 4, 6.5], "extra": [-1, -2, -3]}, vdims=["extra"]
        )
        vspans = VSpans(
            {"x0": [0, 3, 5.5], "x1": [1, 4, 6.5], "extra": [-1, -2, -3]}, vdims=["extra"]
        )
        plot = bokeh_renderer.get_plot(hspans * vspans)
        assert plot.handles["xaxis"].axis_label == "x"
        assert plot.handles["yaxis"].axis_label == "y"

        assert plot.handles["x_range"].start == 0
        assert plot.handles["x_range"].end == 6.5
        assert plot.handles["y_range"].start == 0
        assert plot.handles["y_range"].end == 6.5

    def test_vlines_hlines_overlay_non_annotation(self):
        non_annotation = hv.Curve([], kdims=["time"])
        hspans = HSpans(
            {"y0": [0, 3, 5.5], "y1": [1, 4, 6.5], "extra": [-1, -2, -3]}, vdims=["extra"]
        )
        vspans = VSpans(
            {"x0": [0, 3, 5.5], "x1": [1, 4, 6.5], "extra": [-1, -2, -3]}, vdims=["extra"]
        )
        plot = bokeh_renderer.get_plot(non_annotation * hspans * vspans)
        assert plot.handles["xaxis"].axis_label == "time"
        assert plot.handles["yaxis"].axis_label == "y"

    def test_coloring_hline(self):
        hspans = HSpans({"y0": [1, 3, 5], "y1": [2, 4, 6]}).opts(
            alpha=hv.dim("y0").norm(),
            line_color="red",
            line_dash=hv.dim("y1").bin([0, 3, 6], ["dashed", "solid"]),
        )

        plot = hv.renderer("bokeh").get_plot(hspans)
        assert plot.handles["glyph"].line_color == "red"

        data = plot.handles["glyph_renderer"].data_source.data
        np.testing.assert_allclose(data["alpha"], [0, 0.5, 1])
        assert data["line_dash"] == ["dashed", "solid", "solid"]
