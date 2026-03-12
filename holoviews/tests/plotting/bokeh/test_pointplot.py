from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from bokeh.core.enums import MarkerType
from bokeh.models import (
    CategoricalColorMapper,
    Circle,
    FactorRange,
    LinearColorMapper,
    Scatter,
)

import holoviews as hv
from holoviews.plotting.bokeh.chart import SizebarMixin
from holoviews.plotting.bokeh.util import BOKEH_GE_3_8_0, property_to_dict
from holoviews.streams import Stream
from holoviews.testing import assert_data_equal

from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer


class TestPointPlot(TestBokehPlot):
    def test_points_color_selection_nonselection(self):
        opts = dict(color="green", selection_color="red", nonselection_color="blue")
        points = hv.Points(
            [(i, i * 2, i * 3, chr(65 + i)) for i in range(10)], vdims=["a", "b"]
        ).opts(**opts)
        plot = bokeh_renderer.get_plot(points)
        glyph_renderer = plot.handles["glyph_renderer"]
        assert glyph_renderer.glyph.fill_color == "green"
        assert glyph_renderer.glyph.line_color == "green"
        assert glyph_renderer.selection_glyph.fill_color == "red"
        assert glyph_renderer.selection_glyph.line_color == "red"
        assert glyph_renderer.nonselection_glyph.fill_color == "blue"
        assert glyph_renderer.nonselection_glyph.line_color == "blue"

    def test_points_alpha_selection_nonselection(self):
        opts = dict(alpha=0.8, selection_alpha=1.0, nonselection_alpha=0.2)
        points = hv.Points(
            [(i, i * 2, i * 3, chr(65 + i)) for i in range(10)], vdims=["a", "b"]
        ).opts(**opts)
        plot = bokeh_renderer.get_plot(points)
        glyph_renderer = plot.handles["glyph_renderer"]
        assert glyph_renderer.glyph.fill_alpha == 0.8
        assert glyph_renderer.glyph.line_alpha == 0.8
        assert glyph_renderer.selection_glyph.fill_alpha == 1
        assert glyph_renderer.selection_glyph.line_alpha == 1
        assert glyph_renderer.nonselection_glyph.fill_alpha == 0.2
        assert glyph_renderer.nonselection_glyph.line_alpha == 0.2

    def test_points_alpha_selection_partial(self):
        opts = dict(selection_alpha=1.0, selection_fill_alpha=0.2)
        points = hv.Points(
            [(i, i * 2, i * 3, chr(65 + i)) for i in range(10)], vdims=["a", "b"]
        ).opts(**opts)
        plot = bokeh_renderer.get_plot(points)
        glyph_renderer = plot.handles["glyph_renderer"]
        assert glyph_renderer.glyph.fill_alpha == 1.0
        assert glyph_renderer.glyph.line_alpha == 1.0
        assert glyph_renderer.selection_glyph.fill_alpha == 0.2
        assert glyph_renderer.selection_glyph.line_alpha == 1

    def test_batched_points(self):
        overlay = hv.NdOverlay({i: hv.Points(np.arange(i)) for i in range(1, 100)})
        plot = bokeh_renderer.get_plot(overlay)
        extents = plot.get_extents(overlay, {})
        assert extents == (0, 0, 98, 98)

    def test_batched_points_size_and_color(self):
        opts = {"NdOverlay": dict(legend_limit=0), "Points": dict(size=hv.Cycle(values=[1, 2]))}
        overlay = hv.NdOverlay({i: hv.Points([(i, j) for j in range(2)]) for i in range(2)}).opts(
            opts
        )
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        size = np.array([1, 1, 2, 2])
        color = ["#30a2da", "#30a2da", "#fc4f30", "#fc4f30"]
        assert plot.handles["source"].data["color"] == color
        assert_data_equal(plot.handles["source"].data["size"], size)

    def test_batched_points_line_color_and_color(self):
        opts = {
            "NdOverlay": dict(legend_limit=0),
            "Points": dict(line_color=hv.Cycle(values=["red", "blue"])),
        }
        overlay = hv.NdOverlay({i: hv.Points([(i, j) for j in range(2)]) for i in range(2)}).opts(
            opts
        )
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        line_color = ["red", "red", "blue", "blue"]
        fill_color = ["#30a2da", "#30a2da", "#fc4f30", "#fc4f30"]
        assert plot.handles["source"].data["fill_color"] == fill_color
        assert plot.handles["source"].data["line_color"] == line_color

    def test_batched_points_alpha_and_color(self):
        opts = {"NdOverlay": dict(legend_limit=0), "Points": dict(alpha=hv.Cycle(values=[0.5, 1]))}
        overlay = hv.NdOverlay({i: hv.Points([(i, j) for j in range(2)]) for i in range(2)}).opts(
            opts
        )
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        alpha = np.array([0.5, 0.5, 1.0, 1.0])
        color = ["#30a2da", "#30a2da", "#fc4f30", "#fc4f30"]
        assert_data_equal(plot.handles["source"].data["alpha"], alpha)
        assert plot.handles["source"].data["color"] == color

    def test_batched_points_line_width_and_color(self):
        opts = {
            "NdOverlay": dict(legend_limit=0),
            "Points": dict(line_width=hv.Cycle(values=[0.5, 1])),
        }
        overlay = hv.NdOverlay({i: hv.Points([(i, j) for j in range(2)]) for i in range(2)}).opts(
            opts
        )
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        line_width = np.array([0.5, 0.5, 1.0, 1.0])
        color = ["#30a2da", "#30a2da", "#fc4f30", "#fc4f30"]
        assert_data_equal(plot.handles["source"].data["line_width"], line_width)
        assert plot.handles["source"].data["color"] == color

    def test_points_overlay_datetime_hover(self):
        obj = hv.NdOverlay(
            {
                i: hv.Points((list(pd.date_range("2016-01-01", "2016-01-31")), range(31)))
                for i in range(5)
            },
            kdims=["Test"],
        )
        opts = {"Points": {"tools": ["hover"]}}
        obj = obj.opts(opts)
        self._test_hover_info(
            obj,
            [("Test", "@{Test}"), ("x", "@{x}{%F %T}"), ("y", "@{y}")],
            formatters={"@{x}": "datetime"},
        )

    def test_points_overlay_hover_batched(self):
        obj = hv.NdOverlay({i: hv.Points(np.random.rand(10, 2)) for i in range(5)}, kdims=["Test"])
        opts = {"Points": {"tools": ["hover"]}, "NdOverlay": {"legend_limit": 0}}
        obj = obj.opts(opts)
        self._test_hover_info(obj, [("Test", "@{Test}"), ("x", "@{x}"), ("y", "@{y}")])

    def test_points_overlay_hover(self):
        obj = hv.NdOverlay({i: hv.Points(np.random.rand(10, 2)) for i in range(5)}, kdims=["Test"])
        opts = {"Points": {"tools": ["hover"]}, "NdOverlay": {"legend_limit": 0}}
        obj = obj.opts(opts)
        self._test_hover_info(obj, [("Test", "@{Test}"), ("x", "@{x}"), ("y", "@{y}")])

    def test_points_no_single_item_legend(self):
        points = hv.Points([("A", 1), ("B", 2)], label="A")
        plot = bokeh_renderer.get_plot(points)
        plot.initialize_plot()
        fig = plot.state
        assert len(fig.legend) == 0

    @pytest.mark.parametrize("marker", MarkerType)
    def test_native_marker_legend(self, marker):
        points = hv.Points([(0, 0, "A"), (0, 1, "B")], vdims="color").opts(
            color="color", marker=marker
        )
        plot = bokeh_renderer.get_plot(points)
        assert plot.state.legend[0].items[0].renderers[0].glyph.marker == marker

    def test_points_categorical_xaxis(self):
        points = hv.Points((["A", "B", "C"], (1, 2, 3)))
        plot = bokeh_renderer.get_plot(points)
        x_range = plot.handles["x_range"]
        assert isinstance(x_range, FactorRange)
        assert x_range.factors == ["A", "B", "C"]

    def test_points_categorical_xaxis_mixed_type(self):
        points = hv.Points(range(10))
        points2 = hv.Points((["A", "B", "C", 1, 2.0], (1, 2, 3, 4, 5)))
        plot = bokeh_renderer.get_plot(points * points2)
        x_range = plot.handles["x_range"]
        assert isinstance(x_range, FactorRange)
        assert x_range.factors == [*map(str, range(10)), "A", "B", "C", "2.0"]

    def test_points_categorical_xaxis_invert_axes(self):
        points = hv.Points((["A", "B", "C"], (1, 2, 3))).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(points)
        y_range = plot.handles["y_range"]
        assert isinstance(y_range, FactorRange)
        assert y_range.factors == ["A", "B", "C"]

    def test_points_overlay_categorical_xaxis(self):
        points = hv.Points((["A", "B", "C"], (1, 2, 3)))
        points2 = hv.Points((["B", "C", "D"], (1, 2, 3)))
        plot = bokeh_renderer.get_plot(points * points2)
        x_range = plot.handles["x_range"]
        assert isinstance(x_range, FactorRange)
        assert x_range.factors == ["A", "B", "C", "D"]

    def test_points_overlay_categorical_xaxis_invert_axis(self):
        points = hv.Points((["A", "B", "C"], (1, 2, 3))).opts(invert_xaxis=True)
        points2 = hv.Points((["B", "C", "D"], (1, 2, 3)))
        plot = bokeh_renderer.get_plot(points * points2)
        x_range = plot.handles["x_range"]
        assert isinstance(x_range, FactorRange)
        assert x_range.factors == ["A", "B", "C", "D"][::-1]

    def test_points_overlay_categorical_xaxis_invert_axes(self):
        points = hv.Points((["A", "B", "C"], (1, 2, 3))).opts(invert_axes=True)
        points2 = hv.Points((["B", "C", "D"], (1, 2, 3)))
        plot = bokeh_renderer.get_plot(points * points2)
        y_range = plot.handles["y_range"]
        assert isinstance(y_range, FactorRange)
        assert y_range.factors == ["A", "B", "C", "D"]

    def test_points_padding_square(self):
        points = hv.Points([1, 2, 3]).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == -0.2
        assert x_range.end == 2.2
        assert y_range.start == 0.8
        assert y_range.end == 3.2

    def test_curve_padding_square_per_axis(self):
        curve = hv.Points([1, 2, 3]).opts(padding=((0, 0.1), (0.1, 0.2)))
        plot = bokeh_renderer.get_plot(curve)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == 0
        assert x_range.end == 2.2
        assert y_range.start == 0.8
        assert y_range.end == 3.4

    def test_points_padding_unequal(self):
        points = hv.Points([1, 2, 3]).opts(padding=(0.05, 0.1))
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == -0.1
        assert x_range.end == 2.1
        assert y_range.start == 0.8
        assert y_range.end == 3.2

    def test_points_padding_nonsquare(self):
        points = hv.Points([1, 2, 3]).opts(padding=0.1, width=600)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == -0.1
        assert x_range.end == 2.1
        assert y_range.start == 0.8
        assert y_range.end == 3.2

    def test_points_padding_logx(self):
        points = hv.Points([(1, 1), (2, 2), (3, 3)]).opts(padding=0.1, logx=True)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == 0.89595845984076228
        assert x_range.end == 3.3483695221017129
        assert y_range.start == 0.8
        assert y_range.end == 3.2

    def test_points_padding_logy(self):
        points = hv.Points([1, 2, 3]).opts(padding=0.1, logy=True)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == -0.2
        assert x_range.end == 2.2
        assert y_range.start == 0.89595845984076228
        assert y_range.end == 3.3483695221017129

    def test_points_padding_datetime_square(self):
        points = hv.Points([(np.datetime64(f"2016-04-0{i}"), i) for i in range(1, 4)]).opts(
            padding=0.1
        )
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == np.datetime64("2016-03-31T19:12:00.000000000")
        assert x_range.end == np.datetime64("2016-04-03T04:48:00.000000000")
        assert y_range.start == 0.8
        assert y_range.end == 3.2

    def test_points_padding_datetime_nonsquare(self):
        points = hv.Points([(np.datetime64(f"2016-04-0{i}"), i) for i in range(1, 4)]).opts(
            padding=0.1, width=600
        )
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == np.datetime64("2016-03-31T21:36:00.000000000")
        assert x_range.end == np.datetime64("2016-04-03T02:24:00.000000000")
        assert y_range.start == 0.8
        assert y_range.end == 3.2

    def test_points_padding_hard_xrange(self):
        points = hv.Points([1, 2, 3]).redim.range(x=(0, 3)).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == 0
        assert x_range.end == 3
        assert y_range.start == 0.8
        assert y_range.end == 3.2

    def test_points_padding_soft_xrange(self):
        points = hv.Points([1, 2, 3]).redim.soft_range(x=(0, 3)).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(points)
        x_range, y_range = plot.handles["x_range"], plot.handles["y_range"]
        assert x_range.start == 0
        assert x_range.end == 3
        assert y_range.start == 0.8
        assert y_range.end == 3.2

    def test_points_datetime_hover(self):
        points = hv.Points([(0, 1, dt.datetime(2017, 1, 1))], vdims="date").opts(tools=["hover"])
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles["cds"]
        assert cds.data["date"] == np.datetime64("2017-01-01", "ns")
        hover = plot.handles["hover"]
        assert hover.tooltips == [("x", "@{x}"), ("y", "@{y}"), ("date", "@{date}{%F %T}")]

    def test_points_selected(self):
        points = hv.Points([(0, 0), (1, 1), (2, 2)]).opts(selected=[0, 2])
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles["cds"]
        assert cds.selected.indices == [0, 2]

    def test_points_update_selected(self):
        stream = Stream.define("Selected", selected=[])()
        points = hv.Points([(0, 0), (1, 1), (2, 2)]).apply.opts(selected=stream.param.selected)
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles["cds"]
        assert cds.selected.indices == []
        stream.event(selected=[0, 2])
        assert cds.selected.indices == [0, 2]

    ###########################
    #    Styling mapping      #
    ###########################

    def test_point_color_op(self):
        points = hv.Points([(0, 0, "#000"), (0, 1, "#F00"), (0, 2, "#0F0")], vdims="color").opts(
            color="color"
        )
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert cds.data["color"] == ["#000", "#F00", "#0F0"]
        assert property_to_dict(glyph.fill_color) == {"field": "color"}
        assert property_to_dict(glyph.line_color) == {"field": "color"}

    def test_point_linear_color_op(self):
        points = hv.Points([(0, 0, 0), (0, 1, 1), (0, 2, 2)], vdims="color").opts(color="color")
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        cmapper = plot.handles["color_color_mapper"]
        assert isinstance(cmapper, LinearColorMapper)
        assert cmapper.low == 0
        assert cmapper.high == 2
        assert_data_equal(cds.data["color"], np.array([0, 1, 2]))
        assert property_to_dict(glyph.fill_color) == {"field": "color", "transform": cmapper}
        assert property_to_dict(glyph.line_color) == {"field": "color", "transform": cmapper}

    def test_point_categorical_color_op(self):
        points = hv.Points([(0, 0, "A"), (0, 1, "B"), (0, 2, "C")], vdims="color").opts(
            color="color"
        )
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        cmapper = plot.handles["color_color_mapper"]
        assert isinstance(cmapper, CategoricalColorMapper)
        assert cmapper.factors == ["A", "B", "C"]
        assert cds.data["color"] == ["A", "B", "C"]
        assert property_to_dict(glyph.fill_color) == {"field": "color", "transform": cmapper}
        assert property_to_dict(glyph.line_color) == {"field": "color", "transform": cmapper}

    def test_point_categorical_color_op_legend_with_labels(self):
        labels = {"A": "A point", "B": "B point", "C": "C point"}
        points = hv.Points([(0, 0, "A"), (0, 1, "B"), (0, 2, "C")], vdims="color").opts(
            color="color", show_legend=True, legend_labels=labels
        )
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles["cds"]
        legend = plot.state.legend[0].items[0]
        assert property_to_dict(legend.label) == {"field": "_color_labels"}
        assert cds.data["_color_labels"] == ["A point", "B point", "C point"]

    def test_point_categorical_dtype_color_op(self):
        df = pd.DataFrame(
            dict(
                sample_id=["subject 1", "subject 2", "subject 3", "subject 4"],
                category=["apple", "pear", "apple", "pear"],
                value=[1, 2, 3, 4],
            )
        )
        df["category"] = df["category"].astype("category")
        points = hv.Points(df, ["sample_id", "value"]).opts(color="category")
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        cmapper = plot.handles["color_color_mapper"]
        assert isinstance(cmapper, CategoricalColorMapper)
        assert cmapper.factors == ["apple", "pear"]
        assert_data_equal(
            np.asarray(cds.data["color"]), np.array(["apple", "pear", "apple", "pear"])
        )
        assert property_to_dict(glyph.fill_color) == {"field": "color", "transform": cmapper}
        assert property_to_dict(glyph.line_color) == {"field": "color", "transform": cmapper}

    def test_point_explicit_cmap_color_op(self):
        points = hv.Points([(0, 0), (0, 1), (0, 2)]).opts(
            color="y", cmap={0: "red", 1: "green", 2: "blue"}
        )
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        cmapper = plot.handles["color_color_mapper"]
        assert isinstance(cmapper, CategoricalColorMapper)
        assert cmapper.factors == ["0", "1", "2"]
        assert cmapper.palette == ["red", "green", "blue"]
        assert cds.data["color_str__"] == ["0", "1", "2"]
        assert property_to_dict(glyph.fill_color) == {"field": "color_str__", "transform": cmapper}
        assert property_to_dict(glyph.line_color) == {"field": "color_str__", "transform": cmapper}

    def test_point_line_color_op(self):
        points = hv.Points([(0, 0, "#000"), (0, 1, "#F00"), (0, 2, "#0F0")], vdims="color").opts(
            line_color="color"
        )
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert cds.data["line_color"] == ["#000", "#F00", "#0F0"]
        assert property_to_dict(glyph.fill_color) != {"field": "line_color"}
        assert property_to_dict(glyph.line_color) == {"field": "line_color"}

    def test_point_fill_color_op(self):
        points = hv.Points([(0, 0, "#000"), (0, 1, "#F00"), (0, 2, "#0F0")], vdims="color").opts(
            fill_color="color"
        )
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert cds.data["fill_color"] == ["#000", "#F00", "#0F0"]
        assert property_to_dict(glyph.fill_color) == {"field": "fill_color"}
        assert property_to_dict(glyph.line_color) != {"field": "fill_color"}

    def test_point_angle_op(self):
        points = hv.Points([(0, 0, 0), (0, 1, 45), (0, 2, 90)], vdims="angle").opts(angle="angle")
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert_data_equal(cds.data["angle"], np.array([0, 0.785398, 1.570796]))
        assert property_to_dict(glyph.angle) == {"field": "angle"}

    def test_point_alpha_op(self):
        points = hv.Points([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)], vdims="alpha").opts(
            alpha="alpha"
        )
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert_data_equal(cds.data["alpha"], np.array([0, 0.2, 0.7]))
        assert property_to_dict(glyph.fill_alpha) == {"field": "alpha"}

    def test_point_line_alpha_op(self):
        points = hv.Points([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)], vdims="alpha").opts(
            line_alpha="alpha"
        )
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert_data_equal(cds.data["line_alpha"], np.array([0, 0.2, 0.7]))
        assert property_to_dict(glyph.line_alpha) == {"field": "line_alpha"}
        assert property_to_dict(glyph.fill_alpha) != {"field": "line_alpha"}

    def test_point_fill_alpha_op(self):
        points = hv.Points([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)], vdims="alpha").opts(
            fill_alpha="alpha"
        )
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert_data_equal(cds.data["fill_alpha"], np.array([0, 0.2, 0.7]))
        assert property_to_dict(glyph.line_alpha) != {"field": "fill_alpha"}
        assert property_to_dict(glyph.fill_alpha) == {"field": "fill_alpha"}

    def test_point_size_op(self):
        points = hv.Points([(0, 0, 1), (0, 1, 4), (0, 2, 8)], vdims="size").opts(size="size")
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert_data_equal(cds.data["size"], np.array([1, 4, 8]))
        assert property_to_dict(glyph.size) == {"field": "size"}

    def test_point_line_width_op(self):
        points = hv.Points([(0, 0, 1), (0, 1, 4), (0, 2, 8)], vdims="line_width").opts(
            line_width="line_width"
        )
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert_data_equal(cds.data["line_width"], np.array([1, 4, 8]))
        assert property_to_dict(glyph.line_width) == {"field": "line_width"}

    def test_point_marker_op(self):
        points = hv.Points(
            [(0, 0, "circle"), (0, 1, "triangle"), (0, 2, "square")], vdims="marker"
        ).opts(marker="marker")
        plot = bokeh_renderer.get_plot(points)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert cds.data["marker"] == ["circle", "triangle", "square"]
        assert property_to_dict(glyph.marker) == {"field": "marker"}

    def test_op_ndoverlay_value(self):
        markers = ["circle", "triangle"]
        overlay = hv.NdOverlay(
            {marker: hv.Points(np.arange(i)) for i, marker in enumerate(markers)}, "Marker"
        ).opts("Points", marker="Marker")
        plot = bokeh_renderer.get_plot(overlay)
        for subplot, glyph_type, marker in zip(
            plot.subplots.values(), [Scatter, Scatter], markers, strict=True
        ):
            assert isinstance(subplot.handles["glyph"], glyph_type)
            assert subplot.handles["glyph"].marker == marker

    def test_point_radius(self):
        x, y = 4, 5
        xs = np.arange(x)
        ys = np.arange(y)
        zs = np.arange(x * y).reshape(y, x)
        plot = hv.Points(
            (
                xs,
                ys,
                zs,
            ),
            kdims=["xs", "ys"],
            vdims="zs",
        )
        plot.opts(radius=hv.dim("zs").norm() / 2)

        handles = bokeh_renderer.get_plot(plot).handles
        glyph = handles["glyph"]
        assert isinstance(glyph, Circle)
        assert glyph.radius_dimension == "min"

        norm = zs.T.ravel() / np.max(zs) / 2
        np.testing.assert_array_equal(handles["cds"].data["radius"], norm)

    def test_point_radius_then_size_then_radius(self):
        plot = hv.Points([1, 2, 3])
        plot.opts(radius=1)

        handles = bokeh_renderer.get_plot(plot).handles
        glyph = handles["glyph"]
        assert isinstance(glyph, Circle)

        plot.opts(radius=None, size=1)
        handles = bokeh_renderer.get_plot(plot).handles
        glyph = handles["glyph"]
        assert isinstance(glyph, Scatter)

        plot.opts(radius=1)
        handles = bokeh_renderer.get_plot(plot).handles
        glyph = handles["glyph"]
        assert isinstance(glyph, Circle)


@pytest.mark.skipif(not BOKEH_GE_3_8_0, reason="Needs Bokeh 3.8")
class TestSizeBar:
    def setup_method(self):
        np.random.seed(1)
        N = 100
        x = np.random.random(size=N) * 100
        y = np.random.random(size=N) * 100
        radii = np.random.random(size=N) * 10
        self.plot = hv.Points((x, y, radii), vdims=["radii"]).opts(radius="radii")

    def get_handles(self):
        return bokeh_renderer.get_plot(self.plot).handles

    def get_sizebar(self):
        return self.get_handles().get("sizebar")

    def test_init(self):
        from bokeh.models import SizeBar

        assert self.get_sizebar() is None

        self.plot.opts(sizebar=True)
        assert isinstance(self.get_sizebar(), SizeBar)

    @pytest.mark.parametrize("location", [SizebarMixin.param.sizebar_location.default])
    def test_location(self, location):
        self.plot.opts(sizebar=True, sizebar_location=location)
        handles = self.get_handles()
        assert handles["sizebar"] in getattr(handles["plot"], location)

    @pytest.mark.parametrize("orientation", [SizebarMixin.param.sizebar_orientation.default])
    def test_orientation(self, orientation):
        self.plot.opts(sizebar=True, sizebar_orientation=orientation)
        assert self.get_sizebar().orientation == orientation

    def test_style(self):
        self.plot.opts(sizebar=True, sizebar_color="red", sizebar_alpha=0.1)
        sizebar = self.get_sizebar()
        assert sizebar.glyph_fill_alpha == 0.1
        assert sizebar.glyph_fill_color == "red"

    @pytest.mark.parametrize("bounds", [(0, 10), (0, float("inf"))])
    def test_bounds(self, bounds):
        self.plot.opts(sizebar=True, sizebar_bounds=bounds)
        assert self.get_sizebar().bounds == bounds

    @pytest.mark.parametrize("location", [SizebarMixin.param.sizebar_location.default])
    @pytest.mark.parametrize("orientation", [SizebarMixin.param.sizebar_orientation.default])
    @pytest.mark.parametrize("set_width", [True, False])
    def test_max_size(self, location, orientation, set_width):
        self.plot.opts(sizebar=True, sizebar_location=location, sizebar_orientation=orientation)
        if set_width:
            self.plot.opts(sizebar_opts={"width": 216})  # Using 216 as it will never be a default

        width = self.get_sizebar().width
        match (location, orientation, set_width):
            case ("above" | "below", "horizontal", False):
                assert width == "max"
            case ("left" | "right", "vertical", False):
                assert width == "max"
            case _:
                if set_width:
                    assert width == 216
                else:
                    assert width != "max"

    def test_overlay(self):
        # Mainly just to check it does not raise an exception
        p1 = self.plot.opts(sizebar=True)
        p2 = hv.Curve([1, 2, 3])
        combined = p1 * p2

        bk_element = hv.render(combined)
        assert len(bk_element.renderers) == 2  # the two plots
        assert len(bk_element.below) == 2  # axis and sizebar

    def test_layout(self):
        # Mainly just to check it does not raise an exception
        p1 = self.plot.opts(sizebar=True)
        p2 = hv.Curve([1, 2, 3])
        combined = p1 + p2

        bk_element = hv.render(combined)
        assert len(bk_element.children) == 2
        assert len(bk_element.children[0][0].below) == 2
        assert len(bk_element.children[1][0].below) == 1
