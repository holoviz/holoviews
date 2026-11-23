from __future__ import annotations

import numpy as np
import pytest
from bokeh.models import CategoricalColorMapper, LinearColorMapper

import holoviews as hv
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.testing import assert_data_equal, assert_dict_equal

from .test_plot import TestBokehPlot, bokeh_renderer


class TestLabelsPlot(TestBokehPlot):
    def test_labels_simple(self):
        labels = hv.Labels([(0, 1, "A"), (1, 0, "B")])
        plot = bokeh_renderer.get_plot(labels)
        source = plot.handles["source"]
        glyph = plot.handles["glyph"]
        expected = {"x": np.array([0, 1]), "y": np.array([1, 0]), "Label": ["A", "B"]}
        for k, vals in expected.items():
            np.testing.assert_array_equal(source.data[k], vals)
        assert glyph.x == "x"
        assert glyph.y == "y"
        assert glyph.text == "Label"

    def test_labels_empty(self):
        labels = hv.Labels([])
        plot = bokeh_renderer.get_plot(labels)
        source = plot.handles["source"]
        glyph = plot.handles["glyph"]
        expected = {"x": np.array([]), "y": np.array([]), "Label": []}
        for k, vals in expected.items():
            np.testing.assert_array_equal(source.data[k], vals)
        assert glyph.x == "x"
        assert glyph.y == "y"
        assert glyph.text == "Label"

    def test_labels_formatter(self):
        vdim = hv.Dimension("text", value_format=lambda x: f"{x:.1f}")
        labels = hv.Labels([(0, 1, 0.33333), (1, 0, 0.66666)], vdims=vdim)
        plot = bokeh_renderer.get_plot(labels)
        source = plot.handles["source"]
        glyph = plot.handles["glyph"]
        expected = {"x": np.array([0, 1]), "y": np.array([1, 0]), "text": ["0.3", "0.7"]}
        for k, vals in expected.items():
            np.testing.assert_array_equal(source.data[k], vals)
        assert glyph.x == "x"
        assert glyph.y == "y"
        assert glyph.text == "text"

    def test_labels_inverted(self):
        labels = hv.Labels([(0, 1, "A"), (1, 0, "B")]).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(labels)
        source = plot.handles["source"]
        glyph = plot.handles["glyph"]
        expected = {"x": np.array([0, 1]), "y": np.array([1, 0]), "Label": ["A", "B"]}
        for k, vals in expected.items():
            np.testing.assert_array_equal(source.data[k], vals)
        assert glyph.x == "y"
        assert glyph.y == "x"
        assert glyph.text == "Label"

    @pytest.mark.parametrize("invert_axes", [False, True])
    def test_labels_screen_offsets(self, invert_axes):
        labels = hv.Labels([(1, 2, "A"), (3, 4, "B")]).opts(
            x_offset=8, y_offset=-6, invert_axes=invert_axes
        )
        plot = bokeh_renderer.get_plot(labels)
        glyph = plot.handles["glyph"]

        assert glyph.x_offset == 8
        assert glyph.y_offset == -6
        assert glyph.x == ("y" if invert_axes else "x")
        assert glyph.y == ("x" if invert_axes else "y")
        np.testing.assert_array_equal(plot.handles["source"].data["x"], [1, 3])
        np.testing.assert_array_equal(plot.handles["source"].data["y"], [2, 4])

    def test_labels_screen_and_data_offsets(self):
        labels = hv.Labels([(1, 2, "A")]).opts(xoffset=0.2, yoffset=-0.3, x_offset=8, y_offset=-6)
        plot = bokeh_renderer.get_plot(labels)
        glyph = plot.handles["glyph"]

        assert glyph.x_offset == 8
        assert glyph.y_offset == -6
        assert property_to_dict(glyph.x)["transform"].value == 0.2
        assert property_to_dict(glyph.y)["transform"].value == -0.3

    def test_labels_screen_offsets_from_dimensions(self):
        labels = hv.Labels(
            [(1, 2, "A", 8, -3), (3, 4, "B", -4, 5)], vdims=["text", "dx", "dy"]
        ).opts(x_offset="dx", y_offset=hv.dim("dy") * 2)
        plot = bokeh_renderer.get_plot(labels)
        glyph = plot.handles["glyph"]
        source = plot.handles["source"]

        assert property_to_dict(glyph.x_offset) == {"field": "x_offset"}
        assert property_to_dict(glyph.y_offset) == {"field": "y_offset"}
        np.testing.assert_array_equal(source.data["x_offset"], [8, -4])
        np.testing.assert_array_equal(source.data["y_offset"], [-6, 10])

    def test_labels_screen_offsets_update(self):
        labels = hv.Labels([(1, 2, "A")])
        hmap = hv.HoloMap(
            {
                0: labels.opts(x_offset=8, y_offset=-6, clone=True),
                1: labels.opts(x_offset=-4, y_offset=10, clone=True),
            }
        )
        plot = bokeh_renderer.get_plot(hmap)
        glyph = plot.handles["glyph"]
        plot.update((0,))
        assert glyph.x_offset == 8
        assert glyph.y_offset == -6

        plot.update((1,))
        assert glyph.x_offset == -4
        assert glyph.y_offset == 10

    ###########################
    #    Styling mapping      #
    ###########################

    def test_label_color_op(self):
        labels = hv.Labels([(0, 0, "#000"), (0, 1, "#F00"), (0, 2, "#0F0")], vdims="color").opts(
            text_color="color"
        )
        plot = bokeh_renderer.get_plot(labels)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert cds.data["text_color"] == ["#000", "#F00", "#0F0"]
        assert property_to_dict(glyph.text_color) == {"field": "text_color"}

    def test_label_linear_color_op(self):
        labels = hv.Labels([(0, 0, 0), (0, 1, 1), (0, 2, 2)], vdims="color").opts(
            text_color="color"
        )
        plot = bokeh_renderer.get_plot(labels)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        cmapper = plot.handles["text_color_color_mapper"]
        assert isinstance(cmapper, LinearColorMapper)
        assert cmapper.low == 0
        assert cmapper.high == 2
        np.testing.assert_array_equal(cds.data["text_color"], [0, 1, 2])
        assert property_to_dict(glyph.text_color) == {"field": "text_color", "transform": cmapper}

    def test_label_categorical_color_op(self):
        labels = hv.Labels([(0, 0, "A"), (0, 1, "B"), (0, 2, "C")], vdims="color").opts(
            text_color="color"
        )
        plot = bokeh_renderer.get_plot(labels)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        cmapper = plot.handles["text_color_color_mapper"]
        assert isinstance(cmapper, CategoricalColorMapper)
        assert cmapper.factors == ["A", "B", "C"]
        assert cds.data["text_color"] == ["A", "B", "C"]
        assert property_to_dict(glyph.text_color) == {"field": "text_color", "transform": cmapper}

    def test_label_angle_op(self):
        labels = hv.Labels([(0, 0, 0), (0, 1, 45), (0, 2, 90)], vdims="angle").opts(angle="angle")
        plot = bokeh_renderer.get_plot(labels)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        np.testing.assert_array_almost_equal(cds.data["angle"], [0, 0.785398, 1.570796])
        assert property_to_dict(glyph.angle) == {"field": "angle"}

    def test_label_alpha_op(self):
        labels = hv.Labels([(0, 0, 0), (0, 1, 0.2), (0, 2, 0.7)], vdims="alpha").opts(
            text_alpha="alpha"
        )
        plot = bokeh_renderer.get_plot(labels)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert_data_equal(cds.data["text_alpha"], np.array([0, 0.2, 0.7]))
        assert property_to_dict(glyph.text_alpha) == {"field": "text_alpha"}

    def test_label_font_size_op_strings(self):
        labels = hv.Labels([(0, 0, "10pt"), (0, 1, "4pt"), (0, 2, "8pt")], vdims="size").opts(
            text_font_size="size"
        )
        plot = bokeh_renderer.get_plot(labels)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert cds.data["text_font_size"] == ["10pt", "4pt", "8pt"]
        assert property_to_dict(glyph.text_font_size) == {"field": "text_font_size"}

    def test_label_font_size_op_ints(self):
        labels = hv.Labels([(0, 0, 10), (0, 1, 4), (0, 2, 8)], vdims="size").opts(
            text_font_size="size"
        )
        plot = bokeh_renderer.get_plot(labels)
        cds = plot.handles["cds"]
        glyph = plot.handles["glyph"]
        assert cds.data["text_font_size"] == ["10pt", "4pt", "8pt"]
        assert property_to_dict(glyph.text_font_size) == {"field": "text_font_size"}

    def test_labels_text_color_cycle(self):
        hm = hv.HoloMap(
            {i: hv.Labels([(0, 0 + i, "Label 1"), (1, 1 + i, "Label 2")]) for i in range(3)}
        ).overlay()
        assert isinstance(hm[0].opts["text_color"], hv.Cycle)

    def test_labels_hover_with_vdims(self):
        labels = hv.Labels(
            [(0, 1, "A", 10.5, 100), (1, 0, "B", 20.3, 200)], vdims=["text", "value", "count"]
        ).opts(hover_tooltips=[("Text", "@text"), ("Value", "@value"), ("Count", "@count")])
        plot = bokeh_renderer.get_plot(labels)
        source = plot.handles["source"]
        glyph = plot.handles["glyph"]
        hover = plot.handles["hover"]

        expected = {
            "x": np.array([0, 1]),
            "y": np.array([1, 0]),
            "text": ["A", "B"],
            "value": np.array([10.5, 20.3]),
            "count": np.array([100, 200]),
        }
        assert_dict_equal(source.data, expected)

        assert glyph.x == "x"
        assert glyph.y == "y"
        assert glyph.text == "text"

        assert hover.tooltips == [
            ("Text", "@{text}"),
            ("Value", "@{value}"),
            ("Count", "@{count}"),
        ]
