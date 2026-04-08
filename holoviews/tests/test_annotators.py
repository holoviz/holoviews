from __future__ import annotations

import holoviews as hv
from holoviews.annotators import PathAnnotator, PointAnnotator
from holoviews.element.tiles import EsriStreet
from holoviews.testing import assert_element_equal
from holoviews.tests.plotting.bokeh.test_plot import TestBokehPlot


class Test_annotate(TestBokehPlot):
    def test_compose_annotators(self):
        layout1 = hv.annotate(hv.Points([]), annotations=["Label"])
        layout2 = hv.annotate(hv.Path([]), annotations=["Name"])

        combined = hv.annotate.compose(layout1, layout2)
        overlay = combined.DynamicMap.I[()]
        tables = combined.Annotator.I[()]

        assert isinstance(overlay, hv.Overlay)
        assert len(overlay) == 2
        assert_element_equal(overlay.get(0), hv.Points([], vdims="Label"))
        assert_element_equal(overlay.get(1), hv.Path([], vdims="Name"))

        assert isinstance(tables, hv.Overlay)
        assert len(tables) == 3

    def test_annotate_overlay(self):
        layout = hv.annotate(EsriStreet() * hv.Points([]), annotations=["Label"])

        overlay = layout.DynamicMap.I[()]
        tables = layout.Annotator.PointAnnotator[()]

        assert isinstance(overlay, hv.Overlay)
        assert len(overlay) == 2
        assert isinstance(overlay.get(0), hv.Tiles)
        assert_element_equal(overlay.get(1), hv.Points([], vdims="Label"))

        assert isinstance(tables, hv.Overlay)
        assert len(tables) == 1

    def test_annotated_property(self):
        annotator = hv.annotate.instance()
        annotator(hv.Points([]), annotations=["Label"])
        assert "Label" in annotator.annotated

    def test_selected_property(self):
        annotator = hv.annotate.instance()
        annotator(hv.Points([(1, 2), (2, 3)]), annotations=["Label"])
        annotator.annotator._selection.update(index=[1])
        assert_element_equal(annotator.selected, hv.Points([(2, 3, "")], vdims="Label"))


class TestPointAnnotator(TestBokehPlot):
    def test_add_annotations(self):
        annotator = PointAnnotator(hv.Points([]), annotations=["Label"])
        assert "Label" in annotator.object

    def test_add_name(self):
        annotator = PointAnnotator(name="Test Annotator", annotations=["Label"])
        assert annotator._stream.tooltip == "Test Annotator Tool"
        assert annotator._table.label == "Test Annotator"
        assert annotator.editor._names == ["Test Annotator"]

    def test_annotation_type(self):
        annotator = PointAnnotator(hv.Points([(1, 2)]), annotations={"Int": int})
        expected = hv.Table([(1, 2, 0)], ["x", "y"], vdims=["Int"], label="PointAnnotator")
        assert_element_equal(annotator._table, expected)

    def test_replace_object(self):
        annotator = PointAnnotator(hv.Points([]), annotations=["Label"])
        annotator.object = hv.Points([(1, 2)])
        assert "Label" in annotator.object
        expected = hv.Table([(1, 2, "")], ["x", "y"], vdims=["Label"], label="PointAnnotator")
        assert_element_equal(annotator._table, expected)
        assert annotator._link.target is annotator._table

    def test_stream_update(self):
        annotator = PointAnnotator(hv.Points([(1, 2)]), annotations=["Label"])
        annotator._stream.event(data={"x": [1], "y": [2], "Label": ["A"]})
        assert_element_equal(annotator.object, hv.Points([(1, 2, "A")], vdims=["Label"]))


class TestPathAnnotator(TestBokehPlot):
    def test_add_annotations(self):
        annotator = PathAnnotator(hv.Path([]), annotations=["Label"])
        assert "Label" in annotator.object

    def test_add_name(self):
        annotator = PathAnnotator(name="Test Annotator", annotations=["Label"])
        assert annotator._stream.tooltip == "Test Annotator Tool"
        assert annotator._vertex_stream.tooltip == "Test Annotator Edit Tool"
        assert annotator._table.label == "Test Annotator"
        assert annotator._vertex_table.label == "Test Annotator Vertices"
        assert annotator.editor._names == ["Test Annotator", "Test Annotator Vertices"]

    def test_add_vertex_annotations(self):
        annotator = PathAnnotator(hv.Path([]), vertex_annotations=["Label"])
        assert "Label" in annotator.object

    def test_replace_object(self):
        annotator = PathAnnotator(hv.Path([]), annotations=["Label"], vertex_annotations=["Value"])
        annotator.object = hv.Path([(1, 2), (2, 3), (0, 0)])
        assert "Label" in annotator.object
        expected = hv.Table([("")], kdims=["Label"], label="PathAnnotator")
        assert_element_equal(annotator._table, expected)
        expected = hv.Table([], ["x", "y"], "Value", label="PathAnnotator Vertices")
        assert_element_equal(annotator._vertex_table, expected)
        assert annotator._link.target is annotator._table
        assert annotator._vertex_link.target is annotator._vertex_table
