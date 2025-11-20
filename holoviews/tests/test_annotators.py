from holoviews import Overlay
from holoviews.annotators import PathAnnotator, PointAnnotator, annotate
from holoviews.element import Path, Points, Table
from holoviews.element.tiles import EsriStreet, Tiles
from holoviews.testing import assert_element_equal
from holoviews.tests.plotting.bokeh.test_plot import TestBokehPlot


class Test_annotate(TestBokehPlot):

    def test_compose_annotators(self):
        layout1 = annotate(Points([]), annotations=['Label'])
        layout2 = annotate(Path([]), annotations=['Name'])

        combined = annotate.compose(layout1, layout2)
        overlay = combined.DynamicMap.I[()]
        tables = combined.Annotator.I[()]

        assert isinstance(overlay, Overlay)
        assert len(overlay) == 2
        assert_element_equal(overlay.get(0), Points([], vdims='Label'))
        assert_element_equal(overlay.get(1), Path([], vdims='Name'))

        assert isinstance(tables, Overlay)
        assert len(tables) == 3

    def test_annotate_overlay(self):
        layout = annotate(EsriStreet() * Points([]), annotations=['Label'])

        overlay = layout.DynamicMap.I[()]
        tables = layout.Annotator.PointAnnotator[()]

        assert isinstance(overlay, Overlay)
        assert len(overlay) == 2
        assert isinstance(overlay.get(0), Tiles)
        assert_element_equal(overlay.get(1), Points([], vdims='Label'))

        assert isinstance(tables, Overlay)
        assert len(tables) == 1

    def test_annotated_property(self):
        annotator = annotate.instance()
        annotator(Points([]), annotations=['Label'])
        assert 'Label' in annotator.annotated

    def test_selected_property(self):
        annotator = annotate.instance()
        annotator(Points([(1, 2), (2, 3)]), annotations=['Label'])
        annotator.annotator._selection.update(index=[1])
        assert_element_equal(annotator.selected, Points([(2, 3, '')], vdims='Label'))


class TestPointAnnotator(TestBokehPlot):

    def test_add_annotations(self):
        annotator = PointAnnotator(Points([]), annotations=['Label'])
        assert 'Label' in annotator.object

    def test_add_name(self):
        annotator = PointAnnotator(name="Test Annotator", annotations=['Label'])
        assert annotator._stream.tooltip == 'Test Annotator Tool'
        assert annotator._table.label == "Test Annotator"
        assert annotator.editor._names == ["Test Annotator"]

    def test_annotation_type(self):
        annotator = PointAnnotator(Points([(1, 2)]), annotations={'Int': int})
        expected = Table([(1, 2, 0)], ['x', 'y'], vdims=['Int'], label='PointAnnotator')
        assert_element_equal(annotator._table, expected)

    def test_replace_object(self):
        annotator = PointAnnotator(Points([]), annotations=['Label'])
        annotator.object = Points([(1, 2)])
        assert 'Label' in annotator.object
        expected = Table([(1, 2, '')], ['x', 'y'], vdims=['Label'], label='PointAnnotator')
        assert_element_equal(annotator._table, expected)
        assert annotator._link.target is annotator._table

    def test_stream_update(self):
        annotator = PointAnnotator(Points([(1, 2)]), annotations=['Label'])
        annotator._stream.event(data={'x': [1], 'y': [2], 'Label': ['A']})
        assert_element_equal(annotator.object, Points([(1, 2, 'A')], vdims=['Label']))



class TestPathAnnotator(TestBokehPlot):

    def test_add_annotations(self):
        annotator = PathAnnotator(Path([]), annotations=['Label'])
        assert 'Label' in annotator.object

    def test_add_name(self):
        annotator = PathAnnotator(name="Test Annotator", annotations=['Label'])
        assert annotator._stream.tooltip == 'Test Annotator Tool'
        assert annotator._vertex_stream.tooltip == 'Test Annotator Edit Tool'
        assert annotator._table.label == "Test Annotator"
        assert annotator._vertex_table.label == "Test Annotator Vertices"
        assert annotator.editor._names == ["Test Annotator", "Test Annotator Vertices"]

    def test_add_vertex_annotations(self):
        annotator = PathAnnotator(Path([]), vertex_annotations=['Label'])
        assert 'Label' in annotator.object

    def test_replace_object(self):
        annotator = PathAnnotator(Path([]), annotations=['Label'], vertex_annotations=['Value'])
        annotator.object = Path([(1, 2), (2, 3), (0, 0)])
        assert 'Label' in annotator.object
        expected = Table([('')], kdims=['Label'], label='PathAnnotator')
        assert_element_equal(annotator._table, expected)
        expected = Table([], ['x', 'y'], 'Value', label='PathAnnotator Vertices')
        assert_element_equal(annotator._vertex_table, expected)
        assert annotator._link.target is annotator._table
        assert annotator._vertex_link.target is annotator._vertex_table
