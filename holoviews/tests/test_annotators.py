from holoviews import Overlay
from holoviews.annotators import annotate, PointAnnotator, PathAnnotator
from holoviews.element import Points, Path, Table
from holoviews.element.tiles import Wikipedia, Tiles

from holoviews.tests.plotting.bokeh.testplot import TestBokehPlot


class Test_annotate(TestBokehPlot):

    def test_compose_annotators(self):
        layout1 = annotate(Points([]), annotations=['Label'])
        layout2 = annotate(Path([]), annotations=['Name'])

        combined = annotate.compose(layout1, layout2)
        overlay = combined.DynamicMap.I[()]
        tables = combined.Annotator.I[()]

        self.assertIsInstance(overlay, Overlay)
        self.assertEqual(len(overlay), 2)
        self.assertEqual(overlay.get(0), Points([], vdims='Label'))
        self.assertEqual(overlay.get(1), Path([], vdims='Name'))

        self.assertIsInstance(tables, Overlay)
        self.assertEqual(len(tables), 3)

    def test_annotate_overlay(self):
        layout = annotate(Wikipedia() * Points([]), annotations=['Label'])

        overlay = layout.DynamicMap.I[()]
        tables = layout.Annotator.PointAnnotator[()]

        self.assertIsInstance(overlay, Overlay)
        self.assertEqual(len(overlay), 2)
        self.assertIsInstance(overlay.get(0), Tiles)
        self.assertEqual(overlay.get(1), Points([], vdims='Label'))

        self.assertIsInstance(tables, Overlay)
        self.assertEqual(len(tables), 1)

    def test_annotated_property(self):
        annotator = annotate.instance()
        annotator(Points([]), annotations=['Label'])
        self.assertIn('Label', annotator.annotated)

    def test_selected_property(self):
        annotator = annotate.instance()
        annotator(Points([(1, 2), (2, 3)]), annotations=['Label'])
        annotator.annotator._selection.update(index=[1])
        self.assertEqual(annotator.selected, Points([(2, 3, '')], vdims='Label'))


class TestPointAnnotator(TestBokehPlot):

    def test_add_annotations(self):
        annotator = PointAnnotator(Points([]), annotations=['Label'])
        self.assertIn('Label', annotator.object)

    def test_add_name(self):
        annotator = PointAnnotator(name="Test Annotator", annotations=['Label'])
        self.assertEqual(annotator._stream.tooltip, 'Test Annotator Tool')
        self.assertEqual(annotator._table.label, "Test Annotator")
        self.assertEqual(annotator.editor._names, ["Test Annotator"])

    def test_annotation_type(self):
        annotator = PointAnnotator(Points([(1, 2)]), annotations={'Int': int})
        expected = Table([(1, 2, 0)], ['x', 'y'], vdims=['Int'], label='PointAnnotator')
        self.assertEqual(annotator._table, expected)

    def test_replace_object(self):
        annotator = PointAnnotator(Points([]), annotations=['Label'])
        annotator.object = Points([(1, 2)])
        self.assertIn('Label', annotator.object)
        expected = Table([(1, 2, '')], ['x', 'y'], vdims=['Label'], label='PointAnnotator')
        self.assertEqual(annotator._table, expected)
        self.assertIs(annotator._link.target, annotator._table)

    def test_stream_update(self):
        annotator = PointAnnotator(Points([(1, 2)]), annotations=['Label'])
        annotator._stream.event(data={'x': [1], 'y': [2], 'Label': ['A']})
        self.assertEqual(annotator.object, Points([(1, 2, 'A')], vdims=['Label']))



class TestPathAnnotator(TestBokehPlot):

    def test_add_annotations(self):
        annotator = PathAnnotator(Path([]), annotations=['Label'])
        self.assertIn('Label', annotator.object)

    def test_add_name(self):
        annotator = PathAnnotator(name="Test Annotator", annotations=['Label'])
        self.assertEqual(annotator._stream.tooltip, 'Test Annotator Tool')
        self.assertEqual(annotator._vertex_stream.tooltip, 'Test Annotator Edit Tool')
        self.assertEqual(annotator._table.label, "Test Annotator")
        self.assertEqual(annotator._vertex_table.label, "Test Annotator Vertices")
        self.assertEqual(annotator.editor._names, ["Test Annotator", "Test Annotator Vertices"])

    def test_add_vertex_annotations(self):
        annotator = PathAnnotator(Path([]), vertex_annotations=['Label'])
        self.assertIn('Label', annotator.object)

    def test_replace_object(self):
        annotator = PathAnnotator(Path([]), annotations=['Label'], vertex_annotations=['Value'])
        annotator.object = Path([(1, 2), (2, 3), (0, 0)])
        self.assertIn('Label', annotator.object)
        expected = Table([('')], kdims=['Label'], label='PathAnnotator')
        self.assertEqual(annotator._table, expected)
        expected = Table([], ['x', 'y'], 'Value', label='PathAnnotator Vertices')
        self.assertEqual(annotator._vertex_table, expected)
        self.assertIs(annotator._link.target, annotator._table)
        self.assertIs(annotator._vertex_link.target, annotator._vertex_table)
