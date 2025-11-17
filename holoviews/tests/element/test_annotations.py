import numpy as np
import param
import pytest
from packaging.version import Version

from holoviews import Annotation, Arrow, HLine, Spline, Text, VLine
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase


class AnnotationTests(ComparisonTestCase):
    """
    Tests allowable data formats when constructing
    the basic Element types.
    """

    def test_hline_invalid_constructor(self):
        if Version(param.__version__).release >= (2, 0, 0):
            err = "ClassSelector parameter 'HLine.y' value must be an instance of"
        else:
            err = "ClassSelector parameter 'y' value must be an instance of"
        with pytest.raises(ValueError) as excinfo:
            HLine(None)
        assert err in str(excinfo.value)

    def test_text_string_position(self):
        text = Text('A', 1, 'A')
        Points([('A', 1)]) * text
        self.assertEqual(text.x, 'A')

    def test_hline_dimension_values(self):
        hline = HLine(0)
        self.assertTrue(all(not np.isfinite(v) for v in hline.range(0)))
        self.assertEqual(hline.range(1), (0, 0))

        # Testing numpy inputs
        hline = HLine(np.array([0]))
        self.assertTrue(all(not np.isfinite(v) for v in hline.range(0)))
        self.assertEqual(hline.range(1), (0, 0))

        hline = HLine(np.array(0))
        self.assertTrue(all(not np.isfinite(v) for v in hline.range(0)))
        self.assertEqual(hline.range(1), (0, 0))

    def test_vline_dimension_values(self):
        vline = VLine(0)
        self.assertEqual(vline.range(0), (0, 0))
        self.assertTrue(all(not np.isfinite(v) for v in vline.range(1)))

        # Testing numpy inputs
        vline = VLine(np.array([0]))
        self.assertEqual(vline.range(0), (0, 0))
        self.assertTrue(all(not np.isfinite(v) for v in vline.range(1)))

        vline = VLine(np.array(0))
        self.assertEqual(vline.range(0), (0, 0))
        self.assertTrue(all(not np.isfinite(v) for v in vline.range(1)))

    def test_arrow_redim_range_aux(self):
        annotations = Arrow(0, 0)
        redimmed = annotations.redim.range(x=(-0.5,0.5))
        self.assertEqual(redimmed.kdims[0].range, (-0.5,0.5))

    def test_deep_clone_map_select_redim(self):
        annotations = (Text(0, 0, 'A') + Arrow(0, 0) + HLine(0) + VLine(0))
        selected = annotations.select(x=(0, 5))
        redimmed = selected.redim(x='z')
        relabelled = redimmed.relabel(label='foo', depth=5)
        mapped = relabelled.map(lambda x: x.clone(group='bar'), Annotation)
        kwargs = dict(label='foo', group='bar', extents=(0, None, 5, None), kdims=['z', 'y'])
        self.assertEqual(mapped.Text.I, Text(0, 0, 'A', **kwargs))
        self.assertEqual(mapped.Arrow.I, Arrow(0, 0, **kwargs))
        self.assertEqual(mapped.HLine.I, HLine(0, **kwargs))
        self.assertEqual(mapped.VLine.I, VLine(0, **kwargs))

    def test_spline_clone(self):
        points = [(-0.3, -0.3), (0,0), (0.25, -0.25), (0.3, 0.3)]
        spline = Spline((points,[])).clone()
        self.assertEqual(spline.dimension_values(0), np.array([-0.3, 0, 0.25, 0.3]))
        self.assertEqual(spline.dimension_values(1), np.array([-0.3, 0, -0.25, 0.3]))
