import numpy as np
import pytest

from holoviews import Annotation, Arrow, HLine, Spline, Text, VLine
from holoviews.element import Points
from holoviews.testing import assert_data_equal, assert_element_equal


class AnnotationTests:
    """
    Tests allowable data formats when constructing
    the basic Element types.
    """

    def test_hline_invalid_constructor(self):
        msg = r"ClassSelector parameter 'HLine\.y' value must be an instance of"
        with pytest.raises(ValueError, match=msg):
            HLine(None)

    def test_text_string_position(self):
        text = Text('A', 1, 'A')
        Points([('A', 1)]) * text
        assert text.x == 'A'

    def test_hline_dimension_values(self):
        hline = HLine(0)
        assert all(not np.isfinite(v) for v in hline.range(0))
        assert hline.range(1) == (0, 0)

        # Testing numpy inputs
        hline = HLine(np.array([0]))
        assert all(not np.isfinite(v) for v in hline.range(0))
        assert hline.range(1) == (0, 0)

        hline = HLine(np.array(0))
        assert all(not np.isfinite(v) for v in hline.range(0))
        assert hline.range(1) == (0, 0)

    def test_vline_dimension_values(self):
        vline = VLine(0)
        assert vline.range(0) == (0, 0)
        assert all(not np.isfinite(v) for v in vline.range(1))

        # Testing numpy inputs
        vline = VLine(np.array([0]))
        assert vline.range(0) == (0, 0)
        assert all(not np.isfinite(v) for v in vline.range(1))

        vline = VLine(np.array(0))
        assert vline.range(0) == (0, 0)
        assert all(not np.isfinite(v) for v in vline.range(1))

    def test_arrow_redim_range_aux(self):
        annotations = Arrow(0, 0)
        redimmed = annotations.redim.range(x=(-0.5,0.5))
        assert redimmed.kdims[0].range == (-0.5,0.5)

    def test_deep_clone_map_select_redim(self):
        annotations = (Text(0, 0, 'A') + Arrow(0, 0) + HLine(0) + VLine(0))
        selected = annotations.select(x=(0, 5))
        redimmed = selected.redim(x='z')
        relabelled = redimmed.relabel(label='foo', depth=5)
        mapped = relabelled.map(lambda x: x.clone(group='bar'), Annotation)
        kwargs = dict(label='foo', group='bar', extents=(0, None, 5, None), kdims=['z', 'y'])
        assert_element_equal(mapped.Text.I, Text(0, 0, 'A', **kwargs))
        assert_element_equal(mapped.Arrow.I, Arrow(0, 0, **kwargs))
        assert_element_equal(mapped.HLine.I, HLine(0, **kwargs))
        assert_element_equal(mapped.VLine.I, VLine(0, **kwargs))

    def test_spline_clone(self):
        points = [(-0.3, -0.3), (0,0), (0.25, -0.25), (0.3, 0.3)]
        spline = Spline((points,[])).clone()
        assert_data_equal(spline.dimension_values(0), np.array([-0.3, 0, 0.25, 0.3]))
        assert_data_equal(spline.dimension_values(1), np.array([-0.3, 0, -0.25, 0.3]))
