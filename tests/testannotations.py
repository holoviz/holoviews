import numpy as np

from holoviews import Image, HLine, VLine 
from holoviews.element.comparison import ComparisonTestCase

class AnnotationTests(ComparisonTestCase):
    """
    Tests allowable data formats when constructing
    the basic Element types.
    """

    def test_hline_dimension_values(self):
        hline = HLine(0)
        self.assertEqual(hline.range(0), (None, None))
        self.assertEqual(hline.range(1), (0, 0))

    def test_vline_dimension_values(self):
        hline = VLine(0)
        self.assertEqual(hline.range(0), (0, 0))
        self.assertEqual(hline.range(1), (None, None))
