import numpy as np

from holoviews.element.util import compute_edges
from holoviews.element.comparison import ComparisonTestCase

class TestComputeEdges(ComparisonTestCase):
    """
    Tests for compute_edges function.
    """

    def setUp(self):
        self.array1 = [.5, 1.5, 2.5]
        self.array2 = [.5, 1.0000001, 1.5]
        self.array3 = [1, 2, 4]

    def test_simple_edges(self):
        self.assertEqual(compute_edges(self.array1),
                         np.array([0, 1, 2, 3]))

    def test_close_edges(self):
        self.assertEqual(compute_edges(self.array2),
                         np.array([0.25, 0.75, 1.25, 1.75]))

    def test_uneven_edges(self):
        with self.assertRaisesRegexp(ValueError, "Centered bins"):
            compute_edges(self.array3)
