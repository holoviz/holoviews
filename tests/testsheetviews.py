import unittest
import numpy as np

from holoviews.core import BoundingBox
from holoviews.view import Matrix


# Duplicates testsheetview from topographica

class TestMatrix(unittest.TestCase):

    def setUp(self):
        self.activity1 = np.array([[1,2],[3,4]])
        self.bounds = BoundingBox(radius=0.5)

    def test_init(self):
        Matrix(self.activity1, self.bounds)


if __name__ == "__main__":
    import sys
    import nose
    nose.runmodule(argv=[sys.argv[0], "--logging-level", "ERROR"])
