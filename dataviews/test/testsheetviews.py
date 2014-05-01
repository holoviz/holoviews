import unittest
import numpy as np

from dataviews.boundingregion import BoundingBox
from dataviews import SheetView

# Duplicates testsheetview from topographica

class TestSheetView(unittest.TestCase):

    def setUp(self):
        self.activity1 = np.array([[1,2],[3,4]])
        self.bounds = BoundingBox(radius=0.5)

    def test_init(self):
        SheetView(self.activity1, self.bounds)


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[sys.argv[0], "--logging-level", "ERROR"])
