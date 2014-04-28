import unittest
from numpy.testing import assert_array_almost_equal

from dataviews import SheetStack, SheetOverlay
from dataviews.sheetviews import SheetView


class ViewTestCase(unittest.TestCase):
    """
    The class implements comparisons between View objects for the
    purposes of testing. The most important attribute that needs to be
    compared is the data attribute as this contains the raw data held
    by the View object.
    """

    def __init__(self, *args, **kwargs):
        super(ViewTestCase, self).__init__(*args, **kwargs)

        self.addTypeEqualityFunc(SheetView, self.compare_sheetviews)
        self.addTypeEqualityFunc(SheetStack, self.compare_sheetstack)

        self.addTypeEqualityFunc(SheetOverlay, self.compare_overlays)


    def compare_sheetstack(self, view1, view2, msg):
        self.bounds_check(view1,view2)
        self.compare_stack(view1, view2, msg)


    def compare_stack(self, view1, view2, msg):

        if view1.ndims != view2.ndims:
            raise self.failureException("Stacks have different numbers of dimensions.")

        if view1.dimension_labels != view2.dimension_labels:
            raise self.failureException("Stacks have different dimension labels.")

        if len(view1.keys()) != len(view2.keys()):
            raise self.failureException("Stacks have different numbers of keys.")

        if set(view1.keys()) != set(view2.keys()):
            raise self.failureException("Stacks have different sets of keys.")

        for el1, el2 in zip(view1, view2):
            self.assertEqual(el1,el2)


    def compare_sheetviews(self, view1, view2, msg):
        try:
            assert_array_almost_equal(view1.data, view2.data)
        except AssertionError as e:
            raise self.failureException(e.message)

        self.bounds_check(view1,view2)


    def compare_annotations(self, view1, view2, msg):
        if set(view1.data) != set(view2.data):
            raise self.failureException("Annotations contain different sets of annotations.")


    def compare_overlays(self, view1, view2, msg):

        if len(view1) != len(view2):
            raise self.failureException("Overlays have different lengths.")

        self.bounds_check(view1, view2)

        for (layer1, layer2) in zip(view1, view2):
            self.assertEqual(layer1, layer2)


    def bounds_check(self, view1, view2):
        if view1.bounds.lbrt() != view2.bounds.lbrt():
            raise self.failureException("BoundingBoxes are mismatched.")
