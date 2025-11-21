"""
Test cases for the Comparisons class over the composite types:

Layout (the + operator)
Overlay    (the * operator)

HoloMaps are not tested in this file.
"""

from holoviews import Element
from holoviews.element.comparison import ComparisonTestCase


class CompositeComparisonTestCase(ComparisonTestCase):

    def setUp(self):
        self.el1 = Element('data1')
        self.el2 = Element('data2')
        self.el3 = Element('data3')
        self.el4 = Element('data5', group='ValB')
        self.el5 = Element('data6', label='LabelA')

    #========================#
    # Tests for layout trees #
    #========================#

    def test_layouttree_comparison_equal(self):
        t1 = self.el1 + self.el2
        t2 = self.el1 + self.el2
        self.assertEqual(t1, t2)

    def test_layouttree_comparison_equal_large(self):
        t1 = self.el1 + self.el2 + self.el4 + self.el5
        t2 = self.el1 + self.el2 + self.el4 + self.el5
        self.assertEqual(t1, t2)


    def test_layouttree_comparison_unequal_data(self):
        t1 = self.el1 + self.el2
        t2 = self.el1 + self.el3
        try:
            self.assertEqual(t1, t2)
        except AssertionError as e:
            self.assertEqual(str(e),"'data2' != 'data3'")

    def test_layouttree_comparison_unequal_paths(self):
        t1 = self.el1 + self.el2
        t2 = self.el1 + self.el2.relabel(group='ValA')
        try:
            self.assertEqual(t1, t2)
        except AssertionError as e:
            self.assertEqual(str(e), 'Layouts have mismatched paths.')

    def test_layouttree_comparison_unequal_sizes(self):
        t1 = self.el1 + self.el2
        t2 = self.el1 + self.el2 + self.el3
        try:
            self.assertEqual(t1, t2)
        except AssertionError as e:
            self.assertEqual(str(e), 'Layouts have mismatched path counts.')

    #=============================#
    # Matching tests for Overlays #
    #=============================#

    def test_overlay_comparison_equal(self):
        t1 = self.el1 * self.el2
        t2 = self.el1 * self.el2
        self.assertEqual(t1, t2)

    def test_overlay_comparison_equal_large(self):
        t1 = self.el1 * self.el2 * self.el3 * self.el4
        t2 = self.el1 * self.el2 * self.el3 * self.el4
        self.assertEqual(t1, t2)


    def test_overlay_comparison_unequal_data(self):
        t1 = self.el1 * self.el2
        t2 = self.el1 * self.el3
        try:
            self.assertEqual(t1, t2)
        except AssertionError as e:
            self.assertEqual(str(e),"'data2' != 'data3'")

    def test_overlay_comparison_unequal_paths(self):
        t1 = self.el1 * self.el2
        t2 = self.el1 * self.el2.relabel(group='ValA')
        try:
            self.assertEqual(t1, t2)
        except AssertionError as e:
            self.assertEqual(str(e), 'Overlays have mismatched paths.')

    def test_overlay_comparison_unequal_sizes(self):
        t1 = self.el1 * self.el2
        t2 = self.el1 * self.el2 * self.el3
        try:
            self.assertEqual(t1, t2)
        except AssertionError as e:
            self.assertEqual(str(e), 'Overlays have mismatched path counts.')


    #==================================#
    # Mixed composite comparison tests #
    #==================================#

    def test_composite_comparison_equal(self):
        t1 = (self.el1 * self.el2) + (self.el1 * self.el2)
        t2 = (self.el1 * self.el2) + (self.el1 * self.el2)
        self.assertEqual(t1, t2)

    def test_composite_unequal_data(self):
        t1 = (self.el1 * self.el2) + (self.el1 * self.el2)
        t2 = (self.el1 * self.el2) + (self.el1 * self.el3)
        try:
            self.assertEqual(t1, t2)
        except AssertionError as e:
            self.assertEqual(str(e), "'data2' != 'data3'")

    def test_composite_unequal_paths_outer(self):
        t1 = (self.el1 * self.el2) + (self.el1 * self.el2).relabel(group='ValA')
        t2 = (self.el1 * self.el2) + (self.el1 * self.el3)
        try:
            self.assertEqual(t1, t2)
        except AssertionError as e:
            self.assertEqual(str(e), 'Layouts have mismatched paths.')

    def test_composite_unequal_paths_inner(self):
        t1 = (self.el1 * self.el2) + (self.el1 * self.el2.relabel(group='ValA'))
        t2 = (self.el1 * self.el2) + (self.el1 * self.el3)
        try:
            self.assertEqual(t1, t2)
        except AssertionError as e:
            self.assertEqual(str(e), 'Overlays have mismatched paths.')


    def test_composite_unequal_sizes(self):
        t1 = (self.el1 * self.el2) + (self.el1 * self.el2) + self.el3
        t2 = (self.el1 * self.el2) + (self.el1 * self.el2)
        try:
            self.assertEqual(t1, t2)
        except AssertionError as e:
            self.assertEqual(str(e), 'Layouts have mismatched path counts.')
