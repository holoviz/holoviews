"""
Test cases for the Comparisons class over the composite types:

Layout (the + operator)
Overlay    (the * operator)

HoloMaps are not tested in this file.
"""

import pytest

from holoviews import Element
from holoviews.testing import assert_element_equal


class CompositeComparisonTestCase:

    def setup_method(self):
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
        assert_element_equal(t1, t2)

    def test_layouttree_comparison_equal_large(self):
        t1 = self.el1 + self.el2 + self.el4 + self.el5
        t2 = self.el1 + self.el2 + self.el4 + self.el5
        assert_element_equal(t1, t2)


    def test_layouttree_comparison_unequal_data(self):
        t1 = self.el1 + self.el2
        t2 = self.el1 + self.el3
        msg = "'data2' == 'data3'"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(t1, t2)

    def test_layouttree_comparison_unequal_paths(self):
        t1 = self.el1 + self.el2
        t2 = self.el1 + self.el2.relabel(group='ValA')
        msg = "At index 1 diff"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(t1, t2)

    def test_layouttree_comparison_unequal_sizes(self):
        t1 = self.el1 + self.el2
        t2 = self.el1 + self.el2 + self.el3
        msg = "Right contains one more item"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(t1, t2)

    #=============================#
    # Matching tests for Overlays #
    #=============================#

    def test_overlay_comparison_equal(self):
        t1 = self.el1 * self.el2
        t2 = self.el1 * self.el2
        assert_element_equal(t1, t2)

    def test_overlay_comparison_equal_large(self):
        t1 = self.el1 * self.el2 * self.el3 * self.el4
        t2 = self.el1 * self.el2 * self.el3 * self.el4
        assert_element_equal(t1, t2)


    def test_overlay_comparison_unequal_data(self):
        t1 = self.el1 * self.el2
        t2 = self.el1 * self.el3
        msg = "'data2' == 'data3'"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(t1, t2)

    def test_overlay_comparison_unequal_paths(self):
        t1 = self.el1 * self.el2
        t2 = self.el1 * self.el2.relabel(group='ValA')
        msg = "At index 1 diff"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(t1, t2)

    def test_overlay_comparison_unequal_sizes(self):
        t1 = self.el1 * self.el2
        t2 = self.el1 * self.el2 * self.el3
        msg = "Right contains one more item"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(t1, t2)


    #==================================#
    # Mixed composite comparison tests #
    #==================================#

    def test_composite_comparison_equal(self):
        t1 = (self.el1 * self.el2) + (self.el1 * self.el2)
        t2 = (self.el1 * self.el2) + (self.el1 * self.el2)
        assert_element_equal(t1, t2)

    def test_composite_unequal_data(self):
        t1 = (self.el1 * self.el2) + (self.el1 * self.el2)
        t2 = (self.el1 * self.el2) + (self.el1 * self.el3)
        msg = "'data2' == 'data3'"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(t1, t2)

    def test_composite_unequal_paths_outer(self):
        t1 = (self.el1 * self.el2) + (self.el1 * self.el2).relabel(group='ValA')
        t2 = (self.el1 * self.el2) + (self.el1 * self.el3)
        msg = "At index 1 diff"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(t1, t2)

    def test_composite_unequal_paths_inner(self):
        t1 = (self.el1 * self.el2) + (self.el1 * self.el2.relabel(group='ValA'))
        t2 = (self.el1 * self.el2) + (self.el1 * self.el3)
        msg = "At index 1 diff"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(t1, t2)

    def test_composite_unequal_sizes(self):
        t1 = (self.el1 * self.el2) + (self.el1 * self.el2) + self.el3
        t2 = (self.el1 * self.el2) + (self.el1 * self.el2)
        msg = "Left contains one more item"
        with pytest.raises(AssertionError, match=msg):
            assert_element_equal(t1, t2)
