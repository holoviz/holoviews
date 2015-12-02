# -*- coding: utf-8 -*-
"""
Unit tests of the helper functions in core.utils
"""
import sys, math
import unittest
from unittest import SkipTest

import numpy as np

from holoviews.core.util import sanitize_identifier_fn, find_range, max_range
from holoviews.element.comparison import ComparisonTestCase

py_version = sys.version_info.major

sanitize_identifier = sanitize_identifier_fn.instance()

class TestAllowablePrefix(ComparisonTestCase):
    """
    Tests of allowable and hasprefix method.
    """

    def test_allowable_false_1(self):
        self.assertEqual(sanitize_identifier.allowable('trait_names'), False)

    def test_allowable_false_2(self):
        self.assertEqual(sanitize_identifier.allowable('_repr_png_'), False)

    def test_allowable_false_3(self):
        self.assertEqual(sanitize_identifier.allowable('_ipython_display_'),
                         False)

    def test_allowable_false_underscore(self):
        self.assertEqual(sanitize_identifier.allowable('_foo', True), False)

    def test_allowable_true(self):
        self.assertEqual(sanitize_identifier.allowable('some_string'), True)

    def test_prefix_test1_py2(self):
        if py_version != 2: raise SkipTest
        prefixed = sanitize_identifier.prefixed('_some_string', version=2)
        self.assertEqual(prefixed, True)

    def test_prefix_test2_py2(self):
        if py_version != 2: raise SkipTest
        prefixed = sanitize_identifier.prefixed('some_string', version=2)
        self.assertEqual(prefixed, False)

    def test_prefix_test3_py2(self):
        if py_version != 2: raise SkipTest
        prefixed = sanitize_identifier.prefixed('0some_string', version=2)
        self.assertEqual(prefixed, True)

    def test_prefix_test1_py3(self):
        if py_version != 3: raise SkipTest
        prefixed = sanitize_identifier.prefixed('_some_string', version=3)
        self.assertEqual(prefixed, True)

    def test_prefix_test2_py3(self):
        if py_version != 3: raise SkipTest
        prefixed = sanitize_identifier.prefixed('some_string', version=3)
        self.assertEqual(prefixed, False)

    def test_prefix_test3_py3(self):
        if py_version != 3: raise SkipTest
        prefixed = sanitize_identifier.prefixed('۵some_string', version=3)
        self.assertEqual(prefixed, True)


class TestSanitizationPy2(ComparisonTestCase):
    """
    Tests of sanitize_identifier (Python 2)
    """
    def setUp(self):
        if py_version != 2: raise SkipTest

    def test_simple_pound_sanitized_py2(self):
        sanitized = sanitize_identifier('£', version=2)
        self.assertEqual(sanitized, 'pound')

    def test_simple_digit_sanitized_py2(self):
        sanitized = sanitize_identifier('0', version=2)
        self.assertEqual(sanitized, 'A_0')

    def test_simple_underscore_sanitized_py2(self):
        sanitized = sanitize_identifier('_test', version=2)
        self.assertEqual(sanitized, 'A__test')

    def test_simple_alpha_sanitized_py2(self):
        sanitized = sanitize_identifier('α', version=2)
        self.assertEqual(sanitized, 'alpha')

    def test_simple_a_pound_sanitized_py2(self):
        sanitized = sanitize_identifier('a £', version=2)
        self.assertEqual(sanitized, 'A_pound')

    def test_capital_delta_sanitized_py2(self):
        sanitized = sanitize_identifier('Δ', version=2)
        self.assertEqual(sanitized, 'Delta')

    def test_lowercase_delta_sanitized_py2(self):
        sanitized = sanitize_identifier('δ', version=2)
        self.assertEqual(sanitized, 'delta')

    def test_simple_alpha_beta_sanitized_py2(self):
        sanitized = sanitize_identifier('α β', version=2)
        self.assertEqual(sanitized, 'alpha_beta')

    def test_simple_alpha_beta_underscore_sanitized_py2(self):
        sanitized = sanitize_identifier('α_β', version=2)
        self.assertEqual(sanitized, 'alpha_beta')

    def test_simple_alpha_beta_double_underscore_sanitized_py2(self):
        sanitized = sanitize_identifier('α__β', version=2)
        self.assertEqual(sanitized, 'alpha__beta')

    def test_simple_alpha_beta_mixed_underscore_space_sanitized_py2(self):
        sanitized = sanitize_identifier('α__  β', version=2)
        self.assertEqual(sanitized, 'alpha__beta')

    def test_alpha_times_two_py2(self):
        sanitized = sanitize_identifier('α*2', version=2)
        self.assertEqual(sanitized,  'alpha_times_2')

    def test_arabic_five_sanitized_py2(self):
        """
        Note: There would be a clash if you mixed the languages of
        your digits! E.g arabic ٥ five and urdu ۵ five
        """
        sanitized = sanitize_identifier('٥', version=2)
        self.assertEqual(sanitized, 'five')

    def test_urdu_five_sanitized_py2(self):
        """
        Note: There would be a clash if you mixed the languages of
        your digits! E.g arabic ٥ five and urdu ۵ five
        """
        sanitized = sanitize_identifier('۵', version=2)
        self.assertEqual(sanitized, 'five')

    def test_urdu_a_five_sanitized_py2(self):
        """
        Note: There would be a clash if you mixed the languages of
        your digits! E.g arabic ٥ five and urdu ۵ five
        """
        sanitized = sanitize_identifier('a ۵', version=2)
        self.assertEqual(sanitized, 'A_five')

    def test_umlaut_sanitized_py2(self):
        sanitized = sanitize_identifier('Festkörperphysik', version=2)
        self.assertEqual(sanitized, 'Festkorperphysik')

    def test_power_umlaut_sanitized_py2(self):
        sanitized = sanitize_identifier('^Festkörperphysik', version=2)
        self.assertEqual(sanitized, 'power_Festkorperphysik')

    def test_custom_dollar_removal_py2(self):
        sanitize_identifier.eliminations.extend(['dollar'])
        sanitized = sanitize_identifier('$E$', version=2)
        self.assertEqual(sanitized, 'E')
        sanitize_identifier.eliminations.remove('dollar')


class TestSanitizationPy3(ComparisonTestCase):
    """
    Tests of sanitize_identifier (Python 3)
    """
    def setUp(self):
        if py_version != 3: raise SkipTest

    def test_simple_pound_sanitized_py3(self):
        sanitized = sanitize_identifier('£', version=3)
        self.assertEqual(sanitized, 'pound')

    def test_simple_digit_sanitized_py3(self):
        sanitized = sanitize_identifier('0', version=3)
        self.assertEqual(sanitized, 'A_0')

    def test_simple_underscore_sanitized_py3(self):
        sanitized = sanitize_identifier('_test', version=3)
        self.assertEqual(sanitized, 'A__test')

    def test_simple_alpha_sanitized_py3(self):
        sanitized = sanitize_identifier('α', version=3)
        self.assertEqual(sanitized, 'α')

    def test_simple_a_pound_sanitized_py3(self):
        sanitized = sanitize_identifier('a £', version=3)
        self.assertEqual(sanitized, 'A_pound')

    def test_capital_delta_sanitized_py3(self):
        sanitized = sanitize_identifier('Δ', version=3)
        self.assertEqual(sanitized, 'Δ')

    def test_lowercase_delta_sanitized_py3(self):
        sanitized = sanitize_identifier('δ', version=3)
        self.assertEqual(sanitized, 'δ')

    def test_simple_alpha_beta_sanitized_py3(self):
        sanitized = sanitize_identifier('α β', version=3)
        self.assertEqual(sanitized, 'α_β')

    def test_simple_alpha_beta_underscore_sanitized_py3(self):
        sanitized = sanitize_identifier('α_β', version=3)
        self.assertEqual(sanitized, 'α_β')

    def test_simple_alpha_beta_double_underscore_sanitized_py3(self):
        sanitized = sanitize_identifier('α__β', version=3)
        self.assertEqual(sanitized, 'α__β')

    def test_simple_alpha_beta_mixed_underscore_space_sanitized_py3(self):
        sanitized = sanitize_identifier('α__  β', version=3)
        self.assertEqual(sanitized, 'α__β')

    def test_alpha_times_two_py3(self):
        sanitized = sanitize_identifier('α*2', version=3)
        self.assertEqual(sanitized,  'α_times_2')

    def test_arabic_five_sanitized_py3(self):
        """
        Note: There would be a clash if you mixed the languages of
        your digits! E.g arabic ٥ five and urdu ۵ five
        """
        try:
            sanitize_identifier('٥', version=3)
        except SyntaxError as e:
            assert str(e).startswith("String '٥' cannot be sanitized")

    def test_urdu_five_sanitized_py3(self):
        try:
            sanitize_identifier('۵', version=3)
        except SyntaxError as e:
            assert str(e).startswith("String '۵' cannot be sanitized")

    def test_urdu_a_five_sanitized_py3(self):
        """
        Note: There would be a clash if you mixed the languages of
        your digits! E.g arabic ٥ five and urdu ۵ five
        """
        sanitized = sanitize_identifier('a ۵', version=3)
        self.assertEqual(sanitized, 'A_۵')

    def test_umlaut_sanitized_py3(self):
        sanitized = sanitize_identifier('Festkörperphysik', version=3)
        self.assertEqual(sanitized, 'Festkörperphysik')

    def test_power_umlaut_sanitized_py3(self):
        sanitized = sanitize_identifier('^Festkörperphysik', version=3)
        self.assertEqual(sanitized, 'power_Festkörperphysik')

    def test_custom_dollar_removal_py2(self):
        sanitize_identifier.eliminations.extend(['dollar'])
        sanitized = sanitize_identifier('$E$', version=3)
        self.assertEqual(sanitized, 'E')
        sanitize_identifier.eliminations.remove('dollar')


class TestFindRange(unittest.TestCase):
    """
    Tests for find_range function.
    """

    def setUp(self):
        self.int_vals = [1, 5, 3, 9, 7, 121, 14]
        self.float_vals = [0.38, 0.121, -0.1424, 5.12]
        self.nan_floats = [np.NaN, 0.32, 1.42, -0.32]
        self.str_vals = ["Aardvark", "Zebra", "Platypus", "Wallaby"]

    def test_int_range(self):
        self.assertEqual(find_range(self.int_vals), (1, 121))

    def test_float_range(self):
        self.assertEqual(find_range(self.float_vals), (-0.1424, 5.12))

    def test_nan_range(self):
        self.assertEqual(find_range(self.nan_floats), (-0.32, 1.42))

    def test_str_range(self):
        self.assertEqual(find_range(self.str_vals), ("Aardvark",  "Zebra"))

    def test_soft_range(self):
        self.assertEqual(find_range(self.float_vals, soft_range=(np.NaN, 100)), (-0.1424, 100))


class TestMaxRange(unittest.TestCase):
    """
    Tests for max_range function.
    """

    def setUp(self):
        self.ranges1 = [(-0.2, 0.5), (0, 1), (-0.37, 1.02), (np.NaN, 0.3)]
        self.ranges2 = [(np.NaN, np.NaN), (np.NaN, np.NaN)]

    def test_max_range1(self):
        self.assertEqual(max_range(self.ranges1), (-0.37, 1.02))

    def test_max_range2(self):
        lower, upper = max_range(self.ranges2)
        self.assertTrue(math.isnan(lower))
        self.assertTrue(math.isnan(upper))
