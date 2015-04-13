# -*- coding: utf-8 -*-
"""
Unit tests of the helper functions in core.utils
"""
import sys
from unittest import SkipTest

from holoviews.core.util import sanitize_identifier, allowable
from holoviews.element.comparison import ComparisonTestCase

py_version = sys.version_info.major


class TestAllowablePy2(ComparisonTestCase):
    """
    Tests of allowable (Python 2)
    """
    def setUp(self):
        if py_version != 2: raise SkipTest

    def test_simple_ascii_allowable_py2(self):
        self.assertEqual(allowable('a', version=2), True)

    def test_simple_alpha_allowable_py2(self):
        self.assertEqual(allowable('α', version=2), True)

    def test_simple_number_allowable_py2(self):
        self.assertEqual(allowable('8', version=2), False)

    def test_simple_underscore_allowable_py2(self):
        self.assertEqual(allowable('_', version=2), False)

    def test_simple_space_allowable_py2(self):
        self.assertEqual(allowable(' ', version=2), False)

    def test_arabic_five_allowable_py2(self):
        "Testing arabic five allowed as it will be sanitized"
        self.assertEqual(allowable('٥', version=2), True)

class TestSanitizationPy2(ComparisonTestCase):
    """
    Tests of sanitize_identifier (Python 2)
    """

    def setUp(self):
        if py_version != 2: raise SkipTest

    def test_simple_dollar_sanitized_py2(self):
        sanitized = sanitize_identifier('$', version=2)
        self.assertEqual(sanitized, 'dollar')

    def test_simple_alpha_sanitized_py2(self):
        sanitized = sanitize_identifier('α', version=2)
        self.assertEqual(sanitized, 'alpha')

    def test_simple_a_dollar_sanitized_py2(self):
        sanitized = sanitize_identifier('a $', version=2)
        self.assertEqual(sanitized, 'A_dollar')

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
        self.assertEqual(sanitized, 'Festk_o_diaeresis_rperphysik')

    def test_power_umlaut_sanitized_py2(self):
        sanitized = sanitize_identifier('^Festkörperphysik', version=2)
        self.assertEqual(sanitized, 'power_Festk_o_diaeresis_rperphysik')



class TestAllowablePy3(ComparisonTestCase):
    """
    Tests of allowable (Python 3)
    """
    def setUp(self):
        if py_version != 3: raise SkipTest

    def test_simple_ascii_allowable_py3(self):
        self.assertEqual(allowable('a', version=3), True)

    def test_simple_alpha_allowable_py3(self):
        self.assertEqual(allowable('α', version=3), True)

    def test_simple_number_allowable_py3(self):
        self.assertEqual(allowable('8', version=3), False)

    def test_simple_underscore_allowable_py3(self):
        self.assertEqual(allowable('_', version=3), False)

    def test_simple_space_allowable_py3(self):
        self.assertEqual(allowable(' ', version=3), False)

    def test_arabic_five_allowable_py3(self):
        "Testing arabic five (digit category, cannot start identifier"
        self.assertEqual(allowable('٥', version=3), False)


class TestSanitizationPy3(ComparisonTestCase):
    """
    Tests of sanitize_identifier (Python 3)
    """
    def setUp(self):
        if py_version != 3: raise SkipTest

    def test_simple_dollar_sanitized_py3(self):
        sanitized = sanitize_identifier('$', version=3)
        self.assertEqual(sanitized, 'dollar')

    def test_simple_alpha_sanitized_py3(self):
        sanitized = sanitize_identifier('α', version=3)
        self.assertEqual(sanitized, 'α')

    def test_simple_a_dollar_sanitized_py3(self):
        sanitized = sanitize_identifier('a $', version=3)
        self.assertEqual(sanitized, 'A_dollar')

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
            sanitized = sanitize_identifier('٥', version=3)
        except SyntaxError as e:
            assert str(e).startswith("String '٥' cannot be sanitized")

    def test_urdu_five_sanitized_py3(self):
        try:
            sanitized = sanitize_identifier('۵', version=3)
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
