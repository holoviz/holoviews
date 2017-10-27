# -*- coding: utf-8 -*-
"""
Unit tests of the helper functions in core.utils
"""
import sys, math
import unittest
from unittest import SkipTest

import datetime
import numpy as np
from collections import OrderedDict
try:
    import pandas as pd
except:
    pd = None

from holoviews.core.util import (
    sanitize_identifier_fn, find_range, max_range, wrap_tuple_streams,
    deephash, merge_dimensions, get_path, make_path_unique, compute_density,
    date_range, dt_to_int
)
from holoviews import Dimension, Element
from holoviews.streams import PointerXY
from holoviews.element.comparison import ComparisonTestCase

py_version = sys.version_info.major

sanitize_identifier = sanitize_identifier_fn.instance()


class TestDeepHash(ComparisonTestCase):
    """
    Tests of deephash function used for memoization.
    """

    def test_deephash_list_equality(self):
        self.assertEqual(deephash([1,2,3]), deephash([1,2,3]))

    def test_deephash_list_inequality(self):
        obj1 = [1,2,3]
        obj2 = [1,2,3,4]
        self.assertNotEqual(deephash(obj1), deephash(obj2))

    def test_deephash_set_equality(self):
        self.assertEqual(deephash(set([1,2,3])), deephash(set([1,3,2])))

    def test_deephash_set_inequality(self):
        self.assertNotEqual(deephash(set([1,2,3])), deephash(set([1,3,4])))

    def test_deephash_dict_equality_v1(self):
        self.assertEqual(deephash({1:'a',2:'b'}), deephash({2:'b', 1:'a'}))

    def test_deephash_dict_equality_v2(self):
        self.assertNotEqual(deephash({1:'a',2:'b'}), deephash({2:'b', 1:'c'}))

    def test_deephash_odict_equality_v1(self):
        odict1 = OrderedDict([(1,'a'), (2,'b')])
        odict2 = OrderedDict([(1,'a'), (2,'b')])
        self.assertEqual(deephash(odict1), deephash(odict2))

    def test_deephash_odict_equality_v2(self):
        odict1 = OrderedDict([(1,'a'), (2,'b')])
        odict2 = OrderedDict([(1,'a'), (2,'c')])
        self.assertNotEqual(deephash(odict1), deephash(odict2))

    def test_deephash_numpy_equality(self):
        self.assertEqual(deephash(np.array([1,2,3])),
                         deephash(np.array([1,2,3])))

    def test_deephash_numpy_inequality(self):
        arr1 = np.array([1,2,3])
        arr2 = np.array([1,2,4])
        self.assertNotEqual(deephash(arr1), deephash(arr2))

    def test_deephash_dataframe_equality(self):
        if pd is None: raise SkipTest
        self.assertEqual(deephash(pd.DataFrame({'a':[1,2,3],'b':[4,5,6]})),
                         deephash(pd.DataFrame({'a':[1,2,3],'b':[4,5,6]})))

    def test_deephash_dataframe_inequality(self):
        if pd is None: raise SkipTest
        self.assertNotEqual(deephash(pd.DataFrame({'a':[1,2,3],'b':[4,5,6]})),
                            deephash(pd.DataFrame({'a':[1,2,3],'b':[4,5,8]})))

    def test_deephash_series_equality(self):
        if pd is None: raise SkipTest
        self.assertEqual(deephash(pd.Series([1,2,3])),
                         deephash(pd.Series([1,2,3])))

    def test_deephash_series_inequality(self):
        if pd is None: raise SkipTest
        self.assertNotEqual(deephash(pd.Series([1,2,3])),
                            deephash(pd.Series([1,2,7])))

    def test_deephash_datetime_equality(self):
        dt1 = datetime.datetime(1,2,3)
        dt2 = datetime.datetime(1,2,3)
        self.assertEqual(deephash(dt1), deephash(dt2))

    def test_deephash_datetime_inequality(self):
        dt1 = datetime.datetime(1,2,3)
        dt2 = datetime.datetime(1,2,5)
        self.assertNotEqual(deephash(dt1), deephash(dt2))

    def test_deephash_nested_native_equality(self):
        obj1 = [[1,2], (3,6,7, [True]), 'a', 9.2, 42, {1:3,2:'c'}]
        obj2 = [[1,2], (3,6,7, [True]), 'a', 9.2, 42, {1:3,2:'c'}]
        self.assertEqual(deephash(obj1), deephash(obj2))

    def test_deephash_nested_native_inequality(self):
        obj1 = [[1,2], (3,6,7, [False]), 'a', 9.2, 42, {1:3,2:'c'}]
        obj2 = [[1,2], (3,6,7, [True]), 'a', 9.2, 42, {1:3,2:'c'}]
        self.assertNotEqual(deephash(obj1), deephash(obj2))

    def test_deephash_nested_mixed_equality(self):
        if pd is None: raise SkipTest
        obj1 = [datetime.datetime(1,2,3), set([1,2,3]),
                pd.DataFrame({'a':[1,2],'b':[3,4]}),
                np.array([1,2,3]), {'a':'b', '1':True},
                OrderedDict([(1,'a'),(2,'b')]), np.int64(34)]
        obj2 = [datetime.datetime(1,2,3), set([1,2,3]),
                pd.DataFrame({'a':[1,2],'b':[3,4]}),
                np.array([1,2,3]), {'a':'b', '1':True},
                OrderedDict([(1,'a'),(2,'b')]), np.int64(34)]
        self.assertEqual(deephash(obj1), deephash(obj2))

    def test_deephash_nested_mixed_inequality(self):
        if pd is None: raise SkipTest
        obj1 = [datetime.datetime(1,2,3), set([1,2,3]),
                pd.DataFrame({'a':[1,2],'b':[3,4]}),
                np.array([1,2,3]), {'a':'b', '2':True},
                OrderedDict([(1,'a'),(2,'b')]), np.int64(34)]
        obj2 = [datetime.datetime(1,2,3), set([1,2,3]),
                pd.DataFrame({'a':[1,2],'b':[3,4]}),
                np.array([1,2,3]), {'a':'b', '1':True},
                OrderedDict([(1,'a'),(2,'b')]), np.int64(34)]
        self.assertNotEqual(deephash(obj1), deephash(obj2))


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



class TestWrapTupleStreams(unittest.TestCase):


    def test_no_streams(self):
        result = wrap_tuple_streams((1,2), [],[])
        self.assertEqual(result, (1,2))

    def test_no_streams_two_kdims(self):
        result = wrap_tuple_streams((1,2),
                                    [Dimension('x'), Dimension('y')],
                                    [])
        self.assertEqual(result, (1,2))

    def test_no_streams_none_value(self):
        result = wrap_tuple_streams((1,None),
                                    [Dimension('x'), Dimension('y')],
                                    [])
        self.assertEqual(result, (1,None))

    def test_no_streams_one_stream_substitution(self):
        result = wrap_tuple_streams((None,3),
                                    [Dimension('x'), Dimension('y')],
                                    [PointerXY(x=-5,y=10)])
        self.assertEqual(result, (-5,3))

    def test_no_streams_two_stream_substitution(self):
        result = wrap_tuple_streams((None,None),
                                    [Dimension('x'), Dimension('y')],
                                    [PointerXY(x=0,y=5)])
        self.assertEqual(result, (0,5))


class TestMergeDimensions(unittest.TestCase):

    def test_merge_dimensions(self):
        dimensions = merge_dimensions([[Dimension('A')], [Dimension('A'), Dimension('B')]])
        self.assertEqual(dimensions, [Dimension('A'), Dimension('B')])

    def test_merge_dimensions_with_values(self):
        dimensions = merge_dimensions([[Dimension('A', values=[0, 1])],
                                       [Dimension('A', values=[1, 2]), Dimension('B')]])
        self.assertEqual(dimensions, [Dimension('A'), Dimension('B')])
        self.assertEqual(dimensions[0].values, [0, 1, 2])


class TestTreePathUtils(unittest.TestCase):

    def test_get_path_with_label(self):
        path = get_path(Element('Test', label='A'))
        self.assertEqual(path, ('Element', 'A'))

    def test_get_path_without_label(self):
        path = get_path(Element('Test'))
        self.assertEqual(path, ('Element',))

    def test_get_path_with_custom_group(self):
        path = get_path(Element('Test', group='Custom Group'))
        self.assertEqual(path, ('Custom_Group',))

    def test_get_path_with_custom_group_and_label(self):
        path = get_path(Element('Test', group='Custom Group', label='A'))
        self.assertEqual(path, ('Custom_Group', 'A'))

    def test_get_path_from_item_with_custom_group(self):
        path = get_path((('Custom',), Element('Test')))
        self.assertEqual(path, ('Custom',))

    def test_get_path_from_item_with_custom_group_and_label(self):
        path = get_path((('Custom', 'Path'), Element('Test')))
        self.assertEqual(path, ('Custom',))

    def test_get_path_from_item_with_custom_group_and_matching_label(self):
        path = get_path((('Custom', 'Path'), Element('Test', label='Path')))
        self.assertEqual(path, ('Custom', 'Path'))

    def test_make_path_unique_no_clash(self):
        path = ('Element', 'A')
        new_path = make_path_unique(path, {}, True)
        self.assertEqual(new_path, path)

    def test_make_path_unique_clash_without_label(self):
        path = ('Element',)
        new_path = make_path_unique(path, {path: 1}, True)
        self.assertEqual(new_path, path+('I',))

    def test_make_path_unique_clash_with_label(self):
        path = ('Element', 'A')
        new_path = make_path_unique(path, {path: 1}, True)
        self.assertEqual(new_path, path+('I',))

    def test_make_path_unique_no_clash_old(self):
        path = ('Element', 'A')
        new_path = make_path_unique(path, {}, False)
        self.assertEqual(new_path, path)

    def test_make_path_unique_clash_without_label_old(self):
        path = ('Element',)
        new_path = make_path_unique(path, {path: 1}, False)
        self.assertEqual(new_path, path+('I',))

    def test_make_path_unique_clash_with_label_old(self):
        path = ('Element', 'A')
        new_path = make_path_unique(path, {path: 1}, False)
        self.assertEqual(new_path, path[:-1]+('I',))


class TestDatetimeUtils(unittest.TestCase):

    def test_compute_density_float(self):
        self.assertEqual(compute_density(0, 1, 10), 10)

    def test_compute_us_density_1s_datetime(self):
        start = np.datetime64(datetime.datetime.today())
        end = start+np.timedelta64(1, 's')
        self.assertEqual(compute_density(start, end, 10), 1e-5)

    def test_compute_us_density_10s_datetime(self):
        start = np.datetime64(datetime.datetime.today())
        end = start+np.timedelta64(10, 's')
        self.assertEqual(compute_density(start, end, 10), 1e-6)

    def test_compute_s_density_1s_datetime(self):
        start = np.datetime64(datetime.datetime.today())
        end = start+np.timedelta64(1, 's')
        self.assertEqual(compute_density(start, end, 10, 's'), 10)

    def test_compute_s_density_10s_datetime(self):
        start = np.datetime64(datetime.datetime.today())
        end = start+np.timedelta64(10, 's')
        self.assertEqual(compute_density(start, end, 10, 's'), 1)

    def test_datetime_to_us_int(self):
        dt = datetime.datetime(2017, 1, 1)
        self.assertEqual(dt_to_int(dt), 1483228800000000.0)

    def test_datetime64_to_us_int(self):
        dt = np.datetime64(datetime.datetime(2017, 1, 1))
        self.assertEqual(dt_to_int(dt), 1483228800000000.0)

    def test_timestamp_to_us_int(self):
        dt = pd.Timestamp(datetime.datetime(2017, 1, 1))
        self.assertEqual(dt_to_int(dt), 1483228800000000.0)
    
    def test_datetime_to_s_int(self):
        dt = datetime.datetime(2017, 1, 1)
        self.assertEqual(dt_to_int(dt, 's'), 1483228800.0)

    def test_datetime64_to_s_int(self):
        dt = np.datetime64(datetime.datetime(2017, 1, 1))
        self.assertEqual(dt_to_int(dt, 's'), 1483228800.0)

    def test_timestamp_to_s_int(self):
        dt = pd.Timestamp(datetime.datetime(2017, 1, 1))
        self.assertEqual(dt_to_int(dt, 's'), 1483228800.0)

    def test_date_range_1_hour(self):
        start = np.datetime64(datetime.datetime(2017, 1, 1))
        end = start+np.timedelta64(1, 'h')
        drange = date_range(start, end, 6)
        self.assertEqual(drange[0], start+np.timedelta64(5, 'm'))
        self.assertEqual(drange[-1], end-np.timedelta64(5, 'm'))

    def test_date_range_1_sec(self):
        start = np.datetime64(datetime.datetime(2017, 1, 1))
        end = start+np.timedelta64(1, 's')
        drange = date_range(start, end, 10)
        self.assertEqual(drange[0], start+np.timedelta64(50, 'ms'))
        self.assertEqual(drange[-1], end-np.timedelta64(50, 'ms'))
