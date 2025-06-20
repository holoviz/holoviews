"""
Unit tests of the helper functions in core.utils
"""
import datetime
import math
import unittest
from itertools import product

import numpy as np
import pandas as pd
import pytest

from holoviews import Dimension, Element
from holoviews.core.util import (
    closest_match,
    compute_density,
    compute_edges,
    cross_index,
    date_range,
    deephash,
    dimension_range,
    dt_to_int,
    find_range,
    get_path,
    is_nan,
    is_null_or_na_scalar,
    isfinite,
    make_path_unique,
    max_range,
    merge_dimensions,
    sanitize_identifier_fn,
    search_indices,
    tree_attribute,
    unique_array,
    wrap_tuple_streams,
)
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PointerXY

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
        self.assertEqual(deephash({1,2,3}), deephash({1,3,2}))

    def test_deephash_set_inequality(self):
        self.assertNotEqual(deephash({1,2,3}), deephash({1,3,4}))

    def test_deephash_dict_equality_v1(self):
        self.assertEqual(deephash({1:'a',2:'b'}), deephash({2:'b', 1:'a'}))

    def test_deephash_dict_equality_v2(self):
        self.assertNotEqual(deephash({1:'a',2:'b'}), deephash({2:'b', 1:'c'}))

    def test_deephash_odict_equality_v1(self):
        odict1 = dict([(1,'a'), (2,'b')])
        odict2 = dict([(1,'a'), (2,'b')])
        self.assertEqual(deephash(odict1), deephash(odict2))

    def test_deephash_odict_equality_v2(self):
        odict1 = dict([(1,'a'), (2,'b')])
        odict2 = dict([(1,'a'), (2,'c')])
        self.assertNotEqual(deephash(odict1), deephash(odict2))

    def test_deephash_numpy_equality(self):
        self.assertEqual(deephash(np.array([1,2,3])),
                         deephash(np.array([1,2,3])))

    def test_deephash_numpy_inequality(self):
        arr1 = np.array([1,2,3])
        arr2 = np.array([1,2,4])
        self.assertNotEqual(deephash(arr1), deephash(arr2))

    def test_deephash_dataframe_equality(self):
        self.assertEqual(deephash(pd.DataFrame({'a':[1,2,3],'b':[4,5,6]})),
                         deephash(pd.DataFrame({'a':[1,2,3],'b':[4,5,6]})))

    def test_deephash_dataframe_column_inequality(self):
        self.assertNotEqual(deephash(pd.DataFrame({'a':[1,2,3],'b':[4,5,6]})),
                            deephash(pd.DataFrame({'a':[1,2,3],'c':[4,5,6]})))

    def test_deephash_dataframe_index_inequality(self):
        self.assertNotEqual(deephash(pd.DataFrame({'a':[1,2,3],'b':[4,5,6]})),
                            deephash(pd.DataFrame({'a':[1,2,3],'b':[4,5,6]},
                                                  index=pd.Series([0, 1, 2], name='Index'))))

    def test_deephash_dataframe_inequality(self):
        self.assertNotEqual(deephash(pd.DataFrame({'a':[1,2,3],'b':[4,5,6]})),
                            deephash(pd.DataFrame({'a':[1,2,3],'b':[4,5,8]})))

    def test_deephash_series_equality(self):
        self.assertEqual(deephash(pd.Series([1,2,3])),
                         deephash(pd.Series([1,2,3])))

    def test_deephash_series_name_inequality(self):
        self.assertNotEqual(deephash(pd.Series([1,2,3], name='Foo')),
                            deephash(pd.Series([1,2,3], name='Bar')))

    def test_deephash_series_index_inequality(self):
        self.assertNotEqual(deephash(pd.Series([1,2,3], index=pd.Series([0, 1, 2], name='Index'))),
                            deephash(pd.Series([1,2,3], index=pd.Series([2, 1, 0], name='Index'))))


    def test_deephash_series_index_name_inequality(self):
        self.assertNotEqual(deephash(pd.Series([1,2,3], index=pd.Series([0, 1, 2], name='Foo'))),
                            deephash(pd.Series([1,2,3], index=pd.Series([0, 1, 2], name='Bar'))))

    def test_deephash_series_inequality(self):
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
        obj1 = [datetime.datetime(1,2,3), {1,2,3},
                pd.DataFrame({'a':[1,2],'b':[3,4]}),
                np.array([1,2,3]), {'a':'b', '1':True},
                dict([(1,'a'),(2,'b')]), np.int64(34)]
        obj2 = [datetime.datetime(1,2,3), {1,2,3},
                pd.DataFrame({'a':[1,2],'b':[3,4]}),
                np.array([1,2,3]), {'a':'b', '1':True},
                dict([(1,'a'),(2,'b')]), np.int64(34)]
        self.assertEqual(deephash(obj1), deephash(obj2))

    def test_deephash_nested_mixed_inequality(self):
        obj1 = [datetime.datetime(1,2,3), {1,2,3},
                pd.DataFrame({'a':[1,2],'b':[3,4]}),
                np.array([1,2,3]), {'a':'b', '2':True},
                dict([(1,'a'),(2,'b')]), np.int64(34)]
        obj2 = [datetime.datetime(1,2,3), {1,2,3},
                pd.DataFrame({'a':[1,2],'b':[3,4]}),
                np.array([1,2,3]), {'a':'b', '1':True},
                dict([(1,'a'),(2,'b')]), np.int64(34)]
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

    def test_prefix_test1(self):
        prefixed = sanitize_identifier.prefixed('_some_string')
        self.assertEqual(prefixed, True)

    def test_prefix_test2(self):
        prefixed = sanitize_identifier.prefixed('some_string')
        self.assertEqual(prefixed, False)

    def test_prefix_test3(self):
        prefixed = sanitize_identifier.prefixed('۵some_string')
        self.assertEqual(prefixed, True)


class TestTreeAttribute(ComparisonTestCase):

    def test_simple_lowercase_string(self):
        self.assertEqual(tree_attribute('lowercase'), False)

    def test_simple_uppercase_string(self):
        self.assertEqual(tree_attribute('UPPERCASE'), True)

    def test_underscore_string(self):
        self.assertEqual(tree_attribute('_underscore'), False)



class TestSanitization(ComparisonTestCase):
    """
    Tests of sanitize_identifier
    """
    def test_simple_pound_sanitized(self):
        sanitized = sanitize_identifier('£')
        self.assertEqual(sanitized, 'pound')

    def test_simple_digit_sanitized(self):
        sanitized = sanitize_identifier('0')
        self.assertEqual(sanitized, 'A_0')

    def test_simple_underscore_sanitized(self):
        sanitized = sanitize_identifier('_test')
        self.assertEqual(sanitized, 'A__test')

    def test_simple_alpha_sanitized(self):
        sanitized = sanitize_identifier('α')
        self.assertEqual(sanitized, 'α')

    def test_simple_a_pound_sanitized(self):
        sanitized = sanitize_identifier('a £')
        self.assertEqual(sanitized, 'A_pound')

    def test_capital_delta_sanitized(self):
        sanitized = sanitize_identifier('Δ')
        self.assertEqual(sanitized, 'Δ')

    def test_lowercase_delta_sanitized(self):
        sanitized = sanitize_identifier('δ')
        self.assertEqual(sanitized, 'δ')

    def test_simple_alpha_beta_sanitized(self):
        sanitized = sanitize_identifier('α β')
        self.assertEqual(sanitized, 'α_β')

    def test_simple_alpha_beta_underscore_sanitized(self):
        sanitized = sanitize_identifier('α_β')
        self.assertEqual(sanitized, 'α_β')

    def test_simple_alpha_beta_double_underscore_sanitized(self):
        sanitized = sanitize_identifier('α__β')
        self.assertEqual(sanitized, 'α__β')

    def test_simple_alpha_beta_mixed_underscore_space_sanitized(self):
        sanitized = sanitize_identifier('α__  β')
        self.assertEqual(sanitized, 'α__β')

    def test_alpha_times_two(self):
        sanitized = sanitize_identifier('α*2')
        self.assertEqual(sanitized,  'α_times_2')

    def test_arabic_five_sanitized(self):
        """
        Note: There would be a clash if you mixed the languages of
        your digits! E.g. arabic ٥ five and urdu ۵ five
        """
        try:
            sanitize_identifier('٥')
        except SyntaxError as e:
            assert str(e).startswith("String '٥' cannot be sanitized")

    def test_urdu_five_sanitized(self):
        try:
            sanitize_identifier('۵')
        except SyntaxError as e:
            assert str(e).startswith("String '۵' cannot be sanitized")

    def test_urdu_a_five_sanitized(self):
        """
        Note: There would be a clash if you mixed the languages of
        your digits! E.g. arabic ٥ five and urdu ۵ five
        """
        sanitized = sanitize_identifier('a ۵')
        self.assertEqual(sanitized, 'A_۵')

    def test_umlaut_sanitized(self):
        sanitized = sanitize_identifier('Festkörperphysik')
        self.assertEqual(sanitized, 'Festkörperphysik')

    def test_power_umlaut_sanitized(self):
        sanitized = sanitize_identifier('^Festkörperphysik')
        self.assertEqual(sanitized, 'power_Festkörperphysik')

    def test_custom_dollar_removal_py2(self):
        sanitize_identifier.eliminations.extend(['dollar'])
        sanitized = sanitize_identifier('$E$')
        self.assertEqual(sanitized, 'E')
        sanitize_identifier.eliminations.remove('dollar')


class TestFindRange(unittest.TestCase):
    """
    Tests for find_range function.
    """

    def setUp(self):
        self.int_vals = [1, 5, 3, 9, 7, 121, 14]
        self.float_vals = [0.38, 0.121, -0.1424, 5.12]
        self.nan_floats = [np.nan, 0.32, 1.42, -0.32]
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
        self.assertEqual(find_range(self.float_vals, soft_range=(np.nan, 100)), (-0.1424, 100))


class TestDimensionRange(unittest.TestCase):
    """
    Tests for dimension_range function.
    """

    def setUp(self):
        self.date_range = (np.datetime64(datetime.datetime(2017, 1, 1)),
                           np.datetime64(datetime.datetime(2017, 1, 2)))
        self.date_range2 = (np.datetime64(datetime.datetime(2016, 12, 31)),
                            np.datetime64(datetime.datetime(2017, 1, 3)))

    def test_dimension_range_date_hard_range(self):
        drange = dimension_range(self.date_range2[0], self.date_range2[1],
                                 self.date_range, (None, None))
        self.assertEqual(drange, self.date_range)

    def test_dimension_range_date_soft_range(self):
        drange = dimension_range(self.date_range[0], self.date_range[1],
                                 (None, None), self.date_range2)
        self.assertEqual(drange, self.date_range2)


class TestMaxRange(unittest.TestCase):
    """
    Tests for max_range function.
    """

    def setUp(self):
        self.ranges1 = [(-0.2, 0.5), (0, 1), (-0.37, 1.02), (np.nan, 0.3)]
        self.ranges2 = [(np.nan, np.nan), (np.nan, np.nan)]

    def test_max_range1(self):
        self.assertEqual(max_range(self.ranges1), (-0.37, 1.02))

    def test_max_range2(self):
        lower, upper = max_range(self.ranges2)
        self.assertTrue(math.isnan(lower))
        self.assertTrue(math.isnan(upper))

    def test_max_range3(self):
        periods = [(pd.Period("1990", freq="M"), pd.Period("1991", freq="M"))]
        expected = (np.datetime64("1990", 'ns'), np.datetime64("1991", 'ns'))
        self.assertEqual(max_range(periods), expected)


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
        self.assertEqual(new_path, (*path, 'I'))

    def test_make_path_unique_clash_with_label(self):
        path = ('Element', 'A')
        new_path = make_path_unique(path, {path: 1}, True)
        self.assertEqual(new_path, (*path, 'I'))

    def test_make_path_unique_no_clash_old(self):
        path = ('Element', 'A')
        new_path = make_path_unique(path, {}, False)
        self.assertEqual(new_path, path)

    def test_make_path_unique_clash_without_label_old(self):
        path = ('Element',)
        new_path = make_path_unique(path, {path: 1}, False)
        self.assertEqual(new_path, (*path, 'I'))

    def test_make_path_unique_clash_with_label_old(self):
        path = ('Element', 'A')
        new_path = make_path_unique(path, {path: 1}, False)
        self.assertEqual(new_path, (*path[:-1], 'I'))


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

    def test_datetime64_s_to_ns_int(self):
        dt = np.datetime64(datetime.datetime(2017, 1, 1), 's')
        self.assertEqual(dt_to_int(dt, 'ns'), 1483228800000000000.0)

    def test_datetime64_us_to_ns_int(self):
        dt = np.datetime64(datetime.datetime(2017, 1, 1), 'us')
        self.assertEqual(dt_to_int(dt, 'ns'), 1483228800000000000.0)

    def test_datetime64_to_ns_int(self):
        dt = np.datetime64(datetime.datetime(2017, 1, 1))
        self.assertEqual(dt_to_int(dt, 'ns'), 1483228800000000000.0)

    def test_datetime64_us_to_us_int(self):
        dt = np.datetime64(datetime.datetime(2017, 1, 1), 'us')
        self.assertEqual(dt_to_int(dt), 1483228800000000.0)

    def test_datetime64_s_to_us_int(self):
        dt = np.datetime64(datetime.datetime(2017, 1, 1), 's')
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

    def test_datetime64_us_to_s_int(self):
        dt = np.datetime64(datetime.datetime(2017, 1, 1), 'us')
        self.assertEqual(dt_to_int(dt, 's'), 1483228800.0)

    def test_datetime64_s_to_s_int(self):
        dt = np.datetime64(datetime.datetime(2017, 1, 1), 's')
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

    def test_timezone_to_int(self):
        import pytz
        timezone = pytz.timezone("Europe/Copenhagen")

        values = [
            datetime.datetime(2021, 4, 8, 12, 0, 0, 0),
            datetime.datetime(2021, 4, 8, 12, 0, 0, 0, datetime.timezone.utc),
            datetime.datetime(2021, 4, 8, 12, 0, 0, 0, timezone),
            datetime.date(2021, 4, 8),
            np.datetime64(datetime.datetime(2021, 4, 8, 12, 0, 0, 0)),
        ]

        for value in values:
            x1 = dt_to_int(value)
            x2 = dt_to_int(pd.to_datetime(value))
            self.assertEqual(x1, x2)

class TestNumericUtilities(ComparisonTestCase):

    def test_isfinite_none(self):
        self.assertFalse(isfinite(None))

    def test_isfinite_nan(self):
        self.assertFalse(isfinite(float('NaN')))

    def test_isfinite_inf(self):
        self.assertFalse(isfinite(float('inf')))

    def test_isfinite_float(self):
        self.assertTrue(isfinite(1.2))

    def test_isfinite_float_array_nan(self):
        array = np.array([1.2, 3.0, np.nan])
        self.assertEqual(isfinite(array), np.array([True, True, False]))

    def test_isfinite_float_array_inf(self):
        array = np.array([1.2, 3.0, np.inf])
        self.assertEqual(isfinite(array), np.array([True, True, False]))

    def test_isfinite_datetime(self):
        dt = datetime.datetime(2017, 1, 1)
        self.assertTrue(isfinite(dt))

    def test_isfinite_datetime64(self):
        dt64 = np.datetime64(datetime.datetime(2017, 1, 1))
        self.assertTrue(isfinite(dt64))

    def test_isfinite_datetime64_nat(self):
        dt64 = np.datetime64('NaT')
        self.assertFalse(isfinite(dt64))

    def test_isfinite_timedelta64_nat(self):
        dt64 = np.timedelta64('NaT')
        self.assertFalse(isfinite(dt64))

    def test_isfinite_pandas_timestamp_nat(self):
        dt64 = pd.Timestamp('NaT')
        self.assertFalse(isfinite(dt64))

    def test_isfinite_pandas_period_nat(self):
        dt64 = pd.Period('NaT')
        self.assertFalse(isfinite(dt64))

    def test_isfinite_pandas_period_index(self):
        daily = pd.date_range('2017-1-1', '2017-1-3', freq='D').to_period('D')
        self.assertEqual(isfinite(daily), np.array([True, True, True]))

    def test_isfinite_pandas_period_series(self):
        daily = pd.date_range('2017-1-1', '2017-1-3', freq='D').to_period('D').to_series()
        self.assertEqual(isfinite(daily), np.array([True, True, True]))

    def test_isfinite_pandas_period_index_nat(self):
        daily = pd.date_range('2017-1-1', '2017-1-3', freq='D').to_period('D')
        daily = pd.PeriodIndex([*daily, pd.NaT])
        self.assertEqual(isfinite(daily), np.array([True, True, True, False]))

    def test_isfinite_pandas_period_series_nat(self):
        daily = pd.date_range('2017-1-1', '2017-1-3', freq='D').to_period('D')
        daily = pd.Series([*daily, pd.NaT])
        self.assertEqual(isfinite(daily), np.array([True, True, True, False]))

    def test_isfinite_pandas_timestamp_index(self):
        daily = pd.date_range('2017-1-1', '2017-1-3', freq='D')
        self.assertEqual(isfinite(daily), np.array([True, True, True]))

    def test_isfinite_pandas_timestamp_series(self):
        daily = pd.date_range('2017-1-1', '2017-1-3', freq='D').to_series()
        self.assertEqual(isfinite(daily), np.array([True, True, True]))

    def test_isfinite_pandas_timestamp_index_nat(self):
        daily = pd.date_range('2017-1-1', '2017-1-3', freq='D')
        daily = pd.DatetimeIndex([*daily, pd.NaT])
        self.assertEqual(isfinite(daily), np.array([True, True, True, False]))

    def test_isfinite_pandas_timestamp_series_nat(self):
        daily = pd.date_range('2017-1-1', '2017-1-3', freq='D')
        daily = pd.Series([*daily, pd.NaT])
        self.assertEqual(isfinite(daily), np.array([True, True, True, False]))

    def test_isfinite_datetime64_array(self):
        dt64 = np.array([np.datetime64(datetime.datetime(2017, 1, i)) for i in range(1, 4)])
        self.assertEqual(isfinite(dt64), np.array([True, True, True]))

    def test_isfinite_datetime64_array_with_nat(self):
        dts = [np.datetime64(datetime.datetime(2017, 1, i)) for i in range(1, 4)]
        dt64 = np.array([*dts, np.datetime64('NaT')])
        self.assertEqual(isfinite(dt64), np.array([True, True, True, False]))



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
        self.assertEqual(compute_edges(self.array3),
                         np.array([0.5, 1.5, 3.0, 5.0]))


class TestCrossIndex(ComparisonTestCase):

    def setUp(self):
        self.values1 = ['A', 'B', 'C']
        self.values2 = [1, 2, 3, 4]
        self.values3 = ['?', '!']
        self.values4 = ['x']

    def test_cross_index_full_product(self):
        values = [self.values1, self.values2, self.values3, self.values4]
        cross_product = list(product(*values))
        for i, p in enumerate(cross_product):
            self.assertEqual(cross_index(values, i), p)

    def test_cross_index_depth_1(self):
        values = [self.values1]
        cross_product = list(product(*values))
        for i, p in enumerate(cross_product):
            self.assertEqual(cross_index(values, i), p)

    def test_cross_index_depth_2(self):
        values = [self.values1, self.values2]
        cross_product = list(product(*values))
        for i, p in enumerate(cross_product):
            self.assertEqual(cross_index(values, i), p)

    def test_cross_index_depth_3(self):
        values = [self.values1, self.values2, self.values3]
        cross_product = list(product(*values))
        for i, p in enumerate(cross_product):
            self.assertEqual(cross_index(values, i), p)

    def test_cross_index_large(self):
        values = [[chr(65+i) for i in range(26)], list(range(500)),
                  [chr(97+i) for i in range(26)], [chr(48+i) for i in range(10)]]
        self.assertEqual(cross_index(values, 50001), ('A', 192, 'i', '1'))
        self.assertEqual(cross_index(values, 500001), ('D', 423, 'c', '1'))


class TestClosestMatch(ComparisonTestCase):

    def test_complete_match_overlay(self):
        specs = [(0, ('Curve', 'Curve', 'I')), (1, ('Points', 'Points', 'I'))]
        spec = ('Curve', 'Curve', 'I')
        self.assertEqual(closest_match(spec, specs), 0)
        spec = ('Points', 'Curve', 'I')
        self.assertEqual(closest_match(spec, specs), 1)

    def test_partial_match_overlay(self):
        specs = [(0, ('Curve', 'Curve', 'I')), (1, ('Points', 'Points', 'I'))]
        spec = ('Curve', 'Curve')
        self.assertEqual(closest_match(spec, specs), 0)
        spec = ('Points', 'Points')
        self.assertEqual(closest_match(spec, specs), 1)

    def test_partial_mismatch_overlay(self):
        specs = [(0, ('Curve', 'Curve', 'I')), (1, ('Points', 'Points', 'I'))]
        spec = ('Curve', 'Foo', 'II')
        self.assertEqual(closest_match(spec, specs), 0)
        spec = ('Points', 'Bar', 'III')
        self.assertEqual(closest_match(spec, specs), 1)

    def test_no_match_overlay(self):
        specs = [(0, ('Curve', 'Curve', 'I')), (1, ('Points', 'Points', 'I'))]
        spec = ('Scatter', 'Points', 'II')
        self.assertEqual(closest_match(spec, specs), None)
        spec = ('Path', 'Curve', 'III')
        self.assertEqual(closest_match(spec, specs), None)

    def test_complete_match_ndoverlay(self):
        spec = ('Points', 'Points', '', 1)
        specs = [(0, ('Points', 'Points', '', 0)), (1, ('Points', 'Points', '', 1)),
                 (2, ('Points', 'Points', '', 2))]
        self.assertEqual(closest_match(spec, specs), 1)
        spec = ('Points', 'Points', '', 2)
        self.assertEqual(closest_match(spec, specs), 2)

    def test_partial_match_ndoverlay(self):
        spec = ('Points', 'Points', '', 5)
        specs = [(0, ('Points', 'Points', '', 0)), (1, ('Points', 'Points', '', 1)),
                 (2, ('Points', 'Points', '', 2))]
        self.assertEqual(closest_match(spec, specs), 2)
        spec = ('Points', 'Points', 'Bar', 5)
        self.assertEqual(closest_match(spec, specs), 0)
        spec = ('Points', 'Foo', 'Bar', 5)
        self.assertEqual(closest_match(spec, specs), 0)

    def test_no_match_ndoverlay(self):
        specs = [(0, ('Points', 'Points', '', 0)), (1, ('Points', 'Points', '', 1)),
                 (2, ('Points', 'Points', '', 2))]
        spec = ('Scatter', 'Points', '', 5)
        self.assertEqual(closest_match(spec, specs), None)
        spec = ('Scatter', 'Bar', 'Foo', 5)
        self.assertEqual(closest_match(spec, specs), None)
        spec = ('Scatter', 'Foo', 'Bar', 5)
        self.assertEqual(closest_match(spec, specs), None)


def test_seach_indices_dtype_object():
    values = np.array(["c0", "c0", np.nan], dtype=object)
    source = np.array(["c0", np.nan], dtype=object)
    search_indices(values, source)


def test_unique_array_categorial():
    ser = pd.Series(np.random.choice(["a", "b", "c"], 100)).astype("category")
    res = unique_array([ser])
    assert sorted(res) == ["a", "b", "c"]


def test_is_nan():
    assert is_nan(np.nan) == True
    assert is_nan(None) == True
    assert is_nan(pd.NA) == True
    assert is_nan(pd.NaT) == True
    assert is_nan([1, 1]) == False
    assert is_nan([np.nan]) == True
    assert is_nan([np.nan, np.nan]) == False


def test_is_null_or_na_scalar():
    assert is_null_or_na_scalar(np.nan)
    assert is_null_or_na_scalar(pd.NA)
    assert is_null_or_na_scalar(pd.NaT)
    assert is_null_or_na_scalar(None)
    assert is_null_or_na_scalar(np.datetime64("NAT"))

    assert not is_null_or_na_scalar(datetime.datetime.today())
    assert not is_null_or_na_scalar(pd.Timestamp.now())
    assert not is_null_or_na_scalar("AAAA")
    assert not is_null_or_na_scalar(...)
    assert not is_null_or_na_scalar([1, 2])
    assert not is_null_or_na_scalar((1, 2))
    assert not is_null_or_na_scalar({1, 2})
    assert not is_null_or_na_scalar({"a": 1, "b": 2})
    assert not is_null_or_na_scalar(slice(None))
    assert not is_null_or_na_scalar(np.array([1, 2]))
    assert not is_null_or_na_scalar(pd.DataFrame([1, 2]))


def test_is_null_or_na_scalar_polars():
    pl = pytest.importorskip("polars")
    assert is_null_or_na_scalar(pl.Null)
    assert not is_null_or_na_scalar(pl.DataFrame([1, 2]))
    assert not is_null_or_na_scalar(pl.LazyFrame([1, 2]))
