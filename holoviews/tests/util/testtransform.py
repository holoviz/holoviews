# -*- coding: utf-8 -*-
"""
Unit tests for dim transforms
"""
from __future__ import division

import numpy as np
import pandas as pd

try:
    import dask.dataframe as dd
    import dask.array as da
except:
    da, dd = None, None

from holoviews.core.data import Dataset
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim


class TestDimTransforms(ComparisonTestCase):

    def setUp(self):
        self.linear_ints = pd.Series(np.arange(1, 11))
        self.linear_floats = pd.Series(np.arange(1, 11)/10.)
        self.negative = pd.Series(-self.linear_floats)
        self.repeating = pd.Series(
            ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A']
        )
        self.booleans = self.repeating == 'A'
        self.dataset = Dataset(
            (self.linear_ints, self.linear_floats,
             self.negative, self.repeating, self.booleans),
            ['int', 'float', 'negative', 'categories', 'booleans']
        )

        if dd is None:
            return

        ddf = dd.from_pandas(self.dataset.data, npartitions=2)
        self.dataset_dask = self.dataset.clone(data=ddf)

    # Assertion helpers

    def check_apply(self, expr, expected, skip_dask=False):
        if np.isscalar(expected):
            # Pandas input
            self.assertEqual(
                expr.apply(self.dataset, keep_index=False), expected
            )
            self.assertEqual(
                expr.apply(self.dataset, keep_index=True), expected
            )

            if dd is None:
                return

            # Dask input
            self.assertEqual(
                expr.apply(self.dataset_dask, keep_index=False), expected
            )
            self.assertEqual(
                expr.apply(self.dataset_dask, keep_index=True), expected
            )
            return

        # Make sure expected is a pandas Series
        self.assertIsInstance(expected, pd.Series)

        # Check using dataset backed by pandas DataFrame
        # keep_index=False
        np.testing.assert_equal(
            expr.apply(self.dataset),
            expected.values
        )
        # keep_index=True
        pd.testing.assert_series_equal(
            expr.apply(self.dataset, keep_index=True),
            expected,
            check_names=False
        )

        if skip_dask or dd is None:
            return

        # Check using dataset backed by Dask DataFrame
        expected_dask = dd.from_pandas(expected, npartitions=2)

        # keep_index=False, compute=False
        da.assert_eq(
            expr.apply(self.dataset_dask, compute=False), expected_dask.values
        )
        # keep_index=True, compute=False
        dd.assert_eq(
            expr.apply(self.dataset_dask, keep_index=True, compute=False),
            expected_dask,
            check_names=False
        )
        # keep_index=False, compute=True
        np.testing.assert_equal(
            expr.apply(self.dataset_dask, compute=True),
            expected_dask.values.compute()
        )
        # keep_index=True, compute=True
        pd.testing.assert_series_equal(
            expr.apply(self.dataset_dask, keep_index=True, compute=True),
            expected_dask.compute(),
            check_names=False
        )

    # Unary operators

    def test_abs_transform(self):
        expr = abs(dim('negative'))
        self.check_apply(expr, self.linear_floats)

    def test_neg_transform(self):
        expr = -dim('negative')
        self.check_apply(expr, self.linear_floats)

    def test_inv_transform(self):
        expr = ~dim('booleans')
        self.check_apply(expr, ~self.booleans)

    # Binary operators

    def test_add_transform(self):
        expr = dim('float') + 1
        self.check_apply(expr, self.linear_floats+1)

    def test_div_transform(self):
        expr = dim('int') / 10.
        self.check_apply(expr, self.linear_floats)

    def test_floor_div_transform(self):
        expr = dim('int') // 2
        self.check_apply(expr, self.linear_ints//2)

    def test_mod_transform(self):
        expr = dim('int') % 2
        self.check_apply(expr, self.linear_ints % 2)

    def test_mul_transform(self):
        expr = dim('float') * 10.
        self.check_apply(expr, self.linear_ints.astype('float64'))

    def test_pow_transform(self):
        expr = dim('int') ** 2
        self.check_apply(expr, self.linear_ints ** 2)

    def test_sub_transform(self):
        expr = dim('int') - 10
        self.check_apply(expr, self.linear_ints - 10)

    # Reverse binary operators

    def test_radd_transform(self):
        expr = 1 + dim('float')
        self.check_apply(expr, 1 + self.linear_floats)

    def test_rdiv_transform(self):
        expr = 10. / dim('int')
        self.check_apply(expr, 10. / self.linear_ints)

    def test_rfloor_div_transform(self):
        expr = 2 // dim('int')
        self.check_apply(expr, 2 // self.linear_ints)

    def test_rmod_transform(self):
        expr = 2 % dim('int')
        self.check_apply(expr, 2 % self.linear_ints)

    def test_rmul_transform(self):
        expr = 10. * dim('float')
        self.check_apply(expr, self.linear_ints.astype('float64'))

    def test_rsub_transform(self):
        expr = 10 - dim('int')
        self.check_apply(expr, 10 - self.linear_ints)

    # NumPy operations

    def test_ufunc_transform(self):
        expr = np.sin(dim('float'))
        self.check_apply(expr, np.sin(self.linear_floats))

    def test_astype_transform(self):
        expr = dim('int').astype('float64')
        self.check_apply(expr, self.linear_ints.astype('float64'))

    def test_cumsum_transform(self):
        expr = dim('float').cumsum()
        self.check_apply(expr, self.linear_floats.cumsum())

    def test_max_transform(self):
        expr = dim('float').max()
        self.check_apply(expr, self.linear_floats.max())

    def test_min_transform(self):
        expr = dim('float').min()
        self.check_apply(expr, self.linear_floats.min())

    def test_round_transform(self):
        expr = dim('float').round()
        self.check_apply(expr, self.linear_floats.round())

    def test_sum_transform(self):
        expr = dim('float').sum()
        self.check_apply(expr, self.linear_floats.sum())

    def test_std_transform(self):
        expr = dim('float').std(ddof=0)
        self.check_apply(expr, self.linear_floats.std(ddof=0))

    def test_var_transform(self):
        expr = dim('float').var(ddof=0)
        self.check_apply(expr, self.linear_floats.var(ddof=0))

    def test_log_transform(self):
        expr = dim('float').log()
        self.check_apply(expr, np.log(self.linear_floats))

    def test_log10_transform(self):
        expr = dim('float').log10()
        self.check_apply(expr, np.log10(self.linear_floats))

    # Custom functions

    def test_norm_transform(self):
        expr = dim('int').norm()
        self.check_apply(expr, (self.linear_ints-1)/9.)

    def test_iloc_transform_int(self):
        expr = dim('int').iloc[1]
        self.check_apply(expr, self.linear_ints[1])

    def test_iloc_transform_slice(self):
        expr = dim('int').iloc[1:3]
        self.check_apply(expr, self.linear_ints[1:3], skip_dask=True)

    def test_iloc_transform_list(self):
        expr = dim('int').iloc[[1, 3, 5]]
        self.check_apply(expr, self.linear_ints[[1, 3, 5]], skip_dask=True)

    def test_bin_transform(self):
        expr = dim('int').bin([0, 5, 10])
        expected = pd.Series(
            [2.5, 2.5, 2.5, 2.5, 2.5, 7.5, 7.5, 7.5, 7.5, 7.5]
        )
        self.check_apply(expr, expected)

    def test_bin_transform_with_labels(self):
        expr = dim('int').bin([0, 5, 10], ['A', 'B'])
        expected = pd.Series(
            ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
        )
        self.check_apply(expr, expected)

    def test_categorize_transform_list(self):
        expr = dim('categories').categorize(['circle', 'square', 'triangle'])
        expected = pd.Series(
            (['circle', 'square', 'triangle']*3)+['circle']
        )
        # We skip dask because results will depend on partition structure
        self.check_apply(expr, expected, skip_dask=True)

    def test_categorize_transform_dict(self):
        expr = dim('categories').categorize(
            {'A': 'circle', 'B': 'square', 'C': 'triangle'}
        )
        expected = pd.Series(
            (['circle', 'square', 'triangle'] * 3) + ['circle']
        )
        # We don't skip dask because results are now stable across partitions
        self.check_apply(expr, expected)

    def test_categorize_transform_dict_with_default(self):
        expr = dim('categories').categorize(
            {'A': 'circle', 'B': 'square'}, default='triangle'
        )
        expected = pd.Series(
            (['circle', 'square', 'triangle'] * 3) + ['circle']
        )
        # We don't skip dask because results are stable across partitions
        self.check_apply(expr, expected)

    # Numpy functions

    def test_digitize(self):
        expr = dim('int').digitize([1, 5, 10])
        expected = pd.Series(np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 3])).astype('int64')
        self.check_apply(expr, expected)

    def test_isin(self):
        expr = dim('int').digitize([1, 5, 10]).isin([1, 3])
        expected = pd.Series(
            np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1], dtype='bool')
        )
        self.check_apply(expr, expected)

    # Complex expressions

    def test_multi_operator_expression(self):
        expr = (((dim('float')-2)*3)**2)
        self.check_apply(expr, ((self.linear_floats-2)*3)**2)

    def test_multi_dim_expression(self):
        expr = dim('int')-dim('float')
        self.check_apply(expr, self.linear_ints-self.linear_floats)

    # Repr method

    def test_dim_repr(self):
        self.assertEqual(repr(dim('float')), "dim('float')")

    def test_unary_op_repr(self):
        self.assertEqual(repr(-dim('float')), "-dim('float')")

    def test_binary_op_repr(self):
        self.assertEqual(repr(dim('float')*2), "dim('float')*2")

    def test_reverse_binary_op_repr(self):
        self.assertEqual(repr(1+dim('float')), "1+dim('float')")

    def test_ufunc_expression_repr(self):
        self.assertEqual(repr(np.log(dim('float'))), "dim('float').log()")

    def test_custom_func_repr(self):
        self.assertEqual(repr(dim('float').norm()), "dim('float').norm()")

    def test_multi_operator_expression_repr(self):
        self.assertEqual(repr(((dim('float')-2)*3)**2),
                         "((dim('float')-2)*3)**2")

    # Applies method

    def test_multi_dim_expression_applies(self):
        self.assertEqual((dim('int')-dim('float')).applies(self.dataset),
                         True)

    def test_multi_dim_expression_not_applies(self):
        self.assertEqual((dim('foo')-dim('bar')).applies(self.dataset),
                         False)

    def test_multi_dim_expression_partial_applies(self):
        self.assertEqual((dim('int')-dim('bar')).applies(self.dataset),
                         False)
