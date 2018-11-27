# -*- coding: utf-8 -*-
"""
Unit tests for dim transforms
"""
from __future__ import division

import numpy as np

from holoviews.core.data import Dataset
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim


class TestDimTransforms(ComparisonTestCase):

    def setUp(self):
        self.linear_ints = np.arange(1, 11)
        self.linear_floats = np.arange(1, 11)/10.
        self.negative = -self.linear_floats
        self.repeating = ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A']
        self.dataset = Dataset(
            (self.linear_ints, self.linear_floats, self.negative, self.repeating),
            ['int', 'float', 'negative', 'categories']
        )

    # Unary operators

    def test_abs_transform(self):
        self.assertEqual(abs(dim('negative')).apply(self.dataset), self.linear_floats)

    def test_neg_transform(self):
        self.assertEqual(-dim('negative').apply(self.dataset), self.linear_floats)

    # Binary operators

    def test_add_transform(self):
        self.assertEqual((dim('float')+1).apply(self.dataset), self.linear_floats+1)

    def test_div_transform(self):
        self.assertEqual((dim('int')/10.).apply(self.dataset), self.linear_floats)

    def test_floor_div_transform(self):
        self.assertEqual((dim('int')//2).apply(self.dataset), self.linear_ints//2)

    def test_mod_transform(self):
        self.assertEqual((dim('int')%2).apply(self.dataset), self.linear_ints%2)

    def test_mul_transform(self):
        self.assertEqual((dim('float')*10.).apply(self.dataset), self.linear_ints)

    def test_pow_transform(self):
        self.assertEqual((dim('int')**2).apply(self.dataset), self.linear_ints**2)

    def test_sub_transform(self):
        self.assertEqual((dim('int')-10).apply(self.dataset), self.linear_ints-10)

    # Reverse binary operators

    def test_radd_transform(self):
        self.assertEqual((1+dim('float')).apply(self.dataset), 1+self.linear_floats)

    def test_rdiv_transform(self):
        self.assertEqual((10./dim('int')).apply(self.dataset), 10./self.linear_ints)

    def test_rfloor_div_transform(self):
        self.assertEqual((2//dim('int')).apply(self.dataset), 2//self.linear_ints)

    def test_rmod_transform(self):
        self.assertEqual((2%dim('int')).apply(self.dataset), 2%self.linear_ints)

    def test_rmul_transform(self):
        self.assertEqual((10.*dim('float')).apply(self.dataset), self.linear_ints)

    def test_rsub_transform(self):
        self.assertEqual((10-dim('int')).apply(self.dataset), 10-self.linear_ints)

    # NumPy operations

    def test_ufunc_transform(self):
        self.assertEqual(np.sin(dim('float')).apply(self.dataset), np.sin(self.linear_floats))

    def test_astype_transform(self):
        self.assertEqual(dim('int').astype(str).apply(self.dataset),
                         self.linear_ints.astype(str))

    def test_cumsum_transform(self):
        self.assertEqual(dim('float').cumsum().apply(self.dataset),
                         self.linear_floats.cumsum())

    def test_max_transform(self):
        self.assertEqual(dim('float').max().apply(self.dataset),
                         self.linear_floats.max())

    def test_min_transform(self):
        self.assertEqual(dim('float').min().apply(self.dataset),
                         self.linear_floats.min())
    
    def test_round_transform(self):
        self.assertEqual(dim('float').round().apply(self.dataset),
                         self.linear_floats.round())

    def test_sum_transform(self):
        self.assertEqual(dim('float').sum().apply(self.dataset),
                         self.linear_floats.sum())

    def test_std_transform(self):
        self.assertEqual(dim('float').std().apply(self.dataset),
                         self.linear_floats.std())
        
    def test_var_transform(self):
        self.assertEqual(dim('float').var().apply(self.dataset),
                         self.linear_floats.var())

    # Custom functions

    def test_norm_transform(self):
        self.assertEqual(dim('int').norm().apply(self.dataset),
                         (self.linear_ints-1)/9.)

    def test_bin_transform(self):
        self.assertEqual(dim('int').bin([0, 5, 10]).apply(self.dataset),
                         np.array([2.5, 2.5, 2.5, 2.5, 2.5, 7.5, 7.5, 7.5, 7.5, 7.5]))

    def test_bin_transform_with_labels(self):
        self.assertEqual(dim('int').bin([0, 5, 10], ['A', 'B']).apply(self.dataset),
                         np.array(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']))

    def test_categorize_transform_list(self):
        self.assertEqual(dim('categories').categorize(['circle', 'square', 'triangle']).apply(self.dataset),
                         np.array((['circle', 'square', 'triangle']*3)+['circle']))

    def test_categorize_transform_dict(self):
        self.assertEqual(dim('categories').categorize({'A': 'circle', 'B': 'square', 'C': 'triangle'}).apply(self.dataset),
                         np.array((['circle', 'square', 'triangle']*3)+['circle']))

    def test_categorize_transform_dict_with_default(self):
        self.assertEqual(dim('categories').categorize({'A': 'circle', 'B': 'square'}, default='triangle').apply(self.dataset),
                         np.array((['circle', 'square', 'triangle']*3)+['circle']))

    # Complex expressions

    def test_multi_operator_expression(self):
        self.assertEqual((((dim('float')-2)*3)**2).apply(self.dataset),
                         ((self.linear_floats-2)*3)**2)

    def test_multi_dim_expression(self):
        self.assertEqual((dim('int')-dim('float')).apply(self.dataset),
                         self.linear_ints-self.linear_floats)

    # Repr method

    def test_dim_repr(self):
        self.assertEqual(repr(dim('float')), "'float'")

    def test_unary_op_repr(self):
        self.assertEqual(repr(-dim('float')), "-dim('float')")

    def test_binary_op_repr(self):
        self.assertEqual(repr(dim('float')*2), "dim('float')*2")

    def test_reverse_binary_op_repr(self):
        self.assertEqual(repr(1+dim('float')), "1+dim('float')")

    def test_ufunc_expression_repr(self):
        self.assertEqual(repr(np.log(dim('float'))), "np.log(dim('float'))")

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
