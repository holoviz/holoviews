"""
Unit tests for dim transforms
"""
import pickle
import warnings
from unittest import skipIf

import numpy as np
import pandas as pd
import param

import holoviews as hv

try:
    import dask.array as da
    import dask.dataframe as dd
except ImportError:
    da, dd = None, None

try:
    import xarray as xr
except ImportError:
    xr = None

xr_skip = skipIf(xr is None, "xarray not available")

try:
    import spatialpandas as spd
except ImportError:
    spd = None

try:
    import shapely
except ImportError:
    shapely = None

shapelib_available = skipIf(shapely is None and spd is None,
                            'Neither shapely nor spatialpandas are available')

from holoviews.core.data import Dataset
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim


class Params(param.Parameterized):

    a = param.Number(default=0)


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

        if dd is not None:
            ddf = dd.from_pandas(self.dataset.data, npartitions=2)
            self.dataset_dask = self.dataset.clone(data=ddf)

        if xr is None:
            return

        x = np.arange(2, 62, 3)
        y = np.arange(2, 12, 2)
        array = np.arange(100).reshape(5, 20)
        darray = xr.DataArray(
            data=array,
            coords=dict([('x', x), ('y', y)]),
            dims=['y','x']
        )
        self.dataset_xarray = Dataset(darray, vdims=['z'])
        if da is not None:
            dask_array = da.from_array(array)
            dask_da = xr.DataArray(
                data=dask_array,
                coords=dict([('x', x), ('y', y)]),
                dims=['y','x']
            )
            self.dataset_xarray_dask = Dataset(dask_da, vdims=['z'])

    # Assertion helpers

    def assert_apply(self, expr, expected, skip_dask=False, skip_no_index=False):
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
        if not skip_no_index:
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
        if not skip_no_index:
            da.assert_eq(
                expr.apply(self.dataset_dask, compute=False).compute(),
                expected_dask.values.compute()
            )
        # keep_index=True, compute=False
        dd.assert_eq(
            expr.apply(self.dataset_dask, keep_index=True, compute=False),
            expected_dask,
            check_names=False
        )
        # keep_index=False, compute=True
        if not skip_no_index:
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


    def assert_apply_xarray(self, expr, expected, skip_dask=False, skip_no_index=False):
        import xarray as xr
        if np.isscalar(expected):
            # Pandas input
            self.assertEqual(
                expr.apply(self.dataset_xarray, keep_index=False), expected
            )
            self.assertEqual(
                expr.apply(self.dataset_xarray, keep_index=True), expected
            )
            return

        # Make sure expected is a pandas Series
        self.assertIsInstance(expected, xr.DataArray)

        # Check using dataset backed by pandas DataFrame
        # keep_index=False
        if not skip_no_index:
            np.testing.assert_equal(
                expr.apply(self.dataset_xarray),
                expected.values
            )
        # keep_index=True
        xr.testing.assert_equal(
            expr.apply(self.dataset_xarray, keep_index=True),
            expected
        )

        if skip_dask or da is None:
            return

        # Check using dataset backed by Dask DataFrame
        expected_da = da.from_array(expected.values)
        expected_dask = expected.copy()
        expected_dask.data = expected_da

        # keep_index=False, compute=False
        if not skip_no_index:
            da.assert_eq(
                expr.apply(self.dataset_xarray_dask, compute=False),
                expected_dask.data
            )
        # keep_index=True, compute=False
        xr.testing.assert_equal(
            expr.apply(self.dataset_xarray_dask, keep_index=True, compute=False),
            expected_dask,
        )
        # keep_index=False, compute=True
        if not skip_no_index:
            np.testing.assert_equal(
                expr.apply(self.dataset_xarray_dask, compute=True),
                expected_dask.data.compute()
            )
        # keep_index=True, compute=True
        xr.testing.assert_equal(
            expr.apply(self.dataset_xarray_dask, keep_index=True, compute=True),
            expected_dask.compute(),
        )


    # Unary operators

    def test_abs_transform(self):
        expr = abs(dim('negative'))
        self.assert_apply(expr, self.linear_floats)

    def test_neg_transform(self):
        expr = -dim('negative')
        self.assert_apply(expr, self.linear_floats)

    def test_inv_transform(self):
        expr = ~dim('booleans')
        self.assert_apply(expr, ~self.booleans)

    # Binary operators

    def test_add_transform(self):
        expr = dim('float') + 1
        self.assert_apply(expr, self.linear_floats+1)

    def test_div_transform(self):
        expr = dim('int') / 10.
        self.assert_apply(expr, self.linear_floats)

    def test_floor_div_transform(self):
        expr = dim('int') // 2
        self.assert_apply(expr, self.linear_ints//2)

    def test_mod_transform(self):
        expr = dim('int') % 2
        self.assert_apply(expr, self.linear_ints % 2)

    def test_mul_transform(self):
        expr = dim('float') * 10.
        self.assert_apply(expr, self.linear_ints.astype('float64'))

    def test_pow_transform(self):
        expr = dim('int') ** 2
        self.assert_apply(expr, self.linear_ints ** 2)

    def test_sub_transform(self):
        expr = dim('int') - 10
        self.assert_apply(expr, self.linear_ints - 10)

    # Reverse binary operators

    def test_radd_transform(self):
        expr = 1 + dim('float')
        self.assert_apply(expr, 1 + self.linear_floats)

    def test_rdiv_transform(self):
        expr = 10. / dim('int')
        self.assert_apply(expr, 10. / self.linear_ints)

    def test_rfloor_div_transform(self):
        expr = 2 // dim('int')
        self.assert_apply(expr, 2 // self.linear_ints)

    def test_rmod_transform(self):
        expr = 2 % dim('int')
        self.assert_apply(expr, 2 % self.linear_ints)

    def test_rmul_transform(self):
        expr = 10. * dim('float')
        self.assert_apply(expr, self.linear_ints.astype('float64'))

    def test_rsub_transform(self):
        expr = 10 - dim('int')
        self.assert_apply(expr, 10 - self.linear_ints)

    # NumPy operations

    def test_ufunc_transform(self):
        expr = np.sin(dim('float'))
        self.assert_apply(expr, np.sin(self.linear_floats))

    def test_astype_transform(self):
        expr = dim('int').astype('float64')
        self.assert_apply(expr, self.linear_ints.astype('float64'))

    def test_cumsum_transform(self):
        expr = dim('float').cumsum()
        self.assert_apply(expr, self.linear_floats.cumsum())

    def test_max_transform(self):
        expr = dim('float').max()
        self.assert_apply(expr, self.linear_floats.max())

    def test_min_transform(self):
        expr = dim('float').min()
        self.assert_apply(expr, self.linear_floats.min())

    def test_round_transform(self):
        expr = dim('float').round()
        self.assert_apply(expr, self.linear_floats.round())

    def test_sum_transform(self):
        expr = dim('float').sum()
        self.assert_apply(expr, self.linear_floats.sum())

    def test_std_transform(self):
        expr = dim('float').std(ddof=0)
        self.assert_apply(expr, self.linear_floats.std(ddof=0))

    def test_var_transform(self):
        expr = dim('float').var(ddof=0)
        self.assert_apply(expr, self.linear_floats.var(ddof=0))

    def test_log_transform(self):
        expr = dim('float').log()
        self.assert_apply(expr, np.log(self.linear_floats))

    def test_log10_transform(self):
        expr = dim('float').log10()
        self.assert_apply(expr, np.log10(self.linear_floats))

    # Custom functions

    def test_str_astype(self):
        expr = dim('int').str()
        self.assert_apply(expr, self.linear_ints.astype(str), skip_dask=True)

    def test_norm_transform(self):
        expr = dim('int').norm()
        self.assert_apply(expr, (self.linear_ints-1)/9.)

    def test_iloc_transform_int(self):
        expr = dim('int').iloc[1]
        self.assert_apply(expr, self.linear_ints[1])

    def test_iloc_transform_slice(self):
        expr = dim('int').iloc[1:3]
        self.assert_apply(expr, self.linear_ints[1:3], skip_dask=True)

    def test_iloc_transform_list(self):
        expr = dim('int').iloc[[1, 3, 5]]
        self.assert_apply(expr, self.linear_ints[[1, 3, 5]], skip_dask=True)

    def test_bin_transform(self):
        expr = dim('int').bin([0, 5, 10])
        expected = pd.Series(
            [2.5, 2.5, 2.5, 2.5, 2.5, 7.5, 7.5, 7.5, 7.5, 7.5]
        )
        self.assert_apply(expr, expected)

    def test_bin_transform_with_labels(self):
        expr = dim('int').bin([0, 5, 10], ['A', 'B'])
        expected = pd.Series(
            ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
        )
        self.assert_apply(expr, expected)

    def test_categorize_transform_list(self):
        expr = dim('categories').categorize(['circle', 'square', 'triangle'])
        expected = pd.Series(
            (['circle', 'square', 'triangle']*3)+['circle']
        )
        # We skip dask because results will depend on partition structure
        self.assert_apply(expr, expected, skip_dask=True)

    def test_categorize_transform_dict(self):
        expr = dim('categories').categorize(
            {'A': 'circle', 'B': 'square', 'C': 'triangle'}
        )
        expected = pd.Series(
            (['circle', 'square', 'triangle'] * 3) + ['circle']
        )
        # We don't skip dask because results are now stable across partitions
        self.assert_apply(expr, expected)

    def test_categorize_transform_dict_with_default(self):
        expr = dim('categories').categorize(
            {'A': 'circle', 'B': 'square'}, default='triangle'
        )
        expected = pd.Series(
            (['circle', 'square', 'triangle'] * 3) + ['circle']
        )
        # We don't skip dask because results are stable across partitions
        self.assert_apply(expr, expected)

    # Numpy functions

    def test_digitize(self):
        expr = dim('int').digitize([1, 5, 10])
        expected = pd.Series(np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 3])).astype('int64')
        self.assert_apply(expr, expected)

    def test_isin(self):
        expr = dim('int').digitize([1, 5, 10]).isin([1, 3])
        expected = pd.Series(
            np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1], dtype='bool')
        )
        self.assert_apply(expr, expected)

    # Complex expressions

    def test_multi_operator_expression(self):
        expr = (((dim('float')-2)*3)**2)
        self.assert_apply(expr, ((self.linear_floats-2)*3)**2)

    def test_multi_dim_expression(self):
        expr = dim('int')-dim('float')
        self.assert_apply(expr, self.linear_ints-self.linear_floats)

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

    # Check namespaced expressions

    def test_pandas_namespace_accessor_repr(self):
        self.assertEqual(repr(dim('date').df.dt.year),
                         "dim('date').pd.dt.year")

    def test_pandas_str_accessor(self):
        expr = dim('categories').df.str.lower()
        self.assert_apply(expr, self.repeating.str.lower())

    def test_pandas_chained_methods(self):
        expr = dim('int').df.rolling(1).mean()

        with warnings.catch_warnings():
            # The kwargs is {'axis': None} and is already handled by the code.
            # This context manager can be removed, when it raises an TypeError instead of warning.
            warnings.filterwarnings("ignore", "Passing additional kwargs to Rolling.mean")
            self.assert_apply(expr, self.linear_ints.rolling(1).mean())


    @xr_skip
    def test_xarray_namespace_method_repr(self):
        self.assertEqual(repr(dim('date').xr.quantile(0.95)),
                         "dim('date').xr.quantile(0.95)")

    @xr_skip
    def test_xarray_quantile_method(self):
        expr = dim('z').xr.quantile(0.95)
        self.assert_apply_xarray(expr, self.dataset_xarray.data.z.quantile(0.95), skip_dask=True)

    @xr_skip
    def test_xarray_roll_method(self):
        expr = dim('z').xr.roll({'x': 1}, roll_coords=False)
        self.assert_apply_xarray(expr, self.dataset_xarray.data.z.roll({'x': 1}, roll_coords=False))

    @xr_skip
    def test_xarray_coarsen_method(self):
        expr = dim('z').xr.coarsen({'x': 4}).mean()
        self.assert_apply_xarray(expr, self.dataset_xarray.data.z.coarsen({'x': 4}).mean())

    # Dynamic arguments

    def test_dynamic_mul(self):
        p = Params(a=1)
        expr = dim('float') * p.param.a
        self.assertEqual(list(expr.params.values()), [p.param.a])
        self.assert_apply(expr, self.linear_floats)
        p.a = 2
        self.assert_apply(expr, self.linear_floats*2)

    def test_dynamic_arg(self):
        p = Params(a=1)
        expr = dim('float').round(p.param.a)
        self.assertEqual(list(expr.params.values()), [p.param.a])
        self.assert_apply(expr, np.round(self.linear_floats, 1))
        p.a = 2
        self.assert_apply(expr, np.round(self.linear_floats, 2))

    def test_dynamic_kwarg(self):
        p = Params(a=1)
        expr = dim('float').round(decimals=p.param.a)
        self.assertEqual(list(expr.params.values()), [p.param.a])
        self.assert_apply(expr, np.round(self.linear_floats, 1))
        p.a = 2
        self.assert_apply(expr, np.round(self.linear_floats, 2))

    def test_pickle(self):
        expr = (((dim('float')-2)*3)**2)
        expr2 = pickle.loads(pickle.dumps(expr))
        self.assertEqual(expr, expr2)


@shapelib_available
def test_dataset_transform_by_spatial_select_expr_index_not_0_based():
    """Ensure 'spatial_select' expression works when index not zero-based.
    Use 'spatial_select' defined by four nodes to select index 104, 105.
    Apply expression to dataset.transform to generate new 'flag' column where True
    for the two indexes."""
    df = pd.DataFrame({"a": [7, 3, 0.5, 2, 1, 1], "b": [3, 4, 3, 2, 2, 1]}, index=list(range(101, 107)))
    geometry = np.array(
        [
            [3.0, 1.7],
            [0.3, 1.7],
            [0.3, 2.7],
            [3.0, 2.7]
        ]
    )
    spatial_expr = hv.dim('a', hv.element.selection.spatial_select, hv.dim('b'), geometry=geometry)
    dataset = hv.Dataset(df)
    df_out = dataset.transform(flag=spatial_expr).dframe()
    expected_series = pd.Series(
        {
            101: False,
            102: False,
            103: False,
            104: True,
            105: True,
            106: False}
        )
    pd.testing.assert_series_equal(df_out['flag'], expected_series, check_names=False)
