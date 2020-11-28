import datetime as dt

from collections import OrderedDict
from unittest import SkipTest

import numpy as np

try:
    import pandas as pd
    import xarray as xr
except:
    raise SkipTest("Could not import xarray, skipping XArrayInterface tests.")

from holoviews.core.data import Dataset, concat
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Image, RGB, HSV, QuadMesh

from .testimageinterface import (
    BaseImageElementInterfaceTests, BaseRGBElementInterfaceTests,
    BaseHSVElementInterfaceTests
)
from .testgridinterface import BaseGridInterfaceTests


class XArrayInterfaceTests(BaseGridInterfaceTests):
    """
    Tests for xarray interface
    """

    datatype = 'xarray'
    data_type = xr.Dataset

    __test__ = True

    def get_irregular_dataarray(self, invert_y=True):
        multiplier = -1 if invert_y else 1
        x = np.arange(2, 62, 3)
        y = np.arange(2, 12, 2) * multiplier
        da = xr.DataArray(
            data=[np.arange(100).reshape(5, 20)],
            coords=OrderedDict([('band', [1]), ('x', x), ('y', y)]),
            dims=['band', 'y','x'],
            attrs={'transform': (3, 0, 2, 0, -2, -2)})
        xs, ys = (np.tile(x[:, np.newaxis], len(y)).T,
                  np.tile(y[:, np.newaxis], len(x)))
        return da.assign_coords(**{'xc': xr.DataArray(xs, dims=('y','x')),
                                   'yc': xr.DataArray(ys, dims=('y','x')),})

    def get_multi_dim_irregular_dataset(self):
        temp = 15 + 8 * np.random.randn(2, 2, 4, 3)
        precip = 10 * np.random.rand(2, 2, 4, 3)
        lon = [[-99.83, -99.32], [-99.79, -99.23]]
        lat = [[42.25, 42.21], [42.63, 42.59]]
        return xr.Dataset({'temperature': (['x', 'y', 'z', 'time'],  temp),
                         'precipitation': (['x', 'y', 'z', 'time'], precip)},
                        coords={'lon': (['x', 'y'], lon),
                                'lat': (['x', 'y'], lat),
                                'z': np.arange(4),
                                'time': pd.date_range('2014-09-06', periods=3),
                                'reference_time': pd.Timestamp('2014-09-05')})

    def test_xarray_dataset_irregular_shape(self):
        ds = Dataset(self.get_multi_dim_irregular_dataset())
        shape = ds.interface.shape(ds, gridded=True)
        self.assertEqual(shape, (np.nan, np.nan, 3, 4))

    def test_xarray_irregular_dataset_values(self):
        ds = Dataset(self.get_multi_dim_irregular_dataset())
        values = ds.dimension_values('z', expanded=False)
        self.assertEqual(values, np.array([0, 1, 2, 3]))

    def test_xarray_dataset_with_scalar_dim_canonicalize(self):
        xs = [0, 1]
        ys = [0.1, 0.2, 0.3]
        zs = np.array([[[0, 1], [2, 3], [4, 5]]])
        xrarr = xr.DataArray(zs, coords={'x': xs, 'y': ys, 't': [1]}, dims=['t', 'y', 'x'])
        xrds = xr.Dataset({'v': xrarr})
        ds = Dataset(xrds, kdims=['x', 'y'], vdims=['v'], datatype=['xarray'])
        canonical = ds.dimension_values(2, flat=False)
        self.assertEqual(canonical.ndim, 2)
        expected = np.array([[0, 1], [2, 3], [4, 5]])
        self.assertEqual(canonical, expected)

    def test_xarray_dataset_names_and_units(self):
        xs = [0.1, 0.2, 0.3]
        ys = [0, 1]
        zs = np.array([[0, 1], [2, 3], [4, 5]])
        da = xr.DataArray(zs, coords=[('x_dim', xs), ('y_dim', ys)], name="data_name", dims=['y_dim', 'x_dim'])
        da.attrs['long_name'] = "data long name"
        da.attrs['units'] = "array_unit"
        da.x_dim.attrs['units'] = "x_unit"
        da.y_dim.attrs['long_name'] = "y axis long name"
        dataset = Dataset(da)
        self.assertEqual(dataset.get_dimension("x_dim"), Dimension("x_dim", unit="x_unit"))
        self.assertEqual(dataset.get_dimension("y_dim"), Dimension("y_dim", label="y axis long name"))
        self.assertEqual(dataset.get_dimension("data_name"),
                         Dimension("data_name", label="data long name", unit="array_unit"))

    def test_xarray_dataset_dataarray_vs_dataset(self):
        xs = [0.1, 0.2, 0.3]
        ys = [0, 1]
        zs = np.array([[0, 1], [2, 3], [4, 5]])
        da = xr.DataArray(zs, coords=[('x_dim', xs), ('y_dim', ys)], name="data_name", dims=['y_dim', 'x_dim'])
        da.attrs['long_name'] = "data long name"
        da.attrs['units'] = "array_unit"
        da.x_dim.attrs['units'] = "x_unit"
        da.y_dim.attrs['long_name'] = "y axis long name"
        ds = da.to_dataset()
        dataset_from_da = Dataset(da)
        dataset_from_ds = Dataset(ds)
        self.assertEqual(dataset_from_da, dataset_from_ds)
        # same with reversed names:
        da_rev = xr.DataArray(zs, coords=[('x_dim', xs), ('y_dim', ys)], name="data_name", dims=['x_dim', 'y_dim'])
        da_rev.attrs['long_name'] = "data long name"
        da_rev.attrs['units'] = "array_unit"
        da_rev.x_dim.attrs['units'] = "x_unit"
        da_rev.y_dim.attrs['long_name'] = "y axis long name"
        ds_rev = da_rev.to_dataset()
        dataset_from_da_rev = Dataset(da_rev)
        dataset_from_ds_rev = Dataset(ds_rev)
        self.assertEqual(dataset_from_da_rev, dataset_from_ds_rev)

    def test_xarray_override_dims(self):
        xs = [0.1, 0.2, 0.3]
        ys = [0, 1]
        zs = np.array([[0, 1], [2, 3], [4, 5]])
        da = xr.DataArray(zs, coords=[('x_dim', xs), ('y_dim', ys)], name="data_name", dims=['y_dim', 'x_dim'])
        da.attrs['long_name'] = "data long name"
        da.attrs['units'] = "array_unit"
        da.x_dim.attrs['units'] = "x_unit"
        da.y_dim.attrs['long_name'] = "y axis long name"
        ds = Dataset(da, kdims=["x_dim", "y_dim"], vdims=["z_dim"])
        x_dim = Dimension("x_dim")
        y_dim = Dimension("y_dim")
        z_dim = Dimension("z_dim")
        self.assertEqual(ds.kdims[0], x_dim)
        self.assertEqual(ds.kdims[1], y_dim)
        self.assertEqual(ds.vdims[0], z_dim)
        ds_from_ds = Dataset(da.to_dataset(), kdims=["x_dim", "y_dim"], vdims=["data_name"])
        self.assertEqual(ds_from_ds.kdims[0], x_dim)
        self.assertEqual(ds_from_ds.kdims[1], y_dim)
        data_dim = Dimension("data_name")
        self.assertEqual(ds_from_ds.vdims[0], data_dim)

    def test_xarray_coord_ordering(self):
        data = np.zeros((3,4,5))
        coords = OrderedDict([('b', range(3)), ('c', range(4)), ('a', range(5))])
        darray = xr.DataArray(data, coords=coords, dims=['b', 'c', 'a'])
        dataset = xr.Dataset({'value': darray}, coords=coords)
        ds = Dataset(dataset)
        self.assertEqual(ds.kdims, ['b', 'c', 'a'])

    def test_irregular_and_regular_coordinate_inference(self):
        data = self.get_irregular_dataarray()
        ds = Dataset(data, vdims='Value')
        self.assertEqual(ds.kdims, [Dimension('band'), Dimension('x'), Dimension('y')])
        self.assertEqual(ds.dimension_values(3, flat=False), data.values[:, ::-1].transpose([1, 2, 0]))

    def test_irregular_and_regular_coordinate_inference_inverted(self):
        data = self.get_irregular_dataarray(False)
        ds = Dataset(data, vdims='Value')
        self.assertEqual(ds.kdims, [Dimension('band'), Dimension('x'), Dimension('y')])
        self.assertEqual(ds.dimension_values(3, flat=False), data.values.transpose([1, 2, 0]))
    def test_irregular_and_regular_coordinate_explicit_regular_coords(self):
        data = self.get_irregular_dataarray()
        ds = Dataset(data, ['x', 'y'], vdims='Value')
        self.assertEqual(ds.kdims, [Dimension('x'), Dimension('y')])
        self.assertEqual(ds.dimension_values(2, flat=False), data.values[0, ::-1])

    def test_irregular_and_regular_coordinate_explicit_regular_coords_inverted(self):
        data = self.get_irregular_dataarray(False)
        ds = Dataset(data, ['x', 'y'], vdims='Value')
        self.assertEqual(ds.kdims, [Dimension('x'), Dimension('y')])
        self.assertEqual(ds.dimension_values(2, flat=False), data.values[0])

    def test_irregular_and_regular_coordinate_explicit_irregular_coords(self):
        data = self.get_irregular_dataarray()
        ds = Dataset(data, ['xc', 'yc'], vdims='Value')
        self.assertEqual(ds.kdims, [Dimension('xc'), Dimension('yc')])
        self.assertEqual(ds.dimension_values(2, flat=False), data.values[0])

    def test_irregular_and_regular_coordinate_explicit_irregular_coords_inverted(self):
        data = self.get_irregular_dataarray(False)
        ds = Dataset(data, ['xc', 'yc'], vdims='Value')
        self.assertEqual(ds.kdims, [Dimension('xc'), Dimension('yc')])
        self.assertEqual(ds.dimension_values(2, flat=False), data.values[0])

    def test_concat_grid_3d_shape_mismatch(self):
        arr1 = np.random.rand(3, 2)
        arr2 = np.random.rand(2, 3)
        ds1 = Dataset(([0, 1], [1, 2, 3], arr1), ['x', 'y'], 'z')
        ds2 = Dataset(([0, 1, 2], [1, 2], arr2), ['x', 'y'], 'z')
        hmap = HoloMap({1: ds1, 2: ds2})
        arr = np.full((3, 3, 2), np.NaN)
        arr[:, :2, 0] = arr1
        arr[:2, :, 1] = arr2
        ds = Dataset(([1, 2], [0, 1, 2], [1, 2, 3], arr), ['Default', 'x', 'y'], 'z')
        self.assertEqual(concat(hmap), ds)

    def test_zero_sized_coordinates_range(self):
        da = xr.DataArray(np.empty((2, 0)), dims=('y', 'x'), coords={'x': [], 'y': [0 ,1]}, name='A')
        ds = Dataset(da)
        x0, x1 = ds.range('x')
        self.assertTrue(np.isnan(x0))
        self.assertTrue(np.isnan(x1))
        z0, z1 = ds.range('A')
        self.assertTrue(np.isnan(z0))
        self.assertTrue(np.isnan(z1))

    def test_datetime_bins_range(self):
        xs = [dt.datetime(2018, 1, i) for i in range(1, 11)]
        ys = np.arange(10)
        array = np.random.rand(10, 10)
        ds = QuadMesh((xs, ys, array))
        self.assertEqual(ds.interface.datatype, 'xarray')
        expected = (np.datetime64(dt.datetime(2017, 12, 31, 12, 0)),
                    np.datetime64(dt.datetime(2018, 1, 10, 12, 0)))
        self.assertEqual(ds.range('x'), expected)

    def test_datetime64_bins_range(self):
        xs = [np.datetime64(dt.datetime(2018, 1, i)) for i in range(1, 11)]
        ys = np.arange(10)
        array = np.random.rand(10, 10)
        ds = QuadMesh((xs, ys, array))
        self.assertEqual(ds.interface.datatype, 'xarray')
        expected = (np.datetime64(dt.datetime(2017, 12, 31, 12, 0)),
                    np.datetime64(dt.datetime(2018, 1, 10, 12, 0)))
        self.assertEqual(ds.range('x'), expected)

    def test_select_dropped_dimensions_restoration(self):
        d = np.random.randn(3, 8)
        da = xr.DataArray(d, name='stuff', dims=['chain', 'value'],
            coords=dict(chain=range(d.shape[0]), value=range(d.shape[1])))
        ds = Dataset(da)
        t = ds.select(chain=0)
        self.assertEqual(t.data.dims , dict(chain=1,value=8))
        self.assertEqual(t.data.stuff.shape , (1,8))

    def test_mask_2d_array_transposed(self):
        array = np.random.rand(4, 3)
        da = xr.DataArray(array.T, coords={'x': [0, 1, 2], 'y': [0, 1, 2, 3]}, dims=['x', 'y'])
        ds = Dataset(da, ['x', 'y'], 'z')
        mask = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1]], dtype='bool')
        masked = ds.clone(ds.interface.mask(ds, mask))
        masked_array = masked.dimension_values(2, flat=False)
        expected = array.copy()
        expected[mask] = np.nan
        self.assertEqual(masked_array, expected)

    # Disabled tests for NotImplemented methods
    def test_dataset_array_init_hm(self):
        "Tests support for arrays (homogeneous)"
        raise SkipTest("Not supported")

    def test_dataset_add_dimensions_values_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_reverse_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_vdim_hm_alias(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_vdim_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sort_reverse_vdim_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sample_hm(self):
        raise SkipTest("Not supported")

    def test_dataset_sample_hm_alias(self):
        raise SkipTest("Not supported")



class DaskXArrayInterfaceTest(XArrayInterfaceTests):
    """
    Tests for XArray interface wrapping dask arrays
    """

    def setUp(self):
        try:
            import dask.array # noqa
        except:
            raise SkipTest('Dask could not be imported, cannot test '
                           'dask arrays with XArrayInterface')
        super(DaskXArrayInterfaceTest, self).setUp()

    def init_column_data(self):
        import dask.array
        self.xs = np.array(range(11))
        self.xs_2 = self.xs**2

        self.y_ints = self.xs*2
        dask_y = dask.array.from_array(np.array(self.y_ints), 2)
        self.dataset_hm = Dataset((self.xs, dask_y),
                                  kdims=['x'], vdims=['y'])
        self.dataset_hm_alias = Dataset((self.xs, dask_y),
                                        kdims=[('x', 'X')], vdims=[('y', 'Y')])

    def init_grid_data(self):
        import dask.array
        self.grid_xs = [0, 1]
        self.grid_ys = [0.1, 0.2, 0.3]
        self.grid_zs = np.array([[0, 1], [2, 3], [4, 5]])
        dask_zs = dask.array.from_array(self.grid_zs, 2)
        self.dataset_grid = self.element((self.grid_xs, self.grid_ys,
                                         dask_zs), kdims=['x', 'y'],
                                        vdims=['z'])
        self.dataset_grid_alias = self.element((self.grid_xs, self.grid_ys,
                                               dask_zs), kdims=[('x', 'X'), ('y', 'Y')],
                                              vdims=[('z', 'Z')])
        self.dataset_grid_inv = self.element((self.grid_xs[::-1], self.grid_ys[::-1],
                                             dask_zs), kdims=['x', 'y'],
                                            vdims=['z'])

    def test_xarray_dataset_with_scalar_dim_canonicalize(self):
        import dask.array
        xs = [0, 1]
        ys = [0.1, 0.2, 0.3]
        zs = dask.array.from_array(np.array([[[0, 1], [2, 3], [4, 5]]]), 2)
        xrarr = xr.DataArray(zs, coords={'x': xs, 'y': ys, 't': [1]}, dims=['t', 'y', 'x'])
        xrds = xr.Dataset({'v': xrarr})
        ds = Dataset(xrds, kdims=['x', 'y'], vdims=['v'], datatype=['xarray'])
        canonical = ds.dimension_values(2, flat=False)
        self.assertEqual(canonical.ndim, 2)
        expected = np.array([[0, 1], [2, 3], [4, 5]])
        self.assertEqual(canonical, expected)



class ImageElement_XArrayInterfaceTests(BaseImageElementInterfaceTests):

    datatype = 'xarray'
    data_type = xr.Dataset

    __test__ = True

    def init_data(self):
        self.image = Image((self.xs, self.ys, self.array))
        self.image_inv = Image((self.xs[::-1], self.ys[::-1], self.array[::-1, ::-1]))

    def test_dataarray_dimension_order(self):
        x = np.linspace(-3, 7, 53)
        y = np.linspace(-5, 8, 89)
        z = np.exp(-1*(x**2 + y[:, np.newaxis]**2))
        array = xr.DataArray(z, coords=[y, x], dims=['x', 'y'])
        img = Image(array)
        self.assertEqual(img.kdims, [Dimension('x'), Dimension('y')])

    def test_dataarray_shape(self):
        x = np.linspace(-3, 7, 53)
        y = np.linspace(-5, 8, 89)
        z = np.exp(-1*(x**2 + y[:, np.newaxis]**2))
        array = xr.DataArray(z, coords=[y, x], dims=['x', 'y'])
        img = Image(array, ['x', 'y'])
        self.assertEqual(img.interface.shape(img, gridded=True), (53, 89))

    def test_dataarray_shape_transposed(self):
        x = np.linspace(-3, 7, 53)
        y = np.linspace(-5, 8, 89)
        z = np.exp(-1*(x**2 + y[:, np.newaxis]**2))
        array = xr.DataArray(z, coords=[y, x], dims=['x', 'y'])
        img = Image(array, ['y', 'x'])
        self.assertEqual(img.interface.shape(img, gridded=True), (89, 53))

    def test_select_on_transposed_dataarray(self):
        x = np.linspace(-3, 7, 53)
        y = np.linspace(-5, 8, 89)
        z = np.exp(-1*(x**2 + y[:, np.newaxis]**2))
        array = xr.DataArray(z, coords=[y, x], dims=['x', 'y'])
        img = Image(array)[1:3]
        self.assertEqual(img['z'], Image(array.sel(x=slice(1, 3)))['z'])

    def test_dataarray_with_no_coords(self):
        expected_xs = list(range(2))
        expected_ys = list(range(3))
        zs = np.arange(6).reshape(2, 3)
        xrarr = xr.DataArray(zs, dims=('x','y'))

        img = Image(xrarr)
        self.assertTrue(all(img.data.x == expected_xs))
        self.assertTrue(all(img.data.y == expected_ys))

        img = Image(xrarr, kdims=['x', 'y'])
        self.assertTrue(all(img.data.x == expected_xs))
        self.assertTrue(all(img.data.y == expected_ys))

    def test_dataarray_with_some_coords(self):
        xs = [4.2, 1]
        zs = np.arange(6).reshape(2, 3)
        xrarr = xr.DataArray(zs, dims=('x','y'), coords={'x': xs})

        with self.assertRaises(ValueError):
            Image(xrarr)

        with self.assertRaises(ValueError):
            Image(xrarr, kdims=['x', 'y'])


class RGBElement_XArrayInterfaceTests(BaseRGBElementInterfaceTests):

    datatype = 'xarray'
    data_type = xr.Dataset

    __test__ = True

    def init_data(self):
        self.rgb = RGB((self.xs, self.ys, self.rgb_array[:, :, 0],
                        self.rgb_array[:, :, 1], self.rgb_array[:, :, 2]))


class RGBElement_PackedXArrayInterfaceTests(BaseRGBElementInterfaceTests):

    datatype = 'xarray'
    data_type = xr.Dataset

    __test__ = True

    def init_data(self):
        self.rgb = RGB((self.xs, self.ys, self.rgb_array))


class HSVElement_XArrayInterfaceTest(BaseHSVElementInterfaceTests):

    datatype = 'xarray'
    data_type = xr.Dataset

    __test__ = True

    def init_data(self):
        self.hsv = HSV((self.xs, self.ys, self.hsv_array[:, :, 0],
                        self.hsv_array[:, :, 1], self.hsv_array[:, :, 2]))
