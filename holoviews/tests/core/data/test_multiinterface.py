"""
Tests for the Dataset Element types.
"""

import logging

import numpy as np
import pandas as pd
import pytest
from param import get_logger

from holoviews.core.data import Dataset, MultiInterface
from holoviews.element import Path, Points, Polygons
from holoviews.testing import assert_data_equal, assert_element_equal

try:
    import dask.dataframe as dd
except ImportError:
    dd = None


class GeomTests:
    """
    Test of the MultiInterface.
    """

    datatype = None

    interface = None

    __test__ = False

    def test_array_dataset(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        for i, array in enumerate(mds.split(datatype='array')):
            assert_data_equal(array, arrays[i])

    def test_dict_dataset(self):
        dicts = [{'x': np.arange(i, i+2), 'y': np.arange(i, i+2)} for i in range(2)]
        mds = Path(dicts, kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        for i, cols in enumerate(mds.split(datatype='columns')):
            output = dict(cols)
            expected = dict(dicts[i], geom_type='Line')
            assert output.keys() == expected.keys()
            for k in output:  # noqa: PLC0206
                if k in "xy":
                    np.testing.assert_equal(output[k], expected[k])
                else:
                    assert output[k] == expected[k]

    def test_df_dataset(self):
        dfs = [pd.DataFrame(np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]), columns=['x', 'y'])
                  for i in range(2)]
        mds = Path(dfs, kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        for i, ds in enumerate(mds.split(datatype='dataframe')):
            assert_data_equal(ds, dfs[i])

    def test_array_dataset_add_dimension_scalar(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=[self.datatype]).add_dimension('A', 0, 'Scalar', True)
        assert mds.interface is self.interface
        assert_element_equal(mds, Path([{('x', 'y'): arrays[i], 'A': 'Scalar'} for i in range(2)],
                                   ['x', 'y'], 'A'))

    def test_dict_dataset_add_dimension_scalar(self):
        arrays = [{'x': np.arange(i, i+2), 'y': np.arange(i, i+2)} for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=[self.datatype]).add_dimension('A', 0, 'Scalar', True)
        assert mds.interface is self.interface
        assert_element_equal(mds, Path([dict(arrays[i], A='Scalar') for i in range(2)], ['x', 'y'],
                                   'A', datatype=['multitabular']))

    def test_dict_dataset_add_dimension_values(self):
        arrays = [{'x': np.arange(i, i+2), 'y': np.arange(i, i+2)} for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=[self.datatype]).add_dimension('A', 0, [0,1], True)
        assert mds.interface is self.interface
        assert_element_equal(mds, Path([dict(arrays[i], A=i) for i in range(2)], ['x', 'y'],
                                   'A', datatype=['multitabular']))

    def test_array_length(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        assert len(mds) == 2

    def test_array_length_points(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Points(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        assert len(mds) == 4

    def test_empty_length(self):
        mds = Path([], kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        assert len(mds) == 0

    def test_empty_range(self):
        mds = Path([], kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        x0, _x1 = mds.range(0)
        assert not np.isfinite(x0)
        assert not np.isfinite(x0)
        y0, y1 = mds.range(1)
        assert not np.isfinite(y0)
        assert not np.isfinite(y1)

    def test_array_range(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        assert mds.range(0) == (0, 2)

    def test_array_shape(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        assert mds.shape == (2, 2)

    def test_array_shape_points(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Points(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        assert mds.shape == (4, 2)

    def test_empty_shape(self):
        mds = Path([], kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        assert mds.shape == (0, 2)

    def test_array_values(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        assert_data_equal(mds.dimension_values(0), np.array([0., 1, np.nan, 1, 2]))

    def test_empty_array_values(self):
        mds = Path([], kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        assert_data_equal(mds.dimension_values(0), np.array([]))

    def test_array_values_coordinates_nonexpanded(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        values = mds.dimension_values(0, expanded=False)
        assert_data_equal(values[0], np.array([0., 1]))
        assert_data_equal(values[1], np.array([1, 2]))

    def test_array_values_coordinates_nonexpanded_constant_kdim(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2), np.ones(2)*i]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        assert mds.interface is self.interface
        assert_data_equal(mds.dimension_values(2, expanded=False), np.array([0, 1]))

    def test_scalar_value_isscalar_per_geom(self):
        path = Path([{'x': [1, 2, 3, 4, 5], 'y': [0, 0, 1, 1, 2], 'value': 0},
                     {'x': [5, 4, 3, 2, 1], 'y': [2, 2, 1, 1, 0], 'value': 1}],
                    vdims='value', datatype=[self.datatype])
        assert path.interface is self.interface
        assert path.interface.isscalar(path, 'value', per_geom=True)

    def test_unique_values_isscalar_per_geom(self):
        path = Path([{'x': [1, 2, 3, 4, 5], 'y': [0, 0, 1, 1, 2], 'value': np.full(5, 0)},
                     {'x': [5, 4, 3, 2, 1], 'y': [2, 2, 1, 1, 0], 'value': np.full(5, 1)}],
                    vdims='value', datatype=[self.datatype])
        assert path.interface is self.interface
        assert path.interface.isscalar(path, 'value', per_geom=True)

    def test_scalar_and_unique_values_isscalar_per_geom(self):
        path = Path([{'x': [1, 2, 3, 4, 5], 'y': [0, 0, 1, 1, 2], 'value': 0},
                     {'x': [5, 4, 3, 2, 1], 'y': [2, 2, 1, 1, 0], 'value': np.full(5, 1)}],
                    vdims='value', datatype=[self.datatype])
        assert path.interface is self.interface
        assert path.interface.isscalar(path, 'value', per_geom=True)

    def test_varying_values_not_isscalar_per_geom(self):
        path = Path([{'x': [1, 2, 3, 4, 5], 'y': [0, 0, 1, 1, 2], 'value': np.arange(5)},
                     {'x': [5, 4, 3, 2, 1], 'y': [2, 2, 1, 1, 0], 'value': np.full(5, 1)}],
                    vdims='value', datatype=[self.datatype])
        assert path.interface is self.interface
        assert not path.interface.isscalar(path, 'value', per_geom=True)

    def test_varying_values_and_scalar_not_isscalar_per_geom(self):
        path = Path([{'x': [1, 2, 3, 4, 5], 'y': [0, 0, 1, 1, 2], 'value': np.arange(5)},
                     {'x': [5, 4, 3, 2, 1], 'y': [2, 2, 1, 1, 0], 'value': 1}],
                    vdims='value', datatype=[self.datatype])
        assert path.interface is self.interface
        assert not path.interface.isscalar(path, 'value', per_geom=True)

    def test_scalar_value_dimension_values_expanded(self):
        path = Path([{'x': [1, 2, 3, 4, 5], 'y': [0, 0, 1, 1, 2], 'value': 0},
                     {'x': [5, 4, 3, 2, 1], 'y': [2, 2, 1, 1, 0], 'value': 1}],
                    vdims='value', datatype=[self.datatype])
        assert path.interface is self.interface
        assert_data_equal(path.dimension_values('value'), np.array([0, 0, 0, 0, 0, np.nan, 1, 1, 1, 1, 1]))

    def test_scalar_value_dimension_values_not_expanded(self):
        path = Path([{'x': [1, 2, 3, 4, 5], 'y': [0, 0, 1, 1, 2], 'value': 0},
                     {'x': [5, 4, 3, 2, 1], 'y': [2, 2, 1, 1, 0], 'value': 1}],
                    vdims='value', datatype=[self.datatype])
        assert path.interface is self.interface
        assert_data_equal(path.dimension_values('value', expanded=False),
                         np.array([0, 1]))

    def test_unique_value_dimension_values_expanded(self):
        path = Path([{'x': [1, 2, 3, 4, 5], 'y': [0, 0, 1, 1, 2], 'value': np.full(5, 0)},
                     {'x': [5, 4, 3, 2, 1], 'y': [2, 2, 1, 1, 0], 'value': np.full(5, 1)}],
                    vdims='value', datatype=[self.datatype])
        assert path.interface is self.interface
        assert_data_equal(path.dimension_values('value'), np.array([0, 0, 0, 0, 0, np.nan, 1, 1, 1, 1, 1]))

    def test_unique_value_dimension_values_not_expanded(self):
        path = Path([{'x': [1, 2, 3, 4, 5], 'y': [0, 0, 1, 1, 2], 'value': np.full(5, 0)},
                     {'x': [5, 4, 3, 2, 1], 'y': [2, 2, 1, 1, 0], 'value': np.full(5, 1)}],
                    vdims='value', datatype=[self.datatype])
        assert path.interface is self.interface
        assert_data_equal(path.dimension_values('value', expanded=False),
                         np.array([0, 1]))

    def test_varying_value_dimension_values_expanded(self):
        path = Path([{'x': [1, 2, 3, 4, 5], 'y': [0, 0, 1, 1, 2], 'value': np.arange(5)},
                     {'x': [5, 4, 3, 2, 1], 'y': [2, 2, 1, 1, 0], 'value': np.full(5, 1)}],
                    vdims='value', datatype=[self.datatype])
        assert path.interface is self.interface
        assert_data_equal(path.dimension_values('value'), np.array([0, 1, 2, 3, 4, np.nan, 1, 1, 1, 1, 1]))

    def test_varying_value_dimension_values_not_expanded(self):
        path = Path([{'x': [1, 2, 3, 4, 5], 'y': [0, 0, 1, 1, 2], 'value': np.arange(5)},
                     {'x': [5, 4, 3, 2, 1], 'y': [2, 2, 1, 1, 0], 'value': np.full(5, 1)}],
                    vdims='value', datatype=[self.datatype])
        assert path.interface is self.interface
        values = path.dimension_values('value', expanded=False)
        assert_data_equal(values[0], np.array([0, 1, 2, 3, 4]))
        assert values[1] == 1
        assert isinstance(values[1], np.int_)

    def test_array_redim(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=[self.datatype]).redim(x='x2')
        assert mds.interface is self.interface
        assert_element_equal(mds, Path([arrays[i] for i in range(2)], ['x2', 'y']))

    def test_mixed_dims_raises(self):
        arrays = [{'x': range(10), 'y' if j else 'z': range(10)}
                  for i in range(2) for j in range(2)]
        with pytest.raises(ValueError):  # noqa: PT011
            Path(arrays, kdims=['x', 'y'], datatype=[self.datatype])

    def test_split_into_arrays(self):
        arrays = [np.column_stack([np.arange(i, i+2), np.arange(i, i+2)]) for i in range(2)]
        mds = Path(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        for arr1, arr2 in zip(mds.split(datatype='array'), arrays, strict=True):
            assert_data_equal(arr1, arr2)

    def test_split_empty(self):
        mds = Path([], kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        assert len(mds.split()) == 0

    def test_values_empty(self):
        mds = Path([], kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        assert_data_equal(mds.dimension_values(0), np.array([]))

    def test_dict_groupby_non_scalar(self):
        arrays = [{'x': np.arange(i, i+2), 'y': i} for i in range(2)]
        mds = Dataset(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        with pytest.raises(ValueError):  # noqa: PT011
            mds.groupby('x')

    def test_array_groupby_non_scalar(self):
        arrays = [np.array([(1+i, i), (2+i, i), (3+i, i)]) for i in range(2)]
        mds = Dataset(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        with pytest.raises(ValueError):  # noqa: PT011
            mds.groupby('x')

    def test_array_points_iloc_index_row(self):
        arrays = [np.array([(1+i, i), (2+i, i), (3+i, i)]) for i in range(2)]
        mds = Points(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        assert_element_equal(mds.iloc[1], Points([(2, 0)], ['x', 'y']))

    def test_array_points_iloc_slice_rows(self):
        arrays = [np.array([(1+i, i), (2+i, i), (3+i, i)]) for i in range(2)]
        mds = Points(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        assert_element_equal(mds.iloc[2:4], Points([(3, 0), (2, 1)], ['x', 'y']))

    def test_array_points_iloc_slice_rows_no_start(self):
        arrays = [np.array([(1+i, i), (2+i, i), (3+i, i)]) for i in range(2)]
        mds = Points(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        assert_element_equal(mds.iloc[:4], Points([(1, 0), (2, 0), (3, 0), (2, 1)], ['x', 'y']))

    def test_array_points_iloc_slice_rows_no_stop(self):
        arrays = [np.array([(1+i, i), (2+i, i), (3+i, i)]) for i in range(2)]
        mds = Points(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        assert_element_equal(mds.iloc[2:], Points([(3, 0), (2, 1), (3, 1), (4, 1)], ['x', 'y']))

    def test_array_points_iloc_index_rows(self):
        arrays = [np.array([(1+i, i), (2+i, i), (3+i, i)]) for i in range(2)]
        mds = Points(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        assert_element_equal(mds.iloc[[1, 3, 4]], Points([(2, 0), (2, 1), (3, 1)], ['x', 'y']))

    def test_array_points_iloc_index_rows_index_cols(self):
        arrays = [np.array([(1+i, i), (2+i, i), (3+i, i)]) for i in range(2)]
        mds = Points(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        assert mds.iloc[3, 0] == 2
        assert mds.iloc[3, 1] == 1

    def test_multi_polygon_iloc_index_row(self):
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        holes = [
            [[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]],
            []
        ]
        poly = Polygons([{'x': xs, 'y': ys, 'holes': holes, 'z': 1},
                         {'x': xs[::-1], 'y': ys[::-1], 'z': 2}],
                        ['x', 'y'], 'z', datatype=[self.datatype])
        expected = Polygons([{'x': xs, 'y': ys, 'holes': holes, 'z': 1}],
                            ['x', 'y'], 'z', datatype=[self.datatype])
        assert poly.interface is self.interface
        assert_element_equal(poly.iloc[0], expected)

    def test_multi_polygon_iloc_index_rows(self):
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        holes = [
            [[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]],
            []
        ]
        poly = Polygons([{'x': xs, 'y': ys, 'holes': holes, 'z': 1},
                         {'x': xs[::-1], 'y': ys[::-1], 'z': 2},
                         {'x': xs, 'y': ys, 'z': 3}],
                        ['x', 'y'], 'z', datatype=[self.datatype])
        expected = Polygons([{'x': xs, 'y': ys, 'holes': holes, 'z': 1},
                             {'x': xs, 'y': ys, 'holes': holes, 'z': 3}],
                            ['x', 'y'], 'z', datatype=[self.datatype])
        assert poly.interface is self.interface
        assert_element_equal(poly.iloc[[0, 2]], expected)

    def test_multi_polygon_iloc_slice_rows(self):
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        holes = [
            [[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]],
            []
        ]
        poly = Polygons([{'x': xs, 'y': ys, 'holes': holes, 'z': 1},
                         {'x': xs[::-1], 'y': ys[::-1], 'z': 2},
                         {'x': xs, 'y': ys, 'z': 3}],
                        ['x', 'y'], 'z', datatype=[self.datatype])
        expected = Polygons([{'x': xs[::-1], 'y': ys[::-1], 'z': 2},
                             {'x': xs, 'y': ys, 'holes': holes, 'z': 3}],
                            ['x', 'y'], 'z', datatype=[self.datatype])
        assert poly.interface is self.interface
        assert_element_equal(poly.iloc[1:3], expected)

    def test_polygon_expanded_values(self):
        xs = [1, 2, 3]
        ys = [2, 0, 7]
        poly = Polygons([{'x': xs, 'y': ys, 'z': 1}],
                        ['x', 'y'], 'z', datatype=[self.datatype])
        assert_data_equal(poly.dimension_values(0), np.array([1, 2, 3, 1]))
        assert_data_equal(poly.dimension_values(1), np.array([2, 0, 7, 2]))
        assert_data_equal(poly.dimension_values(2), np.array([1, 1, 1, 1]))

    def test_polygons_expanded_values(self):
        xs = [1, 2, 3]
        ys = [2, 0, 7]
        poly = Polygons([{'x': xs, 'y': ys, 'z': 1},
                         {'x': xs, 'y': ys, 'z': 2}],
                        ['x', 'y'], 'z', datatype=[self.datatype])
        assert_data_equal(poly.dimension_values(0), np.array([1, 2, 3, 1, np.nan, 1, 2, 3, 1]))
        assert_data_equal(poly.dimension_values(1), np.array([2, 0, 7, 2, np.nan, 2, 0, 7, 2]))
        assert_data_equal(poly.dimension_values(2), np.array([1, 1, 1, 1, np.nan, 2, 2, 2, 2]))

    def test_multi_polygon_expanded_values(self):
        xs = [1, 2, 3, np.nan, 1, 2, 3]
        ys = [2, 0, 7, np.nan, 2, 0, 7]
        poly = Polygons([{'x': xs, 'y': ys, 'z': 1}],
                        ['x', 'y'], 'z', datatype=[self.datatype])
        assert_data_equal(poly.dimension_values(0), np.array([1, 2, 3, 1, np.nan, 1, 2, 3, 1]))
        assert_data_equal(poly.dimension_values(1), np.array([2, 0, 7, 2, np.nan, 2, 0, 7, 2]))
        assert_data_equal(poly.dimension_values(2), np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))

    def test_polygon_get_holes(self):
        xs = [1, 2, 3]
        ys = [2, 0, 7]
        holes = [
            [[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5), (2.1, 4.5)]]
        ]
        poly = Polygons([{'x': xs, 'y': ys, 'holes': holes, 'z': 1},
                         {'x': xs[::-1], 'y': ys[::-1], 'z': 2}],
                        ['x', 'y'], 'z', datatype=[self.datatype])
        holes = [
            [[np.array([(1.5, 2), (2, 3), (1.6, 1.6), (1.5, 2)]), np.array(holes[0][1])]],
            [[]]
        ]
        assert poly.interface is self.interface
        np.testing.assert_equal(poly.holes(), holes)

    def test_multi_polygon_get_holes(self):
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        holes = [
            [[(1.5, 2), (2, 3), (1.6, 1.6), (1.5, 2)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]],
            []
        ]
        poly = Polygons([{'x': xs, 'y': ys, 'holes': holes, 'z': 1},
                         {'x': xs[::-1], 'y': ys[::-1], 'z': 2}],
                        ['x', 'y'], 'z', datatype=[self.datatype])
        holes = [
            [[np.array(holes[0][0]), np.array([(2.1, 4.5), (2.5, 5), (2.3, 3.5), (2.1, 4.5)])], []],
            [[], []]
        ]
        assert poly.interface is self.interface
        np.testing.assert_equal(poly.holes(), holes)

    def test_polygon_dtype(self):
        poly = Polygons([{'x': [1, 2, 3], 'y': [2, 0, 7]}], datatype=[self.datatype])
        assert poly.interface is self.interface
        assert poly.interface.dtype(poly, 'x') == np.dtype('int')

    def test_select_from_multi_polygons_with_scalar(self):
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        holes = [
            [[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]],
            []
        ]
        poly = Polygons([{'x': xs, 'y': ys, 'holes': holes, 'z': 1},
                         {'x': xs[::-1], 'y': ys[::-1], 'z': 2}],
                        ['x', 'y'], 'z', datatype=[self.datatype])
        expected = Polygons([{'x': xs[::-1], 'y': ys[::-1], 'z': 2}],
                            ['x', 'y'], 'z', datatype=[self.datatype])
        assert poly.interface is self.interface
        assert_element_equal(poly.select(z=2), expected)

    def test_select_from_multi_polygons_with_slice(self):
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        holes = [
            [[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]],
            []
        ]
        poly = Polygons([{'x': xs, 'y': ys, 'holes': holes, 'z': 1},
                         {'x': xs[::-1], 'y': ys[::-1], 'z': 2},
                         {'x': xs[:3], 'y': ys[:3], 'z': 3}],
                        ['x', 'y'], 'z', datatype=[self.datatype])
        expected = Polygons([{'x': xs[::-1], 'y': ys[::-1], 'z': 2},
                             {'x': xs[:3], 'y': ys[:3], 'z': 3}],
                            ['x', 'y'], 'z', datatype=[self.datatype])
        assert poly.interface is self.interface
        assert_element_equal(poly.select(z=(2, 4)), expected)

    def test_select_from_multi_polygons_with_list(self):
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        holes = [
            [[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]],
            []
        ]
        poly = Polygons([{'x': xs, 'y': ys, 'holes': holes, 'z': 1},
                         {'x': xs[::-1], 'y': ys[::-1], 'z': 2},
                         {'x': xs[:3], 'y': ys[:3], 'z': 3}],
                        ['x', 'y'], 'z', datatype=[self.datatype])
        expected = Polygons([{'x': xs, 'y': ys, 'holes': holes, 'z': 1},
                             {'x': xs[:3], 'y': ys[:3], 'z': 3}],
                            ['x', 'y'], 'z', datatype=[self.datatype])
        assert poly.interface is self.interface
        assert_element_equal(poly.select(z=[1, 3]), expected)

    def test_sort_by_value(self):
        path = Path([{'x': [1, 2, 3, 4, 5], 'y': [0, 0, 1, 1, 2], 'value': 1},
                     {'x': [5, 4, 3, 2, 1], 'y': [2, 2, 1, 1, 0], 'value': 0}],
                    vdims='value', datatype=[self.datatype])
        assert path.interface is self.interface
        sorted = Path([{'x': [5, 4, 3, 2, 1], 'y': [2, 2, 1, 1, 0], 'value': 0},
                     {'x': [1, 2, 3, 4, 5], 'y': [0, 0, 1, 1, 2], 'value': 1}], vdims='value')
        assert_element_equal(path.sort('value'), sorted)


class MultiBaseInterfaceTest(GeomTests):

    datatype = 'multitabular'
    interface = MultiInterface
    subtype = None

    __test__ = False

    def setup_method(self):
        logger = get_logger()
        self._log_level = logger.level
        get_logger().setLevel(logging.ERROR)
        self._subtypes = MultiInterface.subtypes
        MultiInterface.subtypes = [self.subtype]

    def teardown_method(self):
        MultiInterface.subtypes = self._subtypes
        get_logger().setLevel(self._log_level)


class MultiDictInterfaceTest(MultiBaseInterfaceTest):
    """
    Test of the MultiInterface.
    """

    datatype = 'multitabular'
    interface = MultiInterface
    subtype = 'dictionary'

    __test__ = True


def test_narwhals_multidict():
    import narwhals.stable.v2 as nw

    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    pd_el = Path(df, kdims=["A", "B"], vdims=[])
    nw_el = Path(nw.from_native(df), kdims=["A", "B"], vdims=[])
    pd.testing.assert_frame_equal(pd_el.data[0], nw_el.data[0].to_pandas())
