"""
Tests for the spatialpandas interface.
"""
import numpy as np
import pandas as pd
import pytest

from holoviews import Dimension
from holoviews.core.data import (
    DaskSpatialPandasInterface,
    Dataset,
    SpatialPandasInterface,
)
from holoviews.core.data.interface import DataError
from holoviews.core.data.spatialpandas import get_value_array
from holoviews.element import Path, Points, Polygons
from holoviews.testing import assert_data_equal, assert_element_equal

from ...utils import optional_dependencies
from .test_multiinterface import GeomTests

spd, spd_skip = optional_dependencies("spatialpandas")
dd, dask_skip = optional_dependencies("dask.dataframe")

if spd:
    import spatialpandas.geometry as sgeom


class RoundTripTests:

    datatype = None

    interface = None

    __test__ = False

    def test_point_roundtrip(self):
        points = Points([{'x': 0, 'y': 1, 'z': 0},
                         {'x': 1, 'y': 0, 'z': 1}], ['x', 'y'],
                        'z', datatype=[self.datatype])
        assert isinstance(points.data.geometry.dtype, sgeom.PointDtype)
        roundtrip = points.clone(datatype=['multitabular'])
        assert roundtrip.interface.datatype == 'multitabular'
        expected = Points([{'x': 0, 'y': 1, 'z': 0},
                           {'x': 1, 'y': 0, 'z': 1}], ['x', 'y'],
                          'z', datatype=['multitabular'])
        assert_element_equal(roundtrip, expected)

    def test_multi_point_roundtrip(self):
        xs = [1, 2, 3, 2]
        ys = [2, 0, 7, 4]
        points = Points([{'x': xs, 'y': ys, 'z': 0},
                         {'x': xs[::-1], 'y': ys[::-1], 'z': 1}],
                        ['x', 'y'], 'z', datatype=[self.datatype])
        assert isinstance(points.data.geometry.dtype, sgeom.MultiPointDtype)
        roundtrip = points.clone(datatype=['multitabular'])
        assert roundtrip.interface.datatype == 'multitabular'
        expected = Points([{'x': xs, 'y': ys, 'z': 0},
                           {'x': xs[::-1], 'y': ys[::-1], 'z': 1}],
                          ['x', 'y'], 'z', datatype=['multitabular'])
        assert_element_equal(roundtrip, expected)

    def test_line_roundtrip(self):
        xs = [1, 2, 3]
        ys = [2, 0, 7]
        path = Path([{'x': xs, 'y': ys, 'z': 1},
                     {'x': xs[::-1], 'y': ys[::-1], 'z': 2}],
                    ['x', 'y'], 'z', datatype=[self.datatype])
        assert isinstance(path.data.geometry.dtype, sgeom.LineDtype)
        roundtrip = path.clone(datatype=['multitabular'])
        assert roundtrip.interface.datatype == 'multitabular'
        expected = Path([{'x': xs, 'y': ys, 'z': 1},
                         {'x': xs[::-1], 'y': ys[::-1], 'z': 2}],
                        ['x', 'y'], 'z', datatype=['multitabular'])
        assert_element_equal(roundtrip, expected)

    def test_multi_line_roundtrip(self):
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        path = Path([{'x': xs, 'y': ys, 'z': 0},
                     {'x': xs[::-1], 'y': ys[::-1], 'z': 1}],
                    ['x', 'y'], 'z', datatype=[self.datatype])
        assert isinstance(path.data.geometry.dtype, sgeom.MultiLineDtype)
        roundtrip = path.clone(datatype=['multitabular'])
        assert roundtrip.interface.datatype == 'multitabular'
        expected = Path([{'x': xs, 'y': ys, 'z': 0},
                         {'x': xs[::-1], 'y': ys[::-1], 'z': 1}],
                        ['x', 'y'], 'z', datatype=['multitabular'])
        assert_element_equal(roundtrip, expected)

    def test_polygon_roundtrip(self):
        xs = [1, 2, 3]
        ys = [2, 0, 7]
        poly = Polygons([{'x': xs, 'y': ys, 'z': 0},
                         {'x': xs[::-1], 'y': ys[::-1], 'z': 1}],
                        ['x', 'y'], 'z', datatype=[self.datatype])
        assert isinstance(poly.data.geometry.dtype, sgeom.PolygonDtype)
        roundtrip = poly.clone(datatype=['multitabular'])
        assert roundtrip.interface.datatype == 'multitabular'
        expected = Polygons([{'x': [*xs, 1], 'y': [*ys, 2], 'z': 0},
                             {'x': [3, *xs], 'y': [7, *ys], 'z': 1}],
                            ['x', 'y'], 'z', datatype=['multitabular'])
        assert_element_equal(roundtrip, expected)

    def test_multi_polygon_roundtrip(self):
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        holes = [
            [[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]],
            []
        ]
        poly = Polygons([{'x': xs, 'y': ys, 'holes': holes, 'z': 1},
                         {'x': xs[::-1], 'y': ys[::-1], 'z': 2}],
                        ['x', 'y'], 'z', datatype=[self.datatype])
        assert isinstance(poly.data.geometry.dtype, sgeom.MultiPolygonDtype)
        roundtrip = poly.clone(datatype=['multitabular'])
        assert roundtrip.interface.datatype == 'multitabular'
        expected = Polygons([{'x': [1, 2, 3, 1, np.nan, 6, 3, 7, 6],
                              'y': [2, 0, 7, 2, np.nan, 7, 2, 5, 7], 'holes': holes, 'z': 1},
                             {'x': [3, 7, 6, 3, np.nan, 3, 1, 2, 3],
                              'y': [2, 5, 7, 2, np.nan, 7, 2, 0, 7], 'z': 2}],
                            ['x', 'y'], 'z', datatype=['multitabular'])
        assert_element_equal(roundtrip, expected)



@spd_skip
class SpatialPandasTest(GeomTests, RoundTripTests):
    """
    Test of the SpatialPandasInterface.
    """

    datatype = 'spatialpandas'

    interface = SpatialPandasInterface

    __test__ = True

    def test_array_points_iloc_index_rows_index_cols(self):
        arrays = [np.array([(1+i, i), (2+i, i), (3+i, i)]) for i in range(2)]
        mds = Dataset(arrays, kdims=['x', 'y'], datatype=[self.datatype])
        assert mds.interface is self.interface
        with pytest.raises(DataError):
            mds.iloc[3, 0]

    def test_point_constructor(self):
        points = Points([{'x': 0, 'y': 1}, {'x': 1, 'y': 0}], ['x', 'y'],
                        datatype=[self.datatype])
        assert isinstance(points.data.geometry.dtype, sgeom.PointDtype)
        assert_data_equal(points.data.iloc[0, 0].flat_values, np.array([0, 1]))
        assert_data_equal(points.data.iloc[1, 0].flat_values, np.array([1, 0]))

    def test_multi_point_constructor(self):
        xs = [1, 2, 3, 2]
        ys = [2, 0, 7, 4]
        points = Points([{'x': xs, 'y': ys}, {'x': xs[::-1], 'y': ys[::-1]}], ['x', 'y'],
                        datatype=[self.datatype])
        assert isinstance(points.data.geometry.dtype, sgeom.MultiPointDtype)
        assert_data_equal(points.data.iloc[0, 0].buffer_values,
                         np.array([1, 2, 2, 0, 3, 7, 2, 4]))
        assert_data_equal(points.data.iloc[1, 0].buffer_values,
                         np.array([2, 4, 3, 7, 2, 0, 1, 2]))

    def test_line_constructor(self):
        xs = [1, 2, 3]
        ys = [2, 0, 7]
        path = Path([{'x': xs, 'y': ys}, {'x': xs[::-1], 'y': ys[::-1]}],
                    ['x', 'y'], datatype=[self.datatype])
        assert isinstance(path.data.geometry.dtype, sgeom.LineDtype)
        assert_data_equal(path.data.iloc[0, 0].buffer_values,
                         np.array([1, 2, 2, 0, 3, 7]))
        assert_data_equal(path.data.iloc[1, 0].buffer_values,
                         np.array([3, 7, 2, 0, 1, 2]))

    def test_multi_line_constructor(self):
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        path = Path([{'x': xs, 'y': ys}, {'x': xs[::-1], 'y': ys[::-1]}],
                    ['x', 'y'], datatype=[self.datatype])
        assert isinstance(path.data.geometry.dtype, sgeom.MultiLineDtype)
        assert_data_equal(path.data.iloc[0, 0].buffer_values,
                         np.array([1, 2, 2, 0, 3, 7, 6, 7, 7, 5, 3, 2]))
        assert_data_equal(path.data.iloc[1, 0].buffer_values,
                         np.array([3, 2, 7, 5, 6, 7, 3, 7, 2, 0, 1, 2]))

    def test_polygon_constructor(self):
        xs = [1, 2, 3]
        ys = [2, 0, 7]
        holes = [
            [[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]]
        ]
        path = Polygons([{'x': xs, 'y': ys, 'holes': holes}, {'x': xs[::-1], 'y': ys[::-1]}],
                        ['x', 'y'], datatype=[self.datatype])
        assert isinstance(path.data.geometry.dtype, sgeom.PolygonDtype)
        assert_data_equal(path.data.iloc[0, 0].buffer_values,
                         np.array([1., 2., 2., 0., 3., 7., 1., 2., 1.5, 2., 2., 3.,
                                   1.6, 1.6, 1.5, 2., 2.1, 4.5, 2.5, 5., 2.3, 3.5, 2.1, 4.5]))
        assert_data_equal(path.data.iloc[1, 0].buffer_values,
                         np.array([3, 7, 1, 2, 2, 0, 3, 7]))

    def test_multi_polygon_constructor(self):
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        holes = [
            [[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]],
            []
        ]
        path = Polygons([{'x': xs, 'y': ys, 'holes': holes},
                         {'x': xs[::-1], 'y': ys[::-1]}],
                        ['x', 'y'], datatype=[self.datatype])
        assert isinstance(path.data.geometry.dtype, sgeom.MultiPolygonDtype)
        assert_data_equal(path.data.iloc[0, 0].buffer_values,
                         np.array([1., 2., 2., 0., 3., 7., 1., 2., 1.5, 2., 2., 3., 1.6, 1.6,
                                   1.5, 2., 2.1, 4.5, 2.5, 5., 2.3, 3.5, 2.1, 4.5, 6., 7., 3.,
                                   2., 7., 5., 6., 7. ]))
        assert_data_equal(path.data.iloc[1, 0].buffer_values,
                         np.array([3, 2, 7, 5, 6, 7, 3, 2, 3, 7, 1, 2, 2, 0, 3, 7]))

    def test_geometry_array_constructor(self):
        polygons = sgeom.MultiPolygonArray([
            # First Element
            [[[0, 0, 1, 0, 2, 2, -1, 4, 0, 0],         # Filled quadrilateral (CCW order)
              [0.5, 1,  1, 2,  1.5, 1.5,  0.5, 1],     # Triangular hole (CW order)
              [0, 2, 0, 2.5, 0.5, 2.5, 0.5, 2, 0, 2]], # Rectangular hole (CW order)

             [[-0.5, 3, 1.5, 3, 1.5, 4, -0.5, 3]],],   # Filled triangle

            # Second Element
            [[[1.25, 0, 1.25, 2, 4, 2, 4, 0, 1.25, 0],          # Filled rectangle (CCW order)
              [1.5, 0.25, 3.75, 0.25, 3.75, 1.75, 1.5, 1.75, 1.5, 0.25]],]
        ]) # Rectangular hole (CW order)

        path = Polygons(polygons)
        assert isinstance(path.data.geometry.dtype, sgeom.MultiPolygonDtype)


@spd_skip
@dask_skip
class DaskSpatialPandasTest(GeomTests, RoundTripTests):
    """
    Test of the DaskSpatialPandasInterface.
    """

    datatype = 'dask_spatialpandas'

    interface = DaskSpatialPandasInterface

    __test__ = True

    def test_array_points_iloc_index_row(self):
        pytest.skip("Not supported")

    def test_array_points_iloc_index_rows(self):
        pytest.skip("Not supported")

    def test_array_points_iloc_index_rows_index_cols(self):
        pytest.skip("Not supported")

    def test_array_points_iloc_slice_rows(self):
        pytest.skip("Not supported")

    def test_array_points_iloc_slice_rows_no_start(self):
        pytest.skip("Not supported")

    def test_array_points_iloc_slice_rows_no_end(self):
        pytest.skip("Not supported")

    def test_array_points_iloc_slice_rows_no_stop(self):
        pytest.skip("Not supported")

    def test_multi_polygon_iloc_index_row(self):
        pytest.skip("Not supported")

    def test_multi_polygon_iloc_index_rows(self):
        pytest.skip("Not supported")

    def test_multi_polygon_iloc_slice_rows(self):
        pytest.skip("Not supported")

    def test_dict_dataset_add_dimension_values(self):
        pytest.skip("Not supported")

    def test_sort_by_value(self):
        pytest.skip("Not supported")



def test_regression_get_value_array():
    # See: https://github.com/holoviz/holoviews/pull/6519
    df = pd.DataFrame(
        {"Longitude": [0, 1, 2], "Latitude": [0, 1, 2], "Sensor": ["S1", "S2", "S1"]}
    )
    result = get_value_array(df, Dimension("Sensor"), False, None, None, None)

    np.testing.assert_array_equal(df.Sensor, result)
