"""
Unit tests of Path types.
"""
import numpy as np
import pytest

from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.testing import assert_data_equal, assert_element_equal


class PathTests:

    def test_multi_path_list_constructor(self):
        path = Path([[(0, 1), (1, 2)], [(2, 3), (3, 4)]])
        assert path.interface.multi
        assert_data_equal(path.dimension_values(0), np.array([
            0, 1, np.nan, 2, 3]))
        assert_data_equal(path.dimension_values(1), np.array([
            1, 2, np.nan, 3, 4]))

    def test_multi_path_cast_path(self):
        path = Path([[(0, 1), (1, 2)], [(2, 3), (3, 4)]])
        path2 = Path(path)
        assert path2.interface.multi
        assert_data_equal(path2.dimension_values(0), np.array([
            0, 1, np.nan, 2, 3]))
        assert_data_equal(path2.dimension_values(1), np.array([
            1, 2, np.nan, 3, 4]))

    def test_multi_path_tuple(self):
        path = Path(([0, 1], [[1, 3], [2, 4]]))
        assert path.interface.multi
        assert_data_equal(path.dimension_values(0), np.array([
            0, 1, np.nan, 0, 1]))
        assert_data_equal(path.dimension_values(1), np.array([
            1, 2, np.nan, 3, 4]))

    def test_multi_path_unpack_single_paths(self):
        path = Path([Path([(0, 1), (1, 2)]), Path([(2, 3), (3, 4)])])
        assert path.interface.multi
        assert_data_equal(path.dimension_values(0), np.array([
            0, 1, np.nan, 2, 3]))
        assert_data_equal(path.dimension_values(1), np.array([
            1, 2, np.nan, 3, 4]))

    def test_multi_path_unpack_multi_paths(self):
        path = Path([Path([[(0, 1), (1, 2)]]),
                     Path([[(2, 3), (3, 4)], [(4, 5), (5, 6)]])])
        assert path.interface.multi
        assert_data_equal(path.dimension_values(0), np.array([
            0, 1, np.nan, 2, 3, np.nan, 4, 5]))
        assert_data_equal(path.dimension_values(1), np.array([
            1, 2, np.nan, 3, 4, np.nan, 5, 6]))

    def test_single_path_list_constructor(self):
        path = Path([(0, 1), (1, 2), (2, 3), (3, 4)])
        assert_data_equal(path.dimension_values(0), np.array([
            0, 1, 2, 3]))
        assert_data_equal(path.dimension_values(1), np.array([
            1, 2, 3, 4]))

    def test_single_path_tuple_constructor(self):
        path = Path(([0, 1, 2, 3], [1, 2, 3, 4]))
        assert_data_equal(path.dimension_values(0), np.array([
            0, 1, 2, 3]))
        assert_data_equal(path.dimension_values(1), np.array([
            1, 2, 3, 4]))

    def test_multi_path_list_split(self):
        path = Path([[(0, 1), (1, 2)], [(2, 3), (3, 4)]])
        subpaths = path.split()
        assert len(subpaths) == 2
        assert_element_equal(subpaths[0], Path([(0, 1), (1, 2)]))
        assert_element_equal(subpaths[1], Path([(2, 3), (3, 4)]))

    def test_single_path_split(self):
        path = Path(([0, 1, 2, 3], [1, 2, 3, 4]))
        assert_element_equal(path, path.split()[0])

    def test_dataset_groupby_path(self):
        ds = Dataset([(0, 0, 1), (0, 1, 2), (1, 2, 3), (1, 3, 4)], ['group', 'x', 'y'])
        subpaths = ds.groupby('group', group_type=Path)
        assert len(subpaths) == 2
        assert_element_equal(subpaths[0], Path([(0, 1), (1, 2)]))
        assert_element_equal(subpaths[1], Path([(2, 3), (3, 4)]))


class PolygonsTests:

    def setup_method(self):
        xs = [1, 2, 3]
        ys = [2, 0, 7]
        holes = [[[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]]]
        self.single_poly = Polygons([{'x': xs, 'y': ys, 'holes': holes}])

        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        holes = [
            [[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]],
            []
        ]
        self.multi_poly = Polygons([{'x': xs, 'y': ys, 'holes': holes}])
        self.multi_poly_no_hole = Polygons([{'x': xs, 'y': ys}])

        self.distinct_polys = Polygons([
            {'x': xs, 'y': ys, 'holes': holes, 'value': 0},
            {'x': [4, 6, 6], 'y': [0, 2, 1], 'value': 1}], vdims='value')

    def test_single_poly_holes_match(self):
        assert self.single_poly.interface.has_holes(self.single_poly)
        paths = self.single_poly.split(datatype='array')
        holes = self.single_poly.interface.holes(self.single_poly)
        assert len(paths) == len(holes)
        assert len(holes) == 1
        assert len(holes[0]) == 1
        assert len(holes[0][0]) == 2

    def test_multi_poly_holes_match(self):
        assert self.multi_poly.interface.has_holes(self.multi_poly)
        paths = self.multi_poly.split(datatype='array')
        holes = self.multi_poly.interface.holes(self.multi_poly)
        assert len(paths) == len(holes)
        assert len(holes) == 1
        assert len(holes[0]) == 2
        assert len(holes[0][0]) == 2
        assert len(holes[0][1]) == 0

    def test_multi_poly_empty_holes(self):
        poly = Polygons([])
        assert not poly.interface.has_holes(poly)
        assert poly.interface.holes(poly) == []

    def test_multi_poly_no_holes_match(self):
        assert not self.multi_poly_no_hole.interface.has_holes(self.multi_poly_no_hole)
        paths = self.multi_poly_no_hole.split(datatype='array')
        holes = self.multi_poly_no_hole.interface.holes(self.multi_poly_no_hole)
        assert len(paths) == len(holes)
        assert len(holes) == 1
        assert len(holes[0]) == 2
        assert len(holes[0][0]) == 0
        assert len(holes[0][1]) == 0

    def test_distinct_multi_poly_holes_match(self):
        assert self.distinct_polys.interface.has_holes(self.distinct_polys)
        paths = self.distinct_polys.split(datatype='array')
        holes = self.distinct_polys.interface.holes(self.distinct_polys)
        assert len(paths) == len(holes)
        assert len(holes) == 2
        assert len(holes[0]) == 2
        assert len(holes[0][0]) == 2
        assert len(holes[0][1]) == 0
        assert len(holes[1]) == 1
        assert len(holes[1][0]) == 0

    def test_single_poly_hole_validation(self):
        xs = [1, 2, 3]
        ys = [2, 0, 7]
        with pytest.raises(DataError):
            Polygons([{'x': xs, 'y': ys, 'holes': [[], []]}])

    def test_multi_poly_hole_validation(self):
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        with pytest.raises(DataError):
            Polygons([{'x': xs, 'y': ys, 'holes': [[]]}])


class EllipseTests:

    def setup_method(self):
        self.pentagon = np.array([[  0.00000000e+00,   5.00000000e-01],
                                  [  4.75528258e-01,   1.54508497e-01],
                                  [  2.93892626e-01,  -4.04508497e-01],
                                  [ -2.93892626e-01,  -4.04508497e-01],
                                  [ -4.75528258e-01,   1.54508497e-01],
                                  [ -1.22464680e-16,   5.00000000e-01]])

        self.squashed = np.array([[  0.00000000e+00,   1.00000000e+00],
                                  [  4.75528258e-01,   3.09016994e-01],
                                  [  2.93892626e-01,  -8.09016994e-01],
                                  [ -2.93892626e-01,  -8.09016994e-01],
                                  [ -4.75528258e-01,   3.09016994e-01],
                                  [ -1.22464680e-16,   1.00000000e+00]])


    def test_ellipse_simple_constructor(self):
        ellipse = Ellipse(0,0,1, samples=100)
        assert len(ellipse.data[0]) == 100

    def test_ellipse_simple_constructor_pentagon(self):
        ellipse = Ellipse(0,0,1, samples=6)
        assert np.allclose(ellipse.data[0], self.pentagon)

    def test_ellipse_tuple_constructor_squashed(self):
        ellipse = Ellipse(0,0,(1,2), samples=6)
        assert np.allclose(ellipse.data[0], self.squashed)

    def test_ellipse_simple_constructor_squashed_aspect(self):
        ellipse = Ellipse(0,0,2, aspect=0.5, samples=6)
        assert np.allclose(ellipse.data[0], self.squashed)


class BoxTests:

    def setup_method(self):
        self.rotated_square = np.array([[-0.27059805, -0.65328148],
                                        [-0.65328148,  0.27059805],
                                        [ 0.27059805,  0.65328148],
                                        [ 0.65328148, -0.27059805],
                                        [-0.27059805, -0.65328148]])

        self.rotated_rect = np.array([[-0.73253782, -0.8446232 ],
                                      [-1.11522125,  0.07925633],
                                      [ 0.73253782,  0.8446232 ],
                                      [ 1.11522125, -0.07925633],
                                      [-0.73253782, -0.8446232 ]])

    def test_box_simple_constructor_rotated(self):
        box = Box(0,0,1, orientation=np.pi/8)
        assert np.allclose(box.data[0], self.rotated_square)

    def test_box_tuple_constructor_rotated(self):
        box = Box(0,0,(2,1), orientation=np.pi/8)
        assert np.allclose(box.data[0], self.rotated_rect)

    def test_box_aspect_constructor_rotated(self):
        box = Box(0,0,1, aspect=2, orientation=np.pi/8)
        assert np.allclose(box.data[0], self.rotated_rect)
