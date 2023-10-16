"""
Unit tests of Path types.
"""
import numpy as np

from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.element.comparison import ComparisonTestCase


class PathTests(ComparisonTestCase):

    def test_multi_path_list_constructor(self):
        path = Path([[(0, 1), (1, 2)], [(2, 3), (3, 4)]])
        self.assertTrue(path.interface.multi)
        self.assertEqual(path.dimension_values(0), np.array([
            0, 1, np.nan, 2, 3]))
        self.assertEqual(path.dimension_values(1), np.array([
            1, 2, np.nan, 3, 4]))

    def test_multi_path_cast_path(self):
        path = Path([[(0, 1), (1, 2)], [(2, 3), (3, 4)]])
        path2 = Path(path)
        self.assertTrue(path2.interface.multi)
        self.assertEqual(path2.dimension_values(0), np.array([
            0, 1, np.nan, 2, 3]))
        self.assertEqual(path2.dimension_values(1), np.array([
            1, 2, np.nan, 3, 4]))

    def test_multi_path_tuple(self):
        path = Path(([0, 1], [[1, 3], [2, 4]]))
        self.assertTrue(path.interface.multi)
        self.assertEqual(path.dimension_values(0), np.array([
            0, 1, np.nan, 0, 1]))
        self.assertEqual(path.dimension_values(1), np.array([
            1, 2, np.nan, 3, 4]))

    def test_multi_path_unpack_single_paths(self):
        path = Path([Path([(0, 1), (1, 2)]), Path([(2, 3), (3, 4)])])
        self.assertTrue(path.interface.multi)
        self.assertEqual(path.dimension_values(0), np.array([
            0, 1, np.nan, 2, 3]))
        self.assertEqual(path.dimension_values(1), np.array([
            1, 2, np.nan, 3, 4]))

    def test_multi_path_unpack_multi_paths(self):
        path = Path([Path([[(0, 1), (1, 2)]]),
                     Path([[(2, 3), (3, 4)], [(4, 5), (5, 6)]])])
        self.assertTrue(path.interface.multi)
        self.assertEqual(path.dimension_values(0), np.array([
            0, 1, np.nan, 2, 3, np.nan, 4, 5]))
        self.assertEqual(path.dimension_values(1), np.array([
            1, 2, np.nan, 3, 4, np.nan, 5, 6]))

    def test_single_path_list_constructor(self):
        path = Path([(0, 1), (1, 2), (2, 3), (3, 4)])
        self.assertEqual(path.dimension_values(0), np.array([
            0, 1, 2, 3]))
        self.assertEqual(path.dimension_values(1), np.array([
            1, 2, 3, 4]))

    def test_single_path_tuple_constructor(self):
        path = Path(([0, 1, 2, 3], [1, 2, 3, 4]))
        self.assertEqual(path.dimension_values(0), np.array([
            0, 1, 2, 3]))
        self.assertEqual(path.dimension_values(1), np.array([
            1, 2, 3, 4]))

    def test_multi_path_list_split(self):
        path = Path([[(0, 1), (1, 2)], [(2, 3), (3, 4)]])
        subpaths = path.split()
        self.assertEqual(len(subpaths), 2)
        self.assertEqual(subpaths[0], Path([(0, 1), (1, 2)]))
        self.assertEqual(subpaths[1], Path([(2, 3), (3, 4)]))

    def test_single_path_split(self):
        path = Path(([0, 1, 2, 3], [1, 2, 3, 4]))
        self.assertEqual(path, path.split()[0])

    def test_dataset_groupby_path(self):
        ds = Dataset([(0, 0, 1), (0, 1, 2), (1, 2, 3), (1, 3, 4)], ['group', 'x', 'y'])
        subpaths = ds.groupby('group', group_type=Path)
        self.assertEqual(len(subpaths), 2)
        self.assertEqual(subpaths[0], Path([(0, 1), (1, 2)]))
        self.assertEqual(subpaths[1], Path([(2, 3), (3, 4)]))


class PolygonsTests(ComparisonTestCase):

    def setUp(self):
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
        self.assertTrue(self.single_poly.interface.has_holes(self.single_poly))
        paths = self.single_poly.split(datatype='array')
        holes = self.single_poly.interface.holes(self.single_poly)
        self.assertEqual(len(paths), len(holes))
        self.assertEqual(len(holes), 1)
        self.assertEqual(len(holes[0]), 1)
        self.assertEqual(len(holes[0][0]), 2)

    def test_multi_poly_holes_match(self):
        self.assertTrue(self.multi_poly.interface.has_holes(self.multi_poly))
        paths = self.multi_poly.split(datatype='array')
        holes = self.multi_poly.interface.holes(self.multi_poly)
        self.assertEqual(len(paths), len(holes))
        self.assertEqual(len(holes), 1)
        self.assertEqual(len(holes[0]), 2)
        self.assertEqual(len(holes[0][0]), 2)
        self.assertEqual(len(holes[0][1]), 0)

    def test_multi_poly_empty_holes(self):
        poly = Polygons([])
        self.assertFalse(poly.interface.has_holes(poly))
        self.assertEqual(poly.interface.holes(poly), [])

    def test_multi_poly_no_holes_match(self):
        self.assertFalse(self.multi_poly_no_hole.interface.has_holes(self.multi_poly_no_hole))
        paths = self.multi_poly_no_hole.split(datatype='array')
        holes = self.multi_poly_no_hole.interface.holes(self.multi_poly_no_hole)
        self.assertEqual(len(paths), len(holes))
        self.assertEqual(len(holes), 1)
        self.assertEqual(len(holes[0]), 2)
        self.assertEqual(len(holes[0][0]), 0)
        self.assertEqual(len(holes[0][1]), 0)

    def test_distinct_multi_poly_holes_match(self):
        self.assertTrue(self.distinct_polys.interface.has_holes(self.distinct_polys))
        paths = self.distinct_polys.split(datatype='array')
        holes = self.distinct_polys.interface.holes(self.distinct_polys)
        self.assertEqual(len(paths), len(holes))
        self.assertEqual(len(holes), 2)
        self.assertEqual(len(holes[0]), 2)
        self.assertEqual(len(holes[0][0]), 2)
        self.assertEqual(len(holes[0][1]), 0)
        self.assertEqual(len(holes[1]), 1)
        self.assertEqual(len(holes[1][0]), 0)

    def test_single_poly_hole_validation(self):
        xs = [1, 2, 3]
        ys = [2, 0, 7]
        with self.assertRaises(DataError):
            Polygons([{'x': xs, 'y': ys, 'holes': [[], []]}])

    def test_multi_poly_hole_validation(self):
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        with self.assertRaises(DataError):
            Polygons([{'x': xs, 'y': ys, 'holes': [[]]}])


class EllipseTests(ComparisonTestCase):

    def setUp(self):
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
        self.assertEqual(len(ellipse.data[0]), 100)

    def test_ellipse_simple_constructor_pentagon(self):
        ellipse = Ellipse(0,0,1, samples=6)
        self.assertEqual(np.allclose(ellipse.data[0], self.pentagon), True)

    def test_ellipse_tuple_constructor_squashed(self):
        ellipse = Ellipse(0,0,(1,2), samples=6)
        self.assertEqual(np.allclose(ellipse.data[0], self.squashed), True)

    def test_ellipse_simple_constructor_squashed_aspect(self):
        ellipse = Ellipse(0,0,2, aspect=0.5, samples=6)
        self.assertEqual(np.allclose(ellipse.data[0], self.squashed), True)


class BoxTests(ComparisonTestCase):

    def setUp(self):
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
        self.assertEqual(np.allclose(box.data[0], self.rotated_square), True)


    def test_box_tuple_constructor_rotated(self):
        box = Box(0,0,(2,1), orientation=np.pi/8)
        self.assertEqual(np.allclose(box.data[0], self.rotated_rect), True)

    def test_box_aspect_constructor_rotated(self):
        box = Box(0,0,1, aspect=2, orientation=np.pi/8)
        self.assertEqual(np.allclose(box.data[0], self.rotated_rect), True)
