from __future__ import absolute_import, unicode_literals

import numpy as np

from holoviews.element import Polygons
from holoviews.plotting.mpl.util import polygons_to_path_patches

from .testplot import TestMPLPlot


class TestUtils(TestMPLPlot):

    def test_polygon_to_path_patches(self):
        xs = [1, 2, 3, np.nan, 3, 7, 6, np.nan, 0, 0, 0]
        ys = [2, 0, 7, np.nan, 2, 5, 7, np.nan, 0, 1, 0]

        holes = [
            [[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]],
            [],
            []
        ]
        polys = Polygons([{'x': xs, 'y': ys, 'holes': holes}])
        paths = polygons_to_path_patches(polys)


        self.assertEqual(len(paths), 1)
        self.assertEqual(len(paths[0]), 3)
        self.assertEqual(paths[0][0].get_path().vertices, np.array([
            (1, 2), (2, 0), (3, 7), (1, 2),
            (1.5, 2), (2, 3), (1.6, 1.6), (1.5, 2),
            (2.1, 4.5), (2.5, 5), (2.3, 3.5), (2.1, 4.5)]))
        self.assertEqual(paths[0][0].get_path().codes, np.array([1, 2, 2, 79, 1, 2, 2, 79, 1, 2, 2, 79], dtype='uint8'))
        self.assertEqual(paths[0][1].get_path().vertices, np.array([(3, 2), (7, 5), (6, 7),  (3, 2),]))
        self.assertEqual(paths[0][1].get_path().codes, np.array([1, 2, 2, 79], dtype='uint8'))
        self.assertEqual(paths[0][2].get_path().vertices, np.array([(0, 0), (0, 1), (0, 0)]))
        self.assertEqual(paths[0][1].get_path().codes, np.array([1, 2, 2, 79], dtype='uint8'))
