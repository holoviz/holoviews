"""
Unit tests of Raster elements
"""

import numpy as np
from holoviews.element import Raster, Image, Curve
from holoviews.element.comparison import ComparisonTestCase

class TestRaster(ComparisonTestCase):

    def setUp(self):
        self.array1 = np.array([(0, 1, 2), (3, 4, 5)])

    def test_raster_init(self):
        Raster(self.array1)

    def test_image_init(self):
        image = Image(self.array1)
        self.assertEqual(image.xdensity, 3)
        self.assertEqual(image.ydensity, 2)

    def test_raster_index(self):
        raster = Raster(self.array1)
        self.assertEqual(raster[0, 1], 3)

    def test_image_index(self):
        image = Image(self.array1)
        self.assertEqual(image[-.33, -0.25], 3)

    def test_raster_sample(self):
        raster = Raster(self.array1)
        self.assertEqual(raster.sample(y=0),
                         Curve(np.array([(0, 0), (1, 1), (2, 2)]),
                               kdims=['x'], vdims=['z']))

    def test_image_sample(self):
        image = Image(self.array1)
        self.assertEqual(image.sample(y=0.25),
                         Curve(np.array([(-0.333333, 0), (0, 1), (0.333333, 2)]),
                               kdims=['x'], vdims=['z']))
