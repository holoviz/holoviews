"""
Unit tests of Image elements
"""

import numpy as np
from holoviews.element import  Image, Curve
from holoviews.element.comparison import ComparisonTestCase


class TestImage(ComparisonTestCase):

    def setUp(self):
        self.array1 = np.array([(0, 1, 2), (3, 4, 5)])

    def test_image_init(self):
        image = Image(self.array1)
        self.assertEqual(image.xdensity, 3)
        self.assertEqual(image.ydensity, 2)

    def test_image_index(self):
        image = Image(self.array1)
        self.assertEqual(image[-.33, -0.25], 3)


    def test_image_sample(self):
        image = Image(self.array1)
        self.assertEqual(image.sample(y=0.25),
                         Curve(np.array([(-0.333333, 0), (0, 1), (0.333333, 2)]),
                               kdims=['x'], vdims=['z']))

    def test_image_range_masked(self):
        arr = np.random.rand(10,10)-0.5
        arr = np.ma.masked_where(arr<=0, arr)
        rrange = Image(arr).range(2)
        self.assertEqual(rrange, (np.min(arr), np.max(arr)))

    def test_empty_image(self):
        Image([])
        Image(None)
        Image(np.array([]))
        Image(np.zeros((0, 0)))

