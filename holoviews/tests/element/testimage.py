"""
Unit tests of Image elements
"""

import numpy as np
import holoviews as hv
from holoviews.element import  Image, Curve
from ..utils import LoggingComparisonTestCase


class TestImage(LoggingComparisonTestCase):

    def setUp(self):
        super().setUp()
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

    def test_image_rtol_failure(self):
        vals = np.random.rand(20,20)
        xs = np.linspace(0,10,20)
        ys = np.linspace(0,10,20)
        ys[-1] += 0.1
        Image({'vals':vals, 'xs':xs, 'ys':ys}, ['xs','ys'], 'vals')
        substr = ('set a higher tolerance on hv.config.image_rtol or '
                  'the rtol parameter in the Image constructor.')
        self.log_handler.assertEndsWith('WARNING', substr)

    def test_image_rtol_constructor(self):
        vals = np.random.rand(20,20)
        xs = np.linspace(0,10,20)
        ys = np.linspace(0,10,20)
        ys[-1] += 0.01
        Image({'vals':vals, 'xs':xs, 'ys':ys}, ['xs','ys'], 'vals', rtol=10e-2)

    def test_image_rtol_config(self):
        vals = np.random.rand(20,20)
        xs = np.linspace(0,10,20)
        ys = np.linspace(0,10,20)
        ys[-1] += 0.001
        image_rtol = hv.config.image_rtol
        hv.config.image_rtol = 10e-3
        Image({'vals':vals, 'xs':xs, 'ys':ys}, ['xs','ys'], 'vals')
        hv.config.image_rtol = image_rtol

    def test_image_clone(self):
        vals = np.random.rand(20,20)
        xs = np.linspace(0,10,20)
        ys = np.linspace(0,10,20)
        ys[-1] += 0.001
        img = Image({'vals':vals, 'xs':xs, 'ys':ys}, ['xs','ys'], 'vals', rtol=10e-3)
        self.assertEqual(img.clone().rtol, 10e-3)

    def test_image_curvilinear_coords_error(self):
        x = np.arange(-1, 1, 0.1)
        y = np.arange(-1, 1, 0.1)
        X, Y = np.meshgrid(x, y)
        Z = np.sqrt(X**2 + Y**2) * np.cos(X)
        with self.assertRaises(ValueError):
            Image((X, Y, Z))
