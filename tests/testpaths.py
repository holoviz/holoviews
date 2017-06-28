"""
Unit tests of Path types.
"""
import numpy as np
from holoviews import Ellipse, Box
from holoviews.element.comparison import ComparisonTestCase


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
