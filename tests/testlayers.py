import unittest

import numpy as np

from holoviews import Layers, Layer

class CompositeTest(unittest.TestCase):
    "For testing of basic composite view types"

    def setUp(self):
        self.data1 = np.zeros((10, 2))
        self.data2 = np.ones((10, 2))
        self.data3 = np.ones((10, 2)) * 2

        self.view1 = Layer(self.data1, label='view1')
        self.view2 = Layer(self.data2, label='view2')
        self.view3 = Layer(self.data3, label='view3')


class OverlayTest(CompositeTest):

    def test_overlay(self):
        Layers([self.view1, self.view2, self.view3])

    def test_overlay_iter(self):
        views = [self.view1, self.view2, self.view3]
        overlay = Layers(views)
        for el, v in zip(overlay, views):
            self.assertEqual(el, v)

    def test_overlay_labels(self):
        views = [self.view1, self.view2, self.view3]
        overlay = Layers(views)
        self.assertEqual(overlay.labels, [v.label for v in views])


    def test_overlay_add(self):
        overlay = Layers([self.view1])
        overlay.add(self.view2)


    def test_overlay_integer_indexing(self):
        overlay = Layers([self.view1, self.view2, self.view3])
        self.assertEqual(overlay[0], self.view1)
        self.assertEqual(overlay[1], self.view2)
        self.assertEqual(overlay[2], self.view3)
        try:
            overlay[3]
            raise AssertionError("Index should be out of range.")
        except KeyError: pass
