import unittest

import numpy as np

from holoviews import Overlay, Layer

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
        Overlay([self.view1, self.view2, self.view3])

    def test_overlay_iter(self):
        views = [self.view1, self.view2, self.view3]
        overlay = Overlay(views)
        for el, v in zip(overlay, views):
            self.assertEqual(el, v)

    def test_overlay_labels(self):
        views = [self.view1, self.view2, self.view3]
        overlay = Overlay(views)
        self.assertEqual(overlay.labels, [v.label for v in views])

    def test_overlay_set(self):
        new_layers = [self.view1, self.view3]
        overlay = Overlay([self.view2])
        overlay.set(new_layers)
        self.assertEqual(len(overlay), len(new_layers))


    def test_overlay_add(self):
        overlay = Overlay([self.view1])
        overlay.add(self.view2)


    def test_overlay_integer_indexing(self):
        overlay = Overlay([self.view1, self.view2, self.view3])
        self.assertEqual(overlay[0], self.view1)
        self.assertEqual(overlay[1], self.view2)
        self.assertEqual(overlay[2], self.view3)
        try:
            overlay[3]
            raise AssertionError("Index should be out of range.")
        except IndexError: pass


    def test_overlay_str_indexing(self):
        overlay = Overlay([self.view1, self.view2, self.view3])

        self.assertEqual(overlay[self.view1.label], self.view1)
        self.assertEqual(overlay[self.view2.label], self.view2)
        self.assertEqual(overlay[self.view3.label], self.view3)

        try:
            overlay['Invalid key']
            raise AssertionError("Index should be an invalid key.")
        except KeyError: pass