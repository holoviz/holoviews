import unittest

import numpy as np

from holoviews import Curve, Element, NdOverlay, Overlay


class CompositeTest(unittest.TestCase):
    "For testing of basic composite element types"

    def setUp(self):
        self.data1 = np.zeros((10, 2))
        self.data2 = np.ones((10, 2))
        self.data3 = np.ones((10, 2)) * 2

        self.view1 = Element(self.data1, label='view1')
        self.view2 = Element(self.data2, label='view2')
        self.view3 = Element(self.data3, label='view3')


class OverlayTest(CompositeTest):

    def test_overlay(self):
        NdOverlay(list(enumerate([self.view1, self.view2, self.view3])))

    def test_overlay_iter(self):
        views = [self.view1, self.view2, self.view3]
        overlay = NdOverlay(list(enumerate(views)))
        for el, v in zip(overlay, views, strict=None):
            self.assertEqual(el, v)

    def test_overlay_iterable(self):
        # Related to https://github.com/holoviz/holoviews/issues/5315
        c1 = Curve([0, 1])
        c2 = Curve([10, 20])
        Overlay({'a': c1, 'b': c2}.values())

    def test_overlay_integer_indexing(self):
        overlay = NdOverlay(list(enumerate([self.view1, self.view2, self.view3])))
        self.assertEqual(overlay[0], self.view1)
        self.assertEqual(overlay[1], self.view2)
        self.assertEqual(overlay[2], self.view3)
        try:
            overlay[3]
            raise AssertionError("Index should be out of range.")
        except KeyError: pass
