import numpy as np
import pytest

import holoviews as hv
from holoviews.testing import assert_element_equal


class OverlayTest:

    def setup_method(self):
        self.data1 = np.zeros((10, 2))
        self.data2 = np.ones((10, 2))
        self.data3 = np.ones((10, 2)) * 2

        self.view1 = hv.Element(self.data1, label='view1')
        self.view2 = hv.Element(self.data2, label='view2')
        self.view3 = hv.Element(self.data3, label='view3')

    def test_overlay(self):
        hv.NdOverlay(list(enumerate([self.view1, self.view2, self.view3])))

    def test_overlay_iter(self):
        views = [self.view1, self.view2, self.view3]
        overlay = hv.NdOverlay(list(enumerate(views)))
        for el, v in zip(overlay, views, strict=True):
            assert_element_equal(el, v)

    def test_overlay_iterable(self):
        # Related to https://github.com/holoviz/holoviews/issues/5315
        c1 = hv.Curve([0, 1])
        c2 = hv.Curve([10, 20])
        hv.Overlay({'a': c1, 'b': c2}.values())

    def test_overlay_integer_indexing(self):
        overlay = hv.NdOverlay(list(enumerate([self.view1, self.view2, self.view3])))
        assert_element_equal(overlay[0], self.view1)
        assert_element_equal(overlay[1], self.view2)
        assert_element_equal(overlay[2], self.view3)
        with pytest.raises(KeyError):
            overlay[3]
