"""
Test cases for boundingregion
"""

from holoviews.core import AARectangle, BoundingBox


class TestAARectangle:
    def setup_method(self):
        self.left    = -0.1
        self.bottom  = -0.2
        self.right   =  0.3
        self.top     =  0.4
        self.lbrt = (self.left,self.bottom,self.right,self.top)
        self.aar1 = AARectangle((self.left,self.bottom),(self.right,self.top))
        self.aar2 = AARectangle((self.right,self.bottom),(self.left,self.top))

    def test_left(self):
        assert self.left == self.aar1.left()
    def test_right(self):
        assert self.right == self.aar1.right()
    def test_bottom(self):
        assert self.bottom == self.aar1.bottom()
    def test_top(self):
        assert self.top == self.aar1.top()
    def test_lbrt(self):
        assert self.lbrt == self.aar1.lbrt()
    def test_point_order(self):
        assert self.aar1.lbrt() == self.aar2.lbrt()


class TestBoundingBox:
    def setup_method(self):
        self.left    = -0.1
        self.bottom  = -0.2
        self.right   =  0.3
        self.top     =  0.4
        self.lbrt = (self.left,self.bottom,self.right,self.top)

        self.region = BoundingBox(points = ((self.left,self.bottom),(self.right,self.top)))
        self.xc,self.yc = self.region.aarect().centroid()

    def test_way_inside(self):
        assert self.region.contains(0, 0)
    def test_above(self):
        assert not self.region.contains(0, 1)
    def test_below(self):
        assert not self.region.contains(0, -1)
    def test_left_of(self):
        assert not self.region.contains(-1, 0)
    def test_right_of(self):
        assert not self.region.contains(1, 0)

    def test_centroid_x(self):
        assert self.xc == (self.left+self.right)/2.0
    def test_centroid_y(self):
        assert self.yc == (self.bottom+self.top)/2.0

    def test_left_boundary(self):
        assert self.region.contains(self.left, self.yc)
    def test_right_boundary(self):
        assert self.region.contains(self.right, self.yc)
    def test_bottom_boundary(self):
        assert self.region.contains(self.xc, self.bottom)
    def test_top_boundary(self):
        assert self.region.contains(self.xc, self.top)
