import numpy as np

from holoviews import (HoloMap, NdOverlay, Image, Contours, Polygons, Points,
                       Histogram, Curve)
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (operation, transform, threshold,
                                         gradient, contours, histogram,
                                         interpolate_curve)

class ElementOperationTests(ComparisonTestCase):
    """
    Tests allowable data formats when constructing
    the basic Element types.
    """

    def test_operation_element(self):
        img = Image(np.random.rand(10, 10))
        op_img = operation(img, op=lambda x, k: x.clone(x.data*2))
        self.assertEqual(op_img, img.clone(img.data*2, group='Operation'))

    def test_operation_holomap(self):
        hmap = HoloMap({1: Image(np.random.rand(10, 10))})
        op_hmap = operation(hmap, op=lambda x, k: x.clone(x.data*2))
        self.assertEqual(op_hmap.last, hmap.last.clone(hmap.last.data*2, group='Operation'))

    def test_image_transform(self):
        img = Image(np.random.rand(10, 10))
        op_img = transform(img, operator=lambda x: x*2)
        self.assertEqual(op_img, img.clone(img.data*2, group='Transform'))

    def test_image_threshold(self):
        img = Image(np.array([[0, 1, 0], [3, 4, 5.]]))
        op_img = threshold(img)
        self.assertEqual(op_img, img.clone(np.array([[0, 1, 0], [1, 1, 1]]), group='Threshold'))

    def test_image_gradient(self):
        img = Image(np.array([[0, 1, 0], [3, 4, 5.], [6, 7, 8]]))
        op_img = gradient(img)
        self.assertEqual(op_img, img.clone(np.array([[3.162278, 3.162278], [3.162278, 3.162278]]), group='Gradient'))

    def test_image_contours(self):
        img = Image(np.array([[0, 1, 0], [3, 4, 5.], [6, 7, 8]]))
        op_contours = contours(img)
        ndoverlay = NdOverlay(None, kdims=['Levels'])
        ndoverlay[0.5] = Contours([[(-0.5,  0.416667), (-0.25, 0.5)], [(0.25, 0.5), (0.5, 0.45)]],
                                  group='Level', level=0.5, vdims=img.vdims)
        self.assertEqual(op_contours, img*ndoverlay)

    def test_image_contours_filled(self):
        img = Image(np.array([[0, 1, 0], [3, 4, 5.], [6, 7, 8]]))
        op_contours = contours(img, filled=True, levels=[2, 2.5])
        ndoverlay = NdOverlay(None, kdims=['Levels'])
        data = [[(0., 0.333333), (0.5, 0.3), (0.5, 0.25), (0., 0.25),
                 (-0.5, 0.08333333), (-0.5, 0.16666667), (0., 0.33333333)]]
        ndoverlay[0.5] = Polygons(data, group='Level', level=2, vdims=img.vdims)
        self.assertEqual(op_contours, img*ndoverlay)

    def test_points_histogram(self):
        points = Points([float(i) for i in range(10)])
        op_hist = histogram(points, num_bins=3)
        hist = Histogram(([0.1, 0.1, 0.133333], [0, 3, 6, 9]))
        self.assertEqual(op_hist, hist)

    def test_points_histogram_bin_range(self):
        points = Points([float(i) for i in range(10)])
        op_hist = histogram(points, num_bins=3, bin_range=(0, 3))
        hist = Histogram(([0.25, 0.25, 0.5], [0., 1., 2., 3.]))
        self.assertEqual(op_hist, hist)

    def test_points_histogram_not_normed(self):
        points = Points([float(i) for i in range(10)])
        op_hist = histogram(points, num_bins=3, normed=False)
        hist = Histogram(([3, 3, 4], [0, 3, 6, 9]))
        self.assertEqual(op_hist, hist)

    def test_points_histogram_weighted(self):
        points = Points([float(i) for i in range(10)])
        op_hist = histogram(points, num_bins=3, weight_dimension='y')
        hist = Histogram(([0.022222, 0.088889, 0.222222], [0, 3, 6, 9]), vdims=['y'])
        self.assertEqual(op_hist, hist)
    
    def test_points_histogram_mean_weighted(self):
        points = Points([float(i) for i in range(10)])
        op_hist = histogram(points, num_bins=3, weight_dimension='y', mean_weighted=True)
        hist = Histogram(([1.,  4., 7.5], [0, 3, 6, 9]), vdims=['y'])
        self.assertEqual(op_hist, hist)

    def test_interpolate_curve_pre(self):
        interpolated = interpolate_curve(Curve([0, 0.5, 1]), interpolation='steps-pre')
        curve = Curve([(0, 0), (0, 0.5), (1, 0.5), (1, 1), (2, 1)])
        self.assertEqual(interpolated, curve)

    def test_interpolate_curve_mid(self):
        interpolated = interpolate_curve(Curve([0, 0.5, 1]), interpolation='steps-mid')
        curve = Curve([(0, 0), (0.5, 0), (0.5, 0.5), (1.5, 0.5), (1.5, 1), (2, 1)])
        self.assertEqual(interpolated, curve)

    def test_interpolate_curve_post(self):
        interpolated = interpolate_curve(Curve([0, 0.5, 1]), interpolation='steps-post')
        curve = Curve([(0, 0), (1, 0), (1, 0.5), (2, 0.5), (2, 1)])
        self.assertEqual(interpolated, curve)
