import numpy as np
from nose.plugins.attrib import attr

from holoviews import (HoloMap, NdOverlay, NdLayout, GridSpace, Image,
                       Contours, Polygons, Points, Histogram, Curve, Area)
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (operation, transform, threshold,
                                         gradient, contours, histogram,
                                         interpolate_curve)

class OperationTests(ComparisonTestCase):
    """
    Tests allowable data formats when constructing
    the basic Element types.
    """

    def test_operation_element(self):
        img = Image(np.random.rand(10, 10))
        op_img = operation(img, op=lambda x, k: x.clone(x.data*2))
        self.assertEqual(op_img, img.clone(img.data*2, group='Operation'))

    def test_operation_ndlayout(self):
        ndlayout = NdLayout({i: Image(np.random.rand(10, 10)) for i in range(10)})
        op_ndlayout = operation(ndlayout, op=lambda x, k: x.clone(x.data*2))
        doubled = ndlayout.clone({k: v.clone(v.data*2, group='Operation')
                                  for k, v in ndlayout.items()})
        self.assertEqual(op_ndlayout, doubled)

    def test_operation_grid(self):
        grid = GridSpace({i: Image(np.random.rand(10, 10)) for i in range(10)}, kdims=['X'])
        op_grid = operation(grid, op=lambda x, k: x.clone(x.data*2))
        doubled = grid.clone({k: v.clone(v.data*2, group='Operation')
                              for k, v in grid.items()})
        self.assertEqual(op_grid, doubled)

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

    @attr(optional=1) # Requires matplotlib
    def test_image_contours(self):
        img = Image(np.array([[0, 1, 0], [3, 4, 5.], [6, 7, 8]]))
        op_contours = contours(img, levels=[0.5])
        contour = Contours([[(-0.5,  0.416667, 0.5), (-0.25, 0.5, 0.5)],
                             [(0.25, 0.5, 0.5), (0.5, 0.45, 0.5)]],
                            vdims=img.vdims)
        self.assertEqual(op_contours, contour)

    @attr(optional=1) # Requires matplotlib
    def test_image_contours_filled(self):
        img = Image(np.array([[0, 1, 0], [3, 4, 5.], [6, 7, 8]]))
        op_contours = contours(img, filled=True, levels=[2, 2.5])
        data = [[(0., 0.333333, 2.25), (0.5, 0.3, 2.25), (0.5, 0.25, 2.25), (0., 0.25, 2.25),
                 (-0.5, 0.08333333, 2.25), (-0.5, 0.16666667, 2.25), (0., 0.33333333, 2.25)]]
        polys = Polygons(data, vdims=img.vdims)
        self.assertEqual(op_contours, polys)

    def test_points_histogram(self):
        points = Points([float(i) for i in range(10)])
        op_hist = histogram(points, num_bins=3)

        # Make sure that the name and label are as desired
        op_freq_dim = op_hist.get_dimension('x_frequency')
        self.assertEqual(op_freq_dim.label, 'x Frequency')

        # Because the operation labels are now different from the 
        #  default Element label, change back before comparing.
        op_hist = op_hist.redim(x_frequency='Frequency')
        hist = Histogram(([0.1, 0.1, 0.133333], [0, 3, 6, 9]))
        self.assertEqual(op_hist, hist)

    def test_points_histogram_bin_range(self):
        points = Points([float(i) for i in range(10)])
        op_hist = histogram(points, num_bins=3, bin_range=(0, 3))

        # Make sure that the name and label are as desired
        op_freq_dim = op_hist.get_dimension('x_frequency')
        self.assertEqual(op_freq_dim.label, 'x Frequency')

        # Because the operation labels are now different from the 
        #  default Element label, change back before comparing.
        op_hist = op_hist.redim(x_frequency='Frequency')
        hist = Histogram(([0.25, 0.25, 0.5], [0., 1., 2., 3.]))
        self.assertEqual(op_hist, hist)

    def test_points_histogram_not_normed(self):
        points = Points([float(i) for i in range(10)])
        op_hist = histogram(points, num_bins=3, normed=False)

        # Make sure that the name and label are as desired
        op_freq_dim = op_hist.get_dimension('x_frequency')
        self.assertEqual(op_freq_dim.label, 'x Frequency')

        # Because the operation labels are now different from the 
        #  default Element label, change back before comparing.
        op_hist = op_hist.redim(x_frequency='Frequency')
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

    def test_stack_area_overlay(self):
        areas = Area([1, 2, 3]) * Area([1, 2, 3])
        stacked = Area.stack(areas)
        area1 = Area(([0, 1, 2], [1, 2, 3], [0, 0, 0]), vdims=['y', 'Baseline'])
        area2 = Area(([0, 1, 2], [2, 4, 6], [1, 2, 3]), vdims=['y', 'Baseline'])
        self.assertEqual(stacked, area1 * area2)

    def test_stack_area_ndoverlay(self):
        areas = NdOverlay([(0, Area([1, 2, 3])), (1, Area([1, 2, 3]))])
        stacked = Area.stack(areas)
        area1 = Area(([0, 1, 2], [1, 2, 3], [0, 0, 0]), vdims=['y', 'Baseline'])
        area2 = Area(([0, 1, 2], [2, 4, 6], [1, 2, 3]), vdims=['y', 'Baseline'])
        self.assertEqual(stacked, NdOverlay([(0, area1), (1, area2)]))
