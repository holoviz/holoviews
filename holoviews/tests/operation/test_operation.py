import datetime as dt
from unittest import skipIf

import numpy as np

try:
    import matplotlib as mpl
except:
    mpl = None

try:
    import dask.array as da
except:
    da = None

from holoviews import (HoloMap, NdOverlay, NdLayout, GridSpace, Image,
                       Contours, Polygons, Points, Histogram, Curve, Area,
                       QuadMesh, Dataset)
from holoviews.core.data.grid import GridInterface
from holoviews.core.util import pd
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (operation, transform, threshold,
                                         gradient, contours, histogram,
                                         interpolate_curve)

pd_skip = skipIf(pd is None, "Pandas not available")
mpl_skip = skipIf(mpl is None, "Matplotlib is not available")
da_skip = skipIf(da is None, "dask.array is not available")


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

    @mpl_skip
    def test_image_contours(self):
        img = Image(np.array([[0, 1, 0], [3, 4, 5.], [6, 7, 8]]))
        op_contours = contours(img, levels=[0.5])
        contour = Contours([[(-0.166667,  0.333333, 0.5), (-0.333333, 0.277778, 0.5),
                             (np.NaN, np.NaN, 0.5), (0.333333, 0.3, 0.5),
                             (0.166667, 0.333333, 0.5)]],
                            vdims=img.vdims)
        self.assertEqual(op_contours, contour)

    @mpl_skip
    def test_image_contours_no_range(self):
        img = Image(np.zeros((2, 2)))
        op_contours = contours(img, levels=2)
        contour = Contours([], vdims=img.vdims)
        self.assertEqual(op_contours, contour)

    @mpl_skip
    def test_qmesh_contours(self):
        qmesh = QuadMesh(([0, 1, 2], [1, 2, 3], np.array([[0, 1, 0], [3, 4, 5.], [6, 7, 8]])))
        op_contours = contours(qmesh, levels=[0.5])
        contour = Contours([[(0,  1.166667, 0.5), (0.5, 1., 0.5),
                             (np.NaN, np.NaN, 0.5), (1.5, 1., 0.5),
                             (2, 1.1, 0.5)]],
                            vdims=qmesh.vdims)
        self.assertEqual(op_contours, contour)

    @mpl_skip
    def test_qmesh_curvilinear_contours(self):
        x = y = np.arange(3)
        xs, ys = np.meshgrid(x, y)
        zs = np.array([[0, 1, 0], [3, 4, 5.], [6, 7, 8]])
        qmesh = QuadMesh((xs, ys+0.1, zs))
        op_contours = contours(qmesh, levels=[0.5])
        contour = Contours([[(0,  0.266667, 0.5), (0.5, 0.1, 0.5),
                             (np.NaN, np.NaN, 0.5), (1.5, 0.1, 0.5),
                             (2, 0.2, 0.5)]],
                            vdims=qmesh.vdims)
        self.assertEqual(op_contours, contour)

    @mpl_skip
    def test_qmesh_curvilinear_edges_contours(self):
        x = y = np.arange(3)
        xs, ys = np.meshgrid(x, y)
        xs = GridInterface._infer_interval_breaks(xs)
        xs = GridInterface._infer_interval_breaks(xs, 1)
        ys = GridInterface._infer_interval_breaks(ys)
        ys = GridInterface._infer_interval_breaks(ys, 1)
        zs = np.array([[0, 1, 0], [3, 4, 5.], [6, 7, 8]])
        qmesh = QuadMesh((xs, ys+0.1, zs))
        op_contours = contours(qmesh, levels=[0.5])
        contour = Contours([[(0,  0.266667, 0.5), (0.5, 0.1, 0.5),
                             (np.NaN, np.NaN, 0.5), (1.5, 0.1, 0.5),
                             (2, 0.2, 0.5)]],
                            vdims=qmesh.vdims)
        self.assertEqual(op_contours, contour)

    @mpl_skip
    def test_image_contours_filled(self):
        img = Image(np.array([[0, 1, 0], [3, 4, 5.], [6, 7, 8]]))
        op_contours = contours(img, filled=True, levels=[2, 2.5])
        data = [[(0., 0.166667, 2.25), (0.333333, 0.166667, 2.25), (0.333333, 0.2, 2.25), (0., 0.222222, 2.25),
                 (-0.333333, 0.111111, 2.25), (-0.333333, 0.055556, 2.25), (0., 0.166667, 2.25)]]
        polys = Polygons(data, vdims=img.vdims[0].clone(range=(2, 2.5)))
        self.assertEqual(op_contours, polys)

    def test_points_histogram(self):
        points = Points([float(i) for i in range(10)])
        op_hist = histogram(points, num_bins=3, normed=True)

        hist = Histogram(([0, 3, 6, 9], [0.1, 0.1, 0.133333]),
                         vdims=('x_frequency', 'Frequency'))
        self.assertEqual(op_hist, hist)

    def test_dataset_histogram_empty_explicit_bins(self):
        ds = Dataset([np.nan, np.nan], ['x'])
        op_hist = histogram(ds, bins=[0, 1, 2])

        hist = Histogram(([0, 1, 2], [0, 0]), vdims=('x_count', 'Count'))
        self.assertEqual(op_hist, hist)

    @da_skip
    def test_dataset_histogram_dask(self):
        import dask.array as da
        ds = Dataset((da.from_array(np.array(range(10), dtype='f'), chunks=(3)),),
                     ['x'], datatype=['dask'])
        op_hist = histogram(ds, num_bins=3, normed=True)

        hist = Histogram(([0, 3, 6, 9], [0.1, 0.1, 0.133333]),
                         vdims=('x_frequency', 'Frequency'))
        self.assertIsInstance(op_hist.data['x_frequency'], da.Array)
        self.assertEqual(op_hist, hist)

    @da_skip
    def test_dataset_cumulative_histogram_dask(self):
        import dask.array as da
        ds = Dataset((da.from_array(np.array(range(10), dtype='f'), chunks=(3)),),
                     ['x'], datatype=['dask'])
        op_hist = histogram(ds, num_bins=3, cumulative=True, normed=True)

        hist = Histogram(([0, 3, 6, 9], [0.3, 0.6, 1]),
                         vdims=('x_frequency', 'Frequency'))
        self.assertIsInstance(op_hist.data['x_frequency'], da.Array)
        self.assertEqual(op_hist, hist)

    @da_skip
    def test_dataset_weighted_histogram_dask(self):
        import dask.array as da
        ds = Dataset((da.from_array(np.array(range(10), dtype='f'), chunks=3),
                      da.from_array([i/10. for i in range(10)], chunks=3)),
                     ['x', 'y'], datatype=['dask'])
        op_hist = histogram(ds, weight_dimension='y', num_bins=3, normed=True)

        hist = Histogram(([0, 3, 6, 9], [0.022222, 0.088889, 0.222222]),
                         vdims='y')
        self.assertIsInstance(op_hist.data['y'], da.Array)
        self.assertEqual(op_hist, hist)

    def test_points_histogram_bin_range(self):
        points = Points([float(i) for i in range(10)])
        op_hist = histogram(points, num_bins=3, bin_range=(0, 3), normed=True)

        hist = Histogram(([0.25, 0.25, 0.5], [0., 1., 2., 3.]),
                         vdims=('x_frequency', 'Frequency'))
        self.assertEqual(op_hist, hist)

    def test_points_histogram_explicit_bins(self):
        points = Points([float(i) for i in range(10)])
        op_hist = histogram(points, bins=[0, 1, 3], normed=False)

        hist = Histogram(([0, 1, 3], [1, 3]),
                         vdims=('x_count', 'Count'))
        self.assertEqual(op_hist, hist)

    def test_points_histogram_cumulative(self):
        arr = np.arange(4)
        points = Points(arr)
        op_hist = histogram(points, cumulative=True, num_bins=3, normed=False)

        hist = Histogram(([0, 1, 2, 3], [1, 2, 4]),
                         vdims=('x_count', 'Count'))
        self.assertEqual(op_hist, hist)

    def test_points_histogram_not_normed(self):
        points = Points([float(i) for i in range(10)])
        op_hist = histogram(points, num_bins=3, normed=False)

        hist = Histogram(([0, 3, 6, 9], [3, 3, 4]),
                         vdims=('x_count', 'Count'))
        self.assertEqual(op_hist, hist)

    def test_histogram_operation_datetime(self):
        dates = np.array([dt.datetime(2017, 1, i) for i in range(1, 5)])
        op_hist = histogram(Dataset(dates, 'Date'), num_bins=4, normed=True)
        hist_data = {
            'Date': np.array([
                '2017-01-01T00:00:00.000000', '2017-01-01T18:00:00.000000',
                '2017-01-02T12:00:00.000000', '2017-01-03T06:00:00.000000',
                '2017-01-04T00:00:00.000000'], dtype='datetime64[us]'),
            'Date_frequency': np.array([
                3.85802469e-18, 3.85802469e-18, 3.85802469e-18,
                3.85802469e-18])
        }
        hist = Histogram(hist_data, kdims='Date', vdims=('Date_frequency', 'Frequency'))
        self.assertEqual(op_hist, hist)

    def test_histogram_operation_datetime64(self):
        dates = np.array([dt.datetime(2017, 1, i) for i in range(1, 5)]).astype('M')
        op_hist = histogram(Dataset(dates, 'Date'), num_bins=4, normed=True)
        hist_data = {
            'Date': np.array([
                '2017-01-01T00:00:00.000000', '2017-01-01T18:00:00.000000',
                '2017-01-02T12:00:00.000000', '2017-01-03T06:00:00.000000',
                '2017-01-04T00:00:00.000000'], dtype='datetime64[us]'),
            'Date_frequency': np.array([
                3.85802469e-18, 3.85802469e-18, 3.85802469e-18,
                3.85802469e-18])
        }
        hist = Histogram(hist_data, kdims='Date', vdims=('Date_frequency', 'Frequency'))
        self.assertEqual(op_hist, hist)

    @pd_skip
    def test_histogram_operation_pd_period(self):
        dates = pd.date_range('2017-01-01', '2017-01-04', freq='D').to_period('D')
        op_hist = histogram(Dataset(dates, 'Date'), num_bins=4, normed=True)
        hist_data = {
            'Date': np.array([
                '2017-01-01T00:00:00.000000', '2017-01-01T18:00:00.000000',
                '2017-01-02T12:00:00.000000', '2017-01-03T06:00:00.000000',
                '2017-01-04T00:00:00.000000'], dtype='datetime64[us]'),
            'Date_frequency': np.array([
                3.85802469e-18, 3.85802469e-18, 3.85802469e-18,
                3.85802469e-18])
        }
        hist = Histogram(hist_data, kdims='Date', vdims=('Date_frequency', 'Frequency'))
        self.assertEqual(op_hist, hist)

    def test_points_histogram_weighted(self):
        points = Points([float(i) for i in range(10)])
        op_hist = histogram(points, num_bins=3, weight_dimension='y', normed=True)
        hist = Histogram(([0.022222, 0.088889, 0.222222], [0, 3, 6, 9]), vdims=['y'])
        self.assertEqual(op_hist, hist)

    def test_points_histogram_mean_weighted(self):
        points = Points([float(i) for i in range(10)])
        op_hist = histogram(points, num_bins=3, weight_dimension='y',
                            mean_weighted=True, normed=True)
        hist = Histogram(([1.,  4., 7.5], [0, 3, 6, 9]), vdims=['y'])
        self.assertEqual(op_hist, hist)

    def test_interpolate_curve_pre(self):
        interpolated = interpolate_curve(Curve([0, 0.5, 1]), interpolation='steps-pre')
        curve = Curve([(0, 0), (0, 0.5), (1, 0.5), (1, 1), (2, 1)])
        self.assertEqual(interpolated, curve)

    def test_interpolate_curve_pre_with_values(self):
        interpolated = interpolate_curve(Curve([(0, 0, 'A'), (1, 0.5, 'B'), (2, 1, 'C')], vdims=['y', 'z']),
                                         interpolation='steps-pre')
        curve = Curve([(0, 0, 'A'), (0, 0.5, 'B'), (1, 0.5, 'B'), (1, 1, 'C'), (2, 1, 'C')], vdims=['y', 'z'])
        self.assertEqual(interpolated, curve)

    def test_interpolate_datetime_curve_pre(self):
        dates = np.array([dt.datetime(2017, 1, i) for i in range(1, 5)]).astype('M')
        values = [0, 1, 2, 3]
        interpolated = interpolate_curve(Curve((dates, values)), interpolation='steps-pre')
        dates_interp = np.array([
            '2017-01-01T00:00:00', '2017-01-01T00:00:00',
            '2017-01-02T00:00:00', '2017-01-02T00:00:00',
            '2017-01-03T00:00:00', '2017-01-03T00:00:00',
            '2017-01-04T00:00:00'
        ], dtype='datetime64[ns]')
        curve = Curve((dates_interp, [0, 1, 1, 2, 2, 3, 3]))
        self.assertEqual(interpolated, curve)

    def test_interpolate_curve_mid(self):
        interpolated = interpolate_curve(Curve([0, 0.5, 1]), interpolation='steps-mid')
        curve = Curve([(0, 0), (0.5, 0), (0.5, 0.5), (1.5, 0.5), (1.5, 1), (2, 1)])
        self.assertEqual(interpolated, curve)

    def test_interpolate_curve_mid_with_values(self):
        interpolated = interpolate_curve(Curve([(0, 0, 'A'), (1, 0.5, 'B'), (2, 1, 'C')], vdims=['y', 'z']),
                                         interpolation='steps-mid')
        curve = Curve([(0, 0, 'A'), (0.5, 0, 'A'), (0.5, 0.5, 'B'),
                       (1.5, 0.5, 'B'), (1.5, 1, 'C'), (2, 1, 'C')],
                      vdims=['y', 'z'])
        self.assertEqual(interpolated, curve)

    def test_interpolate_datetime_curve_mid(self):
        dates = np.array([dt.datetime(2017, 1, i) for i in range(1, 5)]).astype('M')
        values = [0, 1, 2, 3]
        interpolated = interpolate_curve(Curve((dates, values)), interpolation='steps-mid')
        dates_interp = np.array([
            '2017-01-01T00:00:00', '2017-01-01T12:00:00',
            '2017-01-01T12:00:00', '2017-01-02T12:00:00',
            '2017-01-02T12:00:00', '2017-01-03T12:00:00',
            '2017-01-03T12:00:00', '2017-01-04T00:00:00'
        ], dtype='datetime64[ns]')
        curve = Curve((dates_interp, [0, 0, 1, 1, 2, 2, 3, 3]))
        self.assertEqual(interpolated, curve)

    def test_interpolate_curve_post(self):
        interpolated = interpolate_curve(Curve([0, 0.5, 1]), interpolation='steps-post')
        curve = Curve([(0, 0), (1, 0), (1, 0.5), (2, 0.5), (2, 1)])
        self.assertEqual(interpolated, curve)

    def test_interpolate_curve_post_with_values(self):
        interpolated = interpolate_curve(Curve([(0, 0, 'A'), (1, 0.5, 'B'), (2, 1, 'C')], vdims=['y', 'z']),
                                         interpolation='steps-post')
        curve = Curve([(0, 0, 'A'), (1, 0, 'A'), (1, 0.5, 'B'),
                       (2, 0.5, 'B'), (2, 1, 'C')],
                      vdims=['y', 'z'])
        self.assertEqual(interpolated, curve)

    def test_interpolate_datetime_curve_post(self):
        dates = np.array([dt.datetime(2017, 1, i) for i in range(1, 5)]).astype('M')
        values = [0, 1, 2, 3]
        interpolated = interpolate_curve(Curve((dates, values)), interpolation='steps-post')
        dates_interp = np.array([
            '2017-01-01T00:00:00', '2017-01-02T00:00:00',
            '2017-01-02T00:00:00', '2017-01-03T00:00:00',
            '2017-01-03T00:00:00', '2017-01-04T00:00:00',
            '2017-01-04T00:00:00'
        ], dtype='datetime64[ns]')
        curve = Curve((dates_interp, [0, 0, 1, 1, 2, 2, 3]))
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

    def test_pre_and_postprocess_hooks(self):
        pre_backup = operation._preprocess_hooks
        post_backup = operation._postprocess_hooks
        operation._preprocess_hooks = [lambda op, x: {'label': str(x.id)}]
        operation._postprocess_hooks = [lambda op, x, **kwargs: x.clone(**kwargs)]
        curve = Curve([1, 2, 3])
        self.assertEqual(operation(curve).label, str(curve.id))
        operation._preprocess_hooks = pre_backup
        operation._postprocess_hooks = post_backup
