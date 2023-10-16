import datetime as dt
from unittest import skipIf

import numpy as np
import pandas as pd
import pytest

try:
    import dask.array as da
except ImportError:
    da = None

try:
    import ibis
except ImportError:
    ibis = None

from holoviews import (
    Area,
    Contours,
    Curve,
    Dataset,
    GridSpace,
    Histogram,
    HoloMap,
    Image,
    NdLayout,
    NdOverlay,
    Points,
    Polygons,
    QuadMesh,
    renderer,
)
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
    contours,
    decimate,
    gradient,
    histogram,
    interpolate_curve,
    operation,
    threshold,
    transform,
)

da_skip = skipIf(da is None, "dask.array is not available")
ibis_skip = skipIf(ibis is None, "ibis is not available")


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

    def test_image_contours(self):
        img = Image(np.array([[0, 1, 0], [0, 1, 0]]))
        op_contours = contours(img, levels=[0.5])
        # Note multiple lines which are nan-separated.
        contour = Contours([[(-0.166667, 0.25, 0.5), (-0.1666667, -0.25, 0.5),
                             (np.nan, np.nan, 0.5), (0.1666667, -0.25, 0.5),
                             (0.1666667, 0.25, 0.5)]],
                            vdims=img.vdims)
        self.assertEqual(op_contours, contour)

    def test_image_contours_empty(self):
        img = Image(np.array([[0, 1, 0], [0, 1, 0]]))
        # Contour level outside of data limits
        op_contours = contours(img, levels=[23.0])
        contour = Contours([], vdims=img.vdims)
        self.assertEqual(op_contours, contour)

    def test_image_contours_auto_levels(self):
        z = np.array([[0, 1, 0], [3, 4, 5.], [6, 7, 8]])
        img = Image(z)
        for nlevels in range(3, 20):
            op_contours = contours(img, levels=nlevels)
            levels = [item['z'] for item in op_contours.data]
            assert len(levels) <= nlevels + 2
            assert np.min(levels) <= z.min()
            assert np.max(levels) < z.max()

    def test_image_contours_no_range(self):
        img = Image(np.zeros((2, 2)))
        op_contours = contours(img, levels=2)
        contour = Contours([], vdims=img.vdims)
        self.assertEqual(op_contours, contour)

    def test_image_contours_x_datetime(self):
        x = np.array(['2023-09-01', '2023-09-03', '2023-09-05'], dtype='datetime64')
        y = [14, 15]
        z = np.array([[0, 1, 0], [0, 1, 0]])
        img = Image((x, y, z))
        op_contours = contours(img, levels=[0.5])
        # Note multiple lines which are nan-separated.
        tz = dt.timezone.utc
        expected_x = np.array(
            [dt.datetime(2023, 9, 2, tzinfo=tz), dt.datetime(2023, 9, 2, tzinfo=tz), np.nan,
             dt.datetime(2023, 9, 4, tzinfo=tz), dt.datetime(2023, 9, 4, tzinfo=tz)],
            dtype=object)

        # Separately compare nans and datetimes
        x = op_contours.dimension_values('x')
        mask = np.array([True, True, False, True, True])  # Mask ignoring nans
        np.testing.assert_array_equal(x[mask], expected_x[mask])
        np.testing.assert_array_equal(x[~mask].astype(float), expected_x[~mask].astype(float))

        np.testing.assert_array_almost_equal(op_contours.dimension_values('y').astype(float),
                                             [15, 14, np.nan, 14, 15])
        np.testing.assert_array_almost_equal(op_contours.dimension_values('z'), [0.5]*5)

    def test_image_contours_y_datetime(self):
        x = [14, 15, 16]
        y = np.array(['2023-09-01', '2023-09-03'], dtype='datetime64')
        z = np.array([[0, 1, 0], [0, 1, 0]])
        img = Image((x, y, z))
        op_contours = contours(img, levels=[0.5])
        # Note multiple lines which are nan-separated.
        np.testing.assert_array_almost_equal(op_contours.dimension_values('x').astype(float),
                                             [14.5, 14.5, np.nan, 15.5, 15.5])

        tz = dt.timezone.utc
        expected_y = np.array(
            [dt.datetime(2023, 9, 3, tzinfo=tz), dt.datetime(2023, 9, 1, tzinfo=tz), np.nan,
             dt.datetime(2023, 9, 1, tzinfo=tz), dt.datetime(2023, 9, 3, tzinfo=tz)],
            dtype=object)

        # Separately compare nans and datetimes
        y = op_contours.dimension_values('y')
        mask = np.array([True, True, False, True, True])  # Mask ignoring nans
        np.testing.assert_array_equal(y[mask], expected_y[mask])
        np.testing.assert_array_equal(y[~mask].astype(float), expected_y[~mask].astype(float))

        np.testing.assert_array_almost_equal(op_contours.dimension_values('z'), [0.5]*5)

    def test_image_contours_xy_datetime(self):
        x = np.array(['2023-09-01', '2023-09-03', '2023-09-05'], dtype='datetime64')
        y = np.array(['2023-10-07', '2023-10-08'], dtype='datetime64')
        z = np.array([[0, 1, 0], [0, 1, 0]])
        img = Image((x, y, z))
        op_contours = contours(img, levels=[0.5])
        # Note multiple lines which are nan-separated.

        tz = dt.timezone.utc
        expected_x = np.array(
            [dt.datetime(2023, 9, 2, tzinfo=tz), dt.datetime(2023, 9, 2, tzinfo=tz), np.nan,
             dt.datetime(2023, 9, 4, tzinfo=tz), dt.datetime(2023, 9, 4, tzinfo=tz)],
            dtype=object)
        expected_y = np.array(
            [dt.datetime(2023, 10, 8, tzinfo=tz), dt.datetime(2023, 10, 7, tzinfo=tz), np.nan,
             dt.datetime(2023, 10, 7, tzinfo=tz), dt.datetime(2023, 10, 8, tzinfo=tz)],
            dtype=object)

        # Separately compare nans and datetimes
        x = op_contours.dimension_values('x')
        mask = np.array([True, True, False, True, True])  # Mask ignoring nans
        np.testing.assert_array_equal(x[mask], expected_x[mask])
        np.testing.assert_array_equal(x[~mask].astype(float), expected_x[~mask].astype(float))

        y = op_contours.dimension_values('y')
        np.testing.assert_array_equal(y[mask], expected_y[mask])
        np.testing.assert_array_equal(y[~mask].astype(float), expected_y[~mask].astype(float))

        np.testing.assert_array_almost_equal(op_contours.dimension_values('z'), [0.5]*5)

    def test_image_contours_z_datetime(self):
        z = np.array([['2023-09-10', '2023-09-10'], ['2023-09-10', '2023-09-12']], dtype='datetime64')
        img = Image(z)
        op_contours = contours(img, levels=[np.datetime64('2023-09-11')])
        np.testing.assert_array_almost_equal(op_contours.dimension_values('x'), [0.25, 0.0])
        np.testing.assert_array_almost_equal(op_contours.dimension_values('y'), [0.0, -0.25])
        expected_z = np.array([
            dt.datetime(2023, 9, 11, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2023, 9, 11, 0, 0, tzinfo=dt.timezone.utc)], dtype=object)
        np.testing.assert_array_equal(op_contours.dimension_values('z'), expected_z)

    def test_qmesh_contours(self):
        qmesh = QuadMesh(([0, 1, 2], [1, 2, 3], np.array([[0, 1, 0], [3, 4, 5.], [6, 7, 8]])))
        op_contours = contours(qmesh, levels=[0.5])
        contour = Contours([[(0,  1.166667, 0.5), (0.5, 1., 0.5),
                             (np.nan, np.nan, 0.5), (1.5, 1., 0.5),
                             (2, 1.1, 0.5)]],
                            vdims=qmesh.vdims)
        self.assertEqual(op_contours, contour)

    def test_qmesh_curvilinear_contours(self):
        x = y = np.arange(3)
        xs, ys = np.meshgrid(x, y)
        zs = np.array([[0, 1, 0], [3, 4, 5.], [6, 7, 8]])
        qmesh = QuadMesh((xs, ys+0.1, zs))
        op_contours = contours(qmesh, levels=[0.5])
        contour = Contours([[(0,  0.266667, 0.5), (0.5, 0.1, 0.5),
                             (np.nan, np.nan, 0.5), (1.5, 0.1, 0.5),
                             (2, 0.2, 0.5)]],
                            vdims=qmesh.vdims)
        self.assertEqual(op_contours, contour)

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
                             (np.nan, np.nan, 0.5), (1.5, 0.1, 0.5),
                             (2, 0.2, 0.5)]],
                            vdims=qmesh.vdims)
        self.assertEqual(op_contours, contour)

    def test_image_contours_filled(self):
        img = Image(np.array([[0, 2, 0], [0, 2, 0]]))
        # Two polygons (nan-separated) without holes
        op_contours = contours(img, filled=True, levels=[0.5, 1.5])
        data = [[(-0.25, -0.25, 1), (-0.08333333, -0.25, 1), (-0.08333333, 0.25, 1),
                 (-0.25, 0.25, 1), (-0.25, -0.25, 1), (np.nan, np.nan, 1), (0.08333333, -0.25, 1),
                 (0.25, -0.25, 1), (0.25, 0.25, 1), (0.08333333, 0.25, 1), (0.08333333, -0.25, 1)]]
        polys = Polygons(data, vdims=img.vdims[0].clone(range=(0.5, 1.5)))
        self.assertEqual(op_contours, polys)

    def test_image_contours_filled_with_hole(self):
        img = Image(np.array([[0, 0, 0], [0, 1, 0.], [0, 0, 0]]))
        # Single polygon with hole
        op_contours = contours(img, filled=True, levels=[0.25, 0.75])
        data = [[(-0.25, 0.0, 0.5), (0.0, -0.25, 0.5), (0.25, 0.0, 0.5), (0.0, 0.25, 0.5),
                  (-0.25, 0.0, 0.5)]]
        polys = Polygons(data, vdims=img.vdims[0].clone(range=(0.25, 0.75)))
        self.assertEqual(op_contours, polys)
        expected_holes = [[[np.array([[0.0, -0.08333333], [-0.08333333, 0.0], [0.0,  0.08333333],
                                      [0.08333333, 0.0], [0.0, -0.08333333]])]]]
        np.testing.assert_array_almost_equal(op_contours.holes(), expected_holes)

    def test_image_contours_filled_multi_holes(self):
        img = Image(np.array([[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]))
        # Single polygon with two holes
        op_contours = contours(img, filled=True, levels=[-0.5, 0.5])
        data = [[(-0.4, -0.3333333, 0), (-0.2, -0.3333333, 0), (0, -0.3333333, 0),
                 (0.2, -0.3333333, 0), (0.4, -0.3333333, 0), (0.4, 0, 0), (0.4, 0.3333333, 0),
                 (0.2, 0.3333333, 0), (0, 0.3333333, 0), (-0.2, 0.3333333, 0), (-0.4, 0.3333333, 0),
                 (-0.4, 0, 0), (-0.4, -0.3333333, 0)]]
        polys = Polygons(data, vdims=img.vdims[0].clone(range=(-0.5, 0.5)))
        self.assertEqual(op_contours, polys)
        expected_holes = [[[np.array([[-0.2, -0.16666667], [-0.3, 0], [-0.2, 0.16666667], [-0.1, 0],
                                      [-0.2, -0.16666667]]),
                            np.array([[0.2, -0.16666667], [0.1, 0], [0.2, 0.16666667], [0.3, 0],
                                      [0.2, -0.16666667]])]]]
        np.testing.assert_array_almost_equal(op_contours.holes(), expected_holes)

    def test_image_contours_filled_empty(self):
        img = Image(np.array([[0, 1, 0], [3, 4, 5.], [6, 7, 8]]))
        # Contour level outside of data limits
        op_contours = contours(img, filled=True, levels=[20.0, 23.0])
        polys = Polygons([], vdims=img.vdims[0].clone(range=(20.0, 23.0)))
        self.assertEqual(op_contours, polys)

    def test_image_contours_filled_auto_levels(self):
        z = np.array([[0, 1, 0], [3, 4, 5], [6, 7, 8]])
        img = Image(z)
        for nlevels in range(3, 20):
            op_contours = contours(img, filled=True, levels=nlevels)
            levels = [item['z'] for item in op_contours.data]
            assert len(levels) <= nlevels + 1
            delta = 0.5*(levels[1] - levels[0])
            assert np.min(levels) <= z.min() + delta
            assert np.max(levels) >= z.max() - delta

    def test_image_contours_filled_x_datetime(self):
        x = np.array(['2023-09-01', '2023-09-05', '2023-09-09'], dtype='datetime64')
        y = np.array([6, 7])
        z = np.array([[0, 2, 0], [0, 2, 0]])
        img = Image((x, y, z))
        msg = r'Datetime spatial coordinates are not supported for filled contour calculations.'
        with pytest.raises(RuntimeError, match=msg):
            _ = contours(img, filled=True, levels=[0.5, 1.5])

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

    @ibis_skip
    @pytest.mark.usefixtures('ibis_sqlite_backend')
    def test_dataset_histogram_ibis(self):
        df = pd.DataFrame(dict(x=np.arange(10)))
        t = ibis.memtable(df, name='t')
        ds = Dataset(t, vdims='x')
        op_hist = histogram(ds, dimension='x', num_bins=3, normed=True)

        hist = Histogram(([0, 3, 6, 9], [0.1, 0.1, 0.133333]),
                         vdims=('x_frequency', 'Frequency'))
        self.assertEqual(op_hist, hist)

    @ibis_skip
    @pytest.mark.usefixtures('ibis_sqlite_backend')
    def test_dataset_cumulative_histogram_ibis(self):
        df = pd.DataFrame(dict(x=np.arange(10)))
        t = ibis.memtable(df, name='t')
        ds = Dataset(t, vdims='x')
        op_hist = histogram(ds, num_bins=3, cumulative=True, normed=True)

        hist = Histogram(([0, 3, 6, 9], [0.3, 0.6, 1]),
                         vdims=('x_frequency', 'Frequency'))
        self.assertEqual(op_hist, hist)

    @ibis_skip
    @pytest.mark.usefixtures('ibis_sqlite_backend')
    def test_dataset_histogram_explicit_bins_ibis(self):
        df = pd.DataFrame(dict(x=np.arange(10)))
        t = ibis.memtable(df, name='t')
        ds = Dataset(t, vdims='x')
        op_hist = histogram(ds, bins=[0, 1, 3], normed=False)

        hist = Histogram(([0, 1, 3], [1, 3]),
                         vdims=('x_count', 'Count'))
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

    def test_histogram_categorical(self):
        series = Dataset(pd.Series(['A', 'B', 'C']))
        kwargs = {'bin_range': ('A', 'C'), 'normed': False, 'cumulative': False, 'num_bins': 3}
        with pytest.raises(ValueError):
            histogram(series, **kwargs)

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

    def test_decimate_ordering(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        curve = Curve((x, y))
        decimated = decimate(curve, max_samples=20)
        renderer("bokeh").get_plot(decimated)

        index = decimated.data[()].data.index
        assert np.all(index == np.sort(index))
