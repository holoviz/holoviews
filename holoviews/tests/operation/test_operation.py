import datetime as dt
import random
from importlib.util import find_spec
from unittest import SkipTest, skipIf

import numpy as np
import pandas as pd
import param
import pytest

try:
    import dask.array as da
except ImportError:
    da = None

try:
    import ibis

    from holoviews.core.data.ibis import IBIS_VERSION
except ImportError:
    ibis = IBIS_VERSION = None

try:
    import cudf
except ImportError:
    cudf = None

from holoviews import (
    AdjointLayout,
    Area,
    Contours,
    Curve,
    Dataset,
    Dendrogram,
    Empty,
    GridSpace,
    HeatMap,
    Histogram,
    HoloMap,
    Image,
    Layout,
    NdLayout,
    NdOverlay,
    Points,
    Polygons,
    QuadMesh,
    renderer,
)
from holoviews.core.data.grid import GridInterface
from holoviews.core.operation import Operation
from holoviews.core.options import SkipRendering
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
    chain,
    contours,
    decimate,
    dendrogram,
    gradient,
    histogram,
    interpolate_curve,
    operation,
    threshold,
    transform,
)

mpl = find_spec("matplotlib")
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

    def test_operation_chain(self):
        img = Image(np.random.rand(10, 10))
        op_img = chain(
            img,
            operations=[
                transform.instance(operator=lambda x: x*2),
                transform.instance(operator=lambda x: x*3),
        ])
        self.assertEqual(op_img, img.clone(img.data*6, group='Transform'))

    def test_operation_chain_find(self):
        class CustomOp1(Operation):
            link_inputs = param.Boolean(False)
        class CustomOp2(Operation):
            link_inputs = param.Boolean(True)

        op1 = CustomOp1.instance()
        op2 = CustomOp2.instance()
        ch_op = chain.instance(operations=[op1, op2])
        self.assertIs(ch_op.find(CustomOp1, skip_nonlinked=False), op1)
        self.assertIsNone(ch_op.find(CustomOp1, skip_nonlinked=True))
        self.assertIs(ch_op.find(CustomOp2, skip_nonlinked=False), op2)
        self.assertIs(ch_op.find(CustomOp2, skip_nonlinked=True), op2)

    def test_operation_chain_find_apply(self):
        img = Image(np.random.rand(10, 10))
        tr_op = transform.instance(operator=lambda x: x*2)
        img_apply = img.apply(tr_op, dynamic=False)
        self.assertIs(img_apply.pipeline.find(transform, skip_nonlinked=False), tr_op)

    def test_operation_chain_find_apply_chain(self):
        class CustomOp1(Operation): pass
        class CustomOp2(Operation): pass

        img = Image(np.random.rand(10, 10))
        op1 = CustomOp1.instance()
        op2 = CustomOp2.instance()
        ch_op = chain.instance(operations=[op1,op2])
        img_apply = img.apply(ch_op, dynamic=False)
        self.assertIs(
            img_apply.pipeline.find(CustomOp1, skip_nonlinked=False),
            op1,
        )

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
        if mpl is None:
            raise SkipTest("Matplotlib required to test datetime axes")

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
        if mpl is None:
            raise SkipTest("Matplotlib required to test datetime axes")
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
        if mpl is None:
            raise SkipTest("Matplotlib required to test datetime axes")
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
        if mpl is None:
            raise SkipTest("Matplotlib required to test datetime axes")
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

    def test_dataset_histogram_groupby_range_shared(self):
        x = np.arange(10)
        y = np.arange(10) + 10
        xy = np.concatenate([x, y])
        label = ["x"] * 10 + ["y"] * 10

        ds = Dataset(pd.DataFrame([xy, label], index=["xy", "label"]).T, vdims=["xy", "label"])
        hist = histogram(ds, groupby="label", groupby_range="shared")
        exp = np.linspace(0, 19, 21)
        for k, v in hist.items():
            np.testing.assert_equal(exp, v.data["xy"])
            sel = np.asarray(label) == k
            assert sel.sum() == 10
            assert (v.data["xy_count"][sel] == 1).all()
            assert (v.data["xy_count"][~sel] == 0).all()

    def test_dataset_histogram_groupby_range_separated(self):
        x = np.arange(10)
        y = np.arange(10) + 10
        xy = np.concatenate([x, y])
        label = ["x"] * 10 + ["y"] * 10

        ds = Dataset(pd.DataFrame([xy, label], index=["xy", "label"]).T, vdims=["xy", "label"])
        hist = histogram(ds, groupby="label", groupby_range="separated")

        for idx, v in enumerate(hist):
            exp = np.linspace(idx * 10, 10 * idx + 9, 21)
            np.testing.assert_equal(exp, v.data["xy"])
            assert v.data["xy_count"].sum() == 10

    def test_dataset_histogram_groupby_datetime(self):
        x = pd.date_range("2020-01-01", periods=100)
        y = pd.date_range("2020-01-01", periods=100)
        xy = np.concatenate([x, y])
        label = ["x"] * 100 + ["y"] * 100
        ds = Dataset(pd.DataFrame([xy, label], index=["xy", "label"]).T, vdims=["xy", "label"])
        hist = histogram(ds, groupby="label")

        exp = pd.date_range("2020-01-01", '2020-04-09', periods=21)
        for h in hist:
            np.testing.assert_equal(exp, h.data["xy"])
            assert (h.data["xy_count"] == 5).all()

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
        t = ibis.memtable(df, **({} if IBIS_VERSION >= (11, 0, 0) else {"name": "t"}))
        ds = Dataset(t, vdims='x')
        op_hist = histogram(ds, dimension='x', num_bins=3, normed=True)

        hist = Histogram(([0, 3, 6, 9], [0.1, 0.1, 0.133333]),
                         vdims=('x_frequency', 'Frequency'))
        self.assertEqual(op_hist, hist)

    @ibis_skip
    @pytest.mark.usefixtures('ibis_sqlite_backend')
    def test_dataset_cumulative_histogram_ibis(self):
        df = pd.DataFrame(dict(x=np.arange(10)))
        t = ibis.memtable(df, **({} if IBIS_VERSION >= (11, 0, 0) else {"name": "t"}))
        ds = Dataset(t, vdims='x')
        op_hist = histogram(ds, num_bins=3, cumulative=True, normed=True)

        hist = Histogram(([0, 3, 6, 9], [0.3, 0.6, 1]),
                         vdims=('x_frequency', 'Frequency'))
        self.assertEqual(op_hist, hist)

    @ibis_skip
    @pytest.mark.usefixtures('ibis_sqlite_backend')
    def test_dataset_histogram_explicit_bins_ibis(self):
        df = pd.DataFrame(dict(x=np.arange(10)))
        t = ibis.memtable(df, **({} if IBIS_VERSION >= (11, 0, 0) else {"name": "t"}))
        ds = Dataset(t, vdims='x')
        op_hist = histogram(ds, bins=[0, 1, 3], normed=False)

        hist = Histogram(([0, 1, 3], [1, 3]),
                         vdims=('x_count', 'Count'))
        self.assertEqual(op_hist, hist)

    @pytest.mark.gpu
    def test_dataset_histogram_cudf(self):
        df = pd.DataFrame(dict(x=np.arange(10)))
        t = cudf.from_pandas(df)
        ds = Dataset(t, vdims='x')
        op_hist = histogram(ds, dimension='x', num_bins=3, normed=True)

        hist = Histogram(([0, 3, 6, 9], [0.1, 0.1, 0.133333]),
                         vdims=('x_frequency', 'Frequency'))
        self.assertEqual(op_hist, hist)

    @pytest.mark.gpu
    def test_dataset_cumulative_histogram_cudf(self):
        df = pd.DataFrame(dict(x=np.arange(10)))
        t = cudf.from_pandas(df)
        ds = Dataset(t, vdims='x')
        op_hist = histogram(ds, num_bins=3, cumulative=True, normed=True)

        hist = Histogram(([0, 3, 6, 9], [0.3, 0.6, 1]),
                         vdims=('x_frequency', 'Frequency'))
        self.assertEqual(op_hist, hist)

    @pytest.mark.gpu
    def test_dataset_histogram_explicit_bins_cudf(self):
        df = pd.DataFrame(dict(x=np.arange(10)))
        t = cudf.from_pandas(df)
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

    def test_histogram_narwhals_pandas(self):
        df = pd.DataFrame({'x': range(10)})
        ds = Dataset(df, vdims='x', datatype=["narwhals"])
        op_hist = histogram(ds, num_bins=3, normed=False)

        hist = Histogram(([0, 3, 6, 9], [3, 3, 4]),
                         vdims=('x_count', 'Count'))
        self.assertEqual(op_hist, hist)

    def test_histogram_narwhals_polars(self):
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({'x': range(10)})
        ds = Dataset(df, vdims='x')
        op_hist = histogram(ds, num_bins=3, normed=False)

        hist = Histogram(([0, 3, 6, 9], [3, 3, 4]),
                         vdims=('x_count', 'Count'))
        self.assertEqual(op_hist, hist)

    def test_histogram_narwhals_polars_lazy(self):
        pl = pytest.importorskip("polars")
        df = pl.LazyFrame({'x': range(10)})
        ds = Dataset(df, vdims='x')
        op_hist = histogram(ds, num_bins=3, normed=False)

        hist = Histogram(([0, 3, 6, 9], [3, 3, 4]),
                         vdims=('x_count', 'Count'))
        self.assertEqual(op_hist, hist)

    def test_histogram_narwhals_polars_lazy_groupby(self):
        pl = pytest.importorskip("polars")
        df = pl.LazyFrame({"x": [1, 2], "y": [1, 2]})
        ds = Dataset(df)
        op_hist = histogram(ds, num_bins=2, groupby="x")

        hist1 = Histogram(([1, 1.5, 2], [1, 0]), kdims="y", vdims=[('y_count', 'Count')])
        hist2 = Histogram(([1, 1.5, 2], [0, 1]), kdims="y", vdims=[('y_count', 'Count')])
        expected = NdOverlay({(1,): hist1, (2,): hist2}, kdims=['x'])

        self.assertEqual(op_hist, expected)

    @pytest.mark.usefixtures("mpl_backend")
    def test_histogram_dask_array_mpl(self):
        # Regression test for https://github.com/holoviz/holoviews/issues/5111
        dd = pytest.importorskip("dask.dataframe")

        data = {
            "carrier": ["A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "depdelay": [127.0, 3.0, -3.0, 19.0, 264.0, -6.0, 1.0, 2.0, 83.0],
        }
        flights = dd.from_pandas(pd.DataFrame(data), npartitions=2)

        by = "carrier"
        ds = Dataset(flights, by)
        ds_grouped = ds.groupby(by)
        hists = histogram(ds_grouped, dimension="depdelay")

        # Should not error
        renderer("matplotlib").get_plot(hists)

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

    @pytest.mark.usefixtures("bokeh_backend")
    def test_decimate_ordering(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        curve = Curve((x, y))
        decimated = decimate(curve, max_samples=20)
        renderer("bokeh").get_plot(decimated)

        index = decimated.data[()].data.index
        assert np.all(index == np.sort(index))

    @pytest.mark.usefixtures("bokeh_backend")
    def test_decimate_same_y(self):
        data = pd.DataFrame({'x': np.arange(10), 'y': np.ones(10)})
        points = Points(data, kdims=['x', 'y']).opts(xlim=(0, 10))
        decimated = decimate(points)

        renderer("bokeh").get_plot(decimated)
        output = decimated.data[()].data
        pd.testing.assert_series_equal(data["x"], output["x"])
        pd.testing.assert_series_equal(data["y"], output["y"])


@pytest.mark.usefixtures("bokeh_backend")
class TestDendrogramOperation:

    def setup_class(self):
        pytest.importorskip("scipy")

        random.seed(1)
        self.df = pd.DataFrame(
            [(i, chr(65 + j), random.random()) for j in range(10) for i in range(5)],
            columns=["z", "x", "y"],
        )
        data = [
            [7, 8, 0, 0, 0, 0],
            [8, 6, 0, 0, 0, 0],
            [0, 0, 9, 8, 2, 7],
            [0, 0, 0, 1, 8, 2],
            [0, 7, 0, 7, 0, 0],
        ]
        self.df2 = pd.DataFrame([
            {"cluster": f"clust {i}", "gene": f"gene {j}", "value": data[i][j]}
            for j in range(6)
            for i in range(5)
        ])

        # Based on scanpy.datasets.pbmc68k_reduced
        obs = 'abcdefghij'
        var = 'mtropqnkslmtropqnkslmtropqnkslmtropqnkslmtropqnkslmtropqnkslmtropqnkslmtropqnkslmtropqnkslmtropqnksl'
        data = [
            8, 10, 4, 8, 45, 6, 32, 45, 60, 6, 23, 10, 19, 0, 10, 6, 30, 7, 7,
            6, 8, 10, 16, 0, 10, 6, 7, 21, 7, 34, 8, 10, 23, 10, 10, 6, 7, 7,
            7, 6, 8, 10, 4, 5, 10, 6, 7, 19, 7, 31, 8, 10, 4, 4, 10, 22, 7, 7,
            7, 6, 8, 10, 4, 7, 10, 33, 7, 24, 22, 6, 8, 10, 4, 12, 10, 27, 7,
            7, 7, 6, 8, 10, 13, 25, 10, 6, 21, 7, 7, 6, 8, 10, 4, 7, 41, 6, 7,
            40, 7, 23
        ]
        counts = [353, 491, 1185, 0, 1674, 2763, 710, 1613, 2377, 716]
        df = pd.DataFrame({
            "obs": [x for x in obs for _ in range(10)],
            "var": list(var),
            "data": data,
            "counts": [x for x in counts for _ in range(10)],
         })
        self.ds = Dataset(df, kdims=["obs", "var"], vdims=["data", "counts"])

        self.bokeh_renderer = renderer("bokeh")

    def get_childrens(self, adjoint):
        bk_childrens = self.bokeh_renderer.get_plot(adjoint).handles["plot"].children
        (atop, *_), (amain, *_), (aright, *_) = bk_childrens
        try:
            top = self.bokeh_renderer.get_plot(adjoint["top"]).handles["plot"]
        except SkipRendering:
            right = None
        main = self.bokeh_renderer.get_plot(adjoint["main"]).handles["plot"]
        try:
            right = self.bokeh_renderer.get_plot(adjoint["right"]).handles["plot"]
        except SkipRendering:
            right = None
        return (atop, amain, aright), (top, main, right)

    def test_right_only(self):
        dataset = Dataset(self.df)
        dendro = dendrogram(dataset, adjoint_dims=["x"], main_dim="y")
        assert isinstance(dendro, AdjointLayout)
        assert isinstance(dendro["main"], HeatMap)
        assert isinstance(dendro["right"], Dendrogram)
        assert isinstance(dendro["top"], Empty)
        assert dendro["right"].kdims == ["__dendrogram_x_x", "__dendrogram_y_x"]

    def test_top_only(self):
        dataset = Dataset(self.df)
        dendro = dendrogram(dataset, adjoint_dims=["z"], main_dim="y")
        assert isinstance(dendro, AdjointLayout)
        assert isinstance(dendro["main"], HeatMap)
        assert isinstance(dendro["right"], Empty)
        assert isinstance(dendro["top"], Dendrogram)
        assert dendro["top"].kdims == ["__dendrogram_x_z", "__dendrogram_y_z"]

    @pytest.mark.parametrize("adjoint_dims", [["x", "z"], ["z", "x"]], ids=["xz", "zx"])
    def test_both_xz(self, adjoint_dims):
        dataset = Dataset(self.df)
        dendro = dendrogram(dataset, adjoint_dims=adjoint_dims, main_dim="y")
        assert isinstance(dendro, AdjointLayout)
        assert isinstance(dendro["main"], HeatMap)
        assert isinstance(dendro["right"], Dendrogram)
        assert isinstance(dendro["top"], Dendrogram)
        assert dendro["right"].kdims == ["__dendrogram_x_x", "__dendrogram_y_x"]
        assert dendro["top"].kdims == ["__dendrogram_x_z", "__dendrogram_y_z"]

    def test_point_plot(self):
        dataset = Points(self.df)
        dendro = dendrogram(dataset, adjoint_dims=["x", "z"], main_dim="y")
        assert isinstance(dendro, AdjointLayout)
        assert isinstance(dendro["main"], Points)

    def test_depth_matches_non_adjoint(self):
        # depth dimensions is the orthogonal axis to the main plot
        dataset = Dataset(self.df)
        dendro = dendrogram(dataset, adjoint_dims=["x", "z"], main_dim="y")
        (atop, amain, aright), (top, _main, right) = self.get_childrens(dendro)

        # Verify no shared axis is changing the depth dimension of the right
        assert atop.y_range.start == top.y_range.start
        assert atop.y_range.end == top.y_range.end
        assert atop.y_range.end == top.y_range.end

        # These should be shared with the main plot
        assert atop.x_range.start == amain.x_range.start
        assert atop.x_range.end == amain.x_range.end

        # Verify no shared axis is changing the depth dimension of the right
        assert aright.x_range.start == right.y_range.start
        assert aright.x_range.end == right.y_range.end

        # These should be shared with the main plot
        assert aright.y_range.factors == amain.y_range.factors

    def test_adjoned_False_1dim(self):
        dataset = Dataset(self.df)
        dendro = dendrogram(dataset, adjoint_dims=["x"], main_dim="y", adjoined=False)

        assert isinstance(dendro, Dendrogram)

    def test_adjoned_False_2dim(self):
        dataset = Dataset(self.df)
        dendro = dendrogram(dataset, adjoint_dims=["x", "z"], main_dim="y", adjoined=False)

        assert isinstance(dendro, Layout)
        assert len(dendro) == 2
        assert isinstance(dendro[0], Dendrogram)
        assert isinstance(dendro[1], Dendrogram)

    @pytest.mark.parametrize(
        "adjoint_dims",
        (["cluster"], ["gene"], ["gene", "cluster"]),
        ids=["right", "top", "both"],
    )
    def test_invert_dendrogram(self, adjoint_dims):
        plot = Points(self.df2, kdims=["gene", "cluster"])
        dendro1 = dendrogram(plot, adjoint_dims=adjoint_dims, main_dim="value")
        dendro2 = dendrogram(plot, adjoint_dims=adjoint_dims, main_dim="value", invert=True)

        main1 = self.bokeh_renderer.get_plot(dendro1["main"]).handles["plot"]
        main2 = self.bokeh_renderer.get_plot(dendro2["main"]).handles["plot"]

        match adjoint_dims:
            case ["cluster"]:
                assert main1.x_range.factors == main2.x_range.factors
                assert main1.y_range.factors == main2.y_range.factors[::-1]
            case ["gene"]:
                assert main1.x_range.factors == main2.x_range.factors[::-1]
                assert main1.y_range.factors == main2.y_range.factors
            case _:
                assert main1.y_range.factors == main2.y_range.factors[::-1]
                assert main1.x_range.factors == main2.x_range.factors[::-1]

    @pytest.mark.parametrize("adjoint_dims", (["cluster"], ["gene"],), ids=["right", "top"])
    def test_assure_non_adjoined_axis_is_unchanged_points(self, adjoint_dims):
        # See: https://github.com/holoviz/holoviews/pull/6625#issuecomment-2981268665
        plot = Points(self.df2, kdims=["gene", "cluster"])
        main1 = self.bokeh_renderer.get_plot(plot).handles["plot"]

        dendro = dendrogram(plot, adjoint_dims=adjoint_dims, main_dim="value")
        main2 = self.bokeh_renderer.get_plot(dendro["main"]).handles["plot"]

        match adjoint_dims:
            case ["cluster"]:
                assert main1.x_range.factors == main2.x_range.factors
            case ["gene"]:
                assert main1.y_range.factors == main2.y_range.factors

    def test_assure_non_adjoined_axis_is_unchanged_heatmap(self):
        # Follow up to previous test, see
        # https://github.com/holoviz/holoviews/pull/6669#issuecomment-3237153317
        plot = HeatMap(self.ds)
        main1 = self.bokeh_renderer.get_plot(plot).handles["plot"]

        dendro = dendrogram(plot, adjoint_dims=["obs"], main_dim="data")
        main2 = self.bokeh_renderer.get_plot(dendro["main"]).handles["plot"]

        assert main1.y_range.factors == main2.y_range.factors

    @pytest.mark.parametrize(
        "adjoint_dims",
        (["cluster"], ["gene"], ["gene", "cluster"]),
        ids=["right", "top", "both"],
    )
    def test_gridded_dataset(self, adjoint_dims, rng):
        xr = pytest.importorskip("xarray")

        N = 10
        da = xr.DataArray(
            rng.normal(size=(N, N)),
            name="main",
            dims=("cluster", "gene"),
            coords={
                "cluster": [f"c{i}" for i in range(N)],
                "gene": [f"g{j}" for j in range(N)],
            },
        )

        dendro = dendrogram(Dataset(da), adjoint_dims=adjoint_dims, main_dim="main")
        assert isinstance(dendro, AdjointLayout)

    def test_failed_linkage(self):
        msg = "Could not calculate linkage for dendrogram"
        with pytest.raises(ValueError, match=msg):
            dendrogram(self.ds, adjoint_dims=["obs"], main_dim="counts")

        # This should work
        dendrogram(self.ds, adjoint_dims=["obs"], main_dim="counts", linkage_metric="euclidean")

    def test_not_primary_main_dim(self):
        # Adding hover to have access to the other dimension values
        plot = HeatMap(self.ds).opts(tools=["hover"])
        assert plot.vdims[0] == "data"
        dendro = dendrogram(
            plot,
            adjoint_dims=["obs"],
            main_dim="counts",
            linkage_metric="euclidean"
        )
        (_, amain, _), *_ = self.get_childrens(dendro)
        data = amain.renderers[0].data_source.data
        assert list(data["zvalues"]) == list(map(int, data["data"]))
