import datetime as dt

from unittest import SkipTest, skipIf

import numpy as np
import pandas as pd
import pytest

from holoviews import (
    Dimension, Curve, Points, Image, Dataset, RGB, Path, Graph, TriMesh,
    QuadMesh, NdOverlay, Contours, Spikes, Spread, Area, Rectangles,
    Segments, Polygons, Nodes, DynamicMap, Overlay, ImageStack
)
from holoviews.util import render
from holoviews.streams import Tap
from holoviews.element.comparison import ComparisonTestCase
from numpy import nan
from holoviews.operation import apply_when
from packaging.version import Version

try:
    import datashader as ds
    import dask.dataframe as dd
    import xarray as xr
    from holoviews.operation.datashader import (
        aggregate, regrid, ds_version, stack, directly_connect_edges,
        shade, spread, rasterize, datashade, AggregationOperation,
        inspect, inspect_points, inspect_polygons
    )
except ImportError:
    raise SkipTest('Datashader not available')

try:
    import cudf
    import cupy
except ImportError:
    cudf = None

try:
    import spatialpandas
except ImportError:
    spatialpandas = None

spatialpandas_skip = skipIf(spatialpandas is None, "SpatialPandas not available")
cudf_skip = skipIf(cudf is None, "cuDF not available")


import logging

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

AggregationOperation.vdim_prefix = ''

@pytest.fixture()
def point_data():
    num = 100
    np.random.seed(1)

    dists = {
        cat: pd.DataFrame(
            {
                "x": np.random.normal(x, s, num),
                "y": np.random.normal(y, s, num),
                "s": s,
                "val": val,
                "cat": cat,
            }
        )
        for x, y, s, val, cat in [
            (2, 2, 0.03, 0, "d1"),
            (2, -2, 0.10, 1, "d2"),
            (-2, -2, 0.50, 2, "d3"),
            (-2, 2, 1.00, 3, "d4"),
            (0, 0, 3.00, 4, "d5"),
        ]
    }
    df = pd.concat(dists, ignore_index=True)
    return df


@pytest.fixture()
def point_plot(point_data):
    return Points(point_data)


class DatashaderAggregateTests(ComparisonTestCase):
    """
    Tests for datashader aggregation
    """

    def test_aggregate_points(self):
        points = Points([(0.2, 0.3), (0.4, 0.7), (0, 0.99)])
        img = aggregate(points, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]),
                         vdims=[Dimension('Count', nodata=0)])
        self.assertEqual(img, expected)

    def test_aggregate_points_count_column(self):
        points = Points([(0.2, 0.3, np.NaN), (0.4, 0.7, 22), (0, 0.99,np.NaN)], vdims='z')
        img = aggregate(points, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2, aggregator=ds.count('z'))
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[0, 0], [1, 0]]),
                         vdims=[Dimension('z Count', nodata=0)])
        self.assertEqual(img, expected)

    @cudf_skip
    def test_aggregate_points_cudf(self):
        points = Points([(0.2, 0.3), (0.4, 0.7), (0, 0.99)], datatype=['cuDF'])
        self.assertIsInstance(points.data, cudf.DataFrame)
        img = aggregate(points, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]),
                         vdims=[Dimension('Count', nodata=0)])
        self.assertIsInstance(img.data.Count.data, cupy.ndarray)
        self.assertEqual(img, expected)

    def test_aggregate_zero_range_points(self):
        p = Points([(0, 0), (1, 1)])
        agg = rasterize(p, x_range=(0, 0), y_range=(0, 1), expand=False, dynamic=False,
                        width=2, height=2)
        img = Image(([], [0.25, 0.75], np.zeros((2, 0))), bounds=(0, 0, 0, 1),
                    xdensity=1, vdims=[Dimension('Count', nodata=0)])
        self.assertEqual(agg, img)

    def test_aggregate_points_target(self):
        points = Points([(0.2, 0.3), (0.4, 0.7), (0, 0.99)])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]),
                         vdims=[Dimension('Count', nodata=0)])
        img = aggregate(points, dynamic=False,  target=expected)
        self.assertEqual(img, expected)

    def test_aggregate_points_sampling(self):
        points = Points([(0.2, 0.3), (0.4, 0.7), (0, 0.99)])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]),
                         vdims=[Dimension('Count', nodata=0)])
        img = aggregate(points, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        x_sampling=0.5, y_sampling=0.5)
        self.assertEqual(img, expected)

    def test_aggregate_points_categorical(self):
        points = Points([(0.2, 0.3, 'A'), (0.4, 0.7, 'B'), (0, 0.99, 'C')], vdims='z')
        img = aggregate(points, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2, aggregator=ds.count_cat('z'))
        x = np.array([0.25, 0.75])
        y = np.array([0.25, 0.75])
        a = np.array([[1, 0], [0, 0]])
        b = np.array([[0, 1], [0, 0]])
        c = np.array([[0, 1], [0, 0]])
        xrds = xr.Dataset(
            coords={"x": x, "y": y},
            data_vars={"a": (("x", "y"), a), "b": (("x", "y"), b), "c": (("x", "y"), c)},
        )
        expected = ImageStack(xrds, kdims=["x", "y"], vdims=["a", "b", "c"])
        actual = img.data
        assert (expected.data.to_array("z").values == actual.T.values).all()

    def test_aggregate_points_categorical_zero_range(self):
        points = Points([(0.2, 0.3, 'A'), (0.4, 0.7, 'B'), (0, 0.99, 'C')], vdims='z')
        img = aggregate(points, dynamic=False,  x_range=(0, 0), y_range=(0, 1),
                        aggregator=ds.count_cat('z'), height=2)
        xs, ys = [], [0.25, 0.75]
        params = dict(bounds=(0, 0, 0, 1), xdensity=1)
        expected = NdOverlay({'A': Image((xs, ys, np.zeros((2, 0))), vdims=Dimension('z Count', nodata=0), **params),
                              'B': Image((xs, ys, np.zeros((2, 0))), vdims=Dimension('z Count', nodata=0), **params),
                              'C': Image((xs, ys, np.zeros((2, 0))), vdims=Dimension('z Count', nodata=0), **params)},
                             kdims=['z'])
        self.assertEqual(img, expected)

    def test_aggregate_curve(self):
        curve = Curve([(0.2, 0.3), (0.4, 0.7), (0.8, 0.99)])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [1, 1]]),
                         vdims=[Dimension('Count', nodata=0)])
        img = aggregate(curve, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        self.assertEqual(img, expected)

    def test_aggregate_curve_datetimes(self):
        dates = pd.date_range(start="2016-01-01", end="2016-01-03", freq='1D')
        curve = Curve((dates, [1, 2, 3]))
        img = aggregate(curve, width=2, height=2, dynamic=False)
        bounds = (np.datetime64('2016-01-01T00:00:00.000000'), 1.0,
                  np.datetime64('2016-01-03T00:00:00.000000'), 3.0)
        dates = [np.datetime64('2016-01-01T12:00:00.000000000'),
                 np.datetime64('2016-01-02T12:00:00.000000000')]
        expected = Image((dates, [1.5, 2.5], [[1, 0], [0, 2]]),
                         datatype=['xarray'], bounds=bounds, vdims=Dimension('Count', nodata=0))
        self.assertEqual(img, expected)

    def test_aggregate_curve_datetimes_dask(self):
        df = pd.DataFrame(
            data=np.arange(1000), columns=['a'],
            index=pd.date_range('2019-01-01', freq='1T', periods=1000),
        )
        ddf = dd.from_pandas(df, npartitions=4)
        curve = Curve(ddf, kdims=['index'], vdims=['a'])
        img = aggregate(curve, width=2, height=3, dynamic=False)
        bounds = (np.datetime64('2019-01-01T00:00:00.000000'), 0.0,
                  np.datetime64('2019-01-01T16:39:00.000000'), 999.0)
        dates = [np.datetime64('2019-01-01T04:09:45.000000000'),
                 np.datetime64('2019-01-01T12:29:15.000000000')]
        expected = Image((dates, [166.5, 499.5, 832.5], [[332, 0], [167, 166], [0, 334]]),
                         kdims=['index', 'a'], vdims=Dimension('Count', nodata=0),
                         datatype=['xarray'], bounds=bounds)
        self.assertEqual(img, expected)

    def test_aggregate_curve_datetimes_microsecond_timebase(self):
        dates = pd.date_range(start="2016-01-01", end="2016-01-03", freq='1D')
        xstart = np.datetime64('2015-12-31T23:59:59.723518000', 'us')
        xend = np.datetime64('2016-01-03T00:00:00.276482000', 'us')
        curve = Curve((dates, [1, 2, 3]))
        img = aggregate(curve, width=2, height=2, x_range=(xstart, xend), dynamic=False)
        bounds = (np.datetime64('2015-12-31T23:59:59.723518'), 1.0,
                  np.datetime64('2016-01-03T00:00:00.276482'), 3.0)
        dates = [np.datetime64('2016-01-01T11:59:59.861759000',),
                 np.datetime64('2016-01-02T12:00:00.138241000')]
        expected = Image((dates, [1.5, 2.5], [[1, 0], [0, 2]]),
                         datatype=['xarray'], bounds=bounds, vdims=Dimension('Count', nodata=0))
        self.assertEqual(img, expected)

    def test_aggregate_ndoverlay_count_cat_datetimes_microsecond_timebase(self):
        dates = pd.date_range(start="2016-01-01", end="2016-01-03", freq='1D')
        xstart = np.datetime64('2015-12-31T23:59:59.723518000', 'us')
        xend = np.datetime64('2016-01-03T00:00:00.276482000', 'us')
        curve = Curve((dates, [1, 2, 3]))
        curve2 = Curve((dates, [3, 2, 1]))
        ndoverlay = NdOverlay({0: curve, 1: curve2}, 'Cat')
        imgs = aggregate(ndoverlay, aggregator=ds.count_cat('Cat'), width=2, height=2,
                         x_range=(xstart, xend), dynamic=False)
        bounds = (np.datetime64('2015-12-31T23:59:59.723518'), 1.0,
                  np.datetime64('2016-01-03T00:00:00.276482'), 3.0)
        dates = [np.datetime64('2016-01-01T11:59:59.861759000',),
                 np.datetime64('2016-01-02T12:00:00.138241000')]
        expected = Image((dates, [1.5, 2.5], [[1, 0], [0, 2]]),
                         datatype=['xarray'], bounds=bounds, vdims=Dimension('Count', nodata=0))
        expected2 = Image((dates, [1.5, 2.5], [[0, 1], [1, 1]]),
                         datatype=['xarray'], bounds=bounds, vdims=Dimension('Count', nodata=0))
        self.assertEqual(imgs[0], expected)
        self.assertEqual(imgs[1], expected2)

    def test_aggregate_dt_xaxis_constant_yaxis(self):
        df = pd.DataFrame({'y': np.ones(100)}, index=pd.date_range('1980-01-01', periods=100, freq='1T'))
        img = rasterize(Curve(df), dynamic=False, width=3)
        xs = np.array(['1980-01-01T00:16:30.000000', '1980-01-01T00:49:30.000000',
                       '1980-01-01T01:22:30.000000'], dtype='datetime64[us]')
        ys = np.array([])
        bounds = (np.datetime64('1980-01-01T00:00:00.000000'), 1.0,
                  np.datetime64('1980-01-01T01:39:00.000000'), 1.0)
        expected = Image((xs, ys, np.empty((0, 3))), ['index', 'y'],
                         vdims=Dimension('Count', nodata=0), xdensity=1,
                         ydensity=1, bounds=bounds)
        self.assertEqual(img, expected)

    def test_aggregate_ndoverlay(self):
        ds = Dataset([(0.2, 0.3, 0), (0.4, 0.7, 1), (0, 0.99, 2)], kdims=['x', 'y', 'z'])
        ndoverlay = ds.to(Points, ['x', 'y'], [], 'z').overlay()
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]),
                         vdims=[Dimension('Count', nodata=0)])
        img = aggregate(ndoverlay, dynamic=False, x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        self.assertEqual(img, expected)

    def test_aggregate_path(self):
        path = Path([[(0.2, 0.3), (0.4, 0.7)], [(0.4, 0.7), (0.8, 0.99)]])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 1]]),
                         vdims=[Dimension('Count', nodata=0)])
        img = aggregate(path, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        self.assertEqual(img, expected)

    def test_aggregate_contours_with_vdim(self):
        contours = Contours([[(0.2, 0.3, 1), (0.4, 0.7, 1)], [(0.4, 0.7, 2), (0.8, 0.99, 2)]], vdims='z')
        img = rasterize(contours, dynamic=False)
        self.assertEqual(img.vdims, ['z'])

    def test_aggregate_contours_without_vdim(self):
        contours = Contours([[(0.2, 0.3), (0.4, 0.7)], [(0.4, 0.7), (0.8, 0.99)]])
        img = rasterize(contours, dynamic=False)
        self.assertEqual(img.vdims, [Dimension('Any', nodata=0)])

    def test_aggregate_dframe_nan_path(self):
        path = Path([Path([[(0.2, 0.3), (0.4, 0.7)], [(0.4, 0.7), (0.8, 0.99)]]).dframe()])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 1]]),
                         vdims=[Dimension('Count', nodata=0)])
        img = aggregate(path, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        self.assertEqual(img, expected)

    def test_spikes_aggregate_count(self):
        spikes = Spikes([1, 2, 3])
        agg = rasterize(spikes, width=5, dynamic=False, expand=False)
        expected = Image(np.array([[1, 0, 1, 0, 1]]), vdims=Dimension('Count', nodata=0),
                         xdensity=2.5, ydensity=1, bounds=(1, 0, 3, 0.5))
        self.assertEqual(agg, expected)

    def test_spikes_aggregate_count_dask(self):
        spikes = Spikes([1, 2, 3], datatype=['dask'])
        agg = rasterize(spikes, width=5, dynamic=False, expand=False)
        expected = Image(np.array([[1, 0, 1, 0, 1]]), vdims=Dimension('Count', nodata=0),
                         xdensity=2.5, ydensity=1, bounds=(1, 0, 3, 0.5))
        self.assertEqual(agg, expected)

    def test_spikes_aggregate_dt_count(self):
        spikes = Spikes([dt.datetime(2016, 1, 1),  dt.datetime(2016, 1, 2), dt.datetime(2016, 1, 3)])
        agg = rasterize(spikes, width=5, dynamic=False, expand=False)
        bounds = (np.datetime64('2016-01-01T00:00:00.000000'), 0,
                  np.datetime64('2016-01-03T00:00:00.000000'), 0.5)
        expected = Image(np.array([[1, 0, 1, 0, 1]]), vdims=Dimension('Count', nodata=0), bounds=bounds)
        self.assertEqual(agg, expected)

    def test_spikes_aggregate_dt_count_dask(self):
        spikes = Spikes([dt.datetime(2016, 1, 1),  dt.datetime(2016, 1, 2), dt.datetime(2016, 1, 3)],
                        datatype=['dask'])
        agg = rasterize(spikes, width=5, dynamic=False, expand=False)
        bounds = (np.datetime64('2016-01-01T00:00:00.000000'), 0,
                  np.datetime64('2016-01-03T00:00:00.000000'), 0.5)
        expected = Image(np.array([[1, 0, 1, 0, 1]]), vdims=Dimension('Count', nodata=0), bounds=bounds)
        self.assertEqual(agg, expected)

    def test_spikes_aggregate_spike_length(self):
        spikes = Spikes([1, 2, 3])
        agg = rasterize(spikes, width=5, dynamic=False, expand=False, spike_length=7)
        expected = Image(np.array([[1, 0, 1, 0, 1]]), vdims=Dimension('Count', nodata=0),
                         xdensity=2.5, ydensity=1, bounds=(1, 0, 3, 7.0))
        self.assertEqual(agg, expected)

    def test_spikes_aggregate_with_height_count(self):
        spikes = Spikes([(1, 0.2), (2, 0.8), (3, 0.4)], vdims='y')
        agg = rasterize(spikes, width=5, height=5, y_range=(0, 1), dynamic=False)
        xs = [1.2, 1.6, 2.0, 2.4, 2.8]
        ys = [0.1, 0.3, 0.5, 0.7, 0.9]
        arr = np.array([
            [1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1],
            [0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ])
        expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
        self.assertEqual(agg, expected)

    def test_spikes_aggregate_with_height_count_override(self):
        spikes = Spikes([(1, 0.2), (2, 0.8), (3, 0.4)], vdims='y')
        agg = rasterize(spikes, width=5, height=5, y_range=(0, 1),
                        spike_length=0.3, dynamic=False)
        xs = [1.2, 1.6, 2.0, 2.4, 2.8]
        ys = [0.1, 0.3, 0.5, 0.7, 0.9]
        arr = np.array([[1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]])
        expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
        self.assertEqual(agg, expected)

    def test_rasterize_regrid_and_spikes_overlay(self):
        img = Image(([0.5, 1.5], [0.5, 1.5], [[0, 1], [2, 3]]))
        spikes = Spikes([(0.5, 0.2), (1.5, 0.8), ], vdims='y')

        expected_regrid = Image(([0.25, 0.75, 1.25, 1.75],
                                 [0.25, 0.75, 1.25, 1.75],
                                 [[0, 0, 1, 1],
                                  [0, 0, 1, 1],
                                  [2, 2, 3, 3],
                                  [2, 2, 3, 3]]))
        spikes_arr = np.array([[0, 1, 0, 1],
                               [0, 1, 0, 1],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]])
        expected_spikes = Image(([0.25, 0.75, 1.25, 1.75],
                                 [0.25, 0.75, 1.25, 1.75], spikes_arr), vdims=Dimension('Count', nodata=0))
        overlay = img * spikes
        agg = rasterize(overlay, width=4, height=4, x_range=(0, 2), y_range=(0, 2),
                        spike_length=0.5, upsample=True, dynamic=False)
        self.assertEqual(agg.Image.I, expected_regrid)
        self.assertEqual(agg.Spikes.I, expected_spikes)


    def test_spikes_aggregate_with_height_count_dask(self):
        spikes = Spikes([(1, 0.2), (2, 0.8), (3, 0.4)], vdims='y', datatype=['dask'])
        agg = rasterize(spikes, width=5, height=5, y_range=(0, 1), dynamic=False)
        xs = [1.2, 1.6, 2.0, 2.4, 2.8]
        ys = [0.1, 0.3, 0.5, 0.7, 0.9]
        arr = np.array([
            [1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1],
            [0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ])
        expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
        self.assertEqual(agg, expected)

    def test_spikes_aggregate_with_negative_height_count(self):
        spikes = Spikes([(1, -0.2), (2, -0.8), (3, -0.4)], vdims='y', datatype=['dask'])
        agg = rasterize(spikes, width=5, height=5, y_range=(-1, 0), dynamic=False)
        xs = [1.2, 1.6, 2.0, 2.4, 2.8]
        ys = [-0.9, -0.7, -0.5, -0.3, -0.1]
        arr = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 1, 0, 1]
        ])
        expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
        self.assertEqual(agg, expected)

    def test_spikes_aggregate_with_positive_and_negative_height_count(self):
        spikes = Spikes([(1, -0.2), (2, 0.8), (3, -0.4)], vdims='y', datatype=['dask'])
        agg = rasterize(spikes, width=5, height=5, y_range=(-1, 1), dynamic=False)
        xs = [1.2, 1.6, 2.0, 2.4, 2.8]
        ys = [-0.8, -0.4, 0.0, 0.4, 0.8]
        arr = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ])
        expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
        self.assertEqual(agg, expected)

    def test_rectangles_aggregate_count(self):
        rects = Rectangles([(0, 0, 1, 2), (1, 1, 3, 2)])
        agg = rasterize(rects, width=4, height=4, dynamic=False)
        xs = [0.375, 1.125, 1.875, 2.625]
        ys = [0.25, 0.75, 1.25, 1.75]
        arr = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 2, 1, 1],
            [0, 0, 0, 0]
        ])
        expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
        self.assertEqual(agg, expected)

    def test_rectangles_aggregate_count_cat(self):
        rects = Rectangles([(0, 0, 1, 2, 'A'), (1, 1, 3, 2, 'B')], vdims=['cat'])
        agg = rasterize(rects, width=4, height=4, aggregator=ds.count_cat('cat'),
                        dynamic=False)
        xs = [0.375, 1.125, 1.875, 2.625]
        ys = [0.25, 0.75, 1.25, 1.75]
        arr1 = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0]
        ])
        arr2 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 0]
        ])
        expected1 = Image((xs, ys, arr1), vdims=Dimension('cat Count', nodata=0))
        expected2 = Image((xs, ys, arr2), vdims=Dimension('cat Count', nodata=0))
        expected = NdOverlay({'A': expected1, 'B': expected2}, kdims=['cat'])
        self.assertEqual(agg, expected)

    def test_rectangles_aggregate_sum(self):
        rects = Rectangles([(0, 0, 1, 2, 0.5), (1, 1, 3, 2, 1.5)], vdims=['value'])
        agg = rasterize(rects, width=4, height=4, aggregator='sum', dynamic=False)
        xs = [0.375, 1.125, 1.875, 2.625]
        ys = [0.25, 0.75, 1.25, 1.75]
        arr = np.array([
            [0.5, 0.5, nan, nan],
            [0.5, 0.5, nan, nan],
            [0.5, 2. , 1.5, 1.5],
            [nan, nan, nan, nan]
        ])
        expected = Image((xs, ys, arr), vdims='value')
        self.assertEqual(agg, expected)

    def test_rectangles_aggregate_dt_count(self):
        rects = Rectangles([
            (0, dt.datetime(2016, 1, 2), 4, dt.datetime(2016, 1, 3)),
            (1, dt.datetime(2016, 1, 1), 2, dt.datetime(2016, 1, 5))
        ])
        agg = rasterize(rects, width=4, height=4, dynamic=False)
        xs = [0.5, 1.5, 2.5, 3.5]
        ys = [
            np.datetime64('2016-01-01T12:00:00'), np.datetime64('2016-01-02T12:00:00'),
            np.datetime64('2016-01-03T12:00:00'), np.datetime64('2016-01-04T12:00:00')
        ]
        arr = np.array([
            [0, 1, 1, 0],
            [1, 2, 2, 1],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ])
        bounds = (0.0, np.datetime64('2016-01-01T00:00:00'),
                  4.0, np.datetime64('2016-01-05T00:00:00'))
        expected = Image((xs, ys, arr), bounds=bounds, vdims=Dimension('Count', nodata=0))
        self.assertEqual(agg, expected)

    def test_segments_aggregate_count(self):
        segments = Segments([(0, 1, 4, 1), (1, 0, 1, 4)])
        agg = rasterize(segments, width=4, height=4, dynamic=False)
        xs = [0.5, 1.5, 2.5, 3.5]
        ys = [0.5, 1.5, 2.5, 3.5]
        arr = np.array([
            [0, 1, 0, 0],
            [1, 2, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0]
        ])
        expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
        self.assertEqual(agg, expected)

    def test_segments_aggregate_sum(self, instance=False):
        segments = Segments([(0, 1, 4, 1, 2), (1, 0, 1, 4, 4)], vdims=['value'])
        if instance:
            agg = rasterize.instance(
                width=10, height=10, dynamic=False, aggregator='sum'
            )(segments, width=4, height=4)
        else:
            agg = rasterize(
                segments, width=4, height=4, dynamic=False, aggregator='sum'
            )
        xs = [0.5, 1.5, 2.5, 3.5]
        ys = [0.5, 1.5, 2.5, 3.5]
        na = np.nan
        arr = np.array([
            [na, 4, na, na],
            [2 , 6, 2 , 2 ],
            [na, 4, na, na],
            [na, 4, na, na]
        ])
        expected = Image((xs, ys, arr), vdims='value')
        self.assertEqual(agg, expected)

    def test_segments_aggregate_sum_instance(self):
        self.test_segments_aggregate_sum(instance=True)

    def test_segments_aggregate_dt_count(self):
        segments = Segments([
            (0, dt.datetime(2016, 1, 2), 4, dt.datetime(2016, 1, 2)),
            (1, dt.datetime(2016, 1, 1), 1, dt.datetime(2016, 1, 5))
        ])
        agg = rasterize(segments, width=4, height=4, dynamic=False)
        xs = [0.5, 1.5, 2.5, 3.5]
        ys = [
            np.datetime64('2016-01-01T12:00:00'), np.datetime64('2016-01-02T12:00:00'),
            np.datetime64('2016-01-03T12:00:00'), np.datetime64('2016-01-04T12:00:00')
        ]
        arr = np.array([
            [0, 1, 0, 0],
            [1, 2, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0]
        ])
        bounds = (0.0, np.datetime64('2016-01-01T00:00:00'),
                  4.0, np.datetime64('2016-01-05T00:00:00'))
        expected = Image((xs, ys, arr), bounds=bounds, vdims=Dimension('Count', nodata=0))
        self.assertEqual(agg, expected)

    def test_area_aggregate_simple_count(self):
        area = Area([1, 2, 1])
        agg = rasterize(area, width=4, height=4, y_range=(0, 3), dynamic=False)
        xs = [0.25, 0.75, 1.25, 1.75]
        ys = [0.375, 1.125, 1.875, 2.625]
        arr = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ])
        expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
        self.assertEqual(agg, expected)

    def test_area_aggregate_negative_count(self):
        area = Area([-1, -2, -3])
        agg = rasterize(area, width=4, height=4, y_range=(-3, 0), dynamic=False)
        xs = [0.25, 0.75, 1.25, 1.75]
        ys = [-2.625, -1.875, -1.125, -0.375]
        arr = np.array([
            [0, 0, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ])
        expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
        self.assertEqual(agg, expected)

    def test_area_aggregate_crossover_count(self):
        area = Area([-1, 2, 3])
        agg = rasterize(area, width=4, height=4, y_range=(-3, 3), dynamic=False)
        xs = [0.25, 0.75, 1.25, 1.75]
        ys = [-2.25, -0.75, 0.75, 2.25]
        arr = np.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 1, 1]
        ])
        expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
        self.assertEqual(agg, expected)

    def test_spread_aggregate_symmetric_count(self):
        spread = Spread([(0, 1, 0.8), (1, 2, 0.3), (2, 3, 0.8)])
        agg = rasterize(spread, width=4, height=4, dynamic=False)
        xs = [0.25, 0.75, 1.25, 1.75]
        ys = [0.65, 1.55, 2.45, 3.35]
        arr = np.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1]
        ])
        expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
        self.assertEqual(agg, expected)

    def test_spread_aggregate_assymmetric_count(self):
        spread = Spread([(0, 1, 0.4, 0.8), (1, 2, 0.8, 0.4), (2, 3, 0.5, 1)],
                        vdims=['y', 'pos', 'neg'])
        agg = rasterize(spread, width=4, height=4, dynamic=False)
        xs = [0.25, 0.75, 1.25, 1.75]
        ys = [0.6125, 1.4375, 2.2625, 3.0875]
        arr = np.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1]
        ])
        expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
        self.assertEqual(agg, expected)

    def test_rgb_regrid_packed(self):
        coords = {'x': [1, 2], 'y': [1, 2], 'band': [0, 1, 2]}
        arr = np.array([
            [[255, 10],
             [  0, 30]],
            [[  1,  0],
             [  0,  0]],
            [[127,  0],
             [  0, 68]],
        ]).T
        da = xr.DataArray(data=arr, dims=('x', 'y', 'band'), coords=coords)
        im = RGB(da, ['x', 'y'])
        agg = rasterize(im, width=3, height=3, dynamic=False, upsample=True)
        xs = [0.8333333, 1.5, 2.166666]
        ys = [0.8333333, 1.5, 2.166666]
        arr = np.array([
            [[255, 255, 10],
             [255, 255, 10],
             [  0,   0, 30]],
            [[  1,   1,  0],
             [  1,   1,  0],
             [  0,   0,  0]],
            [[127, 127,  0],
             [127, 127,  0],
             [  0,   0, 68]],
        ]).transpose((1, 2, 0))
        expected = RGB((xs, ys, arr))
        self.assertEqual(agg, expected)

    @spatialpandas_skip
    def test_line_rasterize(self):
        path = Path([[(0, 0), (1, 1), (2, 0)], [(0, 0), (0, 1)]], datatype=['spatialpandas'])
        agg = rasterize(path, width=4, height=4, dynamic=False)
        xs = [0.25, 0.75, 1.25, 1.75]
        ys = [0.125, 0.375, 0.625, 0.875]
        arr = np.array([
            [2, 0, 0, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [1, 0, 1, 0]
        ])
        expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
        self.assertEqual(agg, expected)

    @spatialpandas_skip
    def test_multi_line_rasterize(self):
        path = Path([{'x': [0, 1, 2, np.nan, 0, 0], 'y': [0, 1, 0, np.nan, 0, 1]}],
                    datatype=['spatialpandas'])
        agg = rasterize(path, width=4, height=4, dynamic=False)
        xs = [0.25, 0.75, 1.25, 1.75]
        ys = [0.125, 0.375, 0.625, 0.875]
        arr = np.array([
            [2, 0, 0, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [1, 0, 1, 0]
        ])
        expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
        self.assertEqual(agg, expected)

    @spatialpandas_skip
    def test_ring_rasterize(self):
        path = Path([{'x': [0, 1, 2], 'y': [0, 1, 0], 'geom_type': 'Ring'}], datatype=['spatialpandas'])
        agg = rasterize(path, width=4, height=4, dynamic=False)
        xs = [0.25, 0.75, 1.25, 1.75]
        ys = [0.125, 0.375, 0.625, 0.875]
        arr = np.array([
            [1, 1, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 0, 1, 0]
        ])
        expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
        self.assertEqual(agg, expected)

    @spatialpandas_skip
    def test_polygon_rasterize(self):
        poly = Polygons([
            {'x': [0, 1, 2], 'y': [0, 1, 0],
             'holes': [[[(1.6, 0.2), (1, 0.8), (0.4, 0.2)]]]}
        ])
        agg = rasterize(poly, width=6, height=6, dynamic=False)
        xs = [0.166667, 0.5, 0.833333, 1.166667, 1.5, 1.833333]
        ys = [0.083333, 0.25, 0.416667, 0.583333, 0.75, 0.916667]
        arr = np.array([
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
        self.assertEqual(agg, expected)

    @spatialpandas_skip
    def test_polygon_rasterize_mean_agg(self):
        poly = Polygons([
            {'x': [0, 1, 2], 'y': [0, 1, 0], 'z': 2.4},
            {'x': [0, 0, 1], 'y': [0, 1, 1], 'z': 3.6}
        ], vdims='z')
        agg = rasterize(poly, width=4, height=4, dynamic=False, aggregator='mean')
        xs = [0.25, 0.75, 1.25, 1.75]
        ys = [0.125, 0.375, 0.625, 0.875]
        arr = np.array([
            [ 2.4,  2.4,  2.4,    2.4],
            [ 3.6,  2.4,  2.4,    np.nan],
            [ 3.6,  2.4,  2.4,    np.nan],
            [ 3.6,  3.6,  np.nan, np.nan]])
        expected = Image((xs, ys, arr), vdims='z')
        self.assertEqual(agg, expected)

    @spatialpandas_skip
    def test_multi_poly_rasterize(self):
        poly = Polygons([{'x': [0, 1, 2, np.nan, 0, 0, 1],
                          'y': [0, 1, 0, np.nan, 0, 1, 1]}],
                        datatype=['spatialpandas'])
        agg = rasterize(poly, width=4, height=4, dynamic=False)
        xs = [0.25, 0.75, 1.25, 1.75]
        ys = [0.125, 0.375, 0.625, 0.875]
        arr = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 0]
        ])
        expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
        self.assertEqual(agg, expected)



class DatashaderCatAggregateTests(ComparisonTestCase):

    def setUp(self):
        if ds_version < Version('0.11.0'):
            raise SkipTest('Regridding operations require datashader>=0.11.0')

    def test_aggregate_points_categorical(self):
        points = Points([(0.2, 0.3, 'A'), (0.4, 0.7, 'B'), (0, 0.99, 'C')], vdims='z')
        img = aggregate(points, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2, aggregator=ds.by('z', ds.count()))
        x = np.array([0.25, 0.75])
        y = np.array([0.25, 0.75])
        a = np.array([[1, 0], [0, 0]])
        b = np.array([[0, 1], [0, 0]])
        c = np.array([[0, 1], [0, 0]])
        xrds = xr.Dataset(
            coords={"x": x, "y": y},
            data_vars={"a": (("x", "y"), a), "b": (("x", "y"), b), "c": (("x", "y"), c)},
        )
        expected = ImageStack(xrds, kdims=["x", "y"], vdims=["a", "b", "c"])
        actual = img.data
        assert (expected.data.to_array("z").values == actual.T.values).all()

    def test_aggregate_points_categorical_mean(self):
        points = Points([(0.2, 0.3, 'A', 0.1), (0.4, 0.7, 'B', 0.2), (0, 0.99, 'C', 0.3)], vdims=['cat', 'z'])
        img = aggregate(points, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2, aggregator=ds.by('cat', ds.mean('z')))
        x = np.array([0.25, 0.75])
        y = np.array([0.25, 0.75])
        a = np.array([[0.1, np.nan], [np.nan, np.nan]])
        b = np.array([[np.nan, 0.2], [np.nan, np.nan]])
        c = np.array([[np.nan, 0.3], [np.nan, np.nan]])
        xrds = xr.Dataset(
            coords={"x": x, "y": y},
            data_vars={"a": (("x", "y"), a), "b": (("x", "y"), b), "c": (("x", "y"), c)},
        )
        expected = ImageStack(xrds, kdims=["x", "y"], vdims=["a", "b", "c"])
        actual = img.data
        np.testing.assert_equal(expected.data.to_array("z").values, actual.T.values)


class DatashaderShadeTests(ComparisonTestCase):

    def test_shade_categorical_images_xarray(self):
        xs, ys = [0.25, 0.75], [0.25, 0.75]
        data = NdOverlay({'A': Image((xs, ys, np.array([[1, 0], [0, 0]], dtype='u4')),
                                     datatype=['xarray'], vdims=Dimension('z Count', nodata=0)),
                          'B': Image((xs, ys, np.array([[0, 0], [1, 0]], dtype='u4')),
                                     datatype=['xarray'], vdims=Dimension('z Count', nodata=0)),
                          'C': Image((xs, ys, np.array([[0, 0], [1, 0]], dtype='u4')),
                                     datatype=['xarray'], vdims=Dimension('z Count', nodata=0))},
                         kdims=['z'])
        shaded = shade(data, rescale_discrete_levels=False)
        r = [[228, 120], [66, 120]]
        g = [[26, 109], [150, 109]]
        b = [[28, 95], [129, 95]]
        a = [[40, 0], [255, 0]]
        expected = RGB((xs, ys, r, g, b, a), datatype=['grid'],
                       vdims=RGB.vdims+[Dimension('A', range=(0, 1))])
        self.assertEqual(shaded, expected)

    def test_shade_categorical_images_grid(self):
        xs, ys = [0.25, 0.75], [0.25, 0.75]
        data = NdOverlay({'A': Image((xs, ys, np.array([[1, 0], [0, 0]], dtype='u4')),
                                     datatype=['grid'], vdims=Dimension('z Count', nodata=0)),
                          'B': Image((xs, ys, np.array([[0, 0], [1, 0]], dtype='u4')),
                                     datatype=['grid'], vdims=Dimension('z Count', nodata=0)),
                          'C': Image((xs, ys, np.array([[0, 0], [1, 0]], dtype='u4')),
                                     datatype=['grid'], vdims=Dimension('z Count', nodata=0))},
                         kdims=['z'])
        shaded = shade(data, rescale_discrete_levels=False)
        r = [[228, 120], [66, 120]]
        g = [[26, 109], [150, 109]]
        b = [[28, 95], [129, 95]]
        a = [[40, 0], [255, 0]]
        expected = RGB((xs, ys, r, g, b, a), datatype=['grid'],
                       vdims=RGB.vdims+[Dimension('A', range=(0, 1))])
        self.assertEqual(shaded, expected)

    def test_shade_dt_xaxis_constant_yaxis(self):
        df = pd.DataFrame({'y': np.ones(100)}, index=pd.date_range('1980-01-01', periods=100, freq='1T'))
        rgb = shade(rasterize(Curve(df), dynamic=False, width=3))
        xs = np.array(['1980-01-01T00:16:30.000000', '1980-01-01T00:49:30.000000',
                       '1980-01-01T01:22:30.000000'], dtype='datetime64[us]')
        ys = np.array([])
        bounds = (np.datetime64('1980-01-01T00:00:00.000000'), 1.0,
                  np.datetime64('1980-01-01T01:39:00.000000'), 1.0)
        expected = RGB((xs, ys, np.empty((0, 3, 4))), ['index', 'y'],
                       xdensity=1, ydensity=1, bounds=bounds)
        self.assertEqual(rgb, expected)



class DatashaderRegridTests(ComparisonTestCase):
    """
    Tests for datashader aggregation
    """

    def setUp(self):
        if ds_version <= Version('0.5.0'):
            raise SkipTest('Regridding operations require datashader>=0.6.0')

    def test_regrid_mean(self):
        img = Image((range(10), range(5), np.arange(10) * np.arange(5)[np.newaxis].T))
        regridded = regrid(img, width=2, height=2, dynamic=False)
        expected = Image(([2., 7.], [0.75, 3.25], [[1, 5], [6, 22]]))
        self.assertEqual(regridded, expected)

    def test_regrid_mean_xarray_transposed(self):
        img = Image((range(10), range(5), np.arange(10) * np.arange(5)[np.newaxis].T),
                    datatype=['xarray'])
        img.data = img.data.transpose()
        regridded = regrid(img, width=2, height=2, dynamic=False)
        expected = Image(([2., 7.], [0.75, 3.25], [[1, 5], [6, 22]]))
        self.assertEqual(regridded, expected)

    def test_regrid_rgb_mean(self):
        arr = (np.arange(10) * np.arange(5)[np.newaxis].T).astype('f')
        rgb = RGB((range(10), range(5), arr, arr*2, arr*2))
        regridded = regrid(rgb, width=2, height=2, dynamic=False)
        new_arr = np.array([[1.6, 5.6], [6.4, 22.4]])
        expected = RGB(([2., 7.], [0.75, 3.25], new_arr, new_arr*2, new_arr*2), datatype=['xarray'])
        self.assertEqual(regridded, expected)

    def test_regrid_max(self):
        img = Image((range(10), range(5), np.arange(10) * np.arange(5)[np.newaxis].T))
        regridded = regrid(img, aggregator='max', width=2, height=2, dynamic=False)
        expected = Image(([2., 7.], [0.75, 3.25], [[8, 18], [16, 36]]))
        self.assertEqual(regridded, expected)

    def test_regrid_upsampling(self):
        img = Image(([0.5, 1.5], [0.5, 1.5], [[0, 1], [2, 3]]))
        regridded = regrid(img, width=4, height=4, upsample=True, dynamic=False)
        expected = Image(([0.25, 0.75, 1.25, 1.75], [0.25, 0.75, 1.25, 1.75],
                          [[0, 0, 1, 1],
                           [0, 0, 1, 1],
                           [2, 2, 3, 3],
                           [2, 2, 3, 3]]))
        self.assertEqual(regridded, expected)

    def test_regrid_upsampling_linear(self):
        img = Image(([0.5, 1.5], [0.5, 1.5], [[0, 1], [2, 3]]))
        regridded = regrid(img, width=4, height=4, upsample=True, interpolation='linear', dynamic=False)
        expected = Image(([0.25, 0.75, 1.25, 1.75], [0.25, 0.75, 1.25, 1.75],
                          [[0, 0, 0, 1],
                           [0, 1, 1, 1],
                           [1, 1, 2, 2],
                           [2, 2, 2, 3]]))
        self.assertEqual(regridded, expected)

    def test_regrid_disabled_upsampling(self):
        img = Image(([0.5, 1.5], [0.5, 1.5], [[0, 1], [2, 3]]))
        regridded = regrid(img, width=3, height=3, dynamic=False, upsample=False)
        self.assertEqual(regridded, img)

    def test_regrid_disabled_expand(self):
        img = Image(([0.5, 1.5], [0.5, 1.5], [[0., 1.], [2., 3.]]))
        regridded = regrid(img, width=2, height=2, x_range=(-2, 4), y_range=(-2, 4), expand=False,
                           dynamic=False)
        self.assertEqual(regridded, img)

    def test_regrid_zero_range(self):
        ls = np.linspace(0, 10, 200)
        xx, yy = np.meshgrid(ls, ls)
        img = Image(np.sin(xx)*np.cos(yy), bounds=(0, 0, 1, 1))
        regridded = regrid(img, x_range=(-1, -0.5), y_range=(-1, -0.5), dynamic=False)
        expected = Image(np.zeros((0, 0)), bounds=(0, 0, 0, 0), xdensity=1, ydensity=1)
        self.assertEqual(regridded, expected)



class DatashaderRasterizeTests(ComparisonTestCase):
    """
    Tests for datashader aggregation
    """

    def setUp(self):
        if ds_version <= Version('0.6.4'):
            raise SkipTest('Regridding operations require datashader>=0.7.0')

        self.simplexes = [(0, 1, 2), (3, 2, 1)]
        self.vertices = [(0., 0.), (0., 1.), (1., 0), (1, 1)]
        self.simplexes_vdim = [(0, 1, 2, 0.5), (3, 2, 1, 1.5)]
        self.vertices_vdim = [(0., 0., 1), (0., 1., 2), (1., 0, 3), (1, 1, 4)]

    def test_rasterize_trimesh_no_vdims(self):
        trimesh = TriMesh((self.simplexes, self.vertices))
        img = rasterize(trimesh, width=3, height=3, dynamic=False)
        image = Image(np.array([[True, True, True], [True, True, True], [True, True, True]]),
                      bounds=(0, 0, 1, 1), vdims=Dimension('Any', nodata=0))
        self.assertEqual(img, image)

    def test_rasterize_trimesh_no_vdims_zero_range(self):
        trimesh = TriMesh((self.simplexes, self.vertices))
        img = rasterize(trimesh, height=2, x_range=(0, 0), dynamic=False)
        image = Image(([], [0.25, 0.75], np.zeros((2, 0))),
                      bounds=(0, 0, 0, 1), xdensity=1, vdims=Dimension('Any', nodata=0))
        self.assertEqual(img, image)

    def test_rasterize_trimesh_with_vdims_as_wireframe(self):
        trimesh = TriMesh((self.simplexes_vdim, self.vertices), vdims=['z'])
        img = rasterize(trimesh, width=3, height=3, aggregator='any', interpolation=None, dynamic=False)
        array = np.array([
            [True, True, True],
            [True, True, True],
            [True, True, True]
        ])
        image = Image(array, bounds=(0, 0, 1, 1), vdims=Dimension('Any', nodata=0))
        self.assertEqual(img, image)

    def test_rasterize_trimesh(self):
        trimesh = TriMesh((self.simplexes_vdim, self.vertices), vdims=['z'])
        img = rasterize(trimesh, width=3, height=3, dynamic=False)
        array = np.array([
            [0.5, 1.5, 1.5],
            [0.5, 0.5, 1.5],
            [0.5, 0.5, 0.5]
        ])
        image = Image(array, bounds=(0, 0, 1, 1))
        self.assertEqual(img, image)

    def test_rasterize_pandas_trimesh_implicit_nodes(self):
        simplex_df = pd.DataFrame(self.simplexes, columns=['v0', 'v1', 'v2'])
        vertex_df = pd.DataFrame(self.vertices_vdim, columns=['x', 'y', 'z'])

        trimesh = TriMesh((simplex_df, vertex_df))
        img = rasterize(trimesh, width=3, height=3, dynamic=False)

        array = np.array([
            [2.166667, 2.833333, 3.5     ],
            [1.833333, 2.5,      3.166667],
            [1.5,      2.166667, 2.833333]
        ])
        image = Image(array, bounds=(0, 0, 1, 1))
        self.assertEqual(img, image)

    def test_rasterize_dask_trimesh_implicit_nodes(self):
        simplex_df = pd.DataFrame(self.simplexes, columns=['v0', 'v1', 'v2'])
        vertex_df = pd.DataFrame(self.vertices_vdim, columns=['x', 'y', 'z'])

        simplex_ddf = dd.from_pandas(simplex_df, npartitions=2)
        vertex_ddf = dd.from_pandas(vertex_df, npartitions=2)

        trimesh = TriMesh((simplex_ddf, vertex_ddf))

        ri = rasterize.instance()
        img = ri(trimesh, width=3, height=3, dynamic=False, precompute=True)

        cache = ri._precomputed
        self.assertEqual(len(cache), 1)
        self.assertIn(trimesh._plot_id, cache)
        self.assertIsInstance(cache[trimesh._plot_id]['mesh'], dd.DataFrame)

        array = np.array([
            [2.166667, 2.833333, 3.5     ],
            [1.833333, 2.5,      3.166667],
            [1.5,      2.166667, 2.833333]
        ])
        image = Image(array, bounds=(0, 0, 1, 1))
        self.assertEqual(img, image)

    def test_rasterize_dask_trimesh(self):
        simplex_df = pd.DataFrame(self.simplexes_vdim, columns=['v0', 'v1', 'v2', 'z'])
        vertex_df = pd.DataFrame(self.vertices, columns=['x', 'y'])

        simplex_ddf = dd.from_pandas(simplex_df, npartitions=2)
        vertex_ddf = dd.from_pandas(vertex_df, npartitions=2)

        tri_nodes = Nodes(vertex_ddf, ['x', 'y', 'index'])
        trimesh = TriMesh((simplex_ddf, tri_nodes), vdims=['z'])

        ri = rasterize.instance()
        img = ri(trimesh, width=3, height=3, dynamic=False, precompute=True)

        cache = ri._precomputed
        self.assertEqual(len(cache), 1)
        self.assertIn(trimesh._plot_id, cache)
        self.assertIsInstance(cache[trimesh._plot_id]['mesh'], dd.DataFrame)

        array = np.array([
            [0.5, 1.5, 1.5],
            [0.5, 0.5, 1.5],
            [0.5, 0.5, 0.5]
        ])
        image = Image(array, bounds=(0, 0, 1, 1))
        self.assertEqual(img, image)

    def test_rasterize_dask_trimesh_with_node_vdims(self):
        simplex_df = pd.DataFrame(self.simplexes, columns=['v0', 'v1', 'v2'])
        vertex_df = pd.DataFrame(self.vertices_vdim, columns=['x', 'y', 'z'])

        simplex_ddf = dd.from_pandas(simplex_df, npartitions=2)
        vertex_ddf = dd.from_pandas(vertex_df, npartitions=2)

        tri_nodes = Nodes(vertex_ddf, ['x', 'y', 'index'], ['z'])
        trimesh = TriMesh((simplex_ddf, tri_nodes))

        ri = rasterize.instance()
        img = ri(trimesh, width=3, height=3, dynamic=False, precompute=True)

        cache = ri._precomputed
        self.assertEqual(len(cache), 1)
        self.assertIn(trimesh._plot_id, cache)
        self.assertIsInstance(cache[trimesh._plot_id]['mesh'], dd.DataFrame)

        array = np.array([
            [2.166667, 2.833333, 3.5     ],
            [1.833333, 2.5,      3.166667],
            [1.5,      2.166667, 2.833333]
        ])
        image = Image(array, bounds=(0, 0, 1, 1))
        self.assertEqual(img, image)

    def test_rasterize_trimesh_node_vdim_precedence(self):
        nodes = Points(self.vertices_vdim, vdims=['node_z'])
        trimesh = TriMesh((self.simplexes_vdim, nodes), vdims=['z'])
        img = rasterize(trimesh, width=3, height=3, dynamic=False)

        array = np.array([
            [2.166667, 2.833333, 3.5     ],
            [1.833333, 2.5,      3.166667],
            [1.5,      2.166667, 2.833333]
        ])
        image = Image(array, bounds=(0, 0, 1, 1), vdims='node_z')
        self.assertEqual(img, image)

    def test_rasterize_trimesh_node_explicit_vdim(self):
        nodes = Points(self.vertices_vdim, vdims=['node_z'])
        trimesh = TriMesh((self.simplexes_vdim, nodes), vdims=['z'])
        img = rasterize(trimesh, width=3, height=3, dynamic=False, aggregator=ds.mean('z'))

        array = np.array([
            [0.5, 1.5, 1.5],
            [0.5, 0.5, 1.5],
            [0.5, 0.5, 0.5]
        ])
        image = Image(array, bounds=(0, 0, 1, 1))
        self.assertEqual(img, image)

    def test_rasterize_trimesh_zero_range(self):
        trimesh = TriMesh((self.simplexes_vdim, self.vertices), vdims=['z'])
        img = rasterize(trimesh, x_range=(0, 0), height=2, dynamic=False)
        image = Image(([], [0.25, 0.75], np.zeros((2, 0))),
                      bounds=(0, 0, 0, 1), xdensity=1)
        self.assertEqual(img, image)

    def test_rasterize_trimesh_vertex_vdims(self):
        simplices = [(0, 1, 2), (3, 2, 1)]
        vertices = [(0., 0., 1), (0., 1., 2), (1., 0., 3), (1., 1., 4)]
        trimesh = TriMesh((simplices, Points(vertices, vdims='z')))
        img = rasterize(trimesh, width=3, height=3, dynamic=False)

        array = np.array([
            [2.166667, 2.833333, 3.5     ],
            [1.833333, 2.5,      3.166667],
            [1.5,      2.166667, 2.833333]
        ])
        image = Image(array, bounds=(0, 0, 1, 1), vdims='z')
        self.assertEqual(img, image)

    def test_rasterize_trimesh_ds_aggregator(self):
        trimesh = TriMesh((self.simplexes_vdim, self.vertices), vdims=['z'])
        img = rasterize(trimesh, width=3, height=3, dynamic=False, aggregator=ds.mean('z'))
        array = np.array([
            [0.5, 1.5, 1.5],
            [0.5, 0.5, 1.5],
            [0.5, 0.5, 0.5]
        ])
        image = Image(array, bounds=(0, 0, 1, 1))
        self.assertEqual(img, image)

    def test_rasterize_trimesh_string_aggregator(self):
        trimesh = TriMesh((self.simplexes_vdim, self.vertices), vdims=['z'])
        img = rasterize(trimesh, width=3, height=3, dynamic=False, aggregator='mean')
        array = np.array([
            [0.5, 1.5, 1.5],
            [0.5, 0.5, 1.5],
            [0.5, 0.5, 0.5]
        ])
        image = Image(array, bounds=(0, 0, 1, 1))
        self.assertEqual(img, image)

    def test_rasterize_quadmesh(self):
        qmesh = QuadMesh(([0, 1], [0, 1], np.array([[0, 1], [2, 3]])))
        img = rasterize(qmesh, width=3, height=3, dynamic=False, aggregator=ds.mean('z'))
        image = Image(np.array([[2, 3, 3], [2, 3, 3], [0, 1, 1]]),
                      bounds=(-.5, -.5, 1.5, 1.5))
        self.assertEqual(img, image)

    def test_rasterize_quadmesh_string_aggregator(self):
        qmesh = QuadMesh(([0, 1], [0, 1], np.array([[0, 1], [2, 3]])))
        img = rasterize(qmesh, width=3, height=3, dynamic=False, aggregator='mean')
        image = Image(np.array([[2, 3, 3], [2, 3, 3], [0, 1, 1]]),
                      bounds=(-.5, -.5, 1.5, 1.5))
        self.assertEqual(img, image)

    def test_rasterize_points(self):
        points = Points([(0.2, 0.3), (0.4, 0.7), (0, 0.99)])
        img = rasterize(points, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]),
                         vdims=[Dimension('Count', nodata=0)])
        self.assertEqual(img, expected)

    def test_rasterize_curve(self):
        curve = Curve([(0.2, 0.3), (0.4, 0.7), (0.8, 0.99)])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [1, 1]]),
                         vdims=[Dimension('Count', nodata=0)])
        img = rasterize(curve, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        self.assertEqual(img, expected)

    def test_rasterize_ndoverlay(self):
        ds = Dataset([(0.2, 0.3, 0), (0.4, 0.7, 1), (0, 0.99, 2)], kdims=['x', 'y', 'z'])
        ndoverlay = ds.to(Points, ['x', 'y'], [], 'z').overlay()
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]),
                         vdims=[Dimension('Count', nodata=0)])
        img = rasterize(ndoverlay, dynamic=False, x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        self.assertEqual(img, expected)

    def test_rasterize_path(self):
        path = Path([[(0.2, 0.3), (0.4, 0.7)], [(0.4, 0.7), (0.8, 0.99)]])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 1]]),
                         vdims=[Dimension('Count', nodata=0)])
        img = rasterize(path, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        self.assertEqual(img, expected)

    def test_rasterize_image(self):
        img = Image((range(10), range(5), np.arange(10) * np.arange(5)[np.newaxis].T))
        regridded = regrid(img, width=2, height=2, dynamic=False)
        expected = Image(([2., 7.], [0.75, 3.25], [[1, 5], [6, 22]]))
        self.assertEqual(regridded, expected)

    def test_rasterize_image_string_aggregator(self):
        img = Image((range(10), range(5), np.arange(10) * np.arange(5)[np.newaxis].T))
        regridded = regrid(img, width=2, height=2, dynamic=False, aggregator='mean')
        expected = Image(([2., 7.], [0.75, 3.25], [[1, 5], [6, 22]]))
        self.assertEqual(regridded, expected)

    def test_rasterize_image_expand_default(self):
        # Should use expand=False by default
        assert not regrid.expand

        data = np.arange(100.0).reshape(10, 10)
        c = np.arange(10.0)
        da = xr.DataArray(data, coords=dict(x=c, y=c))
        rast_input = dict(x_range=(-1, 10), y_range=(-1, 10), precompute=True, dynamic=False)
        img = rasterize(Image(da), **rast_input)
        output = img.data["z"].to_numpy()

        np.testing.assert_array_equal(output, data.T)
        assert not np.isnan(output).any()

        # Setting expand=True with the {x,y}_ranges will expand the data with nan's
        img = rasterize(Image(da), expand=True, **rast_input)
        output = img.data["z"].to_numpy()
        assert np.isnan(output).any()

    def test_rasterize_apply_when_instance_with_line_width(self):
        df = pd.DataFrame(
            np.random.multivariate_normal(
            (0, 0), [[0.1, 0.1], [0.1, 1.0]], (100,))
        )
        df.columns = ["a", "b"]

        curve = Curve(df, kdims=["a"], vdims=["b"])
        # line_width is not a parameter
        custom_rasterize = rasterize.instance(line_width=2)
        assert {'line_width': 2} == custom_rasterize._rasterize__instance_kwargs
        output = apply_when(
            curve, operation=custom_rasterize, predicate=lambda x: len(x) > 10
        )
        render(output, "bokeh")
        assert isinstance(output, DynamicMap)
        overlay = output.items()[0][1]
        assert isinstance(overlay, Overlay)
        assert len(overlay) == 2


@pytest.mark.parametrize("agg_input_fn,index_col",
    (
        [ds.first, [311, 433, 309, 482]],
        [ds.last, [491, 483, 417, 482]],
        [ds.min, [311, 433, 309, 482]],
        [ds.max, [404, 433, 417, 482]],
    )
)
def test_rasterize_where_agg_no_column(point_plot, agg_input_fn, index_col):
    agg_fn = ds.where(agg_input_fn("val"))
    rast_input = dict(dynamic=False,  x_range=(-1, 1), y_range=(-1, 1), width=2, height=2)
    img = rasterize(point_plot, aggregator=agg_fn, **rast_input)

    assert list(img.data) == ["index", "s", "val", "cat"]
    assert list(img.vdims) == ["val", "s", "cat"]  # val first and no index

    # N=100 in point_data is chosen to have a big enough sample size
    # so that the index are not the same for the different agg_input_fn
    np.testing.assert_array_equal(img.data["index"].data.flatten(), index_col)

    img_simple = rasterize(point_plot, aggregator=agg_input_fn("val"), **rast_input)
    np.testing.assert_array_equal(img_simple["val"], img["val"])


@pytest.mark.parametrize("agg_input_fn", (ds.first, ds.last, ds.min, ds.max))
def test_rasterize_where_agg_with_column(point_plot, agg_input_fn):
    agg_fn = ds.where(agg_input_fn("val"), "s")
    rast_input = dict(dynamic=False,  x_range=(-1, 1), y_range=(-1, 1), width=2, height=2)
    img = rasterize(point_plot, aggregator=agg_fn, **rast_input)

    assert list(img.data) == ["s"]
    img_no_column = rasterize(point_plot, aggregator=ds.where(agg_input_fn("val")), **rast_input)
    np.testing.assert_array_equal(img["s"], img_no_column["s"])


def test_rasterize_summerize(point_plot):
    agg_fn_count, agg_fn_first = ds.count(), ds.first("val")
    agg_fn = ds.summary(count=agg_fn_count, first=agg_fn_first)
    rast_input = dict(dynamic=False,  x_range=(-1, 1), y_range=(-1, 1), width=2, height=2)
    img_sum = rasterize(point_plot, aggregator=agg_fn, **rast_input)
    img_count = rasterize(point_plot, aggregator=agg_fn_count, **rast_input)
    img_first = rasterize(point_plot, aggregator=agg_fn_first, **rast_input)

    np.testing.assert_array_equal(img_sum["first"], img_first["val"])

    # Count has special handling in AggregationOperation which sets nodata=0
    # this is not done for count in summary.
    np.testing.assert_array_equal(img_sum["count"], np.nan_to_num(img_count["Count"]))


@pytest.mark.parametrize("sel_fn", (ds.first, ds.last, ds.min, ds.max))
def test_rasterize_selector(point_plot, sel_fn):
    rast_input = dict(dynamic=False,  x_range=(-1, 1), y_range=(-1, 1), width=2, height=2)
    img = rasterize(point_plot, selector=sel_fn("val"), **rast_input)

    # Count is from the aggregator
    assert list(img.data) == ["Count", "index", "s", "val", "cat"]
    assert list(img.vdims) == ["Count", "s", "val", "cat"]  # no index

    # The output for the selector should be equal to the output for the aggregator using
    # ds.where
    img_agg = rasterize(point_plot, aggregator=ds.where(sel_fn("val")), **rast_input)
    for c in ["s", "val", "cat"]:
        np.testing.assert_array_equal(img[c], img_agg[c])

    # Checking the count is also the same
    img_count = rasterize(point_plot, **rast_input)
    np.testing.assert_array_equal(img["Count"], img_count["Count"])



class DatashaderSpreadTests(ComparisonTestCase):

    def test_spread_rgb_1px(self):
        arr = np.array([[[0, 0, 0], [0, 1, 1], [0, 1, 1]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.uint8).T*255
        spreaded = spread(RGB(arr))
        arr = np.array([[[0, 0, 1], [0, 0, 1], [0, 0, 1]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]]], dtype=np.uint8).T*255
        self.assertEqual(spreaded, RGB(arr))

    def test_spread_img_1px(self):
        if ds_version < Version('0.12.0'):
            raise SkipTest('Datashader does not support DataArray yet')
        arr = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]]).T
        spreaded = spread(Image(arr))
        arr = np.array([[0, 0, 0], [2, 3, 2], [2, 3, 2]]).T
        self.assertEqual(spreaded, Image(arr))


class DatashaderStackTests(ComparisonTestCase):

    def setUp(self):
        self.rgb1_arr = np.array([[[0, 1], [1, 0]],
                                  [[1, 0], [0, 1]],
                                  [[0, 0], [0, 0]]], dtype=np.uint8).T*255
        self.rgb2_arr = np.array([[[0, 0], [0, 0]],
                                  [[0, 0], [0, 0]],
                                  [[1, 0], [0, 1]]], dtype=np.uint8).T*255
        self.rgb1 = RGB(self.rgb1_arr)
        self.rgb2 = RGB(self.rgb2_arr)


    def test_stack_add_compositor(self):
        combined = stack(self.rgb1*self.rgb2, compositor='add')
        arr = np.array([[[0, 255, 255], [255,0, 0]], [[255, 0, 0], [0, 255, 255]]], dtype=np.uint8)
        expected = RGB(arr)
        self.assertEqual(combined, expected)

    def test_stack_over_compositor(self):
        combined = stack(self.rgb1*self.rgb2, compositor='over')
        self.assertEqual(combined, self.rgb2)

    def test_stack_over_compositor_reverse(self):
        combined = stack(self.rgb2*self.rgb1, compositor='over')
        self.assertEqual(combined, self.rgb1)

    def test_stack_saturate_compositor(self):
        combined = stack(self.rgb1*self.rgb2, compositor='saturate')
        self.assertEqual(combined, self.rgb1)

    def test_stack_saturate_compositor_reverse(self):
        combined = stack(self.rgb2*self.rgb1, compositor='saturate')
        self.assertEqual(combined, self.rgb2)


class GraphBundlingTests(ComparisonTestCase):

    def setUp(self):
        if ds_version <= Version('0.7.0'):
            raise SkipTest('Regridding operations require datashader>=0.7.0')
        self.source = np.arange(8)
        self.target = np.zeros(8)
        self.graph = Graph(((self.source, self.target),))

    def test_directly_connect_paths(self):
        direct = directly_connect_edges(self.graph)._split_edgepaths
        self.assertEqual(direct, self.graph.edgepaths)

class InspectorTests(ComparisonTestCase):
    """
    Tests for inspector operations
    """
    def setUp(self):
        points = Points([(0.2, 0.3), (0.4, 0.7), (0, 0.99)])
        self.pntsimg = rasterize(points, dynamic=False,
                             x_range=(0, 1), y_range=(0, 1), width=4, height=4)
        if spatialpandas is None:
            return

        xs1, xs2, ys1, ys2 = [1, 2, 3], [6, 7, 3], [2, 0, 7], [7, 5, 2]
        holes = [ [[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]],]
        polydata = [{'x': xs1, 'y': ys1, 'holes': holes, 'z': 1},
                    {'x': xs2, 'y': ys2, 'holes': [[]], 'z': 2}]
        self.polysrgb = datashade(Polygons(polydata, vdims=['z'],
                                           datatype=['spatialpandas']),
                                  x_range=(0, 7), y_range=(0, 7), dynamic=False)

    def tearDown(self):
        Tap.x, Tap.y = None, None


    def test_inspect_points_or_polygons(self):
        if spatialpandas is None:
            raise SkipTest('Polygon inspect tests require spatialpandas')
        polys = inspect(self.polysrgb,
                        max_indicators=3, dynamic=False, pixels=1, x=6, y=5)
        self.assertEqual(polys, Polygons([{'x': [6, 3, 7], 'y': [7, 2, 5], 'z': 2}], vdims='z'))
        points = inspect(self.pntsimg, max_indicators=3, dynamic=False, pixels=1, x=-0.1, y=-0.1)
        self.assertEqual(points.dimension_values('x'), np.array([]))
        self.assertEqual(points.dimension_values('y'), np.array([]))

    def test_points_inspection_1px_mask(self):
        points = inspect_points(self.pntsimg, max_indicators=3, dynamic=False, pixels=1, x=-0.1, y=-0.1)
        self.assertEqual(points.dimension_values('x'), np.array([]))
        self.assertEqual(points.dimension_values('y'), np.array([]))

    def test_points_inspection_2px_mask(self):
        points = inspect_points(self.pntsimg, max_indicators=3, dynamic=False, pixels=2, x=-0.1, y=-0.1)
        self.assertEqual(points.dimension_values('x'), np.array([0.2]))
        self.assertEqual(points.dimension_values('y'), np.array([0.3]))

    def test_points_inspection_4px_mask(self):
        points = inspect_points(self.pntsimg, max_indicators=3, dynamic=False, pixels=4, x=-0.1, y=-0.1)
        self.assertEqual(points.dimension_values('x'), np.array([0.2, 0.4]))
        self.assertEqual(points.dimension_values('y'), np.array([0.3, 0.7]))

    def test_points_inspection_5px_mask(self):
        points = inspect_points(self.pntsimg, max_indicators=3, dynamic=False, pixels=5, x=-0.1, y=-0.1)
        self.assertEqual(points.dimension_values('x'), np.array([0.2, 0.4, 0]))
        self.assertEqual(points.dimension_values('y'), np.array([0.3, 0.7, 0.99]))

    def test_inspection_5px_mask_points_df(self):
        inspector = inspect.instance(max_indicators=3, dynamic=False, pixels=5,
                                     x=-0.1, y=-0.1)
        inspector(self.pntsimg)
        self.assertEqual(list(inspector.hits['x']),[0.2,0.4,0.0])
        self.assertEqual(list(inspector.hits['y']),[0.3,0.7,0.99])

    def test_points_inspection_dict_streams(self):
        Tap.x, Tap.y = 0.4, 0.7
        points = inspect_points(self.pntsimg, max_indicators=3, dynamic=True,
                                pixels=1, streams=dict(x=Tap.param.x, y=Tap.param.y))
        self.assertEqual(len(points.streams), 1)
        self.assertEqual(isinstance(points.streams[0], Tap), True)
        self.assertEqual(points.streams[0].x, 0.4)
        self.assertEqual(points.streams[0].y, 0.7)

    def test_points_inspection_dict_streams_instance(self):
        Tap.x, Tap.y = 0.2, 0.3
        inspector = inspect_points.instance(max_indicators=3, dynamic=True, pixels=1,
                                            streams=dict(x=Tap.param.x, y=Tap.param.y))
        points = inspector(self.pntsimg)
        self.assertEqual(len(points.streams), 1)
        self.assertEqual(isinstance(points.streams[0], Tap), True)
        self.assertEqual(points.streams[0].x, 0.2)
        self.assertEqual(points.streams[0].y, 0.3)

    def test_polys_inspection_1px_mask_hit(self):
        if spatialpandas is None:
            raise SkipTest('Polygon inspect tests require spatialpandas')
        polys = inspect_polygons(self.polysrgb,
                                 max_indicators=3, dynamic=False, pixels=1, x=6, y=5)
        self.assertEqual(polys, Polygons([{'x': [6, 3, 7], 'y': [7, 2, 5], 'z': 2}],
                                         vdims='z'))


    def test_inspection_1px_mask_poly_df(self):
        if spatialpandas is None:
            raise SkipTest('Polygon inspect tests require spatialpandas')
        inspector = inspect.instance(max_indicators=3, dynamic=False, pixels=1, x=6, y=5)
        inspector(self.polysrgb)
        self.assertEqual(len(inspector.hits), 1)
        data = [[6.0, 7.0, 3.0, 2.0, 7.0, 5.0, 6.0, 7.0]]
        self.assertEqual(inspector.hits.iloc[0].geometry,
                         spatialpandas.geometry.polygon.Polygon(data))


    def test_polys_inspection_1px_mask_miss(self):
        if spatialpandas is None:
            raise SkipTest('Polygon inspect tests require spatialpandas')
        polys = inspect_polygons(self.polysrgb,
                                 max_indicators=3, dynamic=False, pixels=1, x=0, y=0)
        self.assertEqual(polys, Polygons([], vdims='z'))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_uint_dtype(dtype):
    df = pd.DataFrame(np.arange(2, dtype=dtype), columns=["A"])
    curve = Curve(df)
    img = rasterize(curve, dynamic=False, height=10, width=10)
    assert (np.asarray(img.data["Count"]) == np.eye(10)).all()


def test_uint64_dtype():
    df = pd.DataFrame(np.arange(2, dtype=np.uint64), columns=["A"])
    curve = Curve(df)
    with pytest.raises(TypeError, match="Dtype of uint64 for column A is not supported."):
        rasterize(curve, dynamic=False, height=10, width=10)
