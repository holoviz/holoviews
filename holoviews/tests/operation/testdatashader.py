from unittest import SkipTest

import numpy as np
from holoviews import (Dimension, Curve, Points, Image, Dataset, RGB, Path,
                       Graph, TriMesh, QuadMesh, NdOverlay, Contours)
from holoviews.element.comparison import ComparisonTestCase

try:
    import datashader as ds
    import dask.dataframe as dd
    from holoviews.core.util import pd
    from holoviews.operation.datashader import (
        aggregate, regrid, ds_version, stack, directly_connect_edges,
        shade, rasterize
    )
except:
    raise SkipTest('Datashader not available')


class DatashaderAggregateTests(ComparisonTestCase):
    """
    Tests for datashader aggregation
    """

    def test_aggregate_points(self):
        points = Points([(0.2, 0.3), (0.4, 0.7), (0, 0.99)])
        img = aggregate(points, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]),
                         vdims=['Count'])
        self.assertEqual(img, expected)

    def test_aggregate_zero_range_points(self):
        p = Points([(0, 0), (1, 1)])
        agg = rasterize(p, x_range=(0, 0), y_range=(0, 1), expand=False, dynamic=False)
        img = Image(([], [0.25, 0.75], np.zeros((2, 0))), bounds=(0, 0, 0, 1), xdensity=1, vdims=['Count'])
        self.assertEqual(agg, img)

    def test_aggregate_points_target(self):
        points = Points([(0.2, 0.3), (0.4, 0.7), (0, 0.99)])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]),
                         vdims=['Count'])
        img = aggregate(points, dynamic=False,  target=expected)
        self.assertEqual(img, expected)

    def test_aggregate_points_sampling(self):
        points = Points([(0.2, 0.3), (0.4, 0.7), (0, 0.99)])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]),
                         vdims=['Count'])
        img = aggregate(points, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        x_sampling=0.5, y_sampling=0.5)
        self.assertEqual(img, expected)

    def test_aggregate_points_categorical(self):
        points = Points([(0.2, 0.3, 'A'), (0.4, 0.7, 'B'), (0, 0.99, 'C')], vdims='z')
        img = aggregate(points, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2, aggregator=ds.count_cat('z'))
        xs, ys = [0.25, 0.75], [0.25, 0.75]
        expected = NdOverlay({'A': Image((xs, ys, [[1, 0], [0, 0]]), vdims='z Count'),
                              'B': Image((xs, ys, [[0, 0], [1, 0]]), vdims='z Count'),
                              'C': Image((xs, ys, [[0, 0], [1, 0]]), vdims='z Count')},
                             kdims=['z'])
        self.assertEqual(img, expected)

    def test_aggregate_points_categorical_zero_range(self):
        points = Points([(0.2, 0.3, 'A'), (0.4, 0.7, 'B'), (0, 0.99, 'C')], vdims='z')
        img = aggregate(points, dynamic=False,  x_range=(0, 0), y_range=(0, 1),
                        aggregator=ds.count_cat('z'))
        xs, ys = [], [0.25, 0.75]
        params = dict(bounds=(0, 0, 0, 1), xdensity=1)
        expected = NdOverlay({'A': Image((xs, ys, np.zeros((2, 0))), vdims='z Count', **params),
                              'B': Image((xs, ys, np.zeros((2, 0))), vdims='z Count', **params),
                              'C': Image((xs, ys, np.zeros((2, 0))), vdims='z Count', **params)},
                             kdims=['z'])
        self.assertEqual(img, expected)

    def test_aggregate_curve(self):
        curve = Curve([(0.2, 0.3), (0.4, 0.7), (0.8, 0.99)])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [1, 1]]),
                         vdims=['Count'])
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
                         datatype=['xarray'], bounds=bounds, vdims='Count')
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
        expected = Image((dates, [166.5, 499.5, 832.5], [[333, 0], [167, 166], [0, 334]]),
                         ['index', 'a'], 'Count', datatype=['xarray'], bounds=bounds)
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
                         datatype=['xarray'], bounds=bounds, vdims='Count')
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
                         datatype=['xarray'], bounds=bounds, vdims='Count')
        expected2 = Image((dates, [1.5, 2.5], [[0, 1], [1, 1]]),
                         datatype=['xarray'], bounds=bounds, vdims='Count')
        self.assertEqual(imgs[0], expected)
        self.assertEqual(imgs[1], expected2)

    def test_aggregate_dt_xaxis_constant_yaxis(self):
        df = pd.DataFrame({'y': np.ones(100)}, index=pd.date_range('1980-01-01', periods=100, freq='1T'))
        img = rasterize(Curve(df), dynamic=False)
        xs = np.array(['1980-01-01T00:16:30.000000', '1980-01-01T00:49:30.000000',
                       '1980-01-01T01:22:30.000000'], dtype='datetime64[us]')
        ys = np.array([])
        bounds = (np.datetime64('1980-01-01T00:00:00.000000'), 1.0,
                  np.datetime64('1980-01-01T01:39:00.000000'), 1.0)
        expected = Image((xs, ys, np.empty((0, 3))), ['index', 'y'], 'Count',
                         xdensity=1, ydensity=1, bounds=bounds)
        self.assertEqual(img, expected)

    def test_aggregate_ndoverlay(self):
        ds = Dataset([(0.2, 0.3, 0), (0.4, 0.7, 1), (0, 0.99, 2)], kdims=['x', 'y', 'z'])
        ndoverlay = ds.to(Points, ['x', 'y'], [], 'z').overlay()
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]),
                         vdims=['Count'])
        img = aggregate(ndoverlay, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        self.assertEqual(img, expected)

    def test_aggregate_path(self):
        path = Path([[(0.2, 0.3), (0.4, 0.7)], [(0.4, 0.7), (0.8, 0.99)]])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 1]]),
                         vdims=['Count'])
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
        self.assertEqual(img.vdims, ['Count'])

    def test_aggregate_dframe_nan_path(self):
        path = Path([Path([[(0.2, 0.3), (0.4, 0.7)], [(0.4, 0.7), (0.8, 0.99)]]).dframe()])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 1]]),
                         vdims=['Count'])
        img = aggregate(path, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        self.assertEqual(img, expected)


class DatashaderShadeTests(ComparisonTestCase):

    def test_shade_categorical_images_xarray(self):
        xs, ys = [0.25, 0.75], [0.25, 0.75]
        data = NdOverlay({'A': Image((xs, ys, [[1, 0], [0, 0]]), datatype=['xarray'], vdims='z Count'),
                          'B': Image((xs, ys, [[0, 0], [1, 0]]), datatype=['xarray'], vdims='z Count'),
                          'C': Image((xs, ys, [[0, 0], [1, 0]]), datatype=['xarray'], vdims='z Count')},
                         kdims=['z'])
        shaded = shade(data)
        r = [[228, 255], [66, 255]]
        g = [[26, 255], [150, 255]]
        b = [[28, 255], [129, 255]]
        a = [[40, 0], [255, 0]]
        expected = RGB((xs, ys, r, g, b, a), datatype=['grid'],
                       vdims=RGB.vdims+[Dimension('A', range=(0, 1))])
        self.assertEqual(shaded, expected)

    def test_shade_categorical_images_grid(self):
        xs, ys = [0.25, 0.75], [0.25, 0.75]
        data = NdOverlay({'A': Image((xs, ys, [[1, 0], [0, 0]]), datatype=['grid'], vdims='z Count'),
                          'B': Image((xs, ys, [[0, 0], [1, 0]]), datatype=['grid'], vdims='z Count'),
                          'C': Image((xs, ys, [[0, 0], [1, 0]]), datatype=['grid'], vdims='z Count')},
                         kdims=['z'])
        shaded = shade(data)
        r = [[228, 255], [66, 255]]
        g = [[26, 255], [150, 255]]
        b = [[28, 255], [129, 255]]
        a = [[40, 0], [255, 0]]
        expected = RGB((xs, ys, r, g, b, a), datatype=['grid'],
                       vdims=RGB.vdims+[Dimension('A', range=(0, 1))])
        self.assertEqual(shaded, expected)



class DatashaderRegridTests(ComparisonTestCase):
    """
    Tests for datashader aggregation
    """

    def setUp(self):
        if ds_version <= '0.5.0':
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
        ### This test causes a numba error using 0.35.0 - temporarily disabled ###
        return
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
        if ds_version <= '0.6.4':
            raise SkipTest('Regridding operations require datashader>=0.7.0')

    def test_rasterize_trimesh_no_vdims(self):
        simplices = [(0, 1, 2), (3, 2, 1)]
        vertices = [(0., 0.), (0., 1.), (1., 0), (1, 1)]
        trimesh = TriMesh((simplices, vertices))
        img = rasterize(trimesh, width=3, height=3, dynamic=False)
        image = Image(np.array([[True, True, True], [True, True, True], [True, True, True]]),
                      bounds=(0, 0, 1, 1), vdims='Any')
        self.assertEqual(img, image)

    def test_rasterize_trimesh_no_vdims_zero_range(self):
        simplices = [(0, 1, 2), (3, 2, 1)]
        vertices = [(0., 0.), (0., 1.), (1., 0), (1, 1)]
        trimesh = TriMesh((simplices, vertices))
        img = rasterize(trimesh, height=2, x_range=(0, 0), dynamic=False)
        image = Image(([], [0.25, 0.75], np.zeros((2, 0))),
                      bounds=(0, 0, 0, 1), xdensity=1, vdims='Any')
        self.assertEqual(img, image)

    def test_rasterize_trimesh_with_vdims_as_wireframe(self):
        simplices = [(0, 1, 2, 0.5), (3, 2, 1, 1.5)]
        vertices = [(0., 0.), (0., 1.), (1., 0), (1, 1)]
        trimesh = TriMesh((simplices, vertices), vdims=['z'])
        img = rasterize(trimesh, width=3, height=3, aggregator='any', interpolation=None, dynamic=False)
        image = Image(np.array([[True, True, True], [True, True, True], [True, True, True]]),
                      bounds=(0, 0, 1, 1), vdims='Any')
        self.assertEqual(img, image)

    def test_rasterize_trimesh(self):
        simplices = [(0, 1, 2, 0.5), (3, 2, 1, 1.5)]
        vertices = [(0., 0.), (0., 1.), (1., 0), (1, 1)]
        trimesh = TriMesh((simplices, vertices), vdims=['z'])
        img = rasterize(trimesh, width=3, height=3, dynamic=False)
        image = Image(np.array([[1.5, 1.5, np.NaN], [0.5, 1.5, np.NaN], [np.NaN, np.NaN, np.NaN]]),
                      bounds=(0, 0, 1, 1))
        self.assertEqual(img, image)

    def test_rasterize_trimesh_node_vdim_precedence(self):
        simplices = [(0, 1, 2, 0.5), (3, 2, 1, 1.5)]
        vertices = [(0., 0., 1), (0., 1., 2), (1., 0, 3), (1, 1, 4)]
        trimesh = TriMesh((simplices, Points(vertices, vdims=['node_z'])), vdims=['z'])
        img = rasterize(trimesh, width=3, height=3, dynamic=False)
        image = Image(np.array([[2., 3., np.NaN], [1.5, 2.5, np.NaN], [np.NaN, np.NaN, np.NaN]]),
                      bounds=(0, 0, 1, 1), vdims='node_z')
        self.assertEqual(img, image)

    def test_rasterize_trimesh_node_explicit_vdim(self):
        simplices = [(0, 1, 2, 0.5), (3, 2, 1, 1.5)]
        vertices = [(0., 0., 1), (0., 1., 2), (1., 0, 3), (1, 1, 4)]
        trimesh = TriMesh((simplices, Points(vertices, vdims=['node_z'])), vdims=['z'])
        img = rasterize(trimesh, width=3, height=3, dynamic=False, aggregator=ds.mean('z'))
        image = Image(np.array([[1.5, 1.5, np.NaN], [0.5, 1.5, np.NaN], [np.NaN, np.NaN, np.NaN]]),
                      bounds=(0, 0, 1, 1))
        self.assertEqual(img, image)

    def test_rasterize_trimesh_zero_range(self):
        simplices = [(0, 1, 2, 0.5), (3, 2, 1, 1.5)]
        vertices = [(0., 0.), (0., 1.), (1., 0), (1, 1)]
        trimesh = TriMesh((simplices, vertices), vdims=['z'])
        img = rasterize(trimesh, x_range=(0, 0), height=2, dynamic=False)
        image = Image(([], [0.25, 0.75], np.zeros((2, 0))),
                      bounds=(0, 0, 0, 1), xdensity=1)
        self.assertEqual(img, image)

    def test_rasterize_trimesh_vertex_vdims(self):
        simplices = [(0, 1, 2), (3, 2, 1)]
        vertices = [(0., 0., 1), (0., 1., 2), (1., 0., 3), (1., 1., 4)]
        trimesh = TriMesh((simplices, Points(vertices, vdims='z')))
        img = rasterize(trimesh, width=3, height=3, dynamic=False)
        image = Image(np.array([[2., 3., np.NaN], [1.5, 2.5, np.NaN], [np.NaN, np.NaN, np.NaN]]),
                      bounds=(0, 0, 1, 1), vdims='z')
        self.assertEqual(img, image)

    def test_rasterize_trimesh_ds_aggregator(self):
        simplices = [(0, 1, 2, 0.5), (3, 2, 1, 1.5)]
        vertices = [(0., 0.), (0., 1.), (1., 0), (1, 1)]
        trimesh = TriMesh((simplices, vertices), vdims=['z'])
        img = rasterize(trimesh, width=3, height=3, dynamic=False, aggregator=ds.mean('z'))
        image = Image(np.array([[1.5, 1.5, np.NaN], [0.5, 1.5, np.NaN], [np.NaN, np.NaN, np.NaN]]),
                      bounds=(0, 0, 1, 1))
        self.assertEqual(img, image)

    def test_rasterize_trimesh_string_aggregator(self):
        simplices = [(0, 1, 2, 0.5), (3, 2, 1, 1.5)]
        vertices = [(0., 0.), (0., 1.), (1., 0), (1, 1)]
        trimesh = TriMesh((simplices, vertices), vdims=['z'])
        img = rasterize(trimesh, width=3, height=3, dynamic=False, aggregator='mean')
        image = Image(np.array([[1.5, 1.5, np.NaN], [0.5, 1.5, np.NaN], [np.NaN, np.NaN, np.NaN]]),
                      bounds=(0, 0, 1, 1))
        self.assertEqual(img, image)

    def test_rasterize_quadmesh(self):
        qmesh = QuadMesh(([0, 1], [0, 1], np.array([[0, 1], [2, 3]])))
        img = rasterize(qmesh, width=3, height=3, dynamic=False, aggregator=ds.mean('z'))
        image = Image(np.array([[2., 3., np.NaN], [0, 1, np.NaN], [np.NaN, np.NaN, np.NaN]]),
                      bounds=(-.5, -.5, 1.5, 1.5))
        self.assertEqual(img, image)

    def test_rasterize_quadmesh_string_aggregator(self):
        qmesh = QuadMesh(([0, 1], [0, 1], np.array([[0, 1], [2, 3]])))
        img = rasterize(qmesh, width=3, height=3, dynamic=False, aggregator='mean')
        image = Image(np.array([[2., 3., np.NaN], [0, 1, np.NaN], [np.NaN, np.NaN, np.NaN]]),
                      bounds=(-.5, -.5, 1.5, 1.5))
        self.assertEqual(img, image)

    def test_rasterize_points(self):
        points = Points([(0.2, 0.3), (0.4, 0.7), (0, 0.99)])
        img = rasterize(points, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]),
                         vdims=['Count'])
        self.assertEqual(img, expected)

    def test_rasterize_curve(self):
        curve = Curve([(0.2, 0.3), (0.4, 0.7), (0.8, 0.99)])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [1, 1]]),
                         vdims=['Count'])
        img = rasterize(curve, dynamic=False,  x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        self.assertEqual(img, expected)

    def test_rasterize_ndoverlay(self):
        ds = Dataset([(0.2, 0.3, 0), (0.4, 0.7, 1), (0, 0.99, 2)], kdims=['x', 'y', 'z'])
        ndoverlay = ds.to(Points, ['x', 'y'], [], 'z').overlay()
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]),
                         vdims=['Count'])
        img = rasterize(ndoverlay, dynamic=False, x_range=(0, 1), y_range=(0, 1),
                        width=2, height=2)
        self.assertEqual(img, expected)

    def test_rasterize_path(self):
        path = Path([[(0.2, 0.3), (0.4, 0.7)], [(0.4, 0.7), (0.8, 0.99)]])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 1]]),
                         vdims=['Count'])
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
        if ds_version <= '0.7.0':
            raise SkipTest('Regridding operations require datashader>=0.7.0')
        self.source = np.arange(8)
        self.target = np.zeros(8)
        self.graph = Graph(((self.source, self.target),))

    def test_directly_connect_paths(self):
        direct = directly_connect_edges(self.graph)._split_edgepaths
        self.assertEqual(direct, self.graph.edgepaths)
