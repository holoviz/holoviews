"""
Test cases for the Comparisons class over the Chart elements
"""

from unittest import skipIf

import numpy as np
import pandas as pd
import pytest

from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
    RGB,
    Area,
    BoxWhisker,
    Curve,
    Distribution,
    HSpan,
    Image,
    Path,
    Points,
    Polygons,
    QuadMesh,
    Rectangles,
    Scatter,
    Segments,
    Violin,
    VSpan,
)
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
from holoviews.util.transform import dim

try:
    import datashader as ds
except ImportError:
    ds = None

try:
    import spatialpandas as spd
except ImportError:
    spd = None

try:
    import shapely
except ImportError:
    shapely = None

try:
    import dask.dataframe as dd
except ImportError:
    dd = None

spd_available = skipIf(spd is None, "spatialpandas is not available")
shapelib_available = skipIf(shapely is None and spd is None,
                            'Neither shapely nor spatialpandas are available')
shapely_available = skipIf(shapely is None, 'shapely is not available')
ds_available = skipIf(ds is None, 'datashader not available')
dd_available = pytest.mark.skipif(dd is None, reason='dask.dataframe not available')


class TestIndexExpr(ComparisonTestCase):

    def setUp(self):
        import holoviews.plotting.bokeh # noqa
        super().setUp()
        self._backend = Store.current_backend
        Store.set_current_backend('bokeh')

    def tearDown(self):
        Store.current_backend = self._backend

    def test_index_selection_on_id_column(self):
        # tests issue in https://github.com/holoviz/holoviews/pull/6336
        x, y = np.random.randn(2, 100)
        idx = np.arange(100)

        points = Points(
            {'x': x, 'y': y, 'id': idx}, kdims=['x', 'y'], vdims=['id'], datatype=['dataframe']
        )
        sel, _, _ = points._get_index_selection([3, 7], ['id'])
        assert sel == dim('id').isin([3, 7])


class TestSelection1DExpr(ComparisonTestCase):

    def setUp(self):
        import holoviews.plotting.bokeh # noqa
        super().setUp()
        self._backend = Store.current_backend
        Store.set_current_backend('bokeh')

    def tearDown(self):
        Store.current_backend = self._backend

    def test_area_selection_numeric(self):
        area = Area([3, 2, 1, 3, 4])
        expr, bbox, region = area._get_selection_expr_for_stream_value(bounds=(1, 0, 3, 2))
        self.assertEqual(bbox, {'x': (1, 3)})
        self.assertEqual(expr.apply(area), np.array([False, True, True, True, False]))
        self.assertEqual(region, NdOverlay({0: VSpan(1, 3)}))

    def test_area_selection_numeric_inverted(self):
        area = Area([3, 2, 1, 3, 4]).opts(invert_axes=True)
        expr, bbox, region = area._get_selection_expr_for_stream_value(bounds=(0, 1, 2, 3))
        self.assertEqual(bbox, {'x': (1, 3)})
        self.assertEqual(expr.apply(area), np.array([False, True, True, True, False]))
        self.assertEqual(region, NdOverlay({0: HSpan(1, 3)}))

    def test_area_selection_categorical(self):
        area = Area((['B', 'A', 'C', 'D', 'E'], [3, 2, 1, 3, 4]))
        expr, bbox, region = area._get_selection_expr_for_stream_value(
            bounds=(0, 1, 2, 3), x_selection=['B', 'A', 'C']
        )
        self.assertEqual(bbox, {'x': ['B', 'A', 'C']})
        self.assertEqual(expr.apply(area), np.array([True, True, True, False, False]))
        self.assertEqual(region, NdOverlay({0: VSpan(0, 2)}))

    def test_area_selection_numeric_index_cols(self):
        area = Area([3, 2, 1, 3, 2])
        expr, bbox, region = area._get_selection_expr_for_stream_value(
            bounds=(1, 0, 3, 2), index_cols=['y']
        )
        self.assertEqual(bbox, {'x': (1, 3)})
        self.assertEqual(expr.apply(area), np.array([False, True, True, False, True]))
        self.assertEqual(region, None)

    def test_curve_selection_numeric(self):
        curve = Curve([3, 2, 1, 3, 4])
        expr, bbox, region = curve._get_selection_expr_for_stream_value(bounds=(1, 0, 3, 2))
        self.assertEqual(bbox, {'x': (1, 3)})
        self.assertEqual(expr.apply(curve), np.array([False, True, True, True, False]))
        self.assertEqual(region, NdOverlay({0: VSpan(1, 3)}))

    def test_curve_selection_categorical(self):
        curve = Curve((['B', 'A', 'C', 'D', 'E'], [3, 2, 1, 3, 4]))
        expr, bbox, region = curve._get_selection_expr_for_stream_value(
            bounds=(0, 1, 2, 3), x_selection=['B', 'A', 'C']
        )
        self.assertEqual(bbox, {'x': ['B', 'A', 'C']})
        self.assertEqual(expr.apply(curve), np.array([True, True, True, False, False]))
        self.assertEqual(region, NdOverlay({0: VSpan(0, 2)}))

    def test_curve_selection_numeric_index_cols(self):
        curve = Curve([3, 2, 1, 3, 2])
        expr, bbox, region = curve._get_selection_expr_for_stream_value(
            bounds=(1, 0, 3, 2), index_cols=['y']
        )
        self.assertEqual(bbox, {'x': (1, 3)})
        self.assertEqual(expr.apply(curve), np.array([False, True, True, False, True]))
        self.assertEqual(region, None)

    def test_box_whisker_single(self):
        box_whisker = BoxWhisker(list(range(10)))
        expr, bbox, region = box_whisker._get_selection_expr_for_stream_value(
            bounds=(0, 3, 1, 7)
        )
        self.assertEqual(bbox, {'y': (3, 7)})
        self.assertEqual(expr.apply(box_whisker), np.array([
            False, False, False, True, True, True, True, True, False, False
        ]))
        self.assertEqual(region, NdOverlay({0: HSpan(3, 7)}))

    def test_box_whisker_single_inverted(self):
        box = BoxWhisker(list(range(10))).opts(invert_axes=True)
        expr, bbox, region = box._get_selection_expr_for_stream_value(
            bounds=(3, 0, 7, 1)
        )
        self.assertEqual(bbox, {'y': (3, 7)})
        self.assertEqual(expr.apply(box), np.array([
            False, False, False, True, True, True, True, True, False, False
        ]))
        self.assertEqual(region, NdOverlay({0: VSpan(3, 7)}))

    def test_box_whisker_cats(self):
        box_whisker = BoxWhisker((['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C'], list(range(10))), 'x', 'y')
        expr, bbox, region = box_whisker._get_selection_expr_for_stream_value(
            bounds=(0, 1, 2, 7), x_selection=['A', 'B']
        )
        self.assertEqual(bbox, {'y': (1, 7), 'x': ['A', 'B']})
        self.assertEqual(expr.apply(box_whisker), np.array([
            False, True, True, True, True, False, False, False, False, False
        ]))
        self.assertEqual(region, NdOverlay({0: HSpan(1, 7)}))

    def test_box_whisker_cats_index_cols(self):
        box_whisker = BoxWhisker((['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C'], list(range(10))), 'x', 'y')
        expr, bbox, region = box_whisker._get_selection_expr_for_stream_value(
            bounds=(0, 1, 2, 7), x_selection=['A', 'B'], index_cols=['x']
        )
        self.assertEqual(bbox, {'y': (1, 7), 'x': ['A', 'B']})
        self.assertEqual(expr.apply(box_whisker), np.array([
            True, True, True, True, True, False, False, False, False, False
        ]))
        self.assertEqual(region, None)

    def test_violin_single(self):
        violin = Violin(list(range(10)))
        expr, bbox, region = violin._get_selection_expr_for_stream_value(
            bounds=(0, 3, 1, 7)
        )
        self.assertEqual(bbox, {'y': (3, 7)})
        self.assertEqual(expr.apply(violin), np.array([
            False, False, False, True, True, True, True, True, False, False
        ]))
        self.assertEqual(region, NdOverlay({0: HSpan(3, 7)}))

    def test_violin_single_inverted(self):
        violin = Violin(list(range(10))).opts(invert_axes=True)
        expr, bbox, region = violin._get_selection_expr_for_stream_value(
            bounds=(3, 0, 7, 1)
        )
        self.assertEqual(bbox, {'y': (3, 7)})
        self.assertEqual(expr.apply(violin), np.array([
            False, False, False, True, True, True, True, True, False, False
        ]))
        self.assertEqual(region, NdOverlay({0: VSpan(3, 7)}))

    def test_violin_cats(self):
        violin = Violin((['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C'], list(range(10))), 'x', 'y')
        expr, bbox, region = violin._get_selection_expr_for_stream_value(
            bounds=(0, 1, 2, 7), x_selection=['A', 'B']
        )
        self.assertEqual(bbox, {'y': (1, 7), 'x': ['A', 'B']})
        self.assertEqual(expr.apply(violin), np.array([
            False, True, True, True, True, False, False, False, False, False
        ]))
        self.assertEqual(region, NdOverlay({0: HSpan(1, 7)}))

    def test_violin_cats_index_cols(self):
        violin = Violin((['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C'], list(range(10))), 'x', 'y')
        expr, bbox, region = violin._get_selection_expr_for_stream_value(
            bounds=(0, 1, 2, 7), x_selection=['A', 'B'], index_cols=['x']
        )
        self.assertEqual(bbox, {'y': (1, 7), 'x': ['A', 'B']})
        self.assertEqual(expr.apply(violin), np.array([
            True, True, True, True, True, False, False, False, False, False
        ]))
        self.assertEqual(region, None)

    def test_distribution_single(self):
        dist = Distribution(list(range(10)))
        expr, bbox, region = dist._get_selection_expr_for_stream_value(
            bounds=(3, 0, 7, 1)
        )
        self.assertEqual(bbox, {'Value': (3, 7)})
        self.assertEqual(expr.apply(dist), np.array([
            False, False, False, True, True, True, True, True, False, False
        ]))
        self.assertEqual(region, NdOverlay({0: VSpan(3, 7)}))

    def test_distribution_single_inverted(self):
        dist = Distribution(list(range(10))).opts(invert_axes=True)
        expr, bbox, region = dist._get_selection_expr_for_stream_value(
            bounds=(0, 3, 1, 7)
        )
        self.assertEqual(bbox, {'Value': (3, 7)})
        self.assertEqual(expr.apply(dist), np.array([
            False, False, False, True, True, True, True, True, False, False
        ]))
        self.assertEqual(region, NdOverlay({0: HSpan(3, 7)}))


class TestSelection2DExpr(ComparisonTestCase):

    def setUp(self):
        import holoviews.plotting.bokeh # noqa
        super().setUp()
        self._backend = Store.current_backend
        Store.set_current_backend('bokeh')

    def tearDown(self):
        Store.current_backend = self._backend

    def test_points_selection_numeric(self):
        points = Points([3, 2, 1, 3, 4])
        expr, bbox, region = points._get_selection_expr_for_stream_value(bounds=(1, 0, 3, 2))
        self.assertEqual(bbox, {'x': (1, 3), 'y': (0, 2)})
        self.assertEqual(expr.apply(points), np.array([False, True, True, False, False]))
        self.assertEqual(region, Rectangles([(1, 0, 3, 2)]) * Path([]))

    def test_points_selection_numeric_inverted(self):
        points = Points([3, 2, 1, 3, 4]).opts(invert_axes=True)
        expr, bbox, region = points._get_selection_expr_for_stream_value(bounds=(0, 1, 2, 3))
        self.assertEqual(bbox, {'x': (1, 3), 'y': (0, 2)})
        self.assertEqual(expr.apply(points), np.array([False, True, True, False, False]))
        self.assertEqual(region, Rectangles([(0, 1, 2, 3)]) * Path([]))

    @shapelib_available
    def test_points_selection_geom(self):
        points = Points([3, 2, 1, 3, 4])
        geom = np.array([(-0.1, -0.1), (1.4, 0), (1.4, 2.2), (-0.1, 2.2)])
        expr, bbox, region = points._get_selection_expr_for_stream_value(geometry=geom)
        self.assertEqual(bbox, {'x': np.array([-0.1, 1.4, 1.4, -0.1]),
                                'y': np.array([-0.1, 0, 2.2, 2.2])})
        self.assertEqual(expr.apply(points), np.array([False, True, False, False, False]))
        self.assertEqual(region, Rectangles([]) * Path([[*geom, (-0.1, -0.1)]]))

    @shapelib_available
    def test_points_selection_geom_inverted(self):
        points = Points([3, 2, 1, 3, 4]).opts(invert_axes=True)
        geom = np.array([(-0.1, -0.1), (1.4, 0), (1.4, 2.2), (-0.1, 2.2)])
        expr, bbox, region = points._get_selection_expr_for_stream_value(geometry=geom)
        self.assertEqual(bbox, {'y': np.array([-0.1, 1.4, 1.4, -0.1]),
                                'x': np.array([-0.1, 0, 2.2, 2.2])})
        self.assertEqual(expr.apply(points), np.array([False, False, True, False, False]))
        self.assertEqual(region, Rectangles([]) * Path([[*geom, (-0.1, -0.1)]]))

    def test_points_selection_categorical(self):
        points = Points((['B', 'A', 'C', 'D', 'E'], [3, 2, 1, 3, 4]))
        expr, bbox, region = points._get_selection_expr_for_stream_value(
            bounds=(0, 1, 2, 3), x_selection=['B', 'A', 'C'], y_selection=None
        )
        self.assertEqual(bbox, {'x': ['B', 'A', 'C'], 'y': (1, 3)})
        self.assertEqual(expr.apply(points), np.array([True, True, True, False, False]))
        self.assertEqual(region, Rectangles([(0, 1, 2, 3)]) * Path([]))

    def test_points_selection_numeric_index_cols(self):
        points = Points([3, 2, 1, 3, 2])
        expr, bbox, region = points._get_selection_expr_for_stream_value(
            bounds=(1, 0, 3, 2), index_cols=['y']
        )
        self.assertEqual(bbox, {'x': (1, 3), 'y': (0, 2)})
        self.assertEqual(expr.apply(points), np.array([False, False, True, False, False]))
        self.assertEqual(region, None)

    def test_scatter_selection_numeric(self):
        scatter = Scatter([3, 2, 1, 3, 4])
        expr, bbox, region = scatter._get_selection_expr_for_stream_value(bounds=(1, 0, 3, 2))
        self.assertEqual(bbox, {'x': (1, 3)})
        self.assertEqual(expr.apply(scatter), np.array([False, True, True, True, False]))
        self.assertEqual(region, NdOverlay({0: VSpan(1, 3)}))

    def test_scatter_selection_numeric_inverted(self):
        scatter = Scatter([3, 2, 1, 3, 4]).opts(invert_axes=True)
        expr, bbox, region = scatter._get_selection_expr_for_stream_value(bounds=(0, 1, 2, 3))
        self.assertEqual(bbox, {'x': (1, 3)})
        self.assertEqual(expr.apply(scatter), np.array([False, True, True, True, False]))
        self.assertEqual(region, NdOverlay({0: HSpan(1, 3)}))

    def test_scatter_selection_categorical(self):
        scatter = Scatter((['B', 'A', 'C', 'D', 'E'], [3, 2, 1, 3, 4]))
        expr, bbox, region = scatter._get_selection_expr_for_stream_value(
            bounds=(0, 1, 2, 3), x_selection=['B', 'A', 'C'], y_selection=None
        )
        self.assertEqual(bbox, {'x': ['B', 'A', 'C']})
        self.assertEqual(expr.apply(scatter), np.array([True, True, True, False, False]))
        self.assertEqual(region, NdOverlay({0: VSpan(0, 2)}))

    def test_scatter_selection_numeric_index_cols(self):
        scatter = Scatter([3, 2, 1, 3, 2])
        expr, bbox, region = scatter._get_selection_expr_for_stream_value(
            bounds=(1, 0, 3, 2), index_cols=['y']
        )
        self.assertEqual(bbox, {'x': (1, 3)})
        self.assertEqual(expr.apply(scatter), np.array([False, True, True, False, True]))
        self.assertEqual(region, None)

    def test_image_selection_numeric(self):
        img = Image(([0, 1, 2], [0, 1, 2, 3], np.random.rand(4, 3)))
        expr, bbox, region = img._get_selection_expr_for_stream_value(bounds=(0.5, 1.5, 2.1, 3.1))
        self.assertEqual(bbox, {'x': (0.5, 2.1), 'y': (1.5, 3.1)})
        self.assertEqual(expr.apply(img, expanded=True, flat=False), np.array([
            [False, False, False],
            [False, False, False],
            [False, True, True],
            [False, True, True]
        ]))
        self.assertEqual(region, Rectangles([(0.5, 1.5, 2.1, 3.1)]) * Path([]))

    def test_image_selection_numeric_inverted(self):
        img = Image(([0, 1, 2], [0, 1, 2, 3], np.random.rand(4, 3))).opts(invert_axes=True)
        expr, bbox, region = img._get_selection_expr_for_stream_value(bounds=(1.5, 0.5, 3.1, 2.1))
        self.assertEqual(bbox, {'x': (0.5, 2.1), 'y': (1.5, 3.1)})
        self.assertEqual(expr.apply(img, expanded=True, flat=False), np.array([
            [False, False, False],
            [False, False, False],
            [False, True, True],
            [False, True, True]
        ]))
        self.assertEqual(region, Rectangles([(1.5, 0.5, 3.1, 2.1)]) * Path([]))

    @ds_available
    @spd_available
    def test_img_selection_geom(self):
        img = Image(([0, 1, 2], [0, 1, 2, 3], np.random.rand(4, 3)))
        geom = np.array([(-0.4, -0.1), (0.6, -0.1), (0.4, 1.7), (-0.1, 1.7)])
        expr, bbox, region = img._get_selection_expr_for_stream_value(geometry=geom)
        self.assertEqual(bbox, {'x': np.array([-0.4, 0.6, 0.4, -0.1]),
                                'y': np.array([-0.1, -0.1, 1.7, 1.7])})
        self.assertEqual(expr.apply(img, expanded=True, flat=False), np.array([
            [    1., np.nan, np.nan],
            [    1., np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan]
        ]))
        self.assertEqual(region, Rectangles([]) * Path([[*geom, (-0.4, -0.1)]]))

    @ds_available
    def test_img_selection_geom_inverted(self):
        img = Image(([0, 1, 2], [0, 1, 2, 3], np.random.rand(4, 3))).opts(invert_axes=True)
        geom = np.array([(-0.4, -0.1), (0.6, -0.1), (0.4, 1.7), (-0.1, 1.7)])
        expr, bbox, region = img._get_selection_expr_for_stream_value(geometry=geom)
        self.assertEqual(bbox, {'y': np.array([-0.4, 0.6, 0.4, -0.1]),
                                'x': np.array([-0.1, -0.1, 1.7, 1.7])})
        self.assertEqual(expr.apply(img, expanded=True, flat=False), np.array([
            [ True,  True, False],
            [ False, False, False],
            [ False,  False, False],
            [False, False, False]
        ]))
        self.assertEqual(region, Rectangles([]) * Path([[*geom, (-0.4, -0.1)]]))

    def test_rgb_selection_numeric(self):
        img = RGB(([0, 1, 2], [0, 1, 2, 3], np.random.rand(4, 3, 3)))
        expr, bbox, region = img._get_selection_expr_for_stream_value(bounds=(0.5, 1.5, 2.1, 3.1))
        self.assertEqual(bbox, {'x': (0.5, 2.1), 'y': (1.5, 3.1)})
        self.assertEqual(expr.apply(img, expanded=True, flat=False), np.array([
            [False, False, False],
            [False, False, False],
            [False, True, True],
            [False, True, True]
        ]))
        self.assertEqual(region, Rectangles([(0.5, 1.5, 2.1, 3.1)]) * Path([]))

    def test_rgb_selection_numeric_inverted(self):
        img = RGB(([0, 1, 2], [0, 1, 2, 3], np.random.rand(4, 3, 3))).opts(invert_axes=True)
        expr, bbox, region = img._get_selection_expr_for_stream_value(bounds=(1.5, 0.5, 3.1, 2.1))
        self.assertEqual(bbox, {'x': (0.5, 2.1), 'y': (1.5, 3.1)})
        self.assertEqual(expr.apply(img, expanded=True, flat=False), np.array([
            [False, False, False],
            [False, False, False],
            [False, True, True],
            [False, True, True]
        ]))
        self.assertEqual(region, Rectangles([(1.5, 0.5, 3.1, 2.1)]) * Path([]))

    def test_quadmesh_selection(self):
        n = 4
        coords = np.linspace(-1.5,1.5,n)
        X,Y = np.meshgrid(coords, coords)
        Qx = np.cos(Y) - np.cos(X)
        Qy = np.sin(Y) + np.sin(X)
        Z = np.sqrt(X**2 + Y**2)
        qmesh = QuadMesh((Qx, Qy, Z))
        expr, bbox, region = qmesh._get_selection_expr_for_stream_value(bounds=(0, -0.5, 0.7, 1.5))
        self.assertEqual(bbox, {'x': (0, 0.7), 'y': (-0.5, 1.5)})
        self.assertEqual(expr.apply(qmesh, expanded=True, flat=False), np.array([
            [False, False, False, True],
            [False, False,  True, False],
            [False,  True,  True, False],
            [True,  False, False, False]
        ]))
        self.assertEqual(region, Rectangles([(0, -0.5, 0.7, 1.5)]) * Path([]))

    def test_quadmesh_selection_inverted(self):
        n = 4
        coords = np.linspace(-1.5,1.5,n)
        X,Y = np.meshgrid(coords, coords)
        Qx = np.cos(Y) - np.cos(X)
        Qy = np.sin(Y) + np.sin(X)
        Z = np.sqrt(X**2 + Y**2)
        qmesh = QuadMesh((Qx, Qy, Z)).opts(invert_axes=True)
        expr, bbox, region = qmesh._get_selection_expr_for_stream_value(bounds=(0, -0.5, 0.7, 1.5))
        self.assertEqual(bbox, {'x': (-0.5, 1.5), 'y': (0, 0.7)})
        self.assertEqual(expr.apply(qmesh, expanded=True, flat=False), np.array([
            [False, False, False,  True],
            [False, False,  True,  True],
            [False,  True, False, False],
            [True,  False, False, False]
        ]))
        self.assertEqual(region, Rectangles([(0, -0.5, 0.7, 1.5)]) * Path([]))



class TestSelectionGeomExpr(ComparisonTestCase):

    def setUp(self):
        import holoviews.plotting.bokeh # noqa
        super().setUp()
        self._backend = Store.current_backend
        Store.set_current_backend('bokeh')

    def tearDown(self):
        Store.current_backend = self._backend

    def test_rect_selection_numeric(self):
        rect = Rectangles([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)])
        expr, bbox, region = rect._get_selection_expr_for_stream_value(bounds=(0.5, 0.9, 3.4, 4.9))
        self.assertEqual(bbox, {'x0': (0.5, 3.4), 'y0': (0.9, 4.9), 'x1': (0.5, 3.4), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(rect), np.array([False, True, False]))
        self.assertEqual(region, Rectangles([(0.5, 0.9, 3.4, 4.9)]) * Path([]))
        expr, bbox, region = rect._get_selection_expr_for_stream_value(bounds=(0, 0.9, 3.5, 4.9))
        self.assertEqual(bbox, {'x0': (0, 3.5), 'y0': (0.9, 4.9), 'x1': (0, 3.5), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(rect), np.array([True, True, True]))
        self.assertEqual(region, Rectangles([(0, 0.9, 3.5, 4.9)]) * Path([]))

    def test_rect_selection_numeric_inverted(self):
        rect = Rectangles([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)]).opts(invert_axes=True)
        expr, bbox, region = rect._get_selection_expr_for_stream_value(bounds=(0.9, 0.5, 4.9, 3.4))
        self.assertEqual(bbox, {'x0': (0.5, 3.4), 'y0': (0.9, 4.9), 'x1': (0.5, 3.4), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(rect), np.array([False, True, False]))
        self.assertEqual(region, Rectangles([(0.9, 0.5, 4.9, 3.4)]) * Path([]))
        expr, bbox, region = rect._get_selection_expr_for_stream_value(bounds=(0.9, 0, 4.9, 3.5))
        self.assertEqual(bbox, {'x0': (0, 3.5), 'y0': (0.9, 4.9), 'x1': (0, 3.5), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(rect), np.array([True, True, True]))
        self.assertEqual(region, Rectangles([(0.9, 0, 4.9, 3.5)]) * Path([]))

    @shapely_available
    def test_rect_geom_selection(self):
        rect = Rectangles([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)])
        geom = np.array([(-0.4, -0.1), (2.2, -0.1), (2.2, 4.1), (-0.1, 4.2)])
        expr, bbox, region = rect._get_selection_expr_for_stream_value(geometry=geom)
        self.assertEqual(bbox, {'x0': np.array([-0.4, 2.2, 2.2, -0.1]),
                                'y0': np.array([-0.1, -0.1, 4.1, 4.2]),
                                'x1': np.array([-0.4, 2.2, 2.2, -0.1]),
                                'y1': np.array([-0.1, -0.1, 4.1, 4.2])})
        self.assertEqual(expr.apply(rect), np.array([True, True, False]))
        self.assertEqual(region, Rectangles([]) * Path([[*geom, (-0.4, -0.1)]]))

    @shapely_available
    def test_rect_geom_selection_inverted(self):
        rect = Rectangles([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)]).opts(invert_axes=True)
        geom = np.array([(-0.4, -0.1), (3.2, -0.1), (3.2, 4.1), (-0.1, 4.2)])
        expr, bbox, region = rect._get_selection_expr_for_stream_value(geometry=geom)
        self.assertEqual(bbox, {'y0': np.array([-0.4, 3.2, 3.2, -0.1]),
                                'x0': np.array([-0.1, -0.1, 4.1, 4.2]),
                                'y1': np.array([-0.4, 3.2, 3.2, -0.1]),
                                'x1': np.array([-0.1, -0.1, 4.1, 4.2])})
        self.assertEqual(expr.apply(rect), np.array([True, False, False]))
        self.assertEqual(region, Rectangles([]) * Path([[*geom, (-0.4, -0.1)]]))

    def test_segments_selection_numeric(self):
        segs = Segments([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)])
        expr, bbox, region = segs._get_selection_expr_for_stream_value(bounds=(0.5, 0.9, 3.4, 4.9))
        self.assertEqual(bbox, {'x0': (0.5, 3.4), 'y0': (0.9, 4.9), 'x1': (0.5, 3.4), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(segs), np.array([False, True, False]))
        self.assertEqual(region, Rectangles([(0.5, 0.9, 3.4, 4.9)]) * Path([]))
        expr, bbox, region = segs._get_selection_expr_for_stream_value(bounds=(0, 0.9, 3.5, 4.9))
        self.assertEqual(bbox, {'x0': (0, 3.5), 'y0': (0.9, 4.9), 'x1': (0, 3.5), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(segs), np.array([True, True, True]))
        self.assertEqual(region, Rectangles([(0, 0.9, 3.5, 4.9)]) * Path([]))

    def test_segs_selection_numeric_inverted(self):
        segs = Segments([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)]).opts(invert_axes=True)
        expr, bbox, region = segs._get_selection_expr_for_stream_value(bounds=(0.9, 0.5, 4.9, 3.4))
        self.assertEqual(bbox, {'x0': (0.5, 3.4), 'y0': (0.9, 4.9), 'x1': (0.5, 3.4), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(segs), np.array([False, True, False]))
        self.assertEqual(region, Rectangles([(0.9, 0.5, 4.9, 3.4)]) * Path([]))
        expr, bbox, region = segs._get_selection_expr_for_stream_value(bounds=(0.9, 0, 4.9, 3.5))
        self.assertEqual(bbox, {'x0': (0, 3.5), 'y0': (0.9, 4.9), 'x1': (0, 3.5), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(segs), np.array([True, True, True]))
        self.assertEqual(region, Rectangles([(0.9, 0, 4.9, 3.5)]) * Path([]))

    @shapely_available
    def test_segs_geom_selection(self):
        rect = Segments([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)])
        geom = np.array([(-0.4, -0.1), (2.2, -0.1), (2.2, 4.1), (-0.1, 4.2)])
        expr, bbox, region = rect._get_selection_expr_for_stream_value(geometry=geom)
        self.assertEqual(bbox, {'x0': np.array([-0.4, 2.2, 2.2, -0.1]),
                                'y0': np.array([-0.1, -0.1, 4.1, 4.2]),
                                'x1': np.array([-0.4, 2.2, 2.2, -0.1]),
                                'y1': np.array([-0.1, -0.1, 4.1, 4.2])})
        self.assertEqual(expr.apply(rect), np.array([True, True, False]))
        self.assertEqual(region, Rectangles([]) * Path([[*geom, (-0.4, -0.1)]]))

    @shapely_available
    def test_segs_geom_selection_inverted(self):
        rect = Segments([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)]).opts(invert_axes=True)
        geom = np.array([(-0.4, -0.1), (3.2, -0.1), (3.2, 4.1), (-0.1, 4.2)])
        expr, bbox, region = rect._get_selection_expr_for_stream_value(geometry=geom)
        self.assertEqual(bbox, {'y0': np.array([-0.4, 3.2, 3.2, -0.1]),
                                'x0': np.array([-0.1, -0.1, 4.1, 4.2]),
                                'y1': np.array([-0.4, 3.2, 3.2, -0.1]),
                                'x1': np.array([-0.1, -0.1, 4.1, 4.2])})
        self.assertEqual(expr.apply(rect), np.array([True, False, False]))
        self.assertEqual(region, Rectangles([]) * Path([[*geom, (-0.4, -0.1)]]))


class TestSelectionPolyExpr(ComparisonTestCase):

    def setUp(self):
        import holoviews.plotting.bokeh # noqa
        super().setUp()
        self._backend = Store.current_backend
        Store.set_current_backend('bokeh')

    def tearDown(self):
        Store.current_backend = self._backend

    def test_poly_selection_numeric(self):
        poly = Polygons([
            [(0, 0), (0.2, 0.1), (0.3, 0.4), (0.1, 0.2)],
            [(0.25, -.1), (0.4, 0.2), (0.6, 0.3), (0.5, 0.1)],
            [(0.3, 0.3), (0.5, 0.4), (0.6, 0.5), (0.35, 0.45)]
        ])
        expr, bbox, region = poly._get_selection_expr_for_stream_value(bounds=(0.2, -0.2, 0.6, 0.6))
        self.assertEqual(bbox, {'x': (0.2, 0.6), 'y': (-0.2, 0.6)})
        self.assertEqual(expr.apply(poly, expanded=False), np.array([False, True, True]))
        self.assertEqual(region, Rectangles([(0.2, -0.2, 0.6, 0.6)]) * Path([]))

    def test_poly_selection_numeric_inverted(self):
        poly = Polygons([
            [(0, 0), (0.2, 0.1), (0.3, 0.4), (0.1, 0.2)],
            [(0.25, -.1), (0.4, 0.2), (0.6, 0.3), (0.5, 0.1)],
            [(0.3, 0.3), (0.5, 0.4), (0.6, 0.5), (0.35, 0.45)]
        ]).opts(invert_axes=True)
        expr, bbox, region = poly._get_selection_expr_for_stream_value(bounds=(0.2, -0.2, 0.6, 0.6))
        self.assertEqual(bbox, {'y': (0.2, 0.6), 'x': (-0.2, 0.6)})
        self.assertEqual(expr.apply(poly, expanded=False), np.array([False, False, True]))
        self.assertEqual(region, Rectangles([(0.2, -0.2, 0.6, 0.6)]) * Path([]))

    @shapely_available
    def test_poly_geom_selection(self):
        poly = Polygons([
            [(0, 0), (0.2, 0.1), (0.3, 0.4), (0.1, 0.2)],
            [(0.25, -.1), (0.4, 0.2), (0.6, 0.3), (0.5, 0.1)],
            [(0.3, 0.3), (0.5, 0.4), (0.6, 0.5), (0.35, 0.45)]
        ])
        geom = np.array([(0.2, -0.15), (0.5, 0), (0.75, 0.6), (0.1, 0.45)])
        expr, bbox, region = poly._get_selection_expr_for_stream_value(geometry=geom)
        self.assertEqual(bbox, {'x': np.array([0.2, 0.5, 0.75, 0.1]),
                                'y': np.array([-0.15, 0, 0.6, 0.45])})
        self.assertEqual(expr.apply(poly, expanded=False), np.array([False, True, True]))
        self.assertEqual(region, Rectangles([]) * Path([[*geom, (0.2, -0.15)]]))

    @shapely_available
    def test_poly_geom_selection_inverted(self):
        poly = Polygons([
            [(0, 0), (0.2, 0.1), (0.3, 0.4), (0.1, 0.2)],
            [(0.25, -.1), (0.4, 0.2), (0.6, 0.3), (0.5, 0.1)],
            [(0.3, 0.3), (0.5, 0.4), (0.6, 0.5), (0.35, 0.45)]
        ]).opts(invert_axes=True)
        geom = np.array([(0.2, -0.15), (0.5, 0), (0.75, 0.6), (0.1, 0.6)])
        expr, bbox, region = poly._get_selection_expr_for_stream_value(geometry=geom)
        self.assertEqual(bbox, {'y': np.array([0.2, 0.5, 0.75, 0.1]),
                                'x': np.array([-0.15, 0, 0.6, 0.6])})
        self.assertEqual(expr.apply(poly, expanded=False), np.array([False, False, True]))
        self.assertEqual(region, Rectangles([]) * Path([[*geom, (0.2, -0.15)]]))


class TestSpatialSelectColumnar:
    __test__ = False
    method = None

    geometry_encl = np.array([
        [-1, 0.5],
        [ 1, 0.5],
        [ 0,-1.5],
        [-2,-1.5]
    ], dtype=float)

    pt_mask_encl = np.array([
        0, 0, 0,
        1, 1, 0,
        1, 1, 0,
    ], dtype=bool)

    geometry_noencl = np.array([
        [10, 10],
        [10, 11],
        [11, 11],
        [11, 10]
    ], dtype=float)

    pt_mask_noencl = np.array([
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    ], dtype=bool)

    @pytest.fixture(scope="module")
    def pandas_df(self):
        return pd.DataFrame({
            "x": [-1, 0, 1,
                  -1, 0, 1,
                  -1, 0, 1],
            "y": [ 1, 1, 1,
                   0, 0, 0,
                  -1,-1,-1]
        }, dtype=float)


    @pytest.fixture(scope="function")
    def dask_df(self, pandas_df):
        return dd.from_pandas(pandas_df, npartitions=2)

    @pytest.fixture(scope="function")
    def _method(self):
        return self.method

    @pytest.mark.parametrize("geometry,pt_mask", [(geometry_encl, pt_mask_encl),(geometry_noencl, pt_mask_noencl)])
    class TestSpatialSelectColumnarPtMask:

        def test_pandas(self, geometry, pt_mask, pandas_df, _method):
            mask = spatial_select_columnar(pandas_df.x, pandas_df.y, geometry, _method)
            assert np.array_equal(mask, pt_mask)

        @dd_available
        def test_dask(self, geometry, pt_mask, dask_df, _method):
            mask = spatial_select_columnar(dask_df.x, dask_df.y, geometry, _method)
            assert np.array_equal(mask.compute(), pt_mask)

        def test_numpy(self, geometry, pt_mask, pandas_df, _method):
            mask = spatial_select_columnar(pandas_df.x.to_numpy(copy=True), pandas_df.y.to_numpy(copy=True), geometry, _method)
            assert np.array_equal(mask, pt_mask)

        @pytest.mark.gpu
        def test_cudf(self, geometry, pt_mask, pandas_df, _method, unimport):
            import cudf
            import cupy as cp
            unimport('cuspatial')

            df = cudf.from_pandas(pandas_df)
            mask = spatial_select_columnar(df.x, df.y, geometry, _method)
            assert np.array_equal(cp.asnumpy(mask), pt_mask)

        @pytest.mark.gpu
        def test_cuspatial(self, geometry, pt_mask, pandas_df, _method):
            import cudf
            import cupy as cp

            df = cudf.from_pandas(pandas_df)
            mask = spatial_select_columnar(df.x, df.y, geometry, _method)
            assert np.array_equal(cp.asnumpy(mask), pt_mask)

    @pytest.mark.parametrize("geometry", [geometry_encl, geometry_noencl])
    class TestSpatialSelectColumnarDaskMeta:
        @dd_available
        def test_meta_dtype(self, geometry, dask_df, _method):
            mask = spatial_select_columnar(dask_df.x, dask_df.y, geometry, _method)
            assert mask._meta.dtype == np.bool_


@pytest.mark.skipif(shapely is None, reason='Shapely not available')
class TestSpatialSelectColumnarShapely(TestSpatialSelectColumnar):
    __test__ = True
    method = 'shapely'


@pytest.mark.skipif(spd is None, reason='Spatialpandas not available')
class TestSpatialSelectColumnarSpatialpandas(TestSpatialSelectColumnar):
    __test__ = True
    method = 'spatialpandas'
