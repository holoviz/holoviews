"""
Test cases for the Comparisons class over the Chart elements
"""

from unittest import SkipTest

import numpy as np

from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
    Area, BoxWhisker, Curve, Distribution, HSpan, Image, Points,
    Rectangles, RGB, Scatter, Segments, Violin, VSpan
)
from holoviews.element.comparison import ComparisonTestCase


class TestSelection1DExpr(ComparisonTestCase):

    def setUp(self):
        try:
            import holoviews.plotting.bokeh # noqa
        except:
            raise SkipTest("Bokeh selection tests require bokeh.")
        super(TestSelection1DExpr, self).setUp()
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
        try:
            import holoviews.plotting.bokeh # noqa
        except:
            raise SkipTest("Bokeh selection tests require bokeh.")
        super(TestSelection2DExpr, self).setUp()
        self._backend = Store.current_backend
        Store.set_current_backend('bokeh')

    def tearDown(self):
        Store.current_backend = self._backend

    def test_points_selection_numeric(self):
        points = Points([3, 2, 1, 3, 4])
        expr, bbox, region = points._get_selection_expr_for_stream_value(bounds=(1, 0, 3, 2))
        self.assertEqual(bbox, {'x': (1, 3), 'y': (0, 2)})
        self.assertEqual(expr.apply(points), np.array([False, True, True, False, False]))
        self.assertEqual(region, Rectangles([(1, 0, 3, 2)]))

    def test_points_selection_numeric_inverted(self):
        points = Points([3, 2, 1, 3, 4]).opts(invert_axes=True)
        expr, bbox, region = points._get_selection_expr_for_stream_value(bounds=(0, 1, 2, 3))
        self.assertEqual(bbox, {'x': (1, 3), 'y': (0, 2)})
        self.assertEqual(expr.apply(points), np.array([False, True, True, False, False]))
        self.assertEqual(region, Rectangles([(0, 1, 2, 3)]))

    def test_points_selection_categorical(self):
        points = Points((['B', 'A', 'C', 'D', 'E'], [3, 2, 1, 3, 4]))
        expr, bbox, region = points._get_selection_expr_for_stream_value(
            bounds=(0, 1, 2, 3), x_selection=['B', 'A', 'C'], y_selection=None
        )
        self.assertEqual(bbox, {'x': ['B', 'A', 'C'], 'y': (1, 3)})
        self.assertEqual(expr.apply(points), np.array([True, True, True, False, False]))
        self.assertEqual(region, Rectangles([(0, 1, 2, 3)]))

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
        self.assertEqual(bbox, {'x': (1, 3), 'y': (0, 2)})
        self.assertEqual(expr.apply(scatter), np.array([False, True, True, False, False]))
        self.assertEqual(region, Rectangles([(1, 0, 3, 2)]))

    def test_scatter_selection_numeric_inverted(self):
        scatter = Scatter([3, 2, 1, 3, 4]).opts(invert_axes=True)
        expr, bbox, region = scatter._get_selection_expr_for_stream_value(bounds=(0, 1, 2, 3))
        self.assertEqual(bbox, {'x': (1, 3), 'y': (0, 2)})
        self.assertEqual(expr.apply(scatter), np.array([False, True, True, False, False]))
        self.assertEqual(region, Rectangles([(0, 1, 2, 3)]))

    def test_scatter_selection_categorical(self):
        scatter = Scatter((['B', 'A', 'C', 'D', 'E'], [3, 2, 1, 3, 4]))
        expr, bbox, region = scatter._get_selection_expr_for_stream_value(
            bounds=(0, 1, 2, 3), x_selection=['B', 'A', 'C'], y_selection=None
        )
        self.assertEqual(bbox, {'x': ['B', 'A', 'C'], 'y': (1, 3)})
        self.assertEqual(expr.apply(scatter), np.array([True, True, True, False, False]))
        self.assertEqual(region, Rectangles([(0, 1, 2, 3)]))

    def test_scatter_selection_numeric_index_cols(self):
        scatter = Scatter([3, 2, 1, 3, 2])
        expr, bbox, region = scatter._get_selection_expr_for_stream_value(
            bounds=(1, 0, 3, 2), index_cols=['y']
        )
        self.assertEqual(bbox, {'x': (1, 3), 'y': (0, 2)})
        self.assertEqual(expr.apply(scatter), np.array([False, False, True, False, False]))
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
        self.assertEqual(region, Rectangles([(0.5, 1.5, 2.1, 3.1)]))

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
        self.assertEqual(region, Rectangles([(1.5, 0.5, 3.1, 2.1)]))

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
        self.assertEqual(region, Rectangles([(0.5, 1.5, 2.1, 3.1)]))

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
        self.assertEqual(region, Rectangles([(1.5, 0.5, 3.1, 2.1)]))



class TestSelectionGeomExpr(ComparisonTestCase):

    def setUp(self):
        try:
            import holoviews.plotting.bokeh # noqa
        except:
            raise SkipTest("Bokeh selection tests require bokeh.")
        super(TestSelectionGeomExpr, self).setUp()
        self._backend = Store.current_backend
        Store.set_current_backend('bokeh')

    def tearDown(self):
        Store.current_backend = self._backend

    def test_rect_selection_numeric(self):
        rect = Rectangles([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)])
        expr, bbox, region = rect._get_selection_expr_for_stream_value(bounds=(0.5, 0.9, 3.4, 4.9))
        self.assertEqual(bbox, {'x0': (0.5, 3.4), 'y0': (0.9, 4.9), 'x1': (0.5, 3.4), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(rect), np.array([False, True, False]))
        self.assertEqual(region, Rectangles([(0.5, 0.9, 3.4, 4.9)]))
        expr, bbox, region = rect._get_selection_expr_for_stream_value(bounds=(0, 0.9, 3.5, 4.9))
        self.assertEqual(bbox, {'x0': (0, 3.5), 'y0': (0.9, 4.9), 'x1': (0, 3.5), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(rect), np.array([True, True, True]))
        self.assertEqual(region, Rectangles([(0, 0.9, 3.5, 4.9)]))

    def test_rect_selection_numeric_inverted(self):
        rect = Rectangles([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)]).opts(invert_axes=True)
        expr, bbox, region = rect._get_selection_expr_for_stream_value(bounds=(0.9, 0.5, 4.9, 3.4))
        self.assertEqual(bbox, {'x0': (0.5, 3.4), 'y0': (0.9, 4.9), 'x1': (0.5, 3.4), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(rect), np.array([False, True, False]))
        self.assertEqual(region, Rectangles([(0.9, 0.5, 4.9, 3.4)]))
        expr, bbox, region = rect._get_selection_expr_for_stream_value(bounds=(0.9, 0, 4.9, 3.5))
        self.assertEqual(bbox, {'x0': (0, 3.5), 'y0': (0.9, 4.9), 'x1': (0, 3.5), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(rect), np.array([True, True, True]))
        self.assertEqual(region, Rectangles([(0.9, 0, 4.9, 3.5)]))

    def test_segments_selection_numeric(self):
        segs = Segments([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)])
        expr, bbox, region = segs._get_selection_expr_for_stream_value(bounds=(0.5, 0.9, 3.4, 4.9))
        self.assertEqual(bbox, {'x0': (0.5, 3.4), 'y0': (0.9, 4.9), 'x1': (0.5, 3.4), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(segs), np.array([False, True, False]))
        self.assertEqual(region, Rectangles([(0.5, 0.9, 3.4, 4.9)]))
        expr, bbox, region = segs._get_selection_expr_for_stream_value(bounds=(0, 0.9, 3.5, 4.9))
        self.assertEqual(bbox, {'x0': (0, 3.5), 'y0': (0.9, 4.9), 'x1': (0, 3.5), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(segs), np.array([True, True, True]))
        self.assertEqual(region, Rectangles([(0, 0.9, 3.5, 4.9)]))

    def test_segs_selection_numeric_inverted(self):
        segs = Segments([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)]).opts(invert_axes=True)
        expr, bbox, region = segs._get_selection_expr_for_stream_value(bounds=(0.9, 0.5, 4.9, 3.4))
        self.assertEqual(bbox, {'x0': (0.5, 3.4), 'y0': (0.9, 4.9), 'x1': (0.5, 3.4), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(segs), np.array([False, True, False]))
        self.assertEqual(region, Rectangles([(0.9, 0.5, 4.9, 3.4)]))
        expr, bbox, region = segs._get_selection_expr_for_stream_value(bounds=(0.9, 0, 4.9, 3.5))
        self.assertEqual(bbox, {'x0': (0, 3.5), 'y0': (0.9, 4.9), 'x1': (0, 3.5), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(segs), np.array([True, True, True]))
        self.assertEqual(region, Rectangles([(0.9, 0, 4.9, 3.5)]))
