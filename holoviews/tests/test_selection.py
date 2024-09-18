from unittest import SkipTest, skip, skipIf

import pandas as pd
import panel as pn

import holoviews as hv
from holoviews.core.options import Cycle, Store
from holoviews.element import ErrorBars, Points, Rectangles, Table, VSpan
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.util import linear_gradient
from holoviews.selection import link_selections
from holoviews.streams import SelectionXY

try:
    from holoviews.operation.datashader import datashade, dynspread
except ImportError:
    datashade = None

ds_skip = skipIf(datashade is None, "Datashader not available")


unselected_color = "#ff0000"
box_region_color = linear_gradient(unselected_color, "#000000", 9)[3]
hist_region_color = linear_gradient(unselected_color, "#000000", 9)[1]

class TestLinkSelections(ComparisonTestCase):

    __test__ = False

    def setUp(self):
        self.data = pd.DataFrame(
            {'x': [1, 2, 3],
             'y': [0, 3, 2],
             'e': [1, 1.5, 2],
            },
            columns=['x', 'y', 'e']
        )

    def element_color(self, element):
        raise NotImplementedError

    def check_base_points_like(self, base_points, lnk_sel, data=None):
        if data is None:
            data = self.data

        self.assertEqual(
            self.element_color(base_points),
            lnk_sel.unselected_color
        )
        self.assertEqual(base_points.data, data)

    @staticmethod
    def get_value_with_key_type(d, hvtype):
        for k, v in d.items():
            if isinstance(k, hvtype) or \
                    isinstance(k, hv.DynamicMap) and k.type == hvtype:
                return v

        raise KeyError(f"No key with type {hvtype}")

    @staticmethod
    def expected_selection_color(element, lnk_sel):
        if lnk_sel.selected_color is not None:
            expected_color = lnk_sel.selected_color
        else:
            expected_color = element.opts.get(group='style')[0].get('color')
        return expected_color

    def check_overlay_points_like(self, overlay_points, lnk_sel, data):
        self.assertEqual(
            self.element_color(overlay_points),
            self.expected_selection_color(overlay_points, lnk_sel),
        )

        self.assertEqual(overlay_points.data, data)

    def test_points_selection(self, dynamic=False, show_regions=True):
        points = Points(self.data)
        if dynamic:
            # Convert points to DynamicMap that returns the element
            points = hv.util.Dynamic(points)

        lnk_sel = link_selections.instance(show_regions=show_regions,
                                           unselected_color='#ff0000')
        linked = lnk_sel(points)
        current_obj = linked[()]

        # Check initial state of linked dynamic map
        self.assertIsInstance(current_obj, hv.Overlay)
        unselected, selected, region, region2 = current_obj.values()

        # Check initial base layer
        self.check_base_points_like(unselected, lnk_sel)

        # Check selection layer
        self.check_overlay_points_like(selected, lnk_sel, self.data)

        # Perform selection of second and third point
        selectionxy = TestLinkSelections.get_value_with_key_type(
            lnk_sel._selection_expr_streams, hv.Points
        ).input_streams[0].input_stream.input_streams[0]

        self.assertIsInstance(selectionxy, hv.streams.SelectionXY)
        selectionxy.event(bounds=(0, 1, 5, 5))
        unselected, selected, region, region2 = linked[()].values()

        # Check that base layer is unchanged
        self.check_base_points_like(unselected, lnk_sel)

        # Check selection layer
        self.check_overlay_points_like(selected, lnk_sel, self.data.iloc[1:])

        if show_regions:
            self.assertEqual(region, Rectangles([(0, 1, 5, 5)]))
        else:
            self.assertEqual(region, Rectangles([]))

    def test_points_selection_hide_region(self):
        self.test_points_selection(show_regions=False)

    def test_points_selection_dynamic(self):
        self.test_points_selection(dynamic=True)

    def test_layout_selection_points_table(self):
        points = Points(self.data)
        table = Table(self.data)
        lnk_sel = link_selections.instance(
            selected_color="#aa0000", unselected_color='#ff0000'
        )
        linked = lnk_sel(points + table)

        current_obj = linked[()]

        # Check initial base points
        self.check_base_points_like(
            current_obj[0][()].Points.I,
            lnk_sel
        )

        # Check initial selection points
        self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel,
                                        self.data)

        # Check initial table
        self.assertEqual(
            self.element_color(current_obj[1][()]),
            [lnk_sel.selected_color] * len(self.data)
        )

        # Select first and third point
        selectionxy = TestLinkSelections.get_value_with_key_type(
            lnk_sel._selection_expr_streams, hv.Points
        ).input_streams[0].input_stream.input_streams[0]

        selectionxy.event(bounds=(0, 0, 4, 2))
        current_obj = linked[()]

        # Check base points
        self.check_base_points_like(
            current_obj[0][()].Points.I,
            lnk_sel
        )

        # Check selection points
        self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel,
                                        self.data.iloc[[0, 2]])

        # Check selected table
        self.assertEqual(
            self.element_color(current_obj[1][()]),
            [
                lnk_sel.selected_color,
                lnk_sel.unselected_color,
                lnk_sel.selected_color,
            ]
        )

    def test_overlay_points_errorbars(self, dynamic=False):
        points = Points(self.data)
        error = ErrorBars(self.data, kdims='x', vdims=['y', 'e'])
        lnk_sel = link_selections.instance(unselected_color='#ff0000')
        overlay = points * error

        if dynamic:
            overlay = hv.util.Dynamic(overlay)

        linked = lnk_sel(overlay)
        current_obj = linked[()]

        # Check initial base layers
        self.check_base_points_like(current_obj.Points.I, lnk_sel)
        self.check_base_points_like(current_obj.ErrorBars.I, lnk_sel)

        # Check initial selection layers
        self.check_overlay_points_like(current_obj.Points.II, lnk_sel, self.data)
        self.check_overlay_points_like(current_obj.ErrorBars.II, lnk_sel, self.data)

        # Select first and third point
        selectionxy = TestLinkSelections.get_value_with_key_type(
            lnk_sel._selection_expr_streams, hv.Points
        ).input_streams[0].input_stream.input_streams[0]
        selectionxy.event(bounds=(0, 0, 4, 2))

        current_obj = linked[()]

        # Check base layers haven't changed
        self.check_base_points_like(current_obj.Points.I, lnk_sel)
        self.check_base_points_like(current_obj.ErrorBars.I, lnk_sel)

        # Check selected layers
        self.check_overlay_points_like(current_obj.Points.II, lnk_sel,
                                        self.data.iloc[[0, 2]])
        self.check_overlay_points_like(current_obj.ErrorBars.II, lnk_sel,
                                        self.data.iloc[[0, 2]])

    @ds_skip
    def test_datashade_selection(self):
        points = Points(self.data)
        layout = points + dynspread(datashade(points))

        lnk_sel = link_selections.instance(unselected_color='#ff0000')
        linked = lnk_sel(layout)
        current_obj = linked[()]

        # Check base points layer
        self.check_base_points_like(current_obj[0][()].Points.I, lnk_sel)

        # Check selection layer
        self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel, self.data)

        # Check RGB base layer
        self.assertEqual(
            current_obj[1][()].RGB.I,
            dynspread(
                datashade(points, cmap=lnk_sel.unselected_cmap, alpha=255)
            )[()]
        )

        # Check RGB selection layer
        self.assertEqual(
            current_obj[1][()].RGB.II,
            dynspread(
                datashade(points, cmap=lnk_sel.selected_cmap, alpha=255)
            )[()]
        )

        # Perform selection of second and third point
        selectionxy = TestLinkSelections.get_value_with_key_type(
            lnk_sel._selection_expr_streams, hv.Points
        ).input_streams[0].input_stream.input_streams[0]

        self.assertIsInstance(selectionxy, SelectionXY)
        selectionxy.event(bounds=(0, 1, 5, 5))
        current_obj = linked[()]

        # Check that base points layer is unchanged
        self.check_base_points_like(current_obj[0][()].Points.I, lnk_sel)

        # Check points selection layer
        self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel,
                                        self.data.iloc[1:])

        # Check that base RGB layer is unchanged
        self.assertEqual(
            current_obj[1][()].RGB.I,
            dynspread(
                datashade(points, cmap=lnk_sel.unselected_cmap, alpha=255)
            )[()]
        )

        # Check selection RGB layer
        self.assertEqual(
            current_obj[1][()].RGB.II,
            dynspread(
                datashade(
                    points.iloc[1:], cmap=lnk_sel.selected_cmap, alpha=255
                )
            )[()]
        )

    @ds_skip
    def test_datashade_in_overlay_selection(self):
        points = Points(self.data)
        layout = points * dynspread(datashade(points))

        lnk_sel = link_selections.instance(unselected_color='#ff0000')
        linked = lnk_sel(layout)
        current_obj = linked[()]

        # Check base points layer
        self.check_base_points_like(current_obj[()].Points.I, lnk_sel)

        # Check selection layer
        self.check_overlay_points_like(current_obj[()].Points.II, lnk_sel, self.data)

        # Check RGB base layer
        self.assertEqual(
            current_obj[()].RGB.I,
            dynspread(
                datashade(points, cmap=lnk_sel.unselected_cmap, alpha=255)
            )[()]
        )

        # Check RGB selection layer
        self.assertEqual(
            current_obj[()].RGB.II,
            dynspread(
                datashade(points, cmap=lnk_sel.selected_cmap, alpha=255)
            )[()]
        )

        # Perform selection of second and third point
        selectionxy = TestLinkSelections.get_value_with_key_type(
            lnk_sel._selection_expr_streams, hv.Points
        ).input_streams[0].input_stream.input_streams[0]

        self.assertIsInstance(selectionxy, SelectionXY)
        selectionxy.event(bounds=(0, 1, 5, 5))
        current_obj = linked[()]

        # Check that base points layer is unchanged
        self.check_base_points_like(current_obj[()].Points.I, lnk_sel)

        # Check points selection layer
        self.check_overlay_points_like(current_obj[()].Points.II, lnk_sel,
                                       self.data.iloc[1:])

        # Check that base RGB layer is unchanged
        self.assertEqual(
            current_obj[()].RGB.I,
            dynspread(
                datashade(points, cmap=lnk_sel.unselected_cmap, alpha=255)
            )[()]
        )

        # Check selection RGB layer
        self.assertEqual(
            current_obj[()].RGB.II,
            dynspread(
                datashade(
                    points.iloc[1:], cmap=lnk_sel.selected_cmap, alpha=255
                )
            )[()]
        )

    def test_points_selection_streaming(self):
        buffer = hv.streams.Buffer(self.data.iloc[:2], index=False)
        points = hv.DynamicMap(Points, streams=[buffer])
        lnk_sel = link_selections.instance(unselected_color='#ff0000')
        linked = lnk_sel(points)

        # Perform selection of first and (future) third point
        selectionxy = TestLinkSelections.get_value_with_key_type(
            lnk_sel._selection_expr_streams, hv.Points
        ).input_streams[0].input_stream.input_streams[0]
        self.assertIsInstance(selectionxy, hv.streams.SelectionXY)
        selectionxy.event(bounds=(0, 0, 4, 2))
        current_obj = linked[()]

        # Check initial base layer
        self.check_base_points_like(
            current_obj.Points.I, lnk_sel, self.data.iloc[:2]
        )

        # Check selection layer
        self.check_overlay_points_like(current_obj.Points.II, lnk_sel,
                                        self.data.iloc[[0]])

        # Now stream third point to the DynamicMap
        buffer.send(self.data.iloc[[2]])
        current_obj = linked[()]

        # Check initial base layer
        self.check_base_points_like(
            current_obj.Points.I, lnk_sel, self.data
        )

        # Check selection layer
        self.check_overlay_points_like(current_obj.Points.II, lnk_sel,
                                        self.data.iloc[[0, 2]])

    def do_crossfilter_points_histogram(
            self, selection_mode, cross_filter_mode, selected1, selected2,
            selected3, selected4, points_region1, points_region2,
            points_region3, points_region4, hist_region2, hist_region3,
            hist_region4, show_regions=True, dynamic=False):
        points = Points(self.data)
        hist = points.hist('x', adjoin=False, normed=False, num_bins=5)

        if dynamic:
            # Convert points to DynamicMap that returns the element
            hist_orig = hist
            points = hv.util.Dynamic(points)
        else:
            hist_orig = hist

        lnk_sel = link_selections.instance(
            selection_mode=selection_mode,
            cross_filter_mode=cross_filter_mode,
            show_regions=show_regions,
            selected_color='#00ff00',
            unselected_color='#ff0000'
        )
        linked = lnk_sel(points + hist)
        current_obj = linked[()]

        # Check initial base points
        self.check_base_points_like(
            current_obj[0][()].Points.I,
            lnk_sel
        )

        # Check initial selection overlay points
        self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel,
                                        self.data)

        # Initial region bounds all None
        self.assertEqual(len(current_obj[0][()].Curve.I), 0)

        # Check initial base histogram
        base_hist = current_obj[1][()].Histogram.I
        self.assertEqual(
            self.element_color(base_hist), lnk_sel.unselected_color
        )
        self.assertEqual(base_hist.data, hist_orig.data)

        # Check initial selection overlay Histogram
        selection_hist = current_obj[1][()].Histogram.II
        self.assertEqual(
            self.element_color(selection_hist),
            self.expected_selection_color(selection_hist, lnk_sel)
        )
        self.assertEqual(selection_hist, base_hist)

        # No selection region
        region_hist = current_obj[1][()].NdOverlay.I.last
        self.assertEqual(region_hist.data, [None, None])

        # (1) Perform selection on points of points [1, 2]
        points_selectionxy = TestLinkSelections.get_value_with_key_type(
            lnk_sel._selection_expr_streams, hv.Points
        ).input_streams[0].input_stream.input_streams[0]
        self.assertIsInstance(points_selectionxy, SelectionXY)
        points_selectionxy.event(bounds=(1, 1, 4, 4))

        # Get current object
        current_obj = linked[()]

        # Check base points unchanged
        self.check_base_points_like(
            current_obj[0][()].Points.I,
            lnk_sel
        )

        # Check points selection overlay
        self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel,
                                       self.data.iloc[selected1])

        # Check points region bounds
        region_bounds = current_obj[0][()].Rectangles.I
        self.assertEqual(region_bounds, Rectangles(points_region1))

        if show_regions:
            self.assertEqual(
                self.element_color(region_bounds),
                box_region_color
            )

        # (2) Perform selection on histogram bars [0, 1]
        hist_selectionxy = TestLinkSelections.get_value_with_key_type(
            lnk_sel._selection_expr_streams, hv.Histogram
        ).input_streams[0].input_stream.input_streams[0]

        self.assertIsInstance(hist_selectionxy, SelectionXY)
        hist_selectionxy.event(bounds=(0, 0, 2.5, 2))

        points_unsel, points_sel, points_region, points_region_poly = current_obj[0][()].values()

        # Check points selection overlay
        self.check_overlay_points_like(points_sel, lnk_sel, self.data.iloc[selected2])

        self.assertEqual(points_region, Rectangles(points_region2))

        # Check base histogram unchanged
        base_hist, region_hist, sel_hist = current_obj[1][()].values()
        self.assertEqual(self.element_color(base_hist), lnk_sel.unselected_color)
        self.assertEqual(base_hist.data, hist_orig.data)

        # Check selection region covers first and second bar
        if show_regions:
            self.assertEqual(self.element_color(region_hist.last), hist_region_color)
        if not len(hist_region2) and lnk_sel.selection_mode != 'inverse':
            self.assertEqual(len(region_hist), 1)
        else:
            self.assertEqual(
                region_hist.last.data, [0, 2.5]
            )

        # Check histogram selection overlay
        self.assertEqual(
            self.element_color(sel_hist),
            self.expected_selection_color(sel_hist, lnk_sel)
        )
        self.assertEqual(
            sel_hist.data, hist_orig.pipeline(hist_orig.dataset.iloc[selected2]).data
        )

        # (3) Perform selection on points points [0, 2]
        points_selectionxy = TestLinkSelections.get_value_with_key_type(
            lnk_sel._selection_expr_streams, hv.Points
        ).input_streams[0].input_stream.input_streams[0]

        self.assertIsInstance(points_selectionxy, SelectionXY)
        points_selectionxy.event(bounds=(0, 0, 4, 2.5))

        # Check selection overlay points contains only second point
        self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel,
                                        self.data.iloc[selected3])

        # Check points region bounds
        region_bounds = current_obj[0][()].Rectangles.I
        self.assertEqual(region_bounds, Rectangles(points_region3))

        # Check second and third histogram bars selected
        selection_hist = current_obj[1][()].Histogram.II
        self.assertEqual(
            selection_hist.data, hist_orig.pipeline(hist_orig.dataset.iloc[selected3]).data
        )

        # Check selection region covers first and second bar
        region_hist = current_obj[1][()].NdOverlay.I.last
        if not len(hist_region3) and lnk_sel.selection_mode != 'inverse':
            self.assertEqual(len(region_hist), 1)
        else:
            self.assertEqual(region_hist.data, [0, 2.5])

        # (4) Perform selection of bars [1, 2]
        hist_selectionxy = TestLinkSelections.get_value_with_key_type(
            lnk_sel._selection_expr_streams, hv.Histogram
        ).input_streams[0].input_stream.input_streams[0]

        self.assertIsInstance(hist_selectionxy, SelectionXY)
        hist_selectionxy.event(bounds=(1.5, 0, 3.5, 2))

        # Check points selection overlay
        self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel,
                                       self.data.iloc[selected4])

        # Check points region bounds
        region_bounds = current_obj[0][()].Rectangles.I
        self.assertEqual(region_bounds, Rectangles(points_region4))

        # Check bar selection region
        region_hist = current_obj[1][()].NdOverlay.I.last
        if show_regions:
            self.assertEqual(
                self.element_color(region_hist), hist_region_color
            )
        if not len(hist_region4) and lnk_sel.selection_mode != 'inverse':
            self.assertEqual(len(region_hist), 1)
        elif (lnk_sel.cross_filter_mode != 'overwrite' and not
              (lnk_sel.cross_filter_mode == 'intersect' and lnk_sel.selection_mode == 'overwrite')):
            self.assertEqual(region_hist.data, [0, 3.5])
        else:
            self.assertEqual(region_hist.data, [1.5, 3.5])

        # Check bar selection overlay
        selection_hist = current_obj[1][()].Histogram.II
        self.assertEqual(
            self.element_color(selection_hist),
            self.expected_selection_color(selection_hist, lnk_sel)
        )
        self.assertEqual(
            selection_hist.data, hist_orig.pipeline(hist_orig.dataset.iloc[selected4]).data
        )

    #  cross_filter_mode="overwrite"
    def test_points_histogram_overwrite_overwrite(self, dynamic=False):
        self.do_crossfilter_points_histogram(
            selection_mode="overwrite", cross_filter_mode="overwrite",
            selected1=[1, 2], selected2=[0, 1], selected3=[0, 2], selected4=[1, 2],
            points_region1=[(1, 1, 4, 4)],
            points_region2=[],
            points_region3=[(0, 0, 4, 2.5)],
            points_region4=[],
            hist_region2=[0, 1], hist_region3=[], hist_region4=[1, 2],
            dynamic=dynamic
        )

    def test_points_histogram_overwrite_overwrite_dynamic(self):
        self.test_points_histogram_overwrite_overwrite(dynamic=True)

    def test_points_histogram_intersect_overwrite(self, dynamic=False):
        self.do_crossfilter_points_histogram(
            selection_mode="intersect", cross_filter_mode="overwrite",
            selected1=[1, 2], selected2=[0, 1], selected3=[0, 2], selected4=[1, 2],
            points_region1=[(1, 1, 4, 4)],
            points_region2=[],
            points_region3=[(0, 0, 4, 2.5)],
            points_region4=[],
            hist_region2=[0, 1], hist_region3=[], hist_region4=[1, 2],
            dynamic=dynamic
        )

    def test_points_histogram_intersect_overwrite_dynamic(self):
        self.test_points_histogram_intersect_overwrite(dynamic=True)

    def test_points_histogram_union_overwrite(self, dynamic=False):
        self.do_crossfilter_points_histogram(
            selection_mode="union", cross_filter_mode="overwrite",
            selected1=[1, 2], selected2=[0, 1], selected3=[0, 2], selected4=[1, 2],
            points_region1=[(1, 1, 4, 4)],
            points_region2=[],
            points_region3=[(0, 0, 4, 2.5)],
            points_region4=[],
            hist_region2=[0, 1], hist_region3=[], hist_region4=[1, 2],
            dynamic=dynamic
        )

    def test_points_histogram_union_overwrite_dynamic(self):
        self.test_points_histogram_union_overwrite(dynamic=True)

    #  cross_filter_mode="intersect"
    def test_points_histogram_overwrite_intersect(self, dynamic=False):
        self.do_crossfilter_points_histogram(
            selection_mode="overwrite", cross_filter_mode="intersect",
            selected1=[1, 2], selected2=[1], selected3=[0], selected4=[2],
            points_region1=[(1, 1, 4, 4)],
            points_region2=[(1, 1, 4, 4)],
            points_region3=[(0, 0, 4, 2.5)],
            points_region4=[(0, 0, 4, 2.5)],
            hist_region2=[0, 1], hist_region3=[0, 1], hist_region4=[1, 2],
            dynamic=dynamic
        )

    def test_points_histogram_overwrite_intersect_dynamic(self):
        self.test_points_histogram_overwrite_intersect(dynamic=True)

    def test_points_histogram_overwrite_intersect_hide_region(self, dynamic=False):
        self.do_crossfilter_points_histogram(
            selection_mode="overwrite", cross_filter_mode="intersect",
            selected1=[1, 2], selected2=[1], selected3=[0], selected4=[2],
            points_region1=[],
            points_region2=[],
            points_region3=[],
            points_region4=[],
            hist_region2=[], hist_region3=[], hist_region4=[],
            show_regions=False, dynamic=dynamic
        )

    def test_points_histogram_overwrite_intersect_hide_region_dynamic(self):
        self.test_points_histogram_overwrite_intersect_hide_region(dynamic=True)

    def test_points_histogram_intersect_intersect(self, dynamic=False):
        self.do_crossfilter_points_histogram(
            selection_mode="intersect", cross_filter_mode="intersect",
            selected1=[1, 2], selected2=[1], selected3=[], selected4=[],
            points_region1=[(1, 1, 4, 4)],
            points_region2=[(1, 1, 4, 4)],
            points_region3=[(1, 1, 4, 4), (0, 0, 4, 2.5)],
            points_region4=[(1, 1, 4, 4), (0, 0, 4, 2.5)],
            hist_region2=[0, 1], hist_region3=[0, 1], hist_region4=[1],
            dynamic=dynamic
        )

    def test_points_histogram_intersect_intersect_dynamic(self):
        self.test_points_histogram_intersect_intersect(dynamic=True)

    def test_points_histogram_union_intersect(self, dynamic=False):
        self.do_crossfilter_points_histogram(
            selection_mode="union", cross_filter_mode="intersect",
            selected1=[1, 2], selected2=[1], selected3=[0, 1], selected4=[0, 1, 2],
            points_region1=[(1, 1, 4, 4)],
            points_region2=[(1, 1, 4, 4)],
            points_region3=[(1, 1, 4, 4), (0, 0, 4, 2.5)],
            points_region4=[(1, 1, 4, 4), (0, 0, 4, 2.5)],
            hist_region2=[0, 1], hist_region3=[0, 1], hist_region4=[0, 1, 2],
            dynamic=dynamic
        )

    def test_points_histogram_union_intersect_dynamic(self):
        self.test_points_histogram_union_intersect(dynamic=True)

    def test_points_histogram_inverse_intersect(self, dynamic=False):
        self.do_crossfilter_points_histogram(
            selection_mode="inverse", cross_filter_mode="intersect",
            selected1=[0], selected2=[], selected3=[], selected4=[],
            points_region1=[(1, 1, 4, 4)],
            points_region2=[(1, 1, 4, 4)],
            points_region3=[(1, 1, 4, 4), (0, 0, 4, 2.5)],
            points_region4=[(1, 1, 4, 4), (0, 0, 4, 2.5)],
            hist_region2=[], hist_region3=[], hist_region4=[],
            dynamic=dynamic
        )

    def test_points_histogram_inverse_intersect_dynamic(self):
        self.test_points_histogram_inverse_intersect(dynamic=True)


# Backend implementations
class TestLinkSelectionsPlotly(TestLinkSelections):

    __test__ = True

    def setUp(self):
        try:
            import holoviews.plotting.plotly  # noqa: F401
        except ImportError:
            raise SkipTest("Plotly required to test plotly backend")
        super().setUp()
        self._backend = Store.current_backend
        Store.set_current_backend('plotly')

    def tearDown(self):
        Store.current_backend = self._backend

    def element_color(self, element, color_prop=None):

        if isinstance(element, Table):
            color = element.opts.get('style').kwargs['fill']
        elif isinstance(element, (Rectangles, VSpan)):
            color = element.opts.get('style').kwargs['line_color']
        else:
            color = element.opts.get('style').kwargs['color']

        if isinstance(color, (Cycle, str)):
            return color
        else:
            return list(color)


class TestLinkSelectionsBokeh(TestLinkSelections):

    __test__ = True

    def setUp(self):
        import holoviews.plotting.bokeh # noqa
        super().setUp()
        self._backend = Store.current_backend
        Store.set_current_backend('bokeh')

    def tearDown(self):
        Store.current_backend = self._backend

    def element_color(self, element):
        color = element.opts.get('style').kwargs['color']

        if isinstance(color, str):
            return color
        else:
            return list(color)

    @skip("Coloring Bokeh table not yet supported")
    def test_layout_selection_points_table(self):
        pass

    @skip("Bokeh ErrorBars selection not yet supported")
    def test_overlay_points_errorbars(self):
        pass

    @skip("Bokeh ErrorBars selection not yet supported")
    def test_overlay_points_errorbars_dynamic(self):
        pass

    def test_empty_layout(self):
        # Test for https://github.com/holoviz/holoviews/issues/6106
        df = pd.DataFrame({"x": [1, 2], "y": [1, 2], "cat": ["A", "B"]})

        checkboxes = pn.widgets.CheckBoxGroup(options=['A', 'B'])

        def func(check):
            return hv.Scatter(df[df['cat'].isin(check)])

        ls = hv.link_selections.instance()
        a = ls(hv.DynamicMap(pn.bind(func, checkboxes)))
        b = ls(hv.DynamicMap(pn.bind(func, checkboxes)))

        hv.renderer('bokeh').get_plot(a + b)
        checkboxes.value = ['A']
