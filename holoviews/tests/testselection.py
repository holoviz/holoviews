from unittest import SkipTest, skip, skipIf
import numpy as np
import holoviews as hv
import pandas as pd

from holoviews.core.util import unicode, basestring
from holoviews.core.options import Store
from holoviews.selection import link_selections
from holoviews.element.comparison import ComparisonTestCase

try:
    from holoviews.operation.datashader import datashade, dynspread
except:
    datashade = None

ds_skip = skipIf(datashade is None, "Datashader not available")


class TestLinkSelections(ComparisonTestCase):

    def setUp(self):
        if type(self) is TestLinkSelections:
            # Only run tests in subclasses
            raise SkipTest("Not supported")

        self.data = pd.DataFrame(
            {'x': [1, 2, 3],
             'y': [0, 3, 2],
             'e': [1, 1.5, 2],
             },
            columns=['x', 'y', 'e']
        )

    def element_color(self, element):
        raise NotImplementedError

    def element_visible(self, element):
        raise NotImplementedError

    def check_base_scatter_like(self, base_scatter, lnk_sel, data=None):
        if data is None:
            data = self.data

        self.assertEqual(
            self.element_color(base_scatter),
            lnk_sel.unselected_color
        )
        self.assertTrue(self.element_visible(base_scatter))
        self.assertEqual(base_scatter.data, data)

    def check_overlay_scatter_like(
            self, overlay_scatter, lnk_sel, data, visible
    ):
        self.assertEqual(
            self.element_color(overlay_scatter),
            lnk_sel.selected_color
        )
        self.assertEqual(
            self.element_visible(overlay_scatter),
            visible
        )

        self.assertEqual(overlay_scatter.data, data)

    def test_scatter_selection(self, dynamic=False, show_regions=True):
        scatter = hv.Scatter(self.data, kdims='x', vdims='y')
        if dynamic:
            # Convert scatter to DynamicMap that returns the element
            scatter = hv.util.Dynamic(scatter)

        lnk_sel = link_selections.instance(show_regions=show_regions)
        linked = lnk_sel(scatter)
        current_obj = linked[()]

        # Check initial state of linked dynamic map
        self.assertIsInstance(current_obj, hv.Overlay)

        # Check initial base layer
        self.check_base_scatter_like(current_obj.Scatter.I, lnk_sel)

        # Check selection layer
        self.check_overlay_scatter_like(
            current_obj.Scatter.II, lnk_sel, self.data, visible=False
        )

        # Perform selection of second and third point
        boundsxy = lnk_sel._selection_expr_streams[0]._source_streams[0]
        self.assertIsInstance(boundsxy, hv.streams.BoundsXY)
        boundsxy.event(bounds=(0, 1, 5, 5))
        current_obj = linked[()]

        # Check that base layer is unchanged
        self.check_base_scatter_like(current_obj.Scatter.I, lnk_sel)

        # Check selection layer
        self.check_overlay_scatter_like(
            current_obj.Scatter.II, lnk_sel, self.data.iloc[1:], visible=True
        )

        if show_regions:
            self.assertEqual(
                current_obj.Curve.I,
                hv.Curve(([0, 5, 5, 0, 0], [1, 1, 5, 5, 1]))
            )
        else:
            self.assertEqual(
                current_obj.Curve.I,
                hv.Curve(([], []))
            )

    def test_scatter_selection_hide_region(self):
        self.test_scatter_selection(show_regions=False)

    def test_scatter_selection_dynamic(self):
        self.test_scatter_selection(dynamic=True)

    def test_layout_selection_scatter_table(self):
        scatter = hv.Scatter(self.data, kdims='x', vdims='y')
        table = hv.Table(self.data)
        lnk_sel = link_selections.instance()
        linked = lnk_sel(scatter + table)

        current_obj = linked[()]

        # Check initial base scatter
        self.check_base_scatter_like(
            current_obj[0][()].Scatter.I,
            lnk_sel
        )

        # Check initial selection scatter
        self.check_overlay_scatter_like(
            current_obj[0][()].Scatter.II,
            lnk_sel,
            self.data,
            visible=False,
        )

        # Check initial table
        self.assertEqual(
            self.element_color(current_obj[1][()]),
            [lnk_sel.unselected_color] * len(self.data)
        )

        # Select first and third point
        boundsxy = lnk_sel._selection_expr_streams[0]._source_streams[0]
        boundsxy.event(bounds=(0, 0, 4, 2))
        current_obj = linked[()]

        # Check base scatter
        self.check_base_scatter_like(
            current_obj[0][()].Scatter.I,
            lnk_sel
        )

        # Check selection scatter
        self.check_overlay_scatter_like(
            current_obj[0][()].Scatter.II,
            lnk_sel,
            self.data.iloc[[0, 2]],
            visible=True,
        )

        # Check selected table
        self.assertEqual(
            self.element_color(current_obj[1][()]),
            [
                lnk_sel.selected_color,
                lnk_sel.unselected_color,
                lnk_sel.selected_color,
            ]
        )

    def test_overlay_scatter_errorbars(self, dynamic=False):
        scatter = hv.Scatter(self.data, kdims='x', vdims='y')
        error = hv.ErrorBars(self.data, kdims='x', vdims=['y', 'e'])
        lnk_sel = link_selections.instance()
        overlay = scatter * error
        if dynamic:
            overlay = hv.util.Dynamic(overlay)

        linked = lnk_sel(overlay)
        current_obj = linked[()]

        # Check initial base layers
        self.check_base_scatter_like(current_obj.Scatter.I, lnk_sel)
        self.check_base_scatter_like(current_obj.ErrorBars.I, lnk_sel)

        # Check initial selection layers
        self.check_overlay_scatter_like(
            current_obj.Scatter.II, lnk_sel, self.data, visible=False
        )
        self.check_overlay_scatter_like(
            current_obj.ErrorBars.II, lnk_sel, self.data, visible=False
        )

        # Select first and third point
        boundsxy = lnk_sel._selection_expr_streams[0]._source_streams[0]
        boundsxy.event(bounds=(0, 0, 4, 2))
        current_obj = linked[()]

        # Check base layers haven't changed
        self.check_base_scatter_like(current_obj.Scatter.I, lnk_sel)
        self.check_base_scatter_like(current_obj.ErrorBars.I, lnk_sel)

        # Check selected layers
        self.check_overlay_scatter_like(
            current_obj.Scatter.II, lnk_sel, self.data.iloc[[0, 2]], visible=True
        )
        self.check_overlay_scatter_like(
            current_obj.ErrorBars.II, lnk_sel, self.data.iloc[[0, 2]], visible=True
        )

    def test_overlay_scatter_errorbars_dynamic(self):
        self.test_overlay_scatter_errorbars(dynamic=True)

    @ds_skip
    def test_datashade_selection(self):
        scatter = hv.Scatter(self.data, kdims='x', vdims='y')
        layout = scatter + dynspread(datashade(scatter))

        lnk_sel = link_selections.instance()
        linked = lnk_sel(layout)
        current_obj = linked[()]

        # Check base scatter layer
        self.check_base_scatter_like(current_obj[0][()].Scatter.I, lnk_sel)

        # Check selection layer
        self.check_overlay_scatter_like(
            current_obj[0][()].Scatter.II, lnk_sel, self.data, visible=False
        )

        # Check RGB base layer
        self.assertEqual(
            current_obj[1][()].RGB.I,
            dynspread(
                datashade(scatter, cmap=lnk_sel.unselected_cmap, alpha=255)
            )[()]
        )

        # Check RGB selection layer
        self.assertEqual(
            current_obj[1][()].RGB.II,
            dynspread(
                datashade(scatter, cmap=lnk_sel.selected_cmap, alpha=0)
            )[()]
        )

        # Perform selection of second and third point
        boundsxy = lnk_sel._selection_expr_streams[0]._source_streams[0]
        self.assertIsInstance(boundsxy, hv.streams.BoundsXY)
        boundsxy.event(bounds=(0, 1, 5, 5))
        current_obj = linked[()]

        # Check that base scatter layer is unchanged
        self.check_base_scatter_like(current_obj[0][()].Scatter.I, lnk_sel)

        # Check scatter selection layer
        self.check_overlay_scatter_like(
            current_obj[0][()].Scatter.II, lnk_sel, self.data.iloc[1:], visible=True
        )

        # Check that base RGB layer is unchanged
        self.assertEqual(
            current_obj[1][()].RGB.I,
            dynspread(
                datashade(scatter, cmap=lnk_sel.unselected_cmap, alpha=255)
            )[()]
        )

        # Check selection RGB layer
        self.assertEqual(
            current_obj[1][()].RGB.II,
            dynspread(
                datashade(
                    scatter.iloc[1:], cmap=lnk_sel.selected_cmap, alpha=255
                )
            )[()]
        )

    def test_scatter_selection_streaming(self):
        buffer = hv.streams.Buffer(self.data.iloc[:2], index=False)
        scatter = hv.DynamicMap(hv.Scatter, streams=[buffer])
        lnk_sel = link_selections.instance()
        linked = lnk_sel(scatter)

        # Perform selection of first and (future) third point
        boundsxy = lnk_sel._selection_expr_streams[0]._source_streams[0]
        self.assertIsInstance(boundsxy, hv.streams.BoundsXY)
        boundsxy.event(bounds=(0, 0, 4, 2))
        current_obj = linked[()]

        # Check initial base layer
        self.check_base_scatter_like(
            current_obj.Scatter.I, lnk_sel, self.data.iloc[:2]
        )

        # Check selection layer
        self.check_overlay_scatter_like(
            current_obj.Scatter.II, lnk_sel, self.data.iloc[[0]], visible=True
        )

        # Now stream third point to the DynamicMap
        buffer.send(self.data.iloc[[2]])
        current_obj = linked[()]

        # Check initial base layer
        self.check_base_scatter_like(
            current_obj.Scatter.I, lnk_sel, self.data
        )

        # Check selection layer
        self.check_overlay_scatter_like(
            current_obj.Scatter.II, lnk_sel, self.data.iloc[[0, 2]], visible=True
        )

    def do_crossfilter_scatter_histogram(
            self, element_op, cross_element_op,
            selected1, selected2, selected3, selected4,
            scatter_region1, scatter_region2, scatter_region3, scatter_region4,
            hist_region2, hist_region3, hist_region4, show_regions=True, dynamic=False
    ):
        scatter = hv.Scatter(self.data, kdims='x', vdims='y')
        hist = scatter.hist('x', adjoin=False, normed=False, num_bins=5)

        if dynamic:
            # Convert scatter to DynamicMap that returns the element
            hist_orig = hist
            scatter = hv.util.Dynamic(scatter)
        else:
            hist_orig = hist

        lnk_sel = link_selections.instance(
            element_op=element_op,
            cross_element_op=cross_element_op,
            show_regions=show_regions,
        )
        linked = lnk_sel(scatter + hist)
        current_obj = linked[()]

        # Check initial base scatter
        self.check_base_scatter_like(
            current_obj[0][()].Scatter.I,
            lnk_sel
        )

        # Check initial selection overlay scatter
        self.check_overlay_scatter_like(
            current_obj[0][()].Scatter.II,
            lnk_sel,
            self.data,
            visible=False,
        )

        # Initial region bounds all None
        self.assertEqual(
            list(current_obj[0][()].Curve.I.data['x']),
            []
        )

        # Check initial base histogram
        base_hist = current_obj[1][()].Histogram.I
        self.assertEqual(
            self.element_color(base_hist), lnk_sel.unselected_color
        )
        self.assertTrue(self.element_visible(base_hist))
        self.assertEqual(base_hist.data, hist_orig.data)

        # No selection region
        region_hist = current_obj[1][()].Histogram.II
        self.assertEqual(region_hist.data, hist_orig.pipeline(hist_orig.dataset.iloc[:0]).data)

        # Check initial selection overlay Histogram
        selection_hist = current_obj[1][()].Histogram.III
        self.assertEqual(
            self.element_color(selection_hist), lnk_sel.selected_color
        )
        self.assertFalse(self.element_visible(selection_hist))
        self.assertEqual(selection_hist.data, hist_orig.data)

        # (1) Perform selection on scatter of points [1, 2]
        scatter_boundsxy = lnk_sel._selection_expr_streams[0]._source_streams[0]
        self.assertIsInstance(scatter_boundsxy, hv.streams.BoundsXY)
        scatter_boundsxy.event(bounds=(1, 1, 4, 4))

        # Get current object
        current_obj = linked[()]

        # Check base scatter unchanged
        self.check_base_scatter_like(
            current_obj[0][()].Scatter.I,
            lnk_sel
        )

        # Check scatter selection overlay
        self.check_overlay_scatter_like(
            current_obj[0][()].Scatter.II,
            lnk_sel,
            self.data.iloc[selected1],
            visible=True,
        )

        # Check scatter region bounds
        region_bounds = current_obj[0][()].Curve.I
        self.assertEqual(
            list(region_bounds.data['x']),
            scatter_region1[0]
        )
        self.assertEqual(
            list(region_bounds.data['y']),
            scatter_region1[1]
        )
        if show_regions:
            self.assertEqual(
                self.element_color(region_bounds),
                lnk_sel._region_color
            )

        # Check histogram bars selected
        selection_hist = current_obj[1][()].Histogram.III
        self.assertEqual(
            selection_hist.data, hist_orig.pipeline(hist_orig.dataset.iloc[selected1]).data
        )

        # (2) Perform selection on histogram bars [0, 1]
        hist_boundsxy = lnk_sel._selection_expr_streams[1]._source_streams[0]
        self.assertIsInstance(hist_boundsxy, hv.streams.BoundsXY)
        hist_boundsxy.event(bounds=(0, 0, 2.5, 2))

        # Check scatter selection overlay
        self.check_overlay_scatter_like(
            current_obj[0][()].Scatter.II,
            lnk_sel,
            self.data.iloc[selected2],
            visible=True,
        )

        region_bounds = current_obj[0][()].Curve.I
        self.assertEqual(
            list(region_bounds.data['x']),
            scatter_region2[0]
        )
        self.assertEqual(
            list(region_bounds.data['y']),
            scatter_region2[1]
        )

        # Check base histogram unchanged
        base_hist = current_obj[1][()].Histogram.I
        self.assertEqual(
            self.element_color(base_hist), lnk_sel.unselected_color
        )
        self.assertTrue(self.element_visible(base_hist))
        self.assertEqual(base_hist.data, hist_orig.data)

        # Check selection region covers first and second bar
        region_hist = current_obj[1][()].Histogram.II
        if show_regions:
            self.assertEqual(
                self.element_color(region_hist), lnk_sel._region_color
            )
        self.assertEqual(
            region_hist.data, hist_orig.pipeline(hist_orig.dataset.iloc[hist_region2]).data
        )

        # Check histogram selection overlay
        selection_hist = current_obj[1][()].Histogram.III
        self.assertEqual(
            self.element_color(selection_hist), lnk_sel.selected_color
        )
        self.assertTrue(self.element_visible(selection_hist))
        self.assertEqual(
            selection_hist.data, hist_orig.pipeline(hist_orig.dataset.iloc[selected2]).data
        )

        # (3) Perform selection on scatter points [0, 2]
        scatter_boundsxy = lnk_sel._selection_expr_streams[0]._source_streams[0]
        self.assertIsInstance(scatter_boundsxy, hv.streams.BoundsXY)
        scatter_boundsxy.event(bounds=(0, 0, 4, 2.5))

        # Check selection overlay scatter contains only second point
        self.check_overlay_scatter_like(
            current_obj[0][()].Scatter.II,
            lnk_sel,
            self.data.iloc[selected3],
            visible=True,
        )

        # Check scatter region bounds
        region_bounds = current_obj[0][()].Curve.I
        self.assertEqual(
            list(region_bounds.data['x']),
            scatter_region3[0]
        )
        self.assertEqual(
            list(region_bounds.data['y']),
            scatter_region3[1]
        )

        # Check second and third histogram bars selected
        selection_hist = current_obj[1][()].Histogram.III
        self.assertEqual(
            selection_hist.data, hist_orig.pipeline(hist_orig.dataset.iloc[selected3]).data
        )

        # Check selection region covers first and second bar
        region_hist = current_obj[1][()].Histogram.II
        self.assertEqual(
            region_hist.data,
            hist_orig.pipeline(hist_orig.dataset.iloc[hist_region3]).data
        )

        # (4) Perform selection of bars [1, 2]
        hist_boundsxy = lnk_sel._selection_expr_streams[1]._source_streams[0]
        self.assertIsInstance(hist_boundsxy, hv.streams.BoundsXY)
        hist_boundsxy.event(bounds=(1.5, 0, 3.5, 2))

        # Check scatter selection overlay
        self.check_overlay_scatter_like(
            current_obj[0][()].Scatter.II,
            lnk_sel,
            self.data.iloc[selected4],
            visible=True,
        )

        # Check scatter region bounds
        region_bounds = current_obj[0][()].Curve.I
        self.assertEqual(
            list(region_bounds.data['x']),
            scatter_region4[0]
        )
        self.assertEqual(
            list(region_bounds.data['y']),
            scatter_region4[1]
        )

        # Check bar selection region
        region_hist = current_obj[1][()].Histogram.II
        if show_regions:
            self.assertEqual(
                self.element_color(region_hist), lnk_sel._region_color
            )
        self.assertEqual(
            region_hist.data,
            hist_orig.pipeline(hist_orig.dataset.iloc[hist_region4]).data
        )

        # Check bar selection overlay
        selection_hist = current_obj[1][()].Histogram.III
        self.assertEqual(
            self.element_color(selection_hist), lnk_sel.selected_color
        )
        self.assertTrue(self.element_visible(selection_hist))
        self.assertEqual(
            selection_hist.data, hist_orig.pipeline(hist_orig.dataset.iloc[selected4]).data
        )

    #  cross_element_op="overwrite"
    def test_scatter_histogram_overwrite_overwrite(self, dynamic=False):
        self.do_crossfilter_scatter_histogram(
            element_op="overwrite", cross_element_op="overwrite",
            selected1=[1, 2], selected2=[0, 1], selected3=[0, 2], selected4=[1, 2],
            scatter_region1=([1, 4, 4, 1, 1], [1, 1, 4, 4, 1]),
            scatter_region2=([], []),
            scatter_region3=([0, 4, 4, 0, 0], [0, 0, 2.5, 2.5, 0]),
            scatter_region4=([], []),
            hist_region2=[0, 1], hist_region3=[], hist_region4=[1, 2],
            dynamic=dynamic
        )

    def test_scatter_histogram_overwrite_overwrite_dynamic(self):
        self.test_scatter_histogram_overwrite_overwrite(dynamic=True)

    def test_scatter_histogram_intersect_overwrite(self, dynamic=False):
        self.do_crossfilter_scatter_histogram(
            element_op="intersect", cross_element_op="overwrite",
            selected1=[1, 2], selected2=[0, 1], selected3=[0, 2], selected4=[1, 2],
            scatter_region1=([1, 4, 4, 1, 1], [1, 1, 4, 4, 1]),
            scatter_region2=([], []),
            scatter_region3=([0, 4, 4, 0, 0], [0, 0, 2.5, 2.5, 0]),
            scatter_region4=([], []),
            hist_region2=[0, 1], hist_region3=[], hist_region4=[1, 2],
            dynamic=dynamic
        )

    def test_scatter_histogram_intersect_overwrite_dynamic(self):
        self.test_scatter_histogram_intersect_overwrite(dynamic=True)

    def test_scatter_histogram_union_overwrite(self, dynamic=False):
        self.do_crossfilter_scatter_histogram(
            element_op="union", cross_element_op="overwrite",
            selected1=[1, 2], selected2=[0, 1], selected3=[0, 2], selected4=[1, 2],
            scatter_region1=([1, 4, 4, 1, 1], [1, 1, 4, 4, 1]),
            scatter_region2=([], []),
            scatter_region3=([0, 4, 4, 0, 0], [0, 0, 2.5, 2.5, 0]),
            scatter_region4=([], []),
            hist_region2=[0, 1], hist_region3=[], hist_region4=[1, 2],
            dynamic=dynamic
        )

    def test_scatter_histogram_union_overwrite_dynamic(self):
        self.test_scatter_histogram_union_overwrite(dynamic=True)

    #  cross_element_op="intersect"
    def test_scatter_histogram_overwrite_intersect(self, dynamic=False):
        self.do_crossfilter_scatter_histogram(
            element_op="overwrite", cross_element_op="intersect",
            selected1=[1, 2], selected2=[1], selected3=[0], selected4=[2],
            scatter_region1=([1, 4, 4, 1, 1], [1, 1, 4, 4, 1]),
            scatter_region2=([1, 4, 4, 1, 1], [1, 1, 4, 4, 1]),
            scatter_region3=([0, 4, 4, 0, 0], [0, 0, 2.5, 2.5, 0]),
            scatter_region4=([0, 4, 4, 0, 0], [0, 0, 2.5, 2.5, 0]),
            hist_region2=[0, 1], hist_region3=[0, 1], hist_region4=[1, 2],
            dynamic=dynamic
        )

    def test_scatter_histogram_overwrite_intersect_dynamic(self):
        self.test_scatter_histogram_overwrite_intersect(dynamic=True)

    def test_scatter_histogram_overwrite_intersect_hide_region(self, dynamic=False):
        self.do_crossfilter_scatter_histogram(
            element_op="overwrite", cross_element_op="intersect",
            selected1=[1, 2], selected2=[1], selected3=[0], selected4=[2],
            scatter_region1=([], []),
            scatter_region2=([], []),
            scatter_region3=([], []),
            scatter_region4=([], []),
            hist_region2=[], hist_region3=[], hist_region4=[],
            show_regions=False, dynamic=dynamic
        )

    def test_scatter_histogram_overwrite_intersect_hide_region_dynamic(self):
        self.test_scatter_histogram_overwrite_intersect_hide_region(dynamic=True)

    def test_scatter_histogram_intersect_intersect(self, dynamic=False):
        self.do_crossfilter_scatter_histogram(
            element_op="intersect", cross_element_op="intersect",
            selected1=[1, 2], selected2=[1], selected3=[], selected4=[],
            scatter_region1=([1, 4, 4, 1, 1], [1, 1, 4, 4, 1]),
            scatter_region2=([1, 4, 4, 1, 1], [1, 1, 4, 4, 1]),
            scatter_region3=(
                [1, 4, 4, 1, 1, np.nan, 0, 4, 4, 0, 0],
                [1, 1, 4, 4, 1, np.nan, 0, 0, 2.5, 2.5, 0],
            ),
            scatter_region4=(
                [1, 4, 4, 1, 1, np.nan, 0, 4, 4, 0, 0],
                [1, 1, 4, 4, 1, np.nan, 0, 0, 2.5, 2.5, 0],
            ),
            hist_region2=[0, 1], hist_region3=[0, 1], hist_region4=[1],
            dynamic=dynamic
        )

    def test_scatter_histogram_intersect_intersect_dynamic(self):
        self.test_scatter_histogram_intersect_intersect(dynamic=True)

    def test_scatter_histogram_union_intersect(self, dynamic=False):
        self.do_crossfilter_scatter_histogram(
            element_op="union", cross_element_op="intersect",
            selected1=[1, 2], selected2=[1], selected3=[0, 1], selected4=[0, 1, 2],
            scatter_region1=([1, 4, 4, 1, 1], [1, 1, 4, 4, 1]),
            scatter_region2=([1, 4, 4, 1, 1], [1, 1, 4, 4, 1]),
            scatter_region3=(
                [1, 4, 4, 1, 1, np.nan, 0, 4, 4, 0, 0],
                [1, 1, 4, 4, 1, np.nan, 0, 0, 2.5, 2.5, 0],
            ),
            scatter_region4=(
                [1, 4, 4, 1, 1, np.nan, 0, 4, 4, 0, 0],
                [1, 1, 4, 4, 1, np.nan, 0, 0, 2.5, 2.5, 0],
            ),
            hist_region2=[0, 1], hist_region3=[0, 1], hist_region4=[0, 1, 2],
            dynamic=dynamic
        )

    def test_scatter_histogram_union_intersect_dynamic(self):
        self.test_scatter_histogram_union_intersect(dynamic=True)

    def test_scatter_histogram_difference_intersect(self, dynamic=False):
        self.do_crossfilter_scatter_histogram(
            element_op="difference", cross_element_op="intersect",
            selected1=[0], selected2=[], selected3=[], selected4=[],
            scatter_region1=([1, 4, 4, 1, 1], [1, 1, 4, 4, 1]),
            scatter_region2=([1, 4, 4, 1, 1], [1, 1, 4, 4, 1]),
            scatter_region3=(
                [1, 4, 4, 1, 1, np.nan, 0, 4, 4, 0, 0],
                [1, 1, 4, 4, 1, np.nan, 0, 0, 2.5, 2.5, 0],
            ),
            scatter_region4=(
                [1, 4, 4, 1, 1, np.nan, 0, 4, 4, 0, 0],
                [1, 1, 4, 4, 1, np.nan, 0, 0, 2.5, 2.5, 0],
            ),
            hist_region2=[], hist_region3=[], hist_region4=[],
            dynamic=dynamic
        )

    def test_scatter_histogram_difference_intersect_dynamic(self):
        self.test_scatter_histogram_difference_intersect(dynamic=True)


# Backend implementations
class TestLinkSelectionsPlotly(TestLinkSelections):
    def setUp(self):
        try:
            import holoviews.plotting.plotly # noqa
        except:
            raise SkipTest("Plotly selection tests require plotly.")
        super(TestLinkSelectionsPlotly, self).setUp()
        self._backend = Store.current_backend
        Store.set_current_backend('plotly')

    def tearDown(self):
        Store.current_backend = self._backend

    def element_color(self, element):
        if isinstance(element, hv.Table):
            color = element.opts.get('style').kwargs['fill']
        elif isinstance(element, hv.Bounds):
            color = element.opts.get('style').kwargs['line_color']
        else:
            color = element.opts.get('style').kwargs['color']

        if isinstance(color, (basestring, unicode)):
            return color
        else:
            return list(color)

    def element_visible(self, element):
        return element.opts.get('style').kwargs['visible']


class TestLinkSelectionsBokeh(TestLinkSelections):
    def setUp(self):
        try:
            import holoviews.plotting.bokeh # noqa
        except:
            raise SkipTest("Bokeh selection tests require bokeh.")
        super(TestLinkSelectionsBokeh, self).setUp()
        self._backend = Store.current_backend
        Store.set_current_backend('bokeh')

    def tearDown(self):
        Store.current_backend = self._backend

    def element_color(self, element):
        color = element.opts.get('style').kwargs['color']

        if isinstance(color, (basestring, unicode)):
            return color
        else:
            return list(color)

    def element_visible(self, element):
        return element.opts.get('style').kwargs['alpha'] > 0

    @skip("Coloring Bokeh table not yet supported")
    def test_layout_selection_scatter_table(self):
        pass

    @skip("Bokeh ErrorBars selection not yet supported")
    def test_overlay_scatter_errorbars(self):
        pass

    @skip("Bokeh ErrorBars selection not yet supported")
    def test_overlay_scatter_errorbars_dynamic(self):
        pass
