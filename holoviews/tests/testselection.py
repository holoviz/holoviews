from unittest import SkipTest, skip, skipIf

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

    def check_overlay_scatter_like(self, overlay_scatter, lnk_sel, data):
        self.assertEqual(
            self.element_color(overlay_scatter),
            lnk_sel.selected_color
        )
        self.assertEqual(
            self.element_visible(overlay_scatter),
            len(data) != len(self.data)
        )

        self.assertEqual(overlay_scatter.data, data)

    def test_scatter_selection(self, dynamic=False):
        scatter = hv.Scatter(self.data, kdims='x', vdims='y')
        if dynamic:
            # Convert scatter to DynamicMap that returns the element
            scatter = hv.util.Dynamic(scatter)

        lnk_sel = link_selections.instance()
        linked = lnk_sel(scatter)
        current_obj = linked[()]

        # Check initial state of linked dynamic map
        self.assertIsInstance(current_obj, hv.Overlay)

        # Check initial base layer
        self.check_base_scatter_like(current_obj.Scatter.I, lnk_sel)

        # Check selection layer
        self.check_overlay_scatter_like(current_obj.Scatter.II, lnk_sel, self.data)

        # Perform selection of second and third point
        boundsxy = lnk_sel._selection_expr_streams[0]._source_streams[0]
        self.assertIsInstance(boundsxy, hv.streams.BoundsXY)
        boundsxy.event(bounds=(0, 1, 5, 5))
        current_obj = linked[()]

        # Check that base layer is unchanged
        self.check_base_scatter_like(current_obj.Scatter.I, lnk_sel)

        # Check selection layer
        self.check_overlay_scatter_like(current_obj.Scatter.II, lnk_sel, self.data.iloc[1:])

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
            self.data
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
            self.data.iloc[[0, 2]]
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
            current_obj.Scatter.II, lnk_sel, self.data
        )
        self.check_overlay_scatter_like(
            current_obj.ErrorBars.II, lnk_sel, self.data
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
            current_obj.Scatter.II, lnk_sel, self.data.iloc[[0, 2]]
        )
        self.check_overlay_scatter_like(
            current_obj.ErrorBars.II, lnk_sel, self.data.iloc[[0, 2]]
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
            current_obj[0][()].Scatter.II, lnk_sel, self.data
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
            current_obj[0][()].Scatter.II, lnk_sel, self.data.iloc[1:]
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
            current_obj.Scatter.II, lnk_sel, self.data.iloc[[0]]
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
            current_obj.Scatter.II, lnk_sel, self.data.iloc[[0, 2]]
        )

    def test_crossfilter_scatter_histogram(self, dynamic=False):
        scatter = hv.Scatter(self.data, kdims='x', vdims='y')
        hist = scatter.hist('x', adjoin=False, normed=False, num_bins=5)
        lnk_sel = link_selections.instance(mode="crossfilter")
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
            self.data
        )

        # Initial region bounds all None
        self.assertEqual(
            list(current_obj[0][()].Bounds.I.data[0]['x']),
            [None] * 5
        )

        # Check initial base histogram
        base_hist = current_obj[1][()].Histogram.I
        self.assertEqual(
            self.element_color(base_hist), lnk_sel.unselected_color
        )
        self.assertTrue(self.element_visible(base_hist))
        self.assertEqual(base_hist.data, hist.data)

        # No selection region
        region_hist = current_obj[1][()].Histogram.II
        self.assertEqual(region_hist.data, hist.pipeline(hist.dataset.iloc[:0]).data)

        # Check initial selection overlay Histogram
        selection_hist = current_obj[1][()].Histogram.III
        self.assertEqual(
            self.element_color(selection_hist), lnk_sel.selected_color
        )
        self.assertFalse(self.element_visible(selection_hist))
        self.assertEqual(selection_hist.data, hist.data)

        # Perform selection of second and third point in scatter
        scatter_boundsxy = lnk_sel._selection_expr_streams[0]._source_streams[0]
        self.assertIsInstance(scatter_boundsxy, hv.streams.BoundsXY)
        scatter_boundsxy.event(bounds=(1, 1, 4, 4))

        # Perform selection of first two bars in histogram
        hist_boundsxy = lnk_sel._selection_expr_streams[1]._source_streams[0]
        self.assertIsInstance(hist_boundsxy, hv.streams.BoundsXY)
        hist_boundsxy.event(bounds=(0, 0, 2.5, 2))

        # Get current object
        current_obj = linked[()]

        # Check base scatter unchanged
        self.check_base_scatter_like(
            current_obj[0][()].Scatter.I,
            lnk_sel
        )

        # Check initial selection overlay scatter contains only second point
        self.check_overlay_scatter_like(
            current_obj[0][()].Scatter.II,
            lnk_sel,
            self.data.iloc[[1]]
        )

        # Check scatter region bounds
        region_bounds = current_obj[0][()].Bounds.I
        self.assertEqual(
            list(region_bounds.data[0]['x']),
            [1, 1, 4, 4, 1]
        )
        self.assertEqual(
            list(region_bounds.data[0]['y']),
            [1, 4, 4, 1, 1]
        )
        self.assertEqual(
            self.element_color(region_bounds),
            lnk_sel._region_color
        )

        # Check initial base histogram unchanged
        base_hist = current_obj[1][()].Histogram.I
        self.assertEqual(
            self.element_color(base_hist), lnk_sel.unselected_color
        )
        self.assertTrue(self.element_visible(base_hist))
        self.assertEqual(base_hist.data, hist.data)

        # Check selection region covers first and second bar
        region_hist = current_obj[1][()].Histogram.II
        self.assertEqual(
            self.element_color(region_hist), lnk_sel._region_color
        )
        self.assertEqual(
            region_hist.data, hist.pipeline(hist.dataset.iloc[[0, 1]]).data
        )

        # Check selection overlay covers only second bar
        selection_hist = current_obj[1][()].Histogram.III
        self.assertEqual(
            self.element_color(selection_hist), lnk_sel.selected_color
        )
        self.assertTrue(self.element_visible(selection_hist))
        self.assertEqual(
            selection_hist.data, hist.pipeline(hist.dataset.iloc[[1]]).data
        )


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
