from unittest import SkipTest, skip, skipIf

import holoviews as hv
import pandas as pd

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


# Backend implementations
class TestLinkSelectionsPlotly(TestLinkSelections):
    def setUp(self):
        try:
            import holoviews.plotting.plotly
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
        else:
            color = element.opts.get('style').kwargs['color']

        if isinstance(color, str):
            return color
        else:
            return list(color)

    def element_visible(self, element):
        return element.opts.get('style').kwargs['visible']


class TestLinkSelectionsBokeh(TestLinkSelections):
    def setUp(self):
        try:
            import holoviews.plotting.bokeh
        except:
            raise SkipTest("Bokeh selection tests require bokeh.")
        super(TestLinkSelectionsBokeh, self).setUp()
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
