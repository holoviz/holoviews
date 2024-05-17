import numpy as np

from holoviews.core import DynamicMap, HoloMap, NdOverlay, Overlay
from holoviews.element import Curve, Scatter

from ...utils import LoggingComparisonTestCase
from .test_plot import TestMPLPlot, mpl_renderer

try:
    from holoviews.plotting.mpl import OverlayPlot
except ImportError:
    pass


class TestOverlayPlot(LoggingComparisonTestCase, TestMPLPlot):

    def test_interleaved_overlay(self):
        """
        Test to avoid regression after fix of https://github.com/holoviz/holoviews/issues/41
        """
        o = Overlay([Curve(np.array([[0, 1]])) , Scatter([[1,1]]) , Curve(np.array([[0, 1]]))])
        OverlayPlot(o)

    def test_overlay_empty_layers(self):
        overlay = Curve(range(10)) * NdOverlay()
        plot = mpl_renderer.get_plot(overlay)
        self.assertEqual(len(plot.subplots), 1)
        self.log_handler.assertContains('WARNING', 'is empty and will be skipped during plotting')

    def test_overlay_update_plot_opts(self):
        hmap = HoloMap(
            {0: (Curve([]) * Curve([])).opts(title='A'),
             1: (Curve([]) * Curve([])).opts(title='B')}
        )
        plot = mpl_renderer.get_plot(hmap)
        self.assertEqual(plot.handles['title'].get_text(), 'A')
        plot.update((1,))
        self.assertEqual(plot.handles['title'].get_text(), 'B')

    def test_overlay_update_plot_opts_inherited(self):
        hmap = HoloMap(
            {0: (Curve([]).opts(title='A') * Curve([])),
             1: (Curve([]).opts(title='B') * Curve([]))}
        )
        plot = mpl_renderer.get_plot(hmap)
        self.assertEqual(plot.handles['title'].get_text(), 'A')
        plot.update((1,))
        self.assertEqual(plot.handles['title'].get_text(), 'B')

    def test_overlay_apply_ranges_disabled(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts('Curve', apply_ranges=False)
        plot = mpl_renderer.get_plot(overlay)
        self.assertTrue(all(np.isnan(e) for e in plot.get_extents(overlay, {})))

    def test_overlay_empty_element_extent(self):
        overlay = Curve([]).redim.range(x=(-10, 10)) * Scatter([]).redim.range(y=(-20, 20))
        plot = mpl_renderer.get_plot(overlay)
        extents = plot.get_extents(overlay, {})
        self.assertEqual(extents, (-10, -20, 10, 20))

    def test_dynamic_subplot_remapping(self):
        # Checks that a plot is appropriately updated when reused
        def cb(X):
            return NdOverlay({i: Curve(np.arange(10)+i) for i in range(X-2, X)})
        dmap = DynamicMap(cb, kdims=['X']).redim.range(X=(1, 10))
        plot = mpl_renderer.get_plot(dmap)
        plot.update((3,))
        for i, subplot in enumerate(plot.subplots.values()):
            self.assertEqual(subplot.cyclic_index, i+3)
            self.assertEqual(list(subplot.overlay_dims.values()), [i+1])

    def test_dynamic_subplot_creation(self):
        def cb(X):
            return NdOverlay({i: Curve(np.arange(10)+i) for i in range(X)})
        dmap = DynamicMap(cb, kdims=['X']).redim.range(X=(1, 10))
        plot = mpl_renderer.get_plot(dmap)
        self.assertEqual(len(plot.subplots), 1)
        plot.update((3,))
        self.assertEqual(len(plot.subplots), 3)
        for i, subplot in enumerate(plot.subplots.values()):
            self.assertEqual(subplot.cyclic_index, i)

    def test_overlay_xlabel(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(xlabel='custom x-label')
        axes = mpl_renderer.get_plot(overlay).handles['axis']
        self.assertEqual(axes.get_xlabel(), 'custom x-label')

    def test_overlay_ylabel(self):
        overlay = (Curve(range(10)) * Curve(range(10))).opts(ylabel='custom y-label')
        axes = mpl_renderer.get_plot(overlay).handles['axis']
        self.assertEqual(axes.get_ylabel(), 'custom y-label')

    def test_overlay_xlabel_override_propagated(self):
        overlay = (Curve(range(10)).opts(xlabel='custom x-label') * Curve(range(10)))
        axes = mpl_renderer.get_plot(overlay).handles['axis']
        self.assertEqual(axes.get_xlabel(), 'custom x-label')

    def test_overlay_ylabel_override(self):
        overlay = (Curve(range(10)).opts(ylabel='custom y-label') * Curve(range(10)))
        axes = mpl_renderer.get_plot(overlay).handles['axis']
        self.assertEqual(axes.get_ylabel(), 'custom y-label')



class TestLegends(TestMPLPlot):

    def test_overlay_legend(self):
        overlay = Curve(range(10), label='A') * Curve(range(10), label='B')
        plot = mpl_renderer.get_plot(overlay)
        legend = plot.handles['legend']
        legend_labels = [l.get_text() for l in legend.texts]
        self.assertEqual(legend_labels, ['A', 'B'])

    def test_overlay_legend_with_labels(self):
        overlay = (Curve(range(10), label='A') * Curve(range(10), label='B')).opts(
            legend_labels={'A': 'A Curve', 'B': 'B Curve'})
        plot = mpl_renderer.get_plot(overlay)
        legend = plot.handles['legend']
        legend_labels = [l.get_text() for l in legend.texts]
        self.assertEqual(legend_labels, ['A Curve', 'B Curve'])
