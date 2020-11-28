from __future__ import absolute_import, unicode_literals

import sys

from unittest import SkipTest, skipIf

import numpy as np

from holoviews import NdOverlay, Overlay, Dimension
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.core.options import Store, Cycle
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element import (Image, Scatter, Curve, Points,
                               Area, VectorField, HLine, Path)
from holoviews.operation import operation
from holoviews.plotting.util import (
    compute_overlayable_zorders, get_min_distance, process_cmap,
    initialize_dynamic, split_dmap_overlay, _get_min_distance_numpy,
    bokeh_palette_to_palette, mplcmap_to_palette, color_intervals,
    get_range, get_axis_padding)
from holoviews.streams import PointerX

try:
    from holoviews.plotting.bokeh import util
    bokeh_renderer = Store.renderers['bokeh']
except:
    bokeh_renderer = None

py2_skip = skipIf(sys.version_info.major == 2, "Not supported in python2")



class TestOverlayableZorders(ComparisonTestCase):

    def test_compute_overlayable_zorders_holomap(self):
        hmap = HoloMap({0: Points([])})
        sources = compute_overlayable_zorders(hmap)
        self.assertEqual(sources[0], [hmap, hmap.last])

    def test_compute_overlayable_zorders_with_overlaid_holomap(self):
        points = Points([])
        hmap = HoloMap({0: points})
        curve = Curve([])
        combined = hmap*curve
        sources = compute_overlayable_zorders(combined)
        self.assertEqual(sources[0], [points, combined.last, combined])

    def test_dynamic_compute_overlayable_zorders_two_mixed_layers(self):
        area = Area(range(10))
        dmap = DynamicMap(lambda: Curve(range(10)), kdims=[])
        combined = area*dmap
        combined[()]
        sources = compute_overlayable_zorders(combined)
        self.assertEqual(sources[0], [area])
        self.assertEqual(sources[1], [dmap])

    def test_dynamic_compute_overlayable_zorders_two_mixed_layers_reverse(self):
        area = Area(range(10))
        dmap = DynamicMap(lambda: Curve(range(10)), kdims=[])
        combined = dmap*area
        combined[()]
        sources = compute_overlayable_zorders(combined)
        self.assertEqual(sources[0], [dmap])
        self.assertEqual(sources[1], [area])

    def test_dynamic_compute_overlayable_zorders_two_dynamic_layers(self):
        area = DynamicMap(lambda: Area(range(10)), kdims=[])
        dmap = DynamicMap(lambda: Curve(range(10)), kdims=[])
        combined = area*dmap
        combined[()]
        sources = compute_overlayable_zorders(combined)
        self.assertEqual(sources[0], [area])
        self.assertEqual(sources[1], [dmap])

    def test_dynamic_compute_overlayable_zorders_two_deep_dynamic_layers(self):
        area = DynamicMap(lambda: Area(range(10)), kdims=[])
        curve = DynamicMap(lambda: Curve(range(10)), kdims=[])
        area_redim = area.redim(x='x2')
        curve_redim = curve.redim(x='x2')
        combined = area_redim*curve_redim
        combined[()]
        sources = compute_overlayable_zorders(combined)
        self.assertIn(area_redim, sources[0])
        self.assertIn(area, sources[0])
        self.assertNotIn(curve_redim, sources[0])
        self.assertNotIn(curve, sources[0])
        self.assertIn(curve_redim, sources[1])
        self.assertIn(curve, sources[1])
        self.assertNotIn(area_redim, sources[1])
        self.assertNotIn(area, sources[1])

    def test_dynamic_compute_overlayable_zorders_three_deep_dynamic_layers(self):
        area = DynamicMap(lambda: Area(range(10)), kdims=[])
        curve = DynamicMap(lambda: Curve(range(10)), kdims=[])
        curve2 = DynamicMap(lambda: Curve(range(10)), kdims=[])
        area_redim = area.redim(x='x2')
        curve_redim = curve.redim(x='x2')
        curve2_redim = curve2.redim(x='x3')
        combined = area_redim*curve_redim
        combined1 = (combined*curve2_redim)
        combined1[()]
        sources = compute_overlayable_zorders(combined1)
        self.assertIn(area_redim, sources[0])
        self.assertIn(area, sources[0])
        self.assertNotIn(curve_redim, sources[0])
        self.assertNotIn(curve, sources[0])
        self.assertNotIn(curve2_redim, sources[0])
        self.assertNotIn(curve2, sources[0])

        self.assertIn(curve_redim, sources[1])
        self.assertIn(curve, sources[1])
        self.assertNotIn(area_redim, sources[1])
        self.assertNotIn(area, sources[1])
        self.assertNotIn(curve2_redim, sources[1])
        self.assertNotIn(curve2, sources[1])

        self.assertIn(curve2_redim, sources[2])
        self.assertIn(curve2, sources[2])
        self.assertNotIn(area_redim, sources[2])
        self.assertNotIn(area, sources[2])
        self.assertNotIn(curve_redim, sources[2])
        self.assertNotIn(curve, sources[2])

    def test_dynamic_compute_overlayable_zorders_three_deep_dynamic_layers_cloned(self):
        area = DynamicMap(lambda: Area(range(10)), kdims=[])
        curve = DynamicMap(lambda: Curve(range(10)), kdims=[])
        curve2 = DynamicMap(lambda: Curve(range(10)), kdims=[])
        area_redim = area.redim(x='x2')
        curve_redim = curve.redim(x='x2')
        curve2_redim = curve2.redim(x='x3')
        combined = area_redim*curve_redim
        combined1 = (combined*curve2_redim).redim(y='y2')
        combined1[()]
        sources = compute_overlayable_zorders(combined1)

        self.assertIn(area_redim, sources[0])
        self.assertIn(area, sources[0])
        self.assertNotIn(curve_redim, sources[0])
        self.assertNotIn(curve, sources[0])
        self.assertNotIn(curve2_redim, sources[0])
        self.assertNotIn(curve2, sources[0])

        self.assertIn(curve_redim, sources[1])
        self.assertIn(curve, sources[1])
        self.assertNotIn(area_redim, sources[1])
        self.assertNotIn(area, sources[1])
        self.assertNotIn(curve2_redim, sources[1])
        self.assertNotIn(curve2, sources[1])

        self.assertIn(curve2_redim, sources[2])
        self.assertIn(curve2, sources[2])
        self.assertNotIn(area_redim, sources[2])
        self.assertNotIn(area, sources[2])
        self.assertNotIn(curve_redim, sources[2])
        self.assertNotIn(curve, sources[2])

    def test_dynamic_compute_overlayable_zorders_mixed_dynamic_and_non_dynamic_overlays_reverse(self):
        area1 = Area(range(10))
        area2 = Area(range(10))
        overlay = area1 * area2
        curve = DynamicMap(lambda: Curve(range(10)), kdims=[])
        curve_redim = curve.redim(x='x2')
        combined = curve_redim*overlay
        combined[()]
        sources = compute_overlayable_zorders(combined)

        self.assertIn(curve_redim, sources[0])
        self.assertIn(curve, sources[0])
        self.assertNotIn(overlay, sources[0])

        self.assertIn(area1, sources[1])
        self.assertIn(overlay, sources[1])
        self.assertNotIn(curve_redim, sources[1])
        self.assertNotIn(curve, sources[1])

        self.assertIn(area2, sources[2])
        self.assertIn(overlay, sources[2])
        self.assertNotIn(curve_redim, sources[2])
        self.assertNotIn(curve, sources[2])

    def test_dynamic_compute_overlayable_zorders_mixed_dynamic_and_non_dynamic_ndoverlays(self):
        ndoverlay = NdOverlay({i: Area(range(10+i)) for i in range(2)})
        curve = DynamicMap(lambda: Curve(range(10)), kdims=[])
        curve_redim = curve.redim(x='x2')
        combined = ndoverlay*curve_redim
        combined[()]
        sources = compute_overlayable_zorders(combined)

        self.assertIn(ndoverlay[0], sources[0])
        self.assertIn(ndoverlay, sources[0])
        self.assertNotIn(curve_redim, sources[0])
        self.assertNotIn(curve, sources[0])

        self.assertIn(ndoverlay[1], sources[1])
        self.assertIn(ndoverlay, sources[1])
        self.assertNotIn(curve_redim, sources[1])
        self.assertNotIn(curve, sources[1])

        self.assertIn(curve_redim, sources[2])
        self.assertIn(curve, sources[2])
        self.assertNotIn(ndoverlay, sources[2])


    def test_dynamic_compute_overlayable_zorders_mixed_dynamic_and_dynamic_ndoverlay_with_streams(self):
        ndoverlay = DynamicMap(lambda x: NdOverlay({i: Area(range(10+i)) for i in range(2)}),
                               kdims=[], streams=[PointerX()])
        curve = DynamicMap(lambda: Curve(range(10)), kdims=[])
        curve_redim = curve.redim(x='x2')
        combined = ndoverlay*curve_redim
        combined[()]
        sources = compute_overlayable_zorders(combined)

        self.assertIn(ndoverlay, sources[0])
        self.assertNotIn(curve_redim, sources[0])
        self.assertNotIn(curve, sources[0])

        self.assertIn(ndoverlay, sources[1])
        self.assertNotIn(curve_redim, sources[1])
        self.assertNotIn(curve, sources[1])

        self.assertIn(curve_redim, sources[2])
        self.assertIn(curve, sources[2])
        self.assertNotIn(ndoverlay, sources[2])

    def test_dynamic_compute_overlayable_zorders_mixed_dynamic_and_dynamic_ndoverlay_with_streams_cloned(self):
        ndoverlay = DynamicMap(lambda x: NdOverlay({i: Area(range(10+i)) for i in range(2)}),
                               kdims=[], streams=[PointerX()])
        curve = DynamicMap(lambda: Curve(range(10)), kdims=[])
        curve_redim = curve.redim(x='x2')
        combined = ndoverlay*curve_redim
        combined[()]
        sources = compute_overlayable_zorders(combined.clone())

        self.assertIn(ndoverlay, sources[0])
        self.assertNotIn(curve_redim, sources[0])
        self.assertNotIn(curve, sources[0])

        self.assertIn(ndoverlay, sources[1])
        self.assertNotIn(curve_redim, sources[1])
        self.assertNotIn(curve, sources[1])

        self.assertIn(curve_redim, sources[2])
        self.assertIn(curve, sources[2])
        self.assertNotIn(ndoverlay, sources[2])

    def test_dynamic_compute_overlayable_zorders_mixed_dynamic_and_non_dynamic_ndoverlays_reverse(self):
        ndoverlay = NdOverlay({i: Area(range(10+i)) for i in range(2)})
        curve = DynamicMap(lambda: Curve(range(10)), kdims=[])
        curve_redim = curve.redim(x='x2')
        combined = curve_redim*ndoverlay
        combined[()]
        sources = compute_overlayable_zorders(combined)

        self.assertIn(curve_redim, sources[0])
        self.assertIn(curve, sources[0])
        self.assertNotIn(ndoverlay, sources[0])

        self.assertIn(ndoverlay[0], sources[1])
        self.assertIn(ndoverlay, sources[1])
        self.assertNotIn(curve_redim, sources[1])
        self.assertNotIn(curve, sources[1])

        self.assertIn(ndoverlay[1], sources[2])
        self.assertIn(ndoverlay, sources[2])
        self.assertNotIn(curve_redim, sources[2])
        self.assertNotIn(curve, sources[2])

    def test_dynamic_compute_overlayable_zorders_three_deep_dynamic_layers_reduced(self):
        area = DynamicMap(lambda: Area(range(10)), kdims=[])
        curve = DynamicMap(lambda: Curve(range(10)), kdims=[])
        curve2 = DynamicMap(lambda: Curve(range(10)), kdims=[])
        area_redim = area.redim(x='x2')
        curve_redim = curve.redim(x='x2')
        curve2_redim = curve2.redim(x='x3')
        combined = (area_redim*curve_redim).map(lambda x: x.get(0), Overlay)
        combined1 = combined*curve2_redim
        combined1[()]
        sources = compute_overlayable_zorders(combined1)

        self.assertIn(curve_redim, sources[0])
        self.assertIn(curve, sources[0])
        self.assertIn(area_redim, sources[0])
        self.assertIn(area, sources[0])
        self.assertNotIn(curve2_redim, sources[0])
        self.assertNotIn(curve2, sources[0])

        self.assertIn(curve2_redim, sources[1])
        self.assertIn(curve2, sources[1])
        self.assertNotIn(area_redim, sources[1])
        self.assertNotIn(area, sources[1])
        self.assertNotIn(curve_redim, sources[1])
        self.assertNotIn(curve, sources[1])


    def test_dynamic_compute_overlayable_zorders_three_deep_dynamic_layers_reduced_layers_by_one(self):
        area = DynamicMap(lambda: Area(range(10)), kdims=[])
        area2 = DynamicMap(lambda: Area(range(10)), kdims=[])
        curve = DynamicMap(lambda: Curve(range(10)), kdims=[])
        curve2 = DynamicMap(lambda: Curve(range(10)), kdims=[])
        area_redim = area.redim(x='x2')
        curve_redim = curve.redim(x='x2')
        curve2_redim = curve2.redim(x='x3')
        combined = (area_redim*curve_redim*area2).map(lambda x: x.clone(x.items()[:2]), Overlay)
        combined1 = combined*curve2_redim
        combined1[()]
        sources = compute_overlayable_zorders(combined1)

        self.assertNotIn(curve_redim, sources[0])
        self.assertNotIn(curve, sources[0])
        self.assertNotIn(curve2_redim, sources[0])
        self.assertNotIn(curve2, sources[0])
        self.assertNotIn(area, sources[0])
        self.assertNotIn(area_redim, sources[0])
        self.assertNotIn(area2, sources[0])

        self.assertNotIn(area_redim, sources[1])
        self.assertNotIn(area, sources[1])
        self.assertNotIn(curve2_redim, sources[1])
        self.assertNotIn(curve2, sources[1])
        self.assertNotIn(area2, sources[0])

        self.assertIn(curve2_redim, sources[2])
        self.assertIn(curve2, sources[2])
        self.assertNotIn(area_redim, sources[2])
        self.assertNotIn(area, sources[2])
        self.assertNotIn(area2, sources[0])
        self.assertNotIn(curve_redim, sources[2])
        self.assertNotIn(curve, sources[2])


class TestInitializeDynamic(ComparisonTestCase):

    def test_dynamicmap_default_initializes(self):
        dims = [Dimension('N', default=5, range=(0, 10))]
        dmap = DynamicMap(lambda N: Curve([1, N, 5]), kdims=dims)
        initialize_dynamic(dmap)
        self.assertEqual(dmap.keys(), [5])

    def test_dynamicmap_numeric_values_initializes(self):
        dims = [Dimension('N', values=[10, 5, 0])]
        dmap = DynamicMap(lambda N: Curve([1, N, 5]), kdims=dims)
        initialize_dynamic(dmap)
        self.assertEqual(dmap.keys(), [0])



class TestSplitDynamicMapOverlay(ComparisonTestCase):
    """
    Tests the split_dmap_overlay utility
    """

    def setUp(self):
        self.dmap_element = DynamicMap(lambda: Image([]))
        self.dmap_overlay = DynamicMap(lambda: Overlay([Curve([]), Points([])]))
        self.dmap_ndoverlay = DynamicMap(lambda: NdOverlay({0: Curve([]), 1: Curve([])}))
        self.element = Scatter([])
        self.el1, self.el2 = Path([]), HLine(0)
        self.overlay = Overlay([self.el1, self.el2])
        self.ndoverlay = NdOverlay({0: VectorField([]), 1: VectorField([])})

    def test_dmap_ndoverlay(self):
        test = self.dmap_ndoverlay
        initialize_dynamic(test)
        layers = [self.dmap_ndoverlay, self.dmap_ndoverlay]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_overlay(self):
        test = self.dmap_overlay
        initialize_dynamic(test)
        layers = [self.dmap_overlay, self.dmap_overlay]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_element_mul_dmap_overlay(self):
        test = self.dmap_element * self.dmap_overlay
        initialize_dynamic(test)
        layers = [self.dmap_element, self.dmap_overlay, self.dmap_overlay]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_element_mul_dmap_ndoverlay(self):
        test = self.dmap_element * self.dmap_ndoverlay
        initialize_dynamic(test)
        layers = [self.dmap_element, self.dmap_ndoverlay]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_element_mul_element(self):
        test = self.dmap_element * self.element
        initialize_dynamic(test)
        layers = [self.dmap_element, self.element]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_element_mul_overlay(self):
        test = self.dmap_element * self.overlay
        initialize_dynamic(test)
        layers = [self.dmap_element, self.el1, self.el2]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_element_mul_ndoverlay(self):
        test = self.dmap_element * self.ndoverlay
        initialize_dynamic(test)
        layers = [self.dmap_element, self.ndoverlay]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_overlay_mul_dmap_ndoverlay(self):
        test = self.dmap_overlay * self.dmap_ndoverlay
        initialize_dynamic(test)
        layers = [self.dmap_overlay, self.dmap_overlay, self.dmap_ndoverlay]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_overlay_mul_element(self):
        test = self.dmap_overlay * self.element
        initialize_dynamic(test)
        layers = [self.dmap_overlay, self.dmap_overlay, self.element]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_overlay_mul_overlay(self):
        test = self.dmap_overlay * self.overlay
        initialize_dynamic(test)
        layers = [self.dmap_overlay, self.dmap_overlay, self.el1, self.el2]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_all_combinations(self):
        test = (self.dmap_overlay * self.element * self.dmap_ndoverlay *
                self.overlay * self.dmap_element * self.ndoverlay)
        initialize_dynamic(test)
        layers = [self.dmap_overlay, self.dmap_overlay, self.element,
                  self.dmap_ndoverlay, self.el1, self.el2, self.dmap_element,
                  self.ndoverlay]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_overlay_operation_mul_dmap_ndoverlay(self):
        mapped = operation(self.dmap_overlay)
        test = mapped * self.dmap_ndoverlay
        initialize_dynamic(test)
        layers = [mapped, mapped, self.dmap_ndoverlay]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_overlay_linked_operation_mul_dmap_ndoverlay(self):
        mapped = operation(self.dmap_overlay, link_inputs=True)
        test = mapped * self.dmap_ndoverlay
        initialize_dynamic(test)
        layers = [mapped, mapped, self.dmap_ndoverlay]
        self.assertEqual(split_dmap_overlay(test), layers)

    def test_dmap_overlay_linked_operation_mul_dmap_element_ndoverlay(self):
        mapped = self.dmap_overlay.map(lambda x: x.get(0), Overlay)
        test = mapped * self.element * self.dmap_ndoverlay
        initialize_dynamic(test)
        layers = [mapped, self.element, self.dmap_ndoverlay]
        self.assertEqual(split_dmap_overlay(test), layers)


class TestPlotColorUtils(ComparisonTestCase):

    def test_process_cmap_list_cycle(self):
        colors = process_cmap(['#ffffff', '#959595', '#000000'], 4)
        self.assertEqual(colors, ['#ffffff', '#959595', '#000000', '#ffffff'])

    def test_process_cmap_cycle(self):
        colors = process_cmap(Cycle(values=['#ffffff', '#959595', '#000000']), 4)
        self.assertEqual(colors, ['#ffffff', '#959595', '#000000', '#ffffff'])

    def test_process_cmap_invalid_str(self):
        with self.assertRaises(ValueError):
            process_cmap('NonexistentColorMap', 3)

    def test_process_cmap_invalid_type(self):
        with self.assertRaises(TypeError):
            process_cmap({'A', 'B', 'C'}, 3)


class TestMPLColormapUtils(ComparisonTestCase):

    def setUp(self):
        try:
            import matplotlib.cm # noqa
            import holoviews.plotting.mpl # noqa
        except:
            raise SkipTest("Matplotlib needed to test matplotlib colormap instances")

    def test_mpl_colormap_fire(self):
        colors = process_cmap('fire', 3, provider='matplotlib')
        self.assertEqual(colors, ['#000000', '#ed1400', '#ffffff'])

    def test_mpl_colormap_fire_r(self):
        colors = process_cmap('fire_r', 3, provider='matplotlib')
        self.assertEqual(colors, ['#ffffff', '#eb1300', '#000000'])

    def test_mpl_colormap_name_palette(self):
        colors = process_cmap('Greys', 3, provider='matplotlib')
        self.assertEqual(colors, ['#ffffff', '#959595', '#000000'])

    def test_mpl_colormap_instance(self):
        from matplotlib.cm import get_cmap
        cmap = get_cmap('Greys')
        colors = process_cmap(cmap, 3, provider='matplotlib')
        self.assertEqual(colors, ['#ffffff', '#959595', '#000000'])

    def test_mpl_colormap_categorical(self):
        colors = mplcmap_to_palette('Category20', 3)
        self.assertEqual(colors, ['#1f77b4', '#c5b0d5', '#9edae5'])

    def test_mpl_colormap_categorical_reverse(self):
        colors = mplcmap_to_palette('Category20_r', 3)
        self.assertEqual(colors, ['#1f77b4', '#8c564b', '#9edae5'][::-1])

    def test_mpl_colormap_sequential(self):
        colors = mplcmap_to_palette('YlGn', 3)
        self.assertEqual(colors, ['#ffffe5', '#77c578', '#004529'])

    def test_mpl_colormap_sequential_reverse(self):
        colors = mplcmap_to_palette('YlGn_r', 3)
        self.assertEqual(colors, ['#ffffe5', '#78c679', '#004529'][::-1])

    def test_mpl_colormap_diverging(self):
        colors = mplcmap_to_palette('RdBu', 3)
        self.assertEqual(colors, ['#67001f', '#f6f6f6', '#053061'])

    def test_mpl_colormap_diverging_reverse(self):
        colors = mplcmap_to_palette('RdBu_r', 3)
        self.assertEqual(colors, ['#67001f', '#f7f6f6', '#053061'][::-1])

    def test_mpl_colormap_perceptually_uniform(self):
        colors = mplcmap_to_palette('viridis', 4)
        self.assertEqual(colors, ['#440154', '#30678d', '#35b778', '#fde724'])

    def test_mpl_colormap_perceptually_uniform_reverse(self):
        colors = mplcmap_to_palette('viridis_r', 4)
        self.assertEqual(colors, ['#440154', '#30678d', '#35b778', '#fde724'][::-1])


class TestBokehPaletteUtils(ComparisonTestCase):

    def setUp(self):
        try:
            import bokeh.palettes # noqa
            import holoviews.plotting.bokeh # noqa
        except:
            raise SkipTest('Bokeh required to test bokeh palette utilities')

    def test_bokeh_palette_categorical_palettes_not_interpolated(self):
        # Ensure categorical palettes are not expanded
        categorical = ('accent', 'category20', 'dark2', 'colorblind', 'pastel1',
                       'pastel2', 'set1', 'set2', 'set3', 'paired')
        for cat in categorical:
            self.assertTrue(len(set(bokeh_palette_to_palette(cat))) <= 20)

    @py2_skip
    def test_bokeh_colormap_fire(self):
        colors = process_cmap('fire', 3, provider='bokeh')
        self.assertEqual(colors, ['#000000', '#eb1300', '#ffffff'])

    @py2_skip
    def test_bokeh_colormap_fire_r(self):
        colors = process_cmap('fire_r', 3, provider='bokeh')
        self.assertEqual(colors, ['#ffffff', '#ed1400', '#000000'])

    def test_bokeh_palette_categorical(self):
        colors = bokeh_palette_to_palette('Category20', 3)
        self.assertEqual(colors, ['#1f77b4', '#c5b0d5', '#9edae5'])

    def test_bokeh_palette_categorical_reverse(self):
        colors = bokeh_palette_to_palette('Category20_r', 3)
        self.assertEqual(colors, ['#1f77b4', '#8c564b', '#9edae5'][::-1])

    def test_bokeh_palette_sequential(self):
        colors = bokeh_palette_to_palette('YlGn', 3)
        self.assertEqual(colors, ['#ffffe5', '#78c679', '#004529'])

    def test_bokeh_palette_sequential_reverse(self):
        colors = bokeh_palette_to_palette('YlGn_r', 3)
        self.assertEqual(colors, ['#ffffe5', '#78c679', '#004529'][::-1])

    def test_bokeh_palette_diverging(self):
        colors = bokeh_palette_to_palette('RdBu', 3)
        self.assertEqual(colors, ['#67001f', '#f7f7f7', '#053061'])

    def test_bokeh_palette_diverging_reverse(self):
        colors = bokeh_palette_to_palette('RdBu_r', 3)
        self.assertEqual(colors, ['#67001f', '#f7f7f7', '#053061'][::-1])

    def test_bokeh_palette_uniform_interpolated(self):
        colors = bokeh_palette_to_palette('Viridis', 4)
        self.assertEqual(colors, ['#440154', '#30678D', '#35B778', '#FDE724'])

    def test_bokeh_palette_perceptually_uniform(self):
        colors = bokeh_palette_to_palette('viridis', 4)
        self.assertEqual(colors, ['#440154', '#30678D', '#35B778', '#FDE724'])

    def test_bokeh_palette_perceptually_uniform_reverse(self):
        colors = bokeh_palette_to_palette('viridis_r', 4)
        self.assertEqual(colors, ['#440154', '#30678D', '#35B778', '#FDE724'][::-1])

    def test_color_intervals(self):
        levels = [0, 38, 73, 95, 110, 130, 156]  
        colors = ['#5ebaff', '#00faf4', '#ffffcc', '#ffe775', '#ffc140', '#ff8f20']
        cmap, lims = color_intervals(colors, levels, N=10)
        self.assertEqual(cmap, ['#5ebaff', '#5ebaff', '#00faf4',
                                '#00faf4', '#ffffcc', '#ffe775',
                                '#ffc140', '#ff8f20', '#ff8f20'])

    def test_color_intervals_clipped(self):
        levels = [0, 38, 73, 95, 110, 130, 156, 999]  
        colors = ['#5ebaff', '#00faf4', '#ffffcc', '#ffe775', '#ffc140', '#ff8f20', '#ff6060']
        cmap, lims = color_intervals(colors, levels, clip=(10, 90), N=100)
        self.assertEqual(cmap, ['#5ebaff', '#5ebaff', '#5ebaff', '#00faf4', '#00faf4',
                                '#00faf4', '#00faf4', '#ffffcc'])
        self.assertEqual(lims, (10, 90))


class TestPlotUtils(ComparisonTestCase):

    def test_get_min_distance_float32_type(self):
        xs, ys = (np.arange(0, 2., .2, dtype='float32'),
                  np.arange(0, 2., .2, dtype='float32'))
        X, Y = np.meshgrid(xs, ys)
        dist = get_min_distance(Points((X.flatten(), Y.flatten())))
        self.assertEqual(float(round(dist, 5)), 0.2)

    def test_get_min_distance_int32_type(self):
        xs, ys = (np.arange(0, 10, dtype='int32'),
                  np.arange(0, 10, dtype='int32'))
        X, Y = np.meshgrid(xs, ys)
        dist = get_min_distance(Points((X.flatten(), Y.flatten())))
        self.assertEqual(dist, 1.0)

    def test_get_min_distance_float32_type_no_scipy(self):
        xs, ys = (np.arange(0, 2., .2, dtype='float32'),
                  np.arange(0, 2., .2, dtype='float32'))
        X, Y = np.meshgrid(xs, ys)
        dist = _get_min_distance_numpy(Points((X.flatten(), Y.flatten())))
        self.assertEqual(dist, np.float32(0.2))

    def test_get_min_distance_int32_type_no_scipy(self):
        xs, ys = (np.arange(0, 10, dtype='int32'),
                  np.arange(0, 10, dtype='int32'))
        X, Y = np.meshgrid(xs, ys)
        dist = _get_min_distance_numpy(Points((X.flatten(), Y.flatten())))
        self.assertEqual(dist, 1.0)


class TestRangeUtilities(ComparisonTestCase):

    def test_get_axis_padding_scalar(self):
        padding = get_axis_padding(0.1)
        self.assertEqual(padding, (0.1, 0.1, 0.1))

    def test_get_axis_padding_tuple(self):
        padding = get_axis_padding((0.1, 0.2))
        self.assertEqual(padding, (0.1, 0.2, 0))

    def test_get_axis_padding_tuple_3d(self):
        padding = get_axis_padding((0.1, 0.2, 0.3))
        self.assertEqual(padding, (0.1, 0.2, 0.3))

    def test_get_range_from_element(self):
        dim = Dimension('y', soft_range=(0, 3), range=(0, 2))
        element = Scatter([1, 2, 3], vdims=dim)
        drange, srange, hrange = get_range(element, {}, dim)
        self.assertEqual(drange, (1, 3))
        self.assertEqual(srange, (0, 3))
        self.assertEqual(hrange, (0, 2))

    def test_get_range_from_ranges(self):
        dim = Dimension('y', soft_range=(0, 3), range=(0, 2))
        element = Scatter([1, 2, 3], vdims=dim)
        ranges = {'y': {'soft': (-1, 4), 'hard': (-1, 3), 'data': (-0.5, 2.5)}}
        drange, srange, hrange = get_range(element, ranges, dim)
        self.assertEqual(drange, (-0.5, 2.5))
        self.assertEqual(srange, (-1, 4))
        self.assertEqual(hrange, (-1, 3))



class TestBokehUtils(ComparisonTestCase):

    def setUp(self):
        if not bokeh_renderer:
            raise SkipTest("Bokeh required to test bokeh plot utils.")
        try:
            import pscript # noqa
        except:
            raise SkipTest("Flexx required to test transpiling formatter functions.")


    def test_py2js_funcformatter_single_arg(self):
        def test(x):  return '%s$' % x
        jsfunc = util.py2js_tickformatter(test)
        js_func = ('var x = tick;\nvar formatter;\nformatter = function () {\n'
                   '    return "" + x + "$";\n};\n\nreturn formatter();\n')
        self.assertEqual(jsfunc, js_func)


    def test_py2js_funcformatter_two_args(self):
        def test(x, pos):  return '%s$' % x
        jsfunc = util.py2js_tickformatter(test)
        js_func = ('var x = tick;\nvar formatter;\nformatter = function () {\n'
                   '    return "" + x + "$";\n};\n\nreturn formatter();\n')
        self.assertEqual(jsfunc, js_func)


    def test_py2js_funcformatter_arg_and_kwarg(self):
        def test(x, pos=None):  return '%s$' % x
        jsfunc = util.py2js_tickformatter(test)
        js_func = ('var x = tick;\nvar formatter;\nformatter = function () {\n'
                   '    pos = (pos === undefined) ? null: pos;\n    return "" '
                   '+ x + "$";\n};\n\nreturn formatter();\n')
        self.assertEqual(jsfunc, js_func)
