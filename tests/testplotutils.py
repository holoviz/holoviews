from unittest import SkipTest

from holoviews.core.spaces import DynamicMap, get_sources
from holoviews.core.options import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element import Curve, Area


try:
    from holoviews.plotting.bokeh import util
    bokeh_renderer = Store.renderers['bokeh']
except:
    bokeh_renderer = None


class TestPlotUtils(ComparisonTestCase):

    def test_dynamic_get_sources_two_mixed_layers(self):
        area = Area(range(10))
        dmap = DynamicMap(lambda: Curve(range(10)), kdims=[])
        combined = area*dmap
        combined[()]
        sources = get_sources(combined)
        self.assertEqual(sources[0], [area])
        self.assertEqual(sources[1], [dmap])

    def test_dynamic_get_sources_two_dynamic_layers(self):
        area = DynamicMap(lambda: Area(range(10)), kdims=[])
        dmap = DynamicMap(lambda: Curve(range(10)), kdims=[])
        combined = area*dmap
        combined[()]
        sources = get_sources(combined)
        self.assertEqual(sources[0], [area])
        self.assertEqual(sources[1], [dmap])

    def test_dynamic_get_sources_two_deep_dynamic_layers(self):
        area = DynamicMap(lambda: Area(range(10)), kdims=[])
        curve = DynamicMap(lambda: Curve(range(10)), kdims=[])
        area_redim = area.redim(x='x2')
        curve_redim = curve.redim(x='x2')
        combined = area_redim*curve_redim
        combined[()]
        sources = get_sources(combined)
        self.assertIn(area_redim, sources[0])
        self.assertIn(area, sources[0])
        self.assertNotIn(curve_redim, sources[0])
        self.assertNotIn(curve, sources[0])
        self.assertIn(curve_redim, sources[1])
        self.assertIn(curve, sources[1])
        self.assertNotIn(area_redim, sources[1])
        self.assertNotIn(area, sources[1])

    def test_dynamic_get_sources_three_deep_dynamic_layers(self):
        area = DynamicMap(lambda: Area(range(10)), kdims=[])
        curve = DynamicMap(lambda: Curve(range(10)), kdims=[])
        curve2 = DynamicMap(lambda: Curve(range(10)), kdims=[])
        area_redim = area.redim(x='x2')
        curve_redim = curve.redim(x='x2')
        curve2_redim = curve2.redim(x='x3')
        combined = area_redim*curve_redim
        combined1 = (combined*curve2_redim)
        combined1[()]
        sources = get_sources(combined1)
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

    def test_dynamic_get_sources_three_deep_dynamic_layers_cloned(self):
        area = DynamicMap(lambda: Area(range(10)), kdims=[])
        curve = DynamicMap(lambda: Curve(range(10)), kdims=[])
        curve2 = DynamicMap(lambda: Curve(range(10)), kdims=[])
        area_redim = area.redim(x='x2')
        curve_redim = curve.redim(x='x2')
        curve2_redim = curve2.redim(x='x3')
        combined = area_redim*curve_redim
        combined1 = (combined*curve2_redim).redim(y='y2')
        combined1[()]
        sources = get_sources(combined1)
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

    def test_dynamic_get_sources_mixed_dynamic_and_non_dynamic_overlays(self):
        area1 = Area(range(10))
        area2 = Area(range(10))
        overlay = area1 * area2
        curve = DynamicMap(lambda: Curve(range(10)), kdims=[])
        curve_redim = curve.redim(x='x2')
        combined = overlay*curve_redim
        combined[()]
        sources = get_sources(combined)

        self.assertIn(area1, sources[0])
        self.assertIn(overlay, sources[0])
        self.assertNotIn(curve_redim, sources[0])
        self.assertNotIn(curve, sources[0])

        self.assertIn(area2, sources[1])
        self.assertIn(overlay, sources[1])
        self.assertNotIn(curve_redim, sources[1])
        self.assertNotIn(curve, sources[1])

        self.assertIn(curve_redim, sources[2])
        self.assertIn(curve, sources[2])
        self.assertNotIn(overlay, sources[2])



class TestBokehUtils(ComparisonTestCase):

    def setUp(self):
        if not bokeh_renderer:
            raise SkipTest("Bokeh required to test bokeh plot utils.")


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
