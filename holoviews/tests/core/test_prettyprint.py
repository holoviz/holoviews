"""
Test cases for the pretty printing system.
"""


from holoviews.element.comparison import ComparisonTestCase
from holoviews import Store, Element, Curve, Overlay, Layout
from holoviews.core.pprint import PrettyPrinter

from .test_dimensioned import CustomBackendTestCase, ExampleElement


class PrettyPrintTest(ComparisonTestCase):

    def setUp(self):
        self.element1 = Element(None, group='Value', label='Label')
        self.element2 = Element(None, group='Value', label='')

    def test_element_repr1(self):
        r = PrettyPrinter.pprint(self.element1)
        self.assertEqual(r, ':Element')

    def test_overlay_repr1(self):
        expected = ':Overlay\n   .Value.Label :Element\n   .Value.I     :Element'
        o = self.element1 * self.element2
        r = PrettyPrinter.pprint(o)
        self.assertEqual(r, expected)

    def test_curve_pprint_repr(self):
        # Ensure it isn't a bytes object with the 'b' prefix
        expected = "':Curve   [x]   (y)'"
        r = PrettyPrinter.pprint(Curve([1,2,3]))
        self.assertEqual(repr(r), expected)


class PrettyPrintOptionsTest(CustomBackendTestCase):

    def setUp(self):
        super().setUp()
        self.current_backend = Store.current_backend
        self.pprinter = PrettyPrinter(show_options=True)
        self.register_custom(ExampleElement, 'backend_1', ['plot_custom1'], ['style_custom1'])
        self.register_custom(Overlay, 'backend_1', ['plot_custom1'])
        self.register_custom(Layout, 'backend_1', ['plot_custom1'])
        self.register_custom(ExampleElement, 'backend_2', ['plot_custom2'])
        Store.current_backend = 'backend_1'

    def test_element_options(self):
        element = ExampleElement(None).opts(style_opt1='A', backend='backend_1')
        r = self.pprinter.pprint(element)
        self.assertEqual(r, ":ExampleElement\n | Options(style_opt1='A')")

    def test_element_options_wrapping(self):
        element = ExampleElement(None).opts(plot_opt1='A'*40, style_opt1='B'*40, backend='backend_1')
        r = self.pprinter.pprint(element)
        self.assertEqual(r, ":ExampleElement\n | Options(plot_opt1='AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',\n |         style_opt1='BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')")

    def test_overlay_options(self):
        overlay = (ExampleElement(None) * ExampleElement(None)).opts(plot_opt1='A')
        r = self.pprinter.pprint(overlay)
        self.assertEqual(r, ":Overlay\n | Options(plot_opt1='A')\n   .Element.I  :ExampleElement\n   .Element.II :ExampleElement")

    def test_overlay_nested_options(self):
        overlay = (ExampleElement(None) * ExampleElement(None)).opts('ExampleElement', plot_opt1='A', style_opt1='A')
        r = self.pprinter.pprint(overlay)
        self.assertEqual(r, ":Overlay\n   .Element.I  :ExampleElement\n    | Options(plot_opt1='A', style_opt1='A')\n   .Element.II :ExampleElement\n    | Options(plot_opt1='A', style_opt1='A')")

    def test_layout_options(self):
        overlay = (ExampleElement(None) + ExampleElement(None)).opts(plot_opt1='A')
        r = self.pprinter.pprint(overlay)
        self.assertEqual(r, ":Layout\n | Options(plot_opt1='A')\n   .Element.I  :ExampleElement\n   .Element.II :ExampleElement")

    def test_layout_nested_options(self):
        overlay = (ExampleElement(None) + ExampleElement(None)).opts('ExampleElement', plot_opt1='A', style_opt1='A')
        r = self.pprinter.pprint(overlay)
        self.assertEqual(r, ":Layout\n   .Element.I  :ExampleElement\n    | Options(plot_opt1='A', style_opt1='A')\n   .Element.II :ExampleElement\n    | Options(plot_opt1='A', style_opt1='A')")
