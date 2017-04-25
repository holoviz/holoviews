from unittest import SkipTest

import numpy as np

from holoviews.core import Dimension, NdMapping
from holoviews.element.comparison import ComparisonTestCase

try:
    from holoviews.plotting.bokeh.widgets import BokehServerWidgets
    from bokeh.models.widgets import Select, Slider, AutocompleteInput, TextInput, Div
except:
    BokehServerWidgets = None


class TestBokehServerWidgets(ComparisonTestCase):

    def setUp(self):
        if not BokehServerWidgets:
            raise SkipTest("Bokeh required to test BokehServerWidgets")

    def test_bokeh_server_dynamic_range_int(self):
        dim = Dimension('x', range=(3, 11))
        widget, label, mapping = BokehServerWidgets.create_widget(dim, editable=True)
        self.assertIsInstance(widget, Slider)
        self.assertEqual(widget.value, 3)
        self.assertEqual(widget.start, 3)
        self.assertEqual(widget.end, 11)
        self.assertEqual(widget.step, 1)
        self.assertIsInstance(label, TextInput)
        self.assertEqual(label.title, dim.pprint_label)
        self.assertEqual(label.value, '3')
        self.assertIs(mapping, None)

    def test_bokeh_server_dynamic_range_float(self):
        dim = Dimension('x', range=(3.1, 11.2))
        widget, label, mapping = BokehServerWidgets.create_widget(dim, editable=True)
        self.assertIsInstance(widget, Slider)
        self.assertEqual(widget.value, 3.1)
        self.assertEqual(widget.start, 3.1)
        self.assertEqual(widget.end, 11.2)
        self.assertEqual(widget.step, 0.01)
        self.assertIsInstance(label, TextInput)
        self.assertEqual(label.title, dim.pprint_label)
        self.assertEqual(label.value, '3.1')
        self.assertIs(mapping, None)

    def test_bokeh_server_dynamic_range_float_step(self):
        dim = Dimension('x', range=(3.1, 11.2), step=0.1)
        widget, label, mapping = BokehServerWidgets.create_widget(dim, editable=True)
        self.assertIsInstance(widget, Slider)
        self.assertEqual(widget.value, 3.1)
        self.assertEqual(widget.start, 3.1)
        self.assertEqual(widget.end, 11.2)
        self.assertEqual(widget.step, 0.1)
        self.assertIsInstance(label, TextInput)
        self.assertEqual(label.title, dim.pprint_label)
        self.assertEqual(label.value, '3.1')
        self.assertIs(mapping, None)

    def test_bokeh_server_dynamic_range_not_editable(self):
        dim = Dimension('x', range=(3.1, 11.2))
        widget, label, mapping = BokehServerWidgets.create_widget(dim, editable=False)
        self.assertIsInstance(widget, Slider)
        self.assertEqual(widget.value, 3.1)
        self.assertEqual(widget.start, 3.1)
        self.assertEqual(widget.end, 11.2)
        self.assertEqual(widget.step, 0.01)
        self.assertIsInstance(label, Div)
        self.assertEqual(label.text, '<b>%s</b>' % dim.pprint_value_string(3.1))
        self.assertIs(mapping, None)

    def test_bokeh_server_dynamic_values_int(self):
        values = list(range(3, 11))
        dim = Dimension('x', values=values)
        widget, label, mapping = BokehServerWidgets.create_widget(dim, editable=True)
        self.assertIsInstance(widget, Slider)
        self.assertEqual(widget.value, 0)
        self.assertEqual(widget.start, 0)
        self.assertEqual(widget.end, 7)
        self.assertEqual(widget.step, 1)
        self.assertIsInstance(label, AutocompleteInput)
        self.assertEqual(label.title, dim.pprint_label)
        self.assertEqual(label.value, '3')
        self.assertEqual(mapping, [(v, dim.pprint_value(v)) for v in values])

    def test_bokeh_server_dynamic_values_float_not_editable(self):
        values = list(np.linspace(3.1, 11.2, 7))
        dim = Dimension('x', values=values)
        widget, label, mapping = BokehServerWidgets.create_widget(dim, editable=False)
        self.assertIsInstance(widget, Slider)
        self.assertEqual(widget.value, 0)
        self.assertEqual(widget.start, 0)
        self.assertEqual(widget.end, 6)
        self.assertEqual(widget.step, 1)
        self.assertIsInstance(label, Div)
        self.assertEqual(label.text, '<b>%s</b>' % dim.pprint_value_string(3.1))
        self.assertEqual(mapping, [(v, dim.pprint_value(v)) for v in values])

    def test_bokeh_server_dynamic_values_float_editable(self):
        values = list(np.linspace(3.1, 11.2, 7))
        dim = Dimension('x', values=values)
        widget, label, mapping = BokehServerWidgets.create_widget(dim, editable=True)
        self.assertIsInstance(widget, Slider)
        self.assertEqual(widget.value, 0)
        self.assertEqual(widget.start, 0)
        self.assertEqual(widget.end, 6)
        self.assertEqual(widget.step, 1)
        self.assertIsInstance(label, AutocompleteInput)
        self.assertEqual(label.title, dim.pprint_label)
        self.assertEqual(label.value, '3.1')
        self.assertEqual(mapping, [(v, dim.pprint_value(v)) for v in values])

    def test_bokeh_server_dynamic_values_str_1(self):
        values = [chr(65+i) for i in range(10)]
        dim = Dimension('x', values=values)
        widget, label, mapping = BokehServerWidgets.create_widget(dim, editable=True)
        self.assertIsInstance(widget, Select)
        self.assertEqual(widget.value, 'A')
        self.assertEqual(widget.options, list(zip(values, values)))
        self.assertEqual(widget.title, dim.pprint_label)
        self.assertIs(mapping, None)
        self.assertIs(label, None)

    def test_bokeh_server_dynamic_values_str_2(self):
        keys = [chr(65+i) for i in range(10)]
        ndmap = NdMapping({i: None for i in keys}, kdims=['x'])
        dim = Dimension('x')
        widget, label, mapping = BokehServerWidgets.create_widget(dim, ndmap, editable=True)
        self.assertIsInstance(widget, Select)
        self.assertEqual(widget.value, 'A')
        self.assertEqual(widget.options, list(zip(keys, keys)))
        self.assertEqual(widget.title, dim.pprint_label)
        self.assertEqual(mapping, list(zip(keys, keys)))

    def test_bokeh_server_static_numeric_values(self):
        dim = Dimension('x')
        ndmap = NdMapping({i: None for i in range(3, 12)}, kdims=['x'])
        widget, label, mapping = BokehServerWidgets.create_widget(dim, ndmap, editable=True)
        self.assertIsInstance(widget, Slider)
        self.assertEqual(widget.value, 0)
        self.assertEqual(widget.start, 0)
        self.assertEqual(widget.end, 8)
        self.assertEqual(widget.step, 1)
        self.assertIsInstance(label, AutocompleteInput)
        self.assertEqual(label.title, dim.pprint_label)
        self.assertEqual(label.value, '3')
        self.assertEqual(mapping, [(k, dim.pprint_value(k)) for k in ndmap.keys()])
