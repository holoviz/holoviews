from unittest import SkipTest

import numpy as np

from holoviews import renderer
from holoviews.core import Dimension, NdMapping, DynamicMap, HoloMap
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase

try:
    from holoviews.plotting.bokeh.widgets import BokehServerWidgets
    from bokeh.models.widgets import Select, Slider, AutocompleteInput, TextInput, Div
    bokeh_renderer = renderer('bokeh')
except:
    BokehServerWidgets = None


class TestBokehServerWidgets(ComparisonTestCase):

    def setUp(self):
        if not BokehServerWidgets:
            raise SkipTest("Bokeh required to test BokehServerWidgets")

    def test_bokeh_widgets_server_mode(self):
        dmap = DynamicMap(lambda X: Curve([]), kdims=['X']).redim.range(X=(0, 5))
        widgets = bokeh_renderer.instance(mode='server').get_widget(dmap, None)
        div, widget = widgets.widgets['X']
        self.assertIsInstance(widget, Slider)
        self.assertEqual(widget.value, 0)
        self.assertEqual(widget.start, 0)
        self.assertEqual(widget.end, 5)
        self.assertEqual(widget.step, 1)
        self.assertEqual(widgets.state.sizing_mode, 'fixed')

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
        self.assertEqual(mapping, [(i, (v, dim.pprint_value(v))) for i, v in enumerate(values)])

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
        self.assertEqual(mapping, [(i, (v, dim.pprint_value(v))) for i, v in enumerate(values)])

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
        self.assertEqual(mapping, [(i, (v, dim.pprint_value(v))) for i, v in enumerate(values)])

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
        self.assertEqual(mapping, list(enumerate(zip(keys, keys))))

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
        self.assertEqual(mapping, [(i, (k, dim.pprint_value(k))) for i, k in enumerate(ndmap.keys())])



class TestSelectionWidget(ComparisonTestCase):

    def setUp(self):
        if not BokehServerWidgets:
            raise SkipTest("Bokeh required to test BokehServerWidgets")

    def test_holomap_slider(self):
        hmap = HoloMap({i: Curve([1, 2, 3]) for i in range(10)}, 'X')
        widgets = bokeh_renderer.get_widget(hmap, 'widgets')
        widgets, dimensions, init_dim_vals =  widgets.get_widgets()
        self.assertEqual(len(widgets), 1)
        slider = widgets[0]
        self.assertEqual(slider['type'], 'slider')
        self.assertEqual(slider['dim'], 'X')
        self.assertEqual(slider['dim_idx'], 0)
        self.assertEqual(slider['vals'], repr([repr(float(v)) for v in range(10)]))
        self.assertEqual(slider['labels'], repr([str(v) for v in range(10)]))
        self.assertEqual(slider['step'], 1)
        self.assertEqual(slider['default'], 0)
        self.assertIs(slider['next_dim'], None)
        self.assertEqual(dimensions, ['X'])
        self.assertEqual(init_dim_vals, repr(['0.0']))

    def test_holomap_slider_unsorted(self):
        data = {(i, j): Curve([1, 2, 3]) for i in range(3) for j in range(3)}
        del data[2, 2]
        hmap = HoloMap(data, ['X', 'Y'])
        widgets = bokeh_renderer.get_widget(hmap, 'widgets')
        widgets, dimensions, init_dim_vals =  widgets.get_widgets()
        self.assertEqual(len(widgets), 2)
        slider = widgets[1]
        self.assertEqual(slider['vals'], repr([repr(float(v)) for v in range(3)]))
        self.assertEqual(slider['labels'], repr([str(v) for v in range(3)]))

    def test_holomap_dropdown(self):
        hmap = HoloMap({chr(65+i): Curve([1, 2, 3]) for i in range(10)}, 'X')
        widgets = bokeh_renderer.get_widget(hmap, 'widgets')
        widgets, dimensions, init_dim_vals =  widgets.get_widgets()
        self.assertEqual(len(widgets), 1)
        dropdown = widgets[0]
        self.assertEqual(dropdown['type'], 'dropdown')
        self.assertEqual(dropdown['dim'], 'X')
        self.assertEqual(dropdown['dim_idx'], 0)
        self.assertEqual(dropdown['vals'], repr([chr(65+v) for v in range(10)]))
        self.assertEqual(dropdown['labels'], repr([chr(65+v) for v in range(10)]))
        self.assertEqual(dropdown['step'], 1)
        self.assertEqual(dropdown['default'], 0)
        self.assertIs(dropdown['next_dim'], None)
        self.assertEqual(dimensions, ['X'])
        self.assertEqual(init_dim_vals, repr(['A']))

    def test_holomap_slider_and_dropdown(self):
        hmap = HoloMap({(i, chr(65+i)): Curve([1, 2, 3]) for i in range(10)}, ['X', 'Y'])
        widgets = bokeh_renderer.get_widget(hmap, 'widgets')
        widgets, dimensions, init_dim_vals =  widgets.get_widgets()
        self.assertEqual(len(widgets), 2)

        slider = widgets[0]
        self.assertEqual(slider['type'], 'slider')
        self.assertEqual(slider['dim'], 'X')
        self.assertEqual(slider['dim_idx'], 0)
        self.assertEqual(slider['vals'], repr([repr(float(v)) for v in range(10)]))
        self.assertEqual(slider['labels'], repr([str(v) for v in range(10)]))
        self.assertEqual(slider['step'], 1)
        self.assertEqual(slider['default'], 0)
        self.assertEqual(slider['next_dim'], Dimension('Y'))
        self.assertEqual(eval(slider['next_vals']),
                         {str(float(i)): [chr(65+i)] for i in range(10)})
        
        dropdown = widgets[1]
        self.assertEqual(dropdown['type'], 'dropdown')
        self.assertEqual(dropdown['dim'], 'Y')
        self.assertEqual(dropdown['dim_idx'], 1)
        self.assertEqual(dropdown['vals'], repr([chr(65+v) for v in range(10)]))
        self.assertEqual(dropdown['labels'], repr([chr(65+v) for v in range(10)]))
        self.assertEqual(dropdown['step'], 1)
        self.assertEqual(dropdown['default'], 0)
        self.assertIs(dropdown['next_dim'], None)

        self.assertEqual(dimensions, ['X', 'Y'])
        self.assertEqual(init_dim_vals, repr(['0.0', 'A']))

    def test_dynamicmap_int_range_slider(self):
        hmap = DynamicMap(lambda x: Curve([1, 2, 3]), kdims=[Dimension('X', range=(0, 5))])
        widgets = bokeh_renderer.get_widget(hmap, 'widgets')
        widgets, dimensions, init_dim_vals =  widgets.get_widgets()
        self.assertEqual(len(widgets), 1)
        slider = widgets[0]
        self.assertEqual(slider['type'], 'slider')
        self.assertEqual(slider['dim'], 'X')
        self.assertEqual(slider['dim_idx'], 0)
        self.assertEqual(slider['vals'], "['0.0', '5.0']")
        self.assertEqual(slider['labels'], [])
        self.assertEqual(slider['step'], 1)
        self.assertEqual(slider['default'], 0)
        self.assertIs(slider['next_dim'], None)
        self.assertEqual(dimensions, ['X'])
        self.assertEqual(init_dim_vals, repr([0.0]))

    def test_dynamicmap_float_range_slider(self):
        hmap = DynamicMap(lambda x: Curve([1, 2, 3]), kdims=[Dimension('X', range=(0., 5.))])
        widgets = bokeh_renderer.get_widget(hmap, 'widgets')
        widgets, dimensions, init_dim_vals =  widgets.get_widgets()
        self.assertEqual(len(widgets), 1)
        slider = widgets[0]
        self.assertEqual(slider['type'], 'slider')
        self.assertEqual(slider['dim'], 'X')
        self.assertEqual(slider['dim_idx'], 0)
        self.assertEqual(slider['vals'], "['0.0', '5.0']")
        self.assertEqual(slider['labels'], [])
        self.assertEqual(slider['step'], 0.01)
        self.assertEqual(slider['default'], 0.0)
        self.assertIs(slider['next_dim'], None)
        self.assertEqual(dimensions, ['X'])
        self.assertEqual(init_dim_vals, repr([0.0]))

    def test_dynamicmap_float_range_slider_with_step(self):
        dimension = Dimension('X', range=(0., 5.), step=0.05)
        hmap = DynamicMap(lambda x: Curve([1, 2, 3]), kdims=[dimension])
        widgets = bokeh_renderer.get_widget(hmap, 'widgets')
        widgets, dimensions, init_dim_vals =  widgets.get_widgets()
        self.assertEqual(widgets[0]['step'], 0.05)

    def test_dynamicmap_int_range_slider_with_step(self):
        dimension = Dimension('X', range=(0, 10), step=2)
        hmap = DynamicMap(lambda x: Curve([1, 2, 3]), kdims=[dimension])
        widgets = bokeh_renderer.get_widget(hmap, 'widgets')
        widgets, dimensions, init_dim_vals =  widgets.get_widgets()
        self.assertEqual(widgets[0]['step'], 2)

    def test_dynamicmap_values_slider(self):
        dimension = Dimension('X', values=list(range(10)))
        hmap = DynamicMap(lambda x: Curve([1, 2, 3]), kdims=[dimension])
        widgets = bokeh_renderer.get_widget(hmap, 'widgets')
        widgets, dimensions, init_dim_vals =  widgets.get_widgets()
        self.assertEqual(len(widgets), 1)
        slider = widgets[0]
        self.assertEqual(slider['type'], 'slider')
        self.assertEqual(slider['dim'], 'X')
        self.assertEqual(slider['dim_idx'], 0)
        self.assertEqual(slider['vals'], {i: i for i in range(10)})
        self.assertEqual(slider['labels'], repr([str(i) for i in range(10)]))
        self.assertEqual(slider['step'], 1)
        self.assertEqual(slider['default'], 0)
        self.assertIs(slider['next_dim'], None)
        self.assertEqual(dimensions, ['X'])
        self.assertEqual(init_dim_vals, repr([0.0]))

    def test_dynamicmap_values_dropdown(self):
        values = [chr(65+i) for i in range(10)]
        dimension = Dimension('X', values=values)
        hmap = DynamicMap(lambda x: Curve([1, 2, 3]), kdims=[dimension])
        widgets = bokeh_renderer.get_widget(hmap, 'widgets')
        widgets, dimensions, init_dim_vals =  widgets.get_widgets()
        self.assertEqual(len(widgets), 1)
        dropdown = widgets[0]
        self.assertEqual(dropdown['type'], 'dropdown')
        self.assertEqual(dropdown['dim'], 'X')
        self.assertEqual(dropdown['dim_idx'], 0)
        self.assertEqual(dropdown['vals'], list(range(10)))
        self.assertEqual(dropdown['labels'], repr(values))
        self.assertEqual(dropdown['step'], 1)
        self.assertEqual(dropdown['default'], 0)
        self.assertIs(dropdown['next_dim'], None)
        self.assertEqual(dimensions, ['X'])
        self.assertEqual(init_dim_vals, repr([0.0]))

    def test_dynamicmap_values_default(self):
        values = [chr(65+i) for i in range(10)]
        dimension = Dimension('X', values=values, default='C')
        hmap = DynamicMap(lambda x: Curve([1, 2, 3]), kdims=[dimension])
        widgets = bokeh_renderer.get_widget(hmap, 'widgets')
        widgets, dimensions, init_dim_vals =  widgets.get_widgets()
        self.assertEqual(widgets[0]['default'], '2')
        self.assertEqual(init_dim_vals, repr(['2']))

    def test_dynamicmap_range_default(self):
        dimension = Dimension('X', range=(0., 5.), default=0.05)
        hmap = DynamicMap(lambda x: Curve([1, 2, 3]), kdims=[dimension])
        widgets = bokeh_renderer.get_widget(hmap, 'widgets')
        widgets, dimensions, init_dim_vals =  widgets.get_widgets()
        self.assertEqual(widgets[0]['default'], 0.05)
        self.assertEqual(init_dim_vals, '[0.050000000]')

    def test_holomap_slider_default(self):
        dim = Dimension('X', default=3)
        hmap = HoloMap({i: Curve([1, 2, 3]) for i in range(1, 9)}, dim)
        widgets = bokeh_renderer.get_widget(hmap, 'widgets')
        widgets, dimensions, init_dim_vals =  widgets.get_widgets()
        self.assertEqual(widgets[0]['default'], '2')
        self.assertEqual(init_dim_vals, "['3.0']")

    def test_holomap_slider_bad_default(self):
        dim = Dimension('X', default=42)
        hmap = HoloMap({i: Curve([1, 2, 3]) for i in range(1, 9)}, dim)
        with self.assertRaises(ValueError):
            bokeh_renderer.get_widget(hmap, 'widgets').get_widgets()

    def test_holomap_dropdown_default(self):
        dim = Dimension('X', default='C')
        hmap = HoloMap({chr(65+i): Curve([1, 2, 3]) for i in range(10)}, dim)
        widgets = bokeh_renderer.get_widget(hmap, 'widgets')
        widgets, dimensions, init_dim_vals =  widgets.get_widgets()
        self.assertEqual(widgets[0]['default'], '2')
        self.assertEqual(init_dim_vals, "['C']")

    def test_holomap_dropdown_bad_default(self):
        dim = Dimension('X', default='Z')
        hmap = HoloMap({chr(65+i): Curve([1, 2, 3]) for i in range(10)}, dim)
        with self.assertRaises(ValueError):
            bokeh_renderer.get_widget(hmap, 'widgets').get_widgets()
