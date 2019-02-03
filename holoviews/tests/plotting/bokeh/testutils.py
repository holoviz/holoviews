from unittest import SkipTest
from holoviews.core import Store
from holoviews.element.comparison import ComparisonTestCase

try:
    from holoviews.plotting.bokeh.util import filter_batched_data, glyph_order
    from holoviews.plotting.bokeh.styles import expand_batched_style
    bokeh_renderer = Store.renderers['bokeh']
except:
    bokeh_renderer = None

class TestBokehUtilsInstantiation(ComparisonTestCase):

    def setUp(self):
        if not bokeh_renderer:
            raise SkipTest("Bokeh required to test plot instantiation")

    def test_expand_style_opts_simple(self):
        style = {'line_width': 3}
        opts = ['line_width']
        data, mapping = expand_batched_style(style, opts, {}, nvals=3)
        self.assertEqual(data['line_width'], [3, 3, 3])
        self.assertEqual(mapping, {'line_width': {'field': 'line_width'}})

    def test_expand_style_opts_multiple(self):
        style = {'line_color': 'red', 'line_width': 4}
        opts = ['line_color', 'line_width']
        data, mapping = expand_batched_style(style, opts, {}, nvals=3)
        self.assertEqual(data['line_color'], ['red', 'red', 'red'])
        self.assertEqual(data['line_width'], [4, 4, 4])
        self.assertEqual(mapping, {'line_color': {'field': 'line_color'},
                                   'line_width': {'field': 'line_width'}})

    def test_expand_style_opts_line_color_and_color(self):
        style = {'fill_color': 'red', 'color': 'blue'}
        opts = ['color', 'line_color', 'fill_color']
        data, mapping = expand_batched_style(style, opts, {}, nvals=3)
        self.assertEqual(data['line_color'], ['blue', 'blue', 'blue'])
        self.assertEqual(data['fill_color'], ['red', 'red', 'red'])
        self.assertEqual(mapping, {'line_color': {'field': 'line_color'},
                                   'fill_color': {'field': 'fill_color'}})

    def test_expand_style_opts_line_alpha_and_alpha(self):
        style = {'fill_alpha': 0.5, 'alpha': 0.2}
        opts = ['alpha', 'line_alpha', 'fill_alpha']
        data, mapping = expand_batched_style(style, opts, {}, nvals=3)
        self.assertEqual(data['line_alpha'], [0.2, 0.2, 0.2])
        self.assertEqual(data['fill_alpha'], [0.5, 0.5, 0.5])
        self.assertEqual(mapping, {'line_alpha': {'field': 'line_alpha'},
                                   'fill_alpha': {'field': 'fill_alpha'}})

    def test_expand_style_opts_color_predefined(self):
        style = {'fill_color': 'red'}
        opts = ['color', 'line_color', 'fill_color']
        data, mapping = expand_batched_style(style, opts, {'color': 'color'}, nvals=3)
        self.assertEqual(data['fill_color'], ['red', 'red', 'red'])
        self.assertEqual(mapping, {'fill_color': {'field': 'fill_color'}})

    def test_filter_batched_data(self):
        data = {'line_color': ['red', 'red', 'red']}
        mapping = {'line_color': 'line_color'}
        filter_batched_data(data, mapping)
        self.assertEqual(data, {})
        self.assertEqual(mapping, {'line_color': 'red'})

    def test_filter_batched_data_as_field(self):
        data = {'line_color': ['red', 'red', 'red']}
        mapping = {'line_color': {'field': 'line_color'}}
        filter_batched_data(data, mapping)
        self.assertEqual(data, {})
        self.assertEqual(mapping, {'line_color': 'red'})

    def test_filter_batched_data_heterogeneous(self):
        data = {'line_color': ['red', 'red', 'blue']}
        mapping = {'line_color': {'field': 'line_color'}}
        filter_batched_data(data, mapping)
        self.assertEqual(data, {'line_color': ['red', 'red', 'blue']})
        self.assertEqual(mapping, {'line_color': {'field': 'line_color'}})

    def test_glyph_order(self):
        order = glyph_order(['scatter_1', 'patch_1', 'rect_1'],
                            ['scatter', 'patch'])
        self.assertEqual(order, ['scatter_1', 'patch_1', 'rect_1'])
