import numpy as np

from holoviews.core.data import Dataset
from holoviews.element import Sankey


from .test_plot import TestBokehPlot, bokeh_renderer

class TestSankeyPlot(TestBokehPlot):

    def test_sankey_simple(self):
        sankey = Sankey([
            ('A', 'X', 5), ('A', 'Y', 7), ('A', 'Z', 6),
            ('B', 'X', 2), ('B', 'Y', 9), ('B', 'Z', 4)]
        )
        plot = bokeh_renderer.get_plot(sankey)
        scatter_source = plot.handles['scatter_1_source']
        quad_source = plot.handles['quad_1_source']
        text_source = plot.handles['text_1_source']
        patch_source = plot.handles['patches_1_source']

        scatter_index = np.array(['A', 'B', 'X', 'Y', 'Z'])
        self.assertEqual(scatter_source.data['index'], scatter_index)

        text_data = {'x': np.array([18.75, 18.75, 1003.75, 1003.75, 1003.75]),
                     'y': np.array([1649.49068662, 1899.49068658, 1587.02936107, 1767.33239136, 1968.54451237]),
                     'text': ['A - 18', 'B - 15', 'X - 7', 'Y - 16', 'Z - 10']}
        with open('a.txt', 'w') as f:
            f.write(str(text_source.data) + '\n')
        for k in text_data:
            self.assertEqual(text_source.data[k], text_data[k])

        quad_data = {
            'index': scatter_index,
            'x0': [0, 0, 985.0, 985.0, 985.0],
            'x1': [15, 15, 1000.0, 1000.0, 1000.0],
            'y0': [1524.036141243766, 1794.945232152857, 1538.2414821428135, 1655.8172397185713, 1898.8475427488745],
            'y1': [1774.945232, 2004.036141, 1635.81724, 1878.847543, 2038.241482]
        }
        for k in quad_data:
            self.assertEqual(quad_source.data[k], quad_data[k])

        self.assertEqual(patch_source.data['Value'], np.array([5, 7, 6, 2, 9, 4]))

        renderers = plot.state.renderers
        quad_renderer = plot.handles['quad_1_glyph_renderer']
        text_renderer = plot.handles['text_1_glyph_renderer']
        graph_renderer = plot.handles['glyph_renderer']
        self.assertTrue(renderers.index(graph_renderer)<renderers.index(quad_renderer))
        self.assertTrue(renderers.index(quad_renderer)<renderers.index(text_renderer))


    def test_sankey_label_index(self):
        sankey = Sankey(([
            (0, 2, 5), (0, 3, 7), (0, 4, 6),
            (1, 2, 2), (1, 3, 9), (1, 4, 4)],
            Dataset(enumerate('ABXYZ'), 'index', 'label'))
        ).options(label_index='label', tools=['hover'])
        plot = bokeh_renderer.get_plot(sankey)

        scatter_source = plot.handles['scatter_1_source']
        text_source = plot.handles['text_1_source']
        patch_source = plot.handles['patches_1_source']

        scatter_index = np.arange(5)
        self.assertEqual(scatter_source.data['index'], scatter_index)

        text_data = {'x': np.array([18.75, 18.75, 1003.75, 1003.75, 1003.75]),
                     'y': np.array([1649.490687, 1899.490687, 1587.029361, 1767.332391, 1968.544512]),
                     'text': ['A - 18', 'B - 15', 'X - 7', 'Y - 16', 'Z - 10']}
        for k in text_data:
            self.assertEqual(text_source.data[k], text_data[k])

        patch_data = {'start_values': ['A', 'A', 'A', 'B', 'B', 'B'],
                      'end_values': ['X', 'Y', 'Z', 'X', 'Y', 'Z'],
                      'Value': np.array([5, 7, 6, 2, 9, 4])}
        for k in patch_data:
            self.assertEqual(patch_source.data[k], patch_data[k])

        renderers = plot.state.renderers
        quad_renderer = plot.handles['quad_1_glyph_renderer']
        text_renderer = plot.handles['text_1_glyph_renderer']
        graph_renderer = plot.handles['glyph_renderer']
        self.assertTrue(renderers.index(graph_renderer)<renderers.index(quad_renderer))
        self.assertTrue(renderers.index(quad_renderer)<renderers.index(text_renderer))
