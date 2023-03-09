import numpy as np
import pandas as pd

from holoviews import render
from holoviews.core.data import Dataset, Dimension
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
                     'y': np.array([125.454545, 375.454545,  48.787879, 229.090909, 430.30303 ]),
                     'text': ['A - 18', 'B - 15', 'X - 7', 'Y - 16', 'Z - 10']}
        for k in text_data:
            self.assertEqual(text_source.data[k], text_data[k])

        quad_data = {
            'index': scatter_index,
            'x0': [0, 0, 985.0, 985.0, 985.0],
            'x1': [15, 15, 1000.0, 1000.0, 1000.0],
            'y0': [0.0, 270.9090909090908, -7.105427357601002e-15, 117.57575757575756, 360.6060606060606],
            'y1': [250.909091, 480.0, 97.575758, 340.606061, 500.0]
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
        ).opts(label_index='label', tools=['hover'])
        plot = bokeh_renderer.get_plot(sankey)

        scatter_source = plot.handles['scatter_1_source']
        text_source = plot.handles['text_1_source']
        patch_source = plot.handles['patches_1_source']

        scatter_index = np.arange(5)
        self.assertEqual(scatter_source.data['index'], scatter_index)

        text_data = {'x': np.array([18.75, 18.75, 1003.75, 1003.75, 1003.75]),
                     'y': np.array([125.454545, 375.454545,  48.787879, 229.090909, 430.30303 ]),
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

    def test_dimension_label(self):
        # Ref: https://github.com/holoviz/holoviews/issues/5386
        data = [
            ["source1", "dest1", 3],
            ["source1", "dest2", 1],
        ]
        df = pd.DataFrame(data, columns=["Source", "Dest", "Count"])

        kdims = [Dimension("Source"), Dimension("Dest", label="Dest Label")]
        plot = Sankey(df, kdims=kdims, vdims=["Count"])
        plot = plot.opts(edge_color="Dest Label")

        # To provoke the error in the issue
        render(plot)
