from __future__ import absolute_import

import numpy as np
from holoviews.core.data import Dataset
from holoviews.element import Sankey


from .testplot import TestBokehPlot, bokeh_renderer

class TestSankeyPlot(TestBokehPlot):

    def test_sankey_simple(self):
        sankey = Sankey([
            ('A', 'X', 5), ('A', 'Y', 7), ('A', 'Z', 6),
            ('B', 'X', 2), ('B', 'Y', 9), ('B', 'Z', 4)]
        )
        plot = list(bokeh_renderer.get_plot(sankey).subplots.values())[0]
        scatter_source = plot.handles['scatter_1_source']
        quad_source = plot.handles['quad_1_source']
        text_source = plot.handles['text_1_source']
        patch_source = plot.handles['patches_1_source']

        scatter_index = np.array(['A', 'B', 'X', 'Y', 'Z'])
        self.assertEqual(scatter_source.data['index'], scatter_index)

        text_data = {'x': np.array([18.75, 18.75, 1003.75, 1003.75, 1003.75]),
                     'y': np.array([140.90909091, 390.90909091, 50.90909091,
                                    228.18181818, 427.27272727]),
                     'text': ['A - 18', 'B - 15', 'X - 7', 'Y - 16', 'Z - 10']}
        for k in text_data:
            self.assertEqual(text_source.data[k], text_data[k])

        quad_data = {
            'index': scatter_index,
            'x0': [0, 0, 985.0, 985.0, 985.0],
            'x1': [15, 15, 1000.0, 1000.0, 1000.0],
            'y0': [10.000000000000355, 281.8181818181821, -5.6843418860808015e-14,
                   111.8181818181817, 354.54545454545433],
            'y1': [271.8181818181821, 500.0, 101.8181818181817, 344.54545454545433, 500.0]
        }
        for k in quad_data:
            self.assertEqual(quad_source.data[k], quad_data[k])

        self.assertEqual(patch_source.data['Value'], np.array([5, 7, 6, 2, 9, 4]))
        

    def test_sankey_label_index(self):
        sankey = Sankey(([
            (0, 2, 5), (0, 3, 7), (0, 4, 6),
            (1, 2, 2), (1, 3, 9), (1, 4, 4)],
            Dataset(enumerate('ABXYZ'), 'index', 'label'))
        ).options(label_index='label', tools=['hover'])
        plot = list(bokeh_renderer.get_plot(sankey).subplots.values())[0]

        scatter_source = plot.handles['scatter_1_source']
        text_source = plot.handles['text_1_source']
        patch_source = plot.handles['patches_1_source']
        
        scatter_index = np.arange(5)
        self.assertEqual(scatter_source.data['index'], scatter_index)

        text_data = {'x': np.array([18.75, 18.75, 1003.75, 1003.75, 1003.75]),
                     'y': np.array([140.90909091, 390.90909091, 50.90909091,
                                    228.18181818, 427.27272727]),
                     'text': ['A - 18', 'B - 15', 'X - 7', 'Y - 16', 'Z - 10']}
        for k in text_data:
            self.assertEqual(text_source.data[k], text_data[k])

        patch_data = {'start_values': ['A', 'A', 'A', 'B', 'B', 'B'],
                      'end_values': ['X', 'Y', 'Z', 'X', 'Y', 'Z'],
                      'Value': np.array([5, 7, 6, 2, 9, 4])}
        for k in patch_data:
            self.assertEqual(patch_source.data[k], patch_data[k])
            
