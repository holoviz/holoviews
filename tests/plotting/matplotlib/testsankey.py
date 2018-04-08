import numpy as np

from holoviews.core.data import Dataset
from holoviews.element import Sankey

from .testplot import TestMPLPlot, mpl_renderer


class TestSankeyPlot(TestMPLPlot):

    def test_sankey_simple(self):
        sankey = Sankey([
            ('A', 'X', 5), ('A', 'Y', 7), ('A', 'Z', 6),
            ('B', 'X', 2), ('B', 'Y', 9), ('B', 'Z', 4)]
        )
        plot = list(mpl_renderer.get_plot(sankey).subplots.values())[0]

        rects = plot.handles['rects']
        labels = plot.handles['labels']

        text_data = {'x': np.array([18.75, 18.75, 1003.75, 1003.75, 1003.75]),
                     'y': np.array([140.90909091, 390.90909091, 50.90909091,
                                    228.18181818, 427.27272727]),
                     'text': ['A - 18', 'B - 15', 'X - 7', 'Y - 16', 'Z - 10']}
        for i, text in enumerate(labels):
            self.assertEqual(text.xy[0], text_data['x'][i])
            self.assertEqual(text.xy[1], text_data['y'][i])
            self.assertEqual(text.get_text(), text_data['text'][i])
            
        quad_data = {
            'x0': [0, 0, 985.0, 985.0, 985.0],
            'x1': [15, 15, 1000.0, 1000.0, 1000.0],
            'y0': [10.000000000000355, 281.8181818181821, -5.6843418860808015e-14,
                   111.8181818181817, 354.54545454545433],
            'y1': [271.8181818181821, 500.0, 101.8181818181817, 344.54545454545433, 500.0]
        }
        for i, rect in enumerate(rects.get_paths()):
            x0, x1, y0, y1 = [quad_data[c][i] for c in ('x0', 'x1', 'y0', 'y1')]
            arr = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
            self.assertEqual(rect.vertices, arr)


    def test_sankey_label_index(self):
        sankey = Sankey(([
            (0, 2, 5), (0, 3, 7), (0, 4, 6),
            (1, 2, 2), (1, 3, 9), (1, 4, 4)],
            Dataset(enumerate('ABXYZ'), 'index', 'label'))
        ).options(label_index='label')
        plot = list(mpl_renderer.get_plot(sankey).subplots.values())[0]
        labels = plot.handles['labels']

        text_data = {'x': np.array([18.75, 18.75, 1003.75, 1003.75, 1003.75]),
                     'y': np.array([140.90909091, 390.90909091, 50.90909091,
                                    228.18181818, 427.27272727]),
                     'text': ['A - 18', 'B - 15', 'X - 7', 'Y - 16', 'Z - 10']}
        for i, text in enumerate(labels):
            self.assertEqual(text.xy[0], text_data['x'][i])
            self.assertEqual(text.xy[1], text_data['y'][i])
            self.assertEqual(text.get_text(), text_data['text'][i])
