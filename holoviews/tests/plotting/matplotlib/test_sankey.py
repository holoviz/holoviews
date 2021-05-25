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
        plot = mpl_renderer.get_plot(sankey)

        rects = plot.handles['rects']
        labels = plot.handles['labels']

        text_data = {'x': np.array([18.75, 18.75, 1003.75, 1003.75, 1003.75]),
                     'y': np.array([145.454545, 395.454545,  48.787879, 229.090909, 430.30303]),
                     'text': ['A - 18', 'B - 15', 'X - 7', 'Y - 16', 'Z - 10']}
        for i, text in enumerate(labels):
            self.assertEqual(text.xy[0], text_data['x'][i])
            self.assertEqual(text.xy[1], text_data['y'][i])
            self.assertEqual(text.get_text(), text_data['text'][i])

        quad_data = {
            'x0': [0, 0, 985.0, 985.0, 985.0],
            'x1': [15, 15, 1000.0, 1000.0, 1000.0],
            'y0': [19.999999999999588, 290.90909090909054, 2.842170943040401e-14,
                   117.57575757575768, 360.60606060606057],
            'y1': [270.90909090909054, 500.0, 97.57575757575768, 340.60606060606057, 500.0]
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
        plot = mpl_renderer.get_plot(sankey)
        labels = plot.handles['labels']

        text_data = {'x': np.array([18.75, 18.75, 1003.75, 1003.75, 1003.75]),
                     'y': np.array([145.454545, 395.454545,  48.787879, 229.090909, 430.30303]),
                     'text': ['A - 18', 'B - 15', 'X - 7', 'Y - 16', 'Z - 10']}
        for i, text in enumerate(labels):
            self.assertEqual(text.xy[0], text_data['x'][i])
            self.assertEqual(text.xy[1], text_data['y'][i])
            self.assertEqual(text.get_text(), text_data['text'][i])
