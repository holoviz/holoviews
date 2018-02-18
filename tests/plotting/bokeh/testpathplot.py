import numpy as np

from holoviews.core import NdOverlay
from holoviews.core.options import Cycle
from holoviews.element import Path, Polygons, Contours

from .testplot import TestBokehPlot, bokeh_renderer


class TestPathPlot(TestBokehPlot):

    def test_batched_path_line_color_and_color(self):
        opts = {'NdOverlay': dict(plot=dict(legend_limit=0)),
                'Path': dict(style=dict(line_color=Cycle(values=['red', 'blue'])))}
        overlay = NdOverlay({i: Path([[(i, j) for j in range(2)]])
                             for i in range(2)}).opts(opts)
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        line_color = ['red', 'blue']
        self.assertEqual(plot.handles['source'].data['line_color'], line_color)

    def test_batched_path_alpha_and_color(self):
        opts = {'NdOverlay': dict(plot=dict(legend_limit=0)),
                'Path': dict(style=dict(alpha=Cycle(values=[0.5, 1])))}
        overlay = NdOverlay({i: Path([[(i, j) for j in range(2)]])
                             for i in range(2)}).opts(opts)
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        alpha = [0.5, 1.]
        color = ['#30a2da', '#fc4f30']
        self.assertEqual(plot.handles['source'].data['alpha'], alpha)
        self.assertEqual(plot.handles['source'].data['color'], color)

    def test_batched_path_line_width_and_color(self):
        opts = {'NdOverlay': dict(plot=dict(legend_limit=0)),
                'Path': dict(style=dict(line_width=Cycle(values=[0.5, 1])))}
        overlay = NdOverlay({i: Path([[(i, j) for j in range(2)]])
                             for i in range(2)}).opts(opts)
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        line_width = [0.5, 1.]
        color = ['#30a2da', '#fc4f30']
        self.assertEqual(plot.handles['source'].data['line_width'], line_width)
        self.assertEqual(plot.handles['source'].data['color'], color)

    def test_path_overlay_hover(self):
        obj = NdOverlay({i: Path([np.random.rand(10,2)]) for i in range(5)},
                        kdims=['Test'])
        opts = {'Path': {'tools': ['hover']},
                'NdOverlay': {'legend_limit': 0}}
        obj = obj(plot=opts)
        self._test_hover_info(obj, [('Test', '@{Test}')])

    def test_empty_path_plot(self):
        path = Path([], vdims=['Intensity']).opts(plot=dict(color_index=2))
        plot = bokeh_renderer.get_plot(path)
        source = plot.handles['source']
        self.assertEqual(len(source.data['xs']), 0)
        self.assertEqual(len(source.data['ys']), 0)
        self.assertEqual(len(source.data['Intensity']), 0)

        

class TestPolygonPlot(TestBokehPlot):

    def test_polygons_overlay_hover(self):
        obj = NdOverlay({i: Polygons([np.random.rand(10,2)], vdims=['z'], level=0)
                         for i in range(5)}, kdims=['Test'])
        opts = {'Polygons': {'tools': ['hover']},
                'NdOverlay': {'legend_limit': 0}}
        obj = obj(plot=opts)
        self._test_hover_info(obj, [('Test', '@{Test}'), ('z', '@{z}')])

    def test_polygons_colored(self):
        polygons = NdOverlay({j: Polygons([[(i**j, i) for i in range(10)]], level=j)
                              for j in range(5)})
        plot = bokeh_renderer.get_plot(polygons)
        for i, splot in enumerate(plot.subplots.values()):
            cmapper = splot.handles['color_mapper']
            self.assertEqual(cmapper.low, 0)
            self.assertEqual(cmapper.high, 4)
            source = splot.handles['source']
            self.assertEqual(source.data['Value'], np.array([i]))

    def test_polygons_colored_batched(self):
        polygons = NdOverlay({j: Polygons([[(i**j, i) for i in range(10)]], level=j)
                              for j in range(5)}).opts(plot=dict(legend_limit=0))
        plot = list(bokeh_renderer.get_plot(polygons).subplots.values())[0]
        cmapper = plot.handles['color_mapper']
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 4)
        source = plot.handles['source']
        self.assertEqual(plot.handles['glyph'].fill_color['transform'], cmapper)
        self.assertEqual(source.data['Value'], list(range(5)))

    def test_polygons_colored_batched_unsanitized(self):
        polygons = NdOverlay({j: Polygons([[(i**j, i) for i in range(10)] for i in range(2)],
                                          level=j, vdims=['some ? unescaped name'])
                              for j in range(5)}).opts(plot=dict(legend_limit=0))
        plot = list(bokeh_renderer.get_plot(polygons).subplots.values())[0]
        cmapper = plot.handles['color_mapper']
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 4)
        source = plot.handles['source']
        self.assertEqual(source.data['some_question_mark_unescaped_name'],
                         [j for i in range(5) for j in [i, i]])

    def test_empty_polygons_plot(self):
        poly = Polygons([], vdims=['Intensity'])
        plot = bokeh_renderer.get_plot(poly)
        source = plot.handles['source']
        self.assertEqual(len(source.data['xs']), 0)
        self.assertEqual(len(source.data['ys']), 0)
        self.assertEqual(len(source.data['Intensity']), 0)



class TestContoursPlot(TestBokehPlot):

    def test_empty_contours_plot(self):
        contours = Contours([], vdims=['Intensity'])
        plot = bokeh_renderer.get_plot(contours)
        source = plot.handles['source']
        self.assertEqual(len(source.data['xs']), 0)
        self.assertEqual(len(source.data['ys']), 0)
        self.assertEqual(len(source.data['Intensity']), 0)
