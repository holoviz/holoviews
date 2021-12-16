import datetime as dt

import numpy as np

from holoviews.core import NdOverlay, HoloMap
from holoviews.core.options import Cycle
from holoviews.element import Path, Polygons, Contours
from holoviews.streams import PolyDraw
from holoviews.util.transform import dim

from .testplot import TestBokehPlot, bokeh_renderer

try:
    from bokeh.models import LinearColorMapper, CategoricalColorMapper
except:
    pass


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
        obj = obj.opts(plot=opts)
        self._test_hover_info(obj, [('Test', '@{Test}')])

    def test_path_colored_and_split_with_extra_vdims(self):
        xs = [1, 2, 3, 4]
        ys = xs[::-1]
        color = [0, 0.25, 0.5, 0.75]
        other = ['A', 'B', 'C', 'D']
        data = {'x': xs, 'y': ys, 'color': color, 'other': other}
        path = Path([data], vdims=['color','other']).options(color_index='color', tools=['hover'])
        plot = bokeh_renderer.get_plot(path)
        source = plot.handles['source']

        self.assertEqual(source.data['xs'], [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])])
        self.assertEqual(source.data['ys'], [np.array([4, 3]), np.array([3, 2]), np.array([2, 1])])
        self.assertEqual(source.data['other'], np.array(['A', 'B', 'C']))
        self.assertEqual(source.data['color'], np.array([0, 0.25, 0.5]))

    def test_path_colored_dim_split_with_extra_vdims(self):
        xs = [1, 2, 3, 4]
        ys = xs[::-1]
        color = [0, 0.25, 0.5, 0.75]
        other = ['A', 'B', 'C', 'D']
        data = {'x': xs, 'y': ys, 'color': color, 'other': other}
        path = Path([data], vdims=['color','other']).options(color=dim('color')*2, tools=['hover'])
        plot = bokeh_renderer.get_plot(path)
        source = plot.handles['source']

        self.assertEqual(source.data['xs'], [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])])
        self.assertEqual(source.data['ys'], [np.array([4, 3]), np.array([3, 2]), np.array([2, 1])])
        self.assertEqual(source.data['other'], np.array(['A', 'B', 'C']))
        self.assertEqual(source.data['color'], np.array([0, 0.5, 1]))

    def test_path_colored_by_levels_single_value(self):
        xs = [1, 2, 3, 4]
        ys = xs[::-1]
        color = [998, 999, 998, 998]
        date = np.datetime64(dt.datetime(2018, 8, 1))
        data = {'x': xs, 'y': ys, 'color': color, 'date': date}
        levels = [0, 38, 73, 95, 110, 130, 156, 999]
        colors = ['#5ebaff', '#00faf4', '#ffffcc', '#ffe775', '#ffc140', '#ff8f20', '#ff6060']
        path = Path([data], vdims=['color', 'date']).options(
            color_index='color', color_levels=levels, cmap=colors, tools=['hover'])
        plot = bokeh_renderer.get_plot(path)
        source = plot.handles['source']
        cmapper = plot.handles['color_mapper']

        self.assertEqual(source.data['xs'], [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])])
        self.assertEqual(source.data['ys'], [np.array([4, 3]), np.array([3, 2]), np.array([2, 1])])
        self.assertEqual(source.data['color'], np.array([998, 999, 998]))
        self.assertEqual(source.data['date'], np.array([date]*3))
        self.assertEqual(cmapper.low, 998)
        self.assertEqual(cmapper.high, 999)
        self.assertEqual(cmapper.palette, colors[-1:])

    def test_path_continuously_varying_color_op(self):
        xs = [1, 2, 3, 4]
        ys = xs[::-1]
        color = [998, 999, 998, 994]
        date = np.datetime64(dt.datetime(2018, 8, 1))
        data = {'x': xs, 'y': ys, 'color': color, 'date': date}
        levels = [0, 38, 73, 95, 110, 130, 156, 999]
        colors = ['#5ebaff', '#00faf4', '#ffffcc', '#ffe775', '#ffc140', '#ff8f20', '#ff6060']
        path = Path([data], vdims=['color', 'date']).options(
            color='color', color_levels=levels, cmap=colors, tools=['hover'])
        plot = bokeh_renderer.get_plot(path)
        source = plot.handles['source']
        cmapper = plot.handles['color_color_mapper']

        self.assertEqual(source.data['xs'], [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])])
        self.assertEqual(source.data['ys'], [np.array([4, 3]), np.array([3, 2]), np.array([2, 1])])
        self.assertEqual(source.data['color'], np.array([998, 999, 998]))
        self.assertEqual(source.data['date'], np.array([date]*3))
        self.assertEqual(cmapper.low, 994)
        self.assertEqual(cmapper.high, 999)
        self.assertEqual(cmapper.palette, colors[-1:])
        
    def test_path_continuously_varying_alpha_op(self):
        xs = [1, 2, 3, 4]
        ys = xs[::-1]
        alpha = [0.1, 0.7, 0.3, 0.2]
        data = {'x': xs, 'y': ys, 'alpha': alpha}
        path = Path([data], vdims='alpha').options(alpha='alpha')
        plot = bokeh_renderer.get_plot(path)
        source = plot.handles['source']
        self.assertEqual(source.data['xs'], [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])])
        self.assertEqual(source.data['ys'], [np.array([4, 3]), np.array([3, 2]), np.array([2, 1])])
        self.assertEqual(source.data['alpha'], np.array([0.1, 0.7, 0.3]))

    def test_path_continuously_varying_line_width_op(self):
        xs = [1, 2, 3, 4]
        ys = xs[::-1]
        line_width = [1, 7, 3, 2]
        data = {'x': xs, 'y': ys, 'line_width': line_width}
        path = Path([data], vdims='line_width').options(line_width='line_width')
        plot = bokeh_renderer.get_plot(path)
        source = plot.handles['source']
        self.assertEqual(source.data['xs'], [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])])
        self.assertEqual(source.data['ys'], [np.array([4, 3]), np.array([3, 2]), np.array([2, 1])])
        self.assertEqual(source.data['line_width'], np.array([1, 7, 3]))

    def test_path_continuously_varying_color_legend(self):
        data = {
            "x": [1,2,3,4,5,6,7,8,9],
            "y":   [1,2,3,4,5,6,7,8,9],
            "cat": [0,1,2,0,1,2,0,1,2]
        }

        colors = ["#FF0000", "#00FF00", "#0000FF"]
        levels=[0,1,2,3]

        path = Path(data, vdims="cat").opts(color="cat", cmap=dict(zip(levels, colors)), line_width=4, show_legend=True)
        plot = bokeh_renderer.get_plot(path)
        item = plot.state.legend[0].items[0]
        legend = {'field': 'color_str__'}
        self.assertEqual(item.label, legend)
        self.assertEqual(item.renderers, [plot.handles['glyph_renderer']])

        

class TestPolygonPlot(TestBokehPlot):

    def test_polygons_overlay_hover(self):
        obj = NdOverlay({i: Polygons([{('x', 'y'): np.random.rand(10,2), 'z': 0}], vdims=['z'])
                         for i in range(5)}, kdims=['Test'])
        opts = {'Polygons': {'tools': ['hover']},
                'NdOverlay': {'legend_limit': 0}}
        obj = obj.opts(plot=opts)
        self._test_hover_info(obj, [('Test', '@{Test}'), ('z', '@{z}')])

    def test_polygons_colored(self):
        polygons = NdOverlay({j: Polygons([[(i**j, i, j) for i in range(10)]], vdims='Value')
                              for j in range(5)})
        plot = bokeh_renderer.get_plot(polygons)
        for i, splot in enumerate(plot.subplots.values()):
            cmapper = splot.handles['color_mapper']
            self.assertEqual(cmapper.low, 0)
            self.assertEqual(cmapper.high, 4)
            source = splot.handles['source']
            self.assertEqual(source.data['Value'], np.array([i]))

    def test_polygons_colored_batched(self):
        polygons = NdOverlay({j: Polygons([[(i**j, i, j) for i in range(10)]], vdims='Value')
                              for j in range(5)}).opts(plot=dict(legend_limit=0))
        plot = list(bokeh_renderer.get_plot(polygons).subplots.values())[0]
        cmapper = plot.handles['color_mapper']
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 4)
        source = plot.handles['source']
        self.assertEqual(plot.handles['glyph'].fill_color['transform'], cmapper)
        self.assertEqual(source.data['Value'], list(range(5)))

    def test_polygons_colored_batched_unsanitized(self):
        polygons = NdOverlay({j: Polygons([[(i**j, i, j) for i in range(10)] for i in range(2)],
                                          vdims=['some ? unescaped name'])
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

    def test_polygon_with_hole_plot(self):
        xs = [1, 2, 3]
        ys = [2, 0, 7]
        holes = [[[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]]]
        poly = Polygons([{'x': xs, 'y': ys, 'holes': holes}])
        plot = bokeh_renderer.get_plot(poly)
        source = plot.handles['source']
        self.assertEqual(source.data['xs'], [[[np.array([1, 2, 3, 1]), np.array([1.5, 2, 1.6, 1.5]),
                                              np.array([2.1, 2.5, 2.3, 2.1])]]])
        self.assertEqual(source.data['ys'], [[[np.array([2, 0, 7, 2]), np.array([2, 3, 1.6, 2]),
                                              np.array([4.5, 5, 3.5, 4.5])]]])

    def test_multi_polygon_hole_plot(self):
        xs = [1, 2, 3, np.nan, 3, 7, 6]
        ys = [2, 0, 7, np.nan, 2, 5, 7]
        holes = [
            [[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]],
            []
        ]
        poly = Polygons([{'x': xs, 'y': ys, 'holes': holes}])
        plot = bokeh_renderer.get_plot(poly)
        source = plot.handles['source']
        self.assertEqual(source.data['xs'], [[[np.array([1, 2, 3, 1]), np.array([1.5, 2, 1.6, 1.5]),
                                               np.array([2.1, 2.5, 2.3, 2.1])], [np.array([3, 7, 6, 3])]]])
        self.assertEqual(source.data['ys'], [[[np.array([2, 0, 7, 2]), np.array([2, 3, 1.6, 2]),
                                               np.array([4.5, 5, 3.5, 4.5])], [np.array([2, 5, 7, 2])]]])

    def test_polygons_hover_color_op(self):
        polygons = Polygons([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 'green'},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 'red'}
        ], vdims='color').options(fill_color='color', tools=['hover'])
        plot = bokeh_renderer.get_plot(polygons)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.line_color, 'black')
        self.assertEqual(glyph.fill_color, {'field': 'fill_color'})
        self.assertEqual(cds.data['color'], np.array(['green', 'red']))
        self.assertEqual(cds.data['fill_color'], np.array(['green', 'red']))

    def test_polygons_color_op(self):
        polygons = Polygons([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 'green'},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 'red'}
        ], vdims='color').options(color='color')
        plot = bokeh_renderer.get_plot(polygons)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.line_color, 'black')
        self.assertEqual(glyph.fill_color, {'field': 'color'})
        self.assertEqual(cds.data['color'], np.array(['green', 'red']))

    def test_polygons_linear_color_op(self):
        polygons = Polygons([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 7},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 3}
        ], vdims='color').options(color='color')
        plot = bokeh_renderer.get_plot(polygons)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertEqual(glyph.line_color, 'black')
        self.assertEqual(glyph.fill_color, {'field': 'color', 'transform': cmapper})
        self.assertEqual(cds.data['color'], np.array([7, 3]))
        self.assertIsInstance(cmapper, LinearColorMapper)
        self.assertEqual(cmapper.low, 3)
        self.assertEqual(cmapper.high, 7)

    def test_polygons_categorical_color_op(self):
        polygons = Polygons([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 'b'},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 'a'}
        ], vdims='color').options(color='color')
        plot = bokeh_renderer.get_plot(polygons)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertEqual(glyph.line_color, 'black')
        self.assertEqual(glyph.fill_color, {'field': 'color', 'transform': cmapper})
        self.assertEqual(cds.data['color'], np.array(['b', 'a']))
        self.assertIsInstance(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, ['b', 'a'])

    def test_polygons_alpha_op(self):
        polygons = Polygons([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'alpha': 0.7},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'alpha': 0.3}
        ], vdims='alpha').options(alpha='alpha')
        plot = bokeh_renderer.get_plot(polygons)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.line_alpha, {'field': 'alpha'})
        self.assertEqual(glyph.fill_alpha, {'field': 'alpha'})
        self.assertEqual(cds.data['alpha'], np.array([0.7, 0.3]))

    def test_polygons_line_width_op(self):
        polygons = Polygons([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'line_width': 7},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'line_width': 3}
        ], vdims='line_width').options(line_width='line_width')
        plot = bokeh_renderer.get_plot(polygons)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.line_width, {'field': 'line_width'})
        self.assertEqual(cds.data['line_width'], np.array([7, 3]))

    def test_polygons_holes_initialize(self):
        from bokeh.models import MultiPolygons
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        holes = [
            [[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]],
            []
        ]
        poly = HoloMap({0: Polygons([{'x': xs, 'y': ys, 'holes': holes}]),
                        1: Polygons([{'x': xs, 'y': ys}])})
        plot = bokeh_renderer.get_plot(poly)
        glyph = plot.handles['glyph']
        self.assertTrue(plot._has_holes)
        self.assertIsInstance(glyph, MultiPolygons)

    def test_polygons_no_holes_with_draw_tool(self):
        from bokeh.models import Patches
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        holes = [
            [[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]],
            []
        ]
        poly = HoloMap({0: Polygons([{'x': xs, 'y': ys, 'holes': holes}]),
                        1: Polygons([{'x': xs, 'y': ys}])})
        PolyDraw(source=poly)
        plot = bokeh_renderer.get_plot(poly)
        glyph = plot.handles['glyph']
        self.assertFalse(plot._has_holes)
        self.assertIsInstance(glyph, Patches)



class TestContoursPlot(TestBokehPlot):

    def test_empty_contours_plot(self):
        contours = Contours([], vdims=['Intensity'])
        plot = bokeh_renderer.get_plot(contours)
        source = plot.handles['source']
        self.assertEqual(len(source.data['xs']), 0)
        self.assertEqual(len(source.data['ys']), 0)
        self.assertEqual(len(source.data['Intensity']), 0)

    def test_contours_color_op(self):
        contours = Contours([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 'green'},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 'red'}
        ], vdims='color').options(color='color')
        plot = bokeh_renderer.get_plot(contours)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.line_color, {'field': 'color'})
        self.assertEqual(cds.data['color'], np.array(['green', 'red']))

    def test_contours_linear_color_op(self):
        contours = Contours([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 7},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 3}
        ], vdims='color').options(color='color')
        plot = bokeh_renderer.get_plot(contours)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertEqual(glyph.line_color, {'field': 'color', 'transform': cmapper})
        self.assertEqual(cds.data['color'], np.array([7, 3]))
        self.assertIsInstance(cmapper, LinearColorMapper)
        self.assertEqual(cmapper.low, 3)
        self.assertEqual(cmapper.high, 7)

    def test_contours_empty_path(self):
        import pandas as pd
        contours = Contours([
            pd.DataFrame([], columns=['x', 'y', 'color', 'line_width']),
            pd.DataFrame({'x': np.random.rand(10), 'y': np.random.rand(10),
                          'color': ['red']*10, 'line_width': [3]*10},
                         columns=['x', 'y', 'color', 'line_width'])
        ], vdims=['color', 'line_width']).opts(
            color='color', line_width='line_width')
        plot = bokeh_renderer.get_plot(contours)
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.line_color, 'red')
        self.assertEqual(glyph.line_width, 3)


    def test_contours_linear_color_op_update(self):
        contours = HoloMap({
            0: Contours([
                {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 7},
                {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 3}
            ], vdims='color'),
            1: Contours([
                {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 5},
                {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 2}
            ], vdims='color')}).options(color='color', framewise=True)
        plot = bokeh_renderer.get_plot(contours)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        plot.update((0,))
        self.assertEqual(glyph.line_color, {'field': 'color', 'transform': cmapper})
        self.assertEqual(cds.data['color'], np.array([7, 3]))
        self.assertEqual(cmapper.low, 3)
        self.assertEqual(cmapper.high, 7)
        plot.update((1,))
        self.assertEqual(cds.data['color'], np.array([5, 2]))
        self.assertEqual(cmapper.low, 2)
        self.assertEqual(cmapper.high, 5)

    def test_contours_categorical_color_op(self):
        contours = Contours([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 'b'},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 'a'}
        ], vdims='color').options(color='color')
        plot = bokeh_renderer.get_plot(contours)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertEqual(glyph.line_color, {'field': 'color', 'transform': cmapper})
        self.assertEqual(cds.data['color'], np.array(['b', 'a']))
        self.assertIsInstance(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, ['b', 'a'])

    def test_contours_alpha_op(self):
        contours = Contours([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'alpha': 0.7},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'alpha': 0.3}
        ], vdims='alpha').options(alpha='alpha')
        plot = bokeh_renderer.get_plot(contours)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.line_alpha, {'field': 'alpha'})
        self.assertEqual(cds.data['alpha'], np.array([0.7, 0.3]))

    def test_contours_line_width_op(self):
        contours = Contours([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'line_width': 7},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'line_width': 3}
        ], vdims='line_width').options(line_width='line_width')
        plot = bokeh_renderer.get_plot(contours)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.line_width, {'field': 'line_width'})
        self.assertEqual(cds.data['line_width'], np.array([7, 3]))
