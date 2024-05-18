import datetime as dt

import numpy as np
from bokeh.models import CategoricalColorMapper, ColumnDataSource, LinearColorMapper

from holoviews.element import BoxWhisker
from holoviews.plotting.bokeh.util import property_to_dict

from .test_plot import TestBokehPlot, bokeh_renderer


class TestBoxWhiskerPlot(TestBokehPlot):

    def test_box_whisker_datetime(self):
        times = np.arange(dt.datetime(2017,1,1), dt.datetime(2017,2,1),
                          dt.timedelta(days=1))
        box = BoxWhisker((times, np.random.rand(len(times))), kdims=['Date'])
        plot = bokeh_renderer.get_plot(box)
        formatted = [box.kdims[0].pprint_value(t) for t in times]
        self.assertTrue(all(cds.data['index'][0] in formatted for cds in
                            plot.state.select(ColumnDataSource)
                            if len(cds.data.get('index', []))))

    def test_box_whisker_hover(self):
        xs, ys = np.random.randint(0, 5, 100), np.random.randn(100)
        box = BoxWhisker((xs, ys), 'A').sort().opts(tools=['hover'])
        plot = bokeh_renderer.get_plot(box)
        src = plot.handles['vbar_1_source']
        ys = box.aggregate(function=np.median).dimension_values('y')
        hover_tool = plot.handles['hover']
        self.assertEqual(src.data['y'], ys)
        self.assertIn(plot.handles['vbar_1_glyph_renderer'], hover_tool.renderers)
        self.assertIn(plot.handles['vbar_2_glyph_renderer'], hover_tool.renderers)
        self.assertIn(plot.handles['circle_1_glyph_renderer'], hover_tool.renderers)

    def test_box_whisker_multi_level(self):
        box= BoxWhisker((['A', 'B']*15, [3, 10, 1]*10, np.random.randn(30)),
                        ['Group', 'Category'], 'Value')
        plot = bokeh_renderer.get_plot(box)
        x_range = plot.handles['x_range']
        self.assertEqual(x_range.factors, [
            ('A', '1'), ('A', '3'), ('A', '10'), ('B', '1'), ('B', '3'), ('B', '10')])

    def test_box_whisker_padding_square(self):
        curve = BoxWhisker([1, 2, 3]).opts(padding=0.1)
        plot = bokeh_renderer.get_plot(curve)
        y_range = plot.handles['y_range']
        self.assertEqual(y_range.start, 0.8)
        self.assertEqual(y_range.end, 3.2)

    ###########################
    #    Styling mapping      #
    ###########################

    def test_box_whisker_linear_color_op(self):
        a = np.repeat(np.arange(5), 5)
        b = np.repeat(np.arange(5), 5)
        box = BoxWhisker((a, b, np.arange(25)), ['a', 'b'], 'd').opts(box_color='b')
        plot = bokeh_renderer.get_plot(box)
        source = plot.handles['vbar_1_source']
        cmapper = plot.handles['box_color_color_mapper']
        glyph = plot.handles['vbar_1_glyph']
        self.assertEqual(source.data['box_color'], np.arange(5))
        self.assertTrue(cmapper, LinearColorMapper)
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 4)
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'box_color', 'transform': cmapper})

    def test_box_whisker_categorical_color_op(self):
        a = np.repeat(np.arange(5), 5)
        b = np.repeat(['A', 'B', 'C', 'D', 'E'], 5)
        box = BoxWhisker((a, b, np.arange(25)), ['a', 'b'], 'd').opts(box_color='b')
        plot = bokeh_renderer.get_plot(box)
        source = plot.handles['vbar_1_source']
        glyph = plot.handles['vbar_1_glyph']
        cmapper = plot.handles['box_color_color_mapper']
        self.assertEqual(source.data['box_color'], b[::5])
        self.assertTrue(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, ['A', 'B', 'C', 'D', 'E'])
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'box_color', 'transform': cmapper})

    def test_box_whisker_alpha_op(self):
        a = np.repeat(np.arange(5), 5)
        b = np.repeat(np.arange(5)/10., 5)
        box = BoxWhisker((a, b, np.arange(25)), ['a', 'b'], 'd').opts(box_alpha='b')
        plot = bokeh_renderer.get_plot(box)
        source = plot.handles['vbar_1_source']
        glyph = plot.handles['vbar_1_glyph']
        self.assertEqual(source.data['box_alpha'], np.arange(5)/10.)
        self.assertEqual(property_to_dict(glyph.fill_alpha), {'field': 'box_alpha'})

    def test_box_whisker_line_width_op(self):
        a = np.repeat(np.arange(5), 5)
        b = np.repeat(np.arange(5), 5)
        box = BoxWhisker((a, b, np.arange(25)), ['a', 'b'], 'd').opts(box_line_width='b')
        plot = bokeh_renderer.get_plot(box)
        source = plot.handles['vbar_1_source']
        glyph = plot.handles['vbar_1_glyph']
        self.assertEqual(source.data['box_line_width'], np.arange(5))
        self.assertEqual(property_to_dict(glyph.line_width), {'field': 'box_line_width'})
