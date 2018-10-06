import datetime as dt

import numpy as np

from holoviews.element import BoxWhisker

from .testplot import TestBokehPlot, bokeh_renderer

try:
    from bokeh.models import ColumnDataSource
except:
    pass


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
        box = BoxWhisker((xs, ys), 'A').sort().opts(plot=dict(tools=['hover']))
        plot = bokeh_renderer.get_plot(box)
        src = plot.handles['vbar_1_source']
        ys = box.aggregate(function=np.median).dimension_values('y')
        hover_tool = plot.handles['hover']
        self.assertEqual(src.data['y'], ys)
        self.assertIn(plot.handles['vbar_1_glyph_renderer'], hover_tool.renderers)
        self.assertIn(plot.handles['vbar_2_glyph_renderer'], hover_tool.renderers)
        self.assertIn(plot.handles['circle_1_glyph_renderer'], hover_tool.renderers)
