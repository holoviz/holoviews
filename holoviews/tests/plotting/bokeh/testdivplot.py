from holoviews.element import Div

from .testplot import TestBokehPlot, bokeh_renderer


class TestDivPlot(TestBokehPlot):

    def test_div_plot(self):
        html = '<h1>Test</h1>'
        div = Div(html)
        plot = bokeh_renderer.get_plot(div)
        bkdiv = plot.handles['plot']
        self.assertEqual(bkdiv.text, '&lt;h1&gt;Test&lt;/h1&gt;')

    def test_div_plot_width(self):
        html = '<h1>Test</h1>'
        div = Div(html).options(width=342, height=432, backend='bokeh')
        plot = bokeh_renderer.get_plot(div)
        bkdiv = plot.handles['plot']
        self.assertEqual(bkdiv.width, 342)
        self.assertEqual(bkdiv.height, 432)
