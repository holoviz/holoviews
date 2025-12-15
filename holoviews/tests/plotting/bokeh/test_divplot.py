from holoviews.element import Div

from .test_plot import TestBokehPlot, bokeh_renderer


class TestDivPlot(TestBokehPlot):

    def test_div_plot(self):
        html = '<h1>Test</h1>'
        div = Div(html)
        plot = bokeh_renderer.get_plot(div)
        bkdiv = plot.handles['plot']
        assert bkdiv.text == '&lt;h1&gt;Test&lt;/h1&gt;'

    def test_div_plot_width(self):
        html = '<h1>Test</h1>'
        div = Div(html).opts(width=342, height=432, backend='bokeh')
        plot = bokeh_renderer.get_plot(div)
        bkdiv = plot.handles['plot']
        assert bkdiv.width == 342
        assert bkdiv.height == 432
