from unittest import SkipTest

from holoviews.element import Tiles

from .testplot import TestBokehPlot, bokeh_renderer

try:
    import xyzservices
except ImportError:
    xyzservices = None

class TestTilePlot(TestBokehPlot):

    def test_xyzservices_tileprovider(self):
        if xyzservices is None:
            raise SkipTest("xyzservices not available")
        osm = xyzservices.providers.OpenStreetMap.Mapnik

        tiles = Tiles(osm)
        plot = bokeh_renderer.get_plot(tiles)
        glyph = plot.handles["glyph"]
        assert glyph.attribution == osm.html_attribution
        assert glyph.url == osm.build_url(scale_factor="@2x")
        assert glyph.max_zoom == osm.max_zoom
