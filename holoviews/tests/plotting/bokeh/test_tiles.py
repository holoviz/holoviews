from __future__ import annotations

import holoviews as hv

from ...utils import xyzservices, xyzservices_skip
from .test_plot import TestBokehPlot, bokeh_renderer


@xyzservices_skip
class TestTilePlot(TestBokehPlot):
    def test_xyzservices_tileprovider(self):
        osm = xyzservices.providers.OpenStreetMap.Mapnik

        tiles = hv.Tiles(osm)
        plot = bokeh_renderer.get_plot(tiles)
        glyph = plot.handles["glyph"]
        assert glyph.attribution == osm.html_attribution
        assert glyph.url == osm.build_url(scale_factor="@2x")
        assert glyph.max_zoom == osm.max_zoom
