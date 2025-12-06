import numpy as np
import pytest

from holoviews.element import RGB, Bounds, Points, Tiles
from holoviews.element.tiles import _ATTRIBUTIONS, StamenTerrain
from holoviews.plotting.plotly.util import (
    PLOTLY_GE_6_0_0,
    PLOTLY_MAP,
    PLOTLY_SCATTERMAP,
)
from holoviews.testing import assert_data_equal

from .test_plot import TestPlotlyPlot, plotly_renderer


class TestMapboxTilesPlot(TestPlotlyPlot):
    def setup_method(self):
        super().setup_method()

        # Precompute coordinates
        self.xs = [3000000, 2000000, 1000000]
        self.ys = [-3000000, -2000000, -1000000]
        self.x_range = (-5000000, 4000000)
        self.x_center = sum(self.x_range) / 2.0
        self.y_range = (-3000000, 2000000)
        self.y_center = sum(self.y_range) / 2.0
        self.lon_range, self.lat_range = Tiles.easting_northing_to_lon_lat(self.x_range, self.y_range)
        self.lon_centers, self.lat_centers = Tiles.easting_northing_to_lon_lat(
            [self.x_center], [self.y_center]
        )
        self.lon_center, self.lat_center = self.lon_centers[0], self.lat_centers[0]
        self.lons, self.lats = Tiles.easting_northing_to_lon_lat(self.xs, self.ys)

    def test_mapbox_tiles_defaults(self):
        tiles = Tiles("").redim.range(
            x=self.x_range, y=self.y_range
        )

        fig_dict = plotly_renderer.get_plot_state(tiles)

        # Check dummy trace
        assert len(fig_dict["data"]) == 1
        dummy_trace = fig_dict["data"][0]
        assert dummy_trace["type"] == PLOTLY_SCATTERMAP
        assert dummy_trace["lon"] == []
        assert dummy_trace["lat"] == []
        assert dummy_trace["showlegend"] is False

        # Check mapbox subplot
        subplot = fig_dict["layout"][PLOTLY_MAP]
        assert subplot["style"] == "white-bg"
        assert subplot['center'] == {'lat': self.lat_center, 'lon': self.lon_center}

        # Check that xaxis and yaxis entries are not created
        assert "xaxis" not in fig_dict["layout"]
        assert "yaxis" not in fig_dict["layout"]

        # Check no layers are introduced when an empty tile server string is
        # passed
        layers = fig_dict["layout"][PLOTLY_MAP].get("layers", [])
        assert len(layers) == 0

    def test_styled_mapbox_tiles(self):
        opts = dict(mapstyle="dark") if PLOTLY_GE_6_0_0 else dict(mapboxstyle="dark", accesstoken="token-str")
        tiles = Tiles().opts(**opts).redim.range(x=self.x_range, y=self.y_range)

        fig_dict = plotly_renderer.get_plot_state(tiles)

        # Check mapbox subplot
        subplot = fig_dict["layout"][PLOTLY_MAP]
        assert subplot["style"] == "dark"
        if not PLOTLY_GE_6_0_0:
            assert subplot["accesstoken"] == "token-str"
        assert subplot['center'] == {'lat': self.lat_center, 'lon': self.lon_center}

    def test_raster_layer(self):
        tiles = StamenTerrain().redim.range(
            x=self.x_range, y=self.y_range
        ).opts(alpha=0.7, min_zoom=3, max_zoom=7)

        fig_dict = plotly_renderer.get_plot_state(tiles)

        # Check dummy trace
        assert len(fig_dict["data"]) == 1
        dummy_trace = fig_dict["data"][0]
        assert dummy_trace["type"] == PLOTLY_SCATTERMAP
        assert dummy_trace["lon"] == []
        assert dummy_trace["lat"] == []
        assert dummy_trace["showlegend"] is False

        # Check mapbox subplot
        subplot = fig_dict["layout"][PLOTLY_MAP]
        assert subplot["style"] == "white-bg"
        assert subplot['center'] == {'lat': self.lat_center, 'lon': self.lon_center}

        # Check for raster layer
        layers = fig_dict["layout"][PLOTLY_MAP].get("layers", [])
        assert len(layers) == 1
        layer = layers[0]
        assert layer["source"][0].lower() == tiles.data.lower()
        assert layer["opacity"] == 0.7
        assert layer["sourcetype"] == "raster"
        assert layer["minzoom"] == 3
        assert layer["maxzoom"] == 7
        assert layer["sourceattribution"] == _ATTRIBUTIONS[('stamen', 'png')]

    # xyzservices input
    def test_xyzservices_tileprovider(self):
        xyzservices = pytest.importorskip("xyzservices")
        osm = xyzservices.providers.OpenStreetMap.Mapnik
        tiles = Tiles(osm, name="xyzservices").redim.range(
            x=self.x_range, y=self.y_range
        )

        fig_dict = plotly_renderer.get_plot_state(tiles)
        # Check mapbox subplot
        layers = fig_dict["layout"][PLOTLY_MAP].get("layers", [])
        assert len(layers) == 1
        layer = layers[0]
        assert layer["source"][0].lower() == osm.build_url(scale_factor="@2x")
        assert layer["maxzoom"] == osm.max_zoom
        assert layer["sourceattribution"] == osm.html_attribution

    def test_overlay(self):
        # Base layer is mapbox vector layer
        opts = dict(mapstyle="dark") if PLOTLY_GE_6_0_0 else dict(mapboxstyle="dark", accesstoken="token-str")
        tiles = Tiles("").opts(**opts)

        # Raster tile layer
        stamen_raster = StamenTerrain().opts(alpha=0.7)

        # RGB layer
        rgb_data = np.random.rand(10, 10, 3)
        rgb = RGB(
            rgb_data,
            bounds=(self.x_range[0], self.y_range[0], self.x_range[1], self.y_range[1])
        ).opts(
            opacity=0.5
        )

        # Points layer
        points = Points([(0, 0), (self.x_range[1], self.y_range[1])]).opts(
            show_legend=True
        )

        # Bounds
        bounds = Bounds((self.x_range[0], self.y_range[0], 0, 0))

        # Overlay
        overlay = (tiles * stamen_raster * rgb * points * bounds).redim.range(
            x=self.x_range, y=self.y_range
        )

        # Render to plotly figure dictionary
        fig_dict = plotly_renderer.get_plot_state(overlay, numpy_convert=True)

        # Check number of traces and layers
        traces = fig_dict["data"]
        subplot = fig_dict["layout"][PLOTLY_MAP]
        layers = subplot["layers"]

        assert len(traces) == 5
        assert len(layers) == 2

        # Check vector layer
        dummy_trace = traces[0]
        assert dummy_trace["type"] == PLOTLY_SCATTERMAP
        assert dummy_trace["lon"] == []
        assert dummy_trace["lat"] == []
        assert not dummy_trace["showlegend"]

        assert subplot["style"] == "dark"
        if not PLOTLY_GE_6_0_0:
            assert subplot["accesstoken"] == "token-str"
        assert subplot['center'] == {'lat': self.lat_center, 'lon': self.lon_center}

        # Check raster layer
        dummy_trace = traces[1]
        raster_layer = layers[0]
        assert dummy_trace["type"] == PLOTLY_SCATTERMAP
        assert dummy_trace["lon"] == []
        assert dummy_trace["lat"] == []
        assert not dummy_trace["showlegend"]

        # Check raster_layer
        assert raster_layer["below"] == "traces"
        assert raster_layer["opacity"] == 0.7
        assert raster_layer["sourcetype"] == "raster"
        assert raster_layer["source"][0].lower() == stamen_raster.data.lower()

        # Check RGB layer
        dummy_trace = traces[2]
        rgb_layer = layers[1]
        assert dummy_trace["type"] == PLOTLY_SCATTERMAP
        assert dummy_trace["lon"] == [None]
        assert dummy_trace["lat"] == [None]
        assert not dummy_trace["showlegend"]

        # Check rgb_layer
        assert rgb_layer["below"] == "traces"
        assert rgb_layer["opacity"] == 0.5
        assert rgb_layer["sourcetype"] == "image"
        assert rgb_layer["source"].startswith("data:image/png;base64,iVBOR")
        assert rgb_layer["coordinates"] == [
            [self.lon_range[0], self.lat_range[1]],
            [self.lon_range[1], self.lat_range[1]],
            [self.lon_range[1], self.lat_range[0]],
            [self.lon_range[0], self.lat_range[0]]
        ]

        # Check Points layer
        points_trace = traces[3]
        assert points_trace["type"] == PLOTLY_SCATTERMAP
        assert_data_equal(points_trace["lon"], np.array([0, self.lon_range[1]]))
        assert_data_equal(points_trace["lat"], np.array([0, self.lat_range[1]]))
        assert points_trace["mode"] == "markers"
        assert points_trace["showlegend"]

        # Check Bounds layer
        bounds_trace = traces[4]
        assert bounds_trace["type"] == PLOTLY_SCATTERMAP
        assert_data_equal(bounds_trace["lon"], np.array([
            self.lon_range[0], self.lon_range[0], 0, 0, self.lon_range[0]
        ]))
        assert_data_equal(bounds_trace["lat"], np.array([
            self.lat_range[0], 0, 0, self.lat_range[0], self.lat_range[0]
        ]))
        assert bounds_trace["mode"] == "lines"
        assert points_trace["showlegend"] is True

        # No xaxis/yaxis
        assert "xaxis" not in fig_dict["layout"]
        assert "yaxis" not in fig_dict["layout"]
