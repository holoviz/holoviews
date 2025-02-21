import numpy as np
import pytest

from holoviews.element import RGB, Bounds, Points, Tiles
from holoviews.element.tiles import _ATTRIBUTIONS, StamenTerrain
from holoviews.plotting.plotly.util import (
    PLOTLY_GE_6_0_0,
    PLOTLY_MAP,
    PLOTLY_SCATTERMAP,
)

from .test_plot import TestPlotlyPlot, plotly_renderer


class TestMapboxTilesPlot(TestPlotlyPlot):
    def setUp(self):
        super().setUp()

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
        self.assertEqual(len(fig_dict["data"]), 1)
        dummy_trace = fig_dict["data"][0]
        self.assertEqual(dummy_trace["type"], PLOTLY_SCATTERMAP)
        self.assertEqual(dummy_trace["lon"], [])
        self.assertEqual(dummy_trace["lat"], [])
        self.assertEqual(dummy_trace["showlegend"], False)

        # Check mapbox subplot
        subplot = fig_dict["layout"][PLOTLY_MAP]
        self.assertEqual(subplot["style"], "white-bg")
        self.assertEqual(
            subplot['center'], {'lat': self.lat_center, 'lon': self.lon_center}
        )

        # Check that xaxis and yaxis entries are not created
        self.assertNotIn("xaxis", fig_dict["layout"])
        self.assertNotIn("yaxis", fig_dict["layout"])

        # Check no layers are introduced when an empty tile server string is
        # passed
        layers = fig_dict["layout"][PLOTLY_MAP].get("layers", [])
        self.assertEqual(len(layers), 0)

    def test_styled_mapbox_tiles(self):
        opts = dict(mapstyle="dark") if PLOTLY_GE_6_0_0 else dict(mapboxstyle="dark", accesstoken="token-str")
        tiles = Tiles().opts(**opts).redim.range(x=self.x_range, y=self.y_range)

        fig_dict = plotly_renderer.get_plot_state(tiles)

        # Check mapbox subplot
        subplot = fig_dict["layout"][PLOTLY_MAP]
        self.assertEqual(subplot["style"], "dark")
        if not PLOTLY_GE_6_0_0:
            self.assertEqual(subplot["accesstoken"], "token-str")
        self.assertEqual(
            subplot['center'], {'lat': self.lat_center, 'lon': self.lon_center}
        )

    def test_raster_layer(self):
        tiles = StamenTerrain().redim.range(
            x=self.x_range, y=self.y_range
        ).opts(alpha=0.7, min_zoom=3, max_zoom=7)

        fig_dict = plotly_renderer.get_plot_state(tiles)

        # Check dummy trace
        self.assertEqual(len(fig_dict["data"]), 1)
        dummy_trace = fig_dict["data"][0]
        self.assertEqual(dummy_trace["type"], PLOTLY_SCATTERMAP)
        self.assertEqual(dummy_trace["lon"], [])
        self.assertEqual(dummy_trace["lat"], [])
        self.assertEqual(dummy_trace["showlegend"], False)

        # Check mapbox subplot
        subplot = fig_dict["layout"][PLOTLY_MAP]
        self.assertEqual(subplot["style"], "white-bg")
        self.assertEqual(
            subplot['center'], {'lat': self.lat_center, 'lon': self.lon_center}
        )

        # Check for raster layer
        layers = fig_dict["layout"][PLOTLY_MAP].get("layers", [])
        self.assertEqual(len(layers), 1)
        layer = layers[0]
        self.assertEqual(layer["source"][0].lower(), tiles.data.lower())
        self.assertEqual(layer["opacity"], 0.7)
        self.assertEqual(layer["sourcetype"], "raster")
        self.assertEqual(layer["minzoom"], 3)
        self.assertEqual(layer["maxzoom"], 7)
        self.assertEqual(layer["sourceattribution"], _ATTRIBUTIONS[('stamen', 'png')])

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
        self.assertEqual(len(layers), 1)
        layer = layers[0]
        self.assertEqual(layer["source"][0].lower(), osm.build_url(scale_factor="@2x"))
        self.assertEqual(layer["maxzoom"], osm.max_zoom)
        self.assertEqual(layer["sourceattribution"], osm.html_attribution)

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

        self.assertEqual(len(traces), 5)
        self.assertEqual(len(layers), 2)

        # Check vector layer
        dummy_trace = traces[0]
        self.assertEqual(dummy_trace["type"], PLOTLY_SCATTERMAP)
        self.assertEqual(dummy_trace["lon"], [])
        self.assertEqual(dummy_trace["lat"], [])
        self.assertFalse(dummy_trace["showlegend"])

        self.assertEqual(subplot["style"], "dark")
        if not PLOTLY_GE_6_0_0:
            self.assertEqual(subplot["accesstoken"], "token-str")
        self.assertEqual(
            subplot['center'], {'lat': self.lat_center, 'lon': self.lon_center}
        )

        # Check raster layer
        dummy_trace = traces[1]
        raster_layer = layers[0]
        self.assertEqual(dummy_trace["type"], PLOTLY_SCATTERMAP)
        self.assertEqual(dummy_trace["lon"], [])
        self.assertEqual(dummy_trace["lat"], [])
        self.assertFalse(dummy_trace["showlegend"])

        # Check raster_layer
        self.assertEqual(raster_layer["below"], "traces")
        self.assertEqual(raster_layer["opacity"], 0.7)
        self.assertEqual(raster_layer["sourcetype"], "raster")
        self.assertEqual(raster_layer["source"][0].lower(), stamen_raster.data.lower())

        # Check RGB layer
        dummy_trace = traces[2]
        rgb_layer = layers[1]
        self.assertEqual(dummy_trace["type"], PLOTLY_SCATTERMAP)
        self.assertEqual(dummy_trace["lon"], [None])
        self.assertEqual(dummy_trace["lat"], [None])
        self.assertFalse(dummy_trace["showlegend"])

        # Check rgb_layer
        self.assertEqual(rgb_layer["below"], "traces")
        self.assertEqual(rgb_layer["opacity"], 0.5)
        self.assertEqual(rgb_layer["sourcetype"], "image")
        self.assertTrue(rgb_layer["source"].startswith("data:image/png;base64,iVBOR"))
        self.assertEqual(rgb_layer["coordinates"], [
            [self.lon_range[0], self.lat_range[1]],
            [self.lon_range[1], self.lat_range[1]],
            [self.lon_range[1], self.lat_range[0]],
            [self.lon_range[0], self.lat_range[0]]
        ])

        # Check Points layer
        points_trace = traces[3]
        self.assertEqual(points_trace["type"], PLOTLY_SCATTERMAP)
        self.assertEqual(points_trace["lon"], np.array([0, self.lon_range[1]]))
        self.assertEqual(points_trace["lat"], np.array([0, self.lat_range[1]]))
        self.assertEqual(points_trace["mode"], "markers")
        self.assertTrue(points_trace.get("showlegend", True))

        # Check Bounds layer
        bounds_trace = traces[4]
        self.assertEqual(bounds_trace["type"], PLOTLY_SCATTERMAP)
        self.assertEqual(bounds_trace["lon"], np.array([
            self.lon_range[0], self.lon_range[0], 0, 0, self.lon_range[0]
        ]))
        self.assertEqual(bounds_trace["lat"], np.array([
            self.lat_range[0], 0, 0, self.lat_range[0], self.lat_range[0]
        ]))
        self.assertEqual(bounds_trace["mode"], "lines")
        self.assertTrue(points_trace["showlegend"], False)

        # No xaxis/yaxis
        self.assertNotIn("xaxis", fig_dict["layout"])
        self.assertNotIn("yaxis", fig_dict["layout"])
