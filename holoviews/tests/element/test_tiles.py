import numpy as np
import pandas as pd

from holoviews import Tiles


class TestCoordinateConversion:
    def test_spot_check_lonlat_to_eastingnorthing(self):
        # Anchor implementation with a few hard-coded known values.
        # Generated ad-hoc from https://epsg.io/transform#s_srs=4326&t_srs=3857
        easting, northing = Tiles.lon_lat_to_easting_northing(0, 0)
        np.testing.assert_array_almost_equal(easting, 0)
        np.testing.assert_array_almost_equal(northing, 0)

        easting, northing = Tiles.lon_lat_to_easting_northing(20, 10)
        np.testing.assert_array_almost_equal(easting, 2226389.82, decimal=2)
        np.testing.assert_array_almost_equal(northing, 1118889.97, decimal=2)

        easting, northing = Tiles.lon_lat_to_easting_northing(-33, -18)
        np.testing.assert_array_almost_equal(easting, -3673543.20, decimal=2)
        np.testing.assert_array_almost_equal(northing, -2037548.54, decimal=2)

        easting, northing = Tiles.lon_lat_to_easting_northing(85, -75)
        np.testing.assert_array_almost_equal(easting, 9462156.72, decimal=2)
        np.testing.assert_array_almost_equal(northing, -12932243.11, decimal=2)

        easting, northing = Tiles.lon_lat_to_easting_northing(180, 85)
        np.testing.assert_array_almost_equal(easting, 20037508.34, decimal=2)
        np.testing.assert_array_almost_equal(northing, 19971868.88, decimal=2)

    def test_spot_check_eastingnorthing_to_lonlat(self):
        # Anchor implementation with a few hard-coded known values.
        # Generated ad-hoc from https://epsg.io/transform#s_srs=3857&t_srs=4326

        lon, lat = Tiles.easting_northing_to_lon_lat(0, 0)
        np.testing.assert_array_almost_equal(lon, 0)
        np.testing.assert_array_almost_equal(lat, 0)

        lon, lat = Tiles.easting_northing_to_lon_lat(1230020, -432501)
        np.testing.assert_array_almost_equal(lon, 11.0494578, decimal=2)
        np.testing.assert_array_almost_equal(lat, -3.8822487, decimal=2)

        lon, lat = Tiles.easting_northing_to_lon_lat(-2130123, 1829312)
        np.testing.assert_array_almost_equal(lon, -19.1352205, decimal=2)
        np.testing.assert_array_almost_equal(lat, 16.2122187, decimal=2)

        lon, lat = Tiles.easting_northing_to_lon_lat(-1000000, 5000000)
        np.testing.assert_array_almost_equal(lon, -8.9831528, decimal=2)
        np.testing.assert_array_almost_equal(lat, 40.9162745, decimal=2)

        lon, lat = Tiles.easting_northing_to_lon_lat(-20037508.34, 20037508.34)
        np.testing.assert_array_almost_equal(lon, -180.0, decimal=2)
        np.testing.assert_array_almost_equal(lat, 85.0511288, decimal=2)

    def test_check_lonlat_to_eastingnorthing_identity(self):
        for lon in np.linspace(-180, 180, 100):
            for lat in np.linspace(-85, 85, 100):
                easting, northing = Tiles.lon_lat_to_easting_northing(lon, lat)
                new_lon, new_lat = Tiles.easting_northing_to_lon_lat(easting, northing)
                np.testing.assert_array_almost_equal(lon, new_lon, decimal=2)
                np.testing.assert_array_almost_equal(lat, new_lat, decimal=2)

    def test_check_eastingnorthing_to_lonlat_identity(self):
        for easting in np.linspace(-20037508.34, 20037508.34, 100):
            for northing in np.linspace(-20037508.34, 20037508.34, 100):
                lon, lat = Tiles.easting_northing_to_lon_lat(easting, northing)
                new_easting, new_northing = Tiles.lon_lat_to_easting_northing(lon, lat)
                np.testing.assert_array_almost_equal(easting, new_easting, decimal=2)
                np.testing.assert_array_almost_equal(northing, new_northing, decimal=2)

    def check_array_type_preserved(self, constructor, array_type, check):
        lons, lats = np.meshgrid(
            np.linspace(-180, 180, 100), np.linspace(-85, 85, 100)
        )
        lons = lons.flatten()
        lats = lats.flatten()

        array_lons = constructor(lons)
        array_lats = constructor(lats)

        assert isinstance(array_lons, array_type)
        assert isinstance(array_lats, array_type)

        eastings, northings = Tiles.lon_lat_to_easting_northing(
            array_lons, array_lats
        )
        assert isinstance(eastings, array_type)
        assert isinstance(northings, array_type)

        new_lons, new_lats = Tiles.easting_northing_to_lon_lat(
            eastings, northings
        )
        assert isinstance(new_lons, array_type)
        assert isinstance(new_lats, array_type)

        check(array_lons, new_lons)
        check(array_lats, new_lats)

    def test_check_numpy_array(self):
        self.check_array_type_preserved(
            np.array, np.ndarray,
            lambda a, b: np.testing.assert_array_almost_equal(a, b, decimal=2)
        )

    def test_pandas_series(self):
        self.check_array_type_preserved(
            pd.Series, pd.Series,
            lambda a, b: pd.testing.assert_series_equal(
                a, b, check_exact=False,
            )
        )
