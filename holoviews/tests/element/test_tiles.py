import numpy as np
import pandas as pd
from holoviews import Tiles

from holoviews.element.comparison import ComparisonTestCase


class TestCoordinateConversion(ComparisonTestCase):
    def test_spot_check_lonlat_to_eastingnorthing(self):
        # Anchor implementation with a few hard-coded known values.
        # Generated ad-hoc from https://epsg.io/transform#s_srs=4326&t_srs=3857
        easting, northing = Tiles.lon_lat_to_easting_northing(0, 0)
        self.assertAlmostEqual(easting, 0)
        self.assertAlmostEqual(northing, 0)

        easting, northing = Tiles.lon_lat_to_easting_northing(20, 10)
        self.assertAlmostEqual(easting, 2226389.82, places=2)
        self.assertAlmostEqual(northing, 1118889.97, places=2)

        easting, northing = Tiles.lon_lat_to_easting_northing(-33, -18)
        self.assertAlmostEqual(easting, -3673543.20, places=2)
        self.assertAlmostEqual(northing, -2037548.54, places=2)

        easting, northing = Tiles.lon_lat_to_easting_northing(85, -75)
        self.assertAlmostEqual(easting, 9462156.72, places=2)
        self.assertAlmostEqual(northing, -12932243.11, places=2)

        easting, northing = Tiles.lon_lat_to_easting_northing(180, 85)
        self.assertAlmostEqual(easting, 20037508.34, places=2)
        self.assertAlmostEqual(northing, 19971868.88, places=2)

    def test_spot_check_eastingnorthing_to_lonlat(self):
        # Anchor implementation with a few hard-coded known values.
        # Generated ad-hoc from https://epsg.io/transform#s_srs=3857&t_srs=4326

        lon, lat = Tiles.easting_northing_to_lon_lat(0, 0)
        self.assertAlmostEqual(lon, 0)
        self.assertAlmostEqual(lat, 0)

        lon, lat = Tiles.easting_northing_to_lon_lat(1230020, -432501)
        self.assertAlmostEqual(lon, 11.0494578, places=2)
        self.assertAlmostEqual(lat, -3.8822487, places=2)

        lon, lat = Tiles.easting_northing_to_lon_lat(-2130123, 1829312)
        self.assertAlmostEqual(lon, -19.1352205, places=2)
        self.assertAlmostEqual(lat, 16.2122187, places=2)

        lon, lat = Tiles.easting_northing_to_lon_lat(-1000000, 5000000)
        self.assertAlmostEqual(lon, -8.9831528, places=2)
        self.assertAlmostEqual(lat, 40.9162745, places=2)

        lon, lat = Tiles.easting_northing_to_lon_lat(-20037508.34, 20037508.34)
        self.assertAlmostEqual(lon, -180.0, places=2)
        self.assertAlmostEqual(lat, 85.0511288, places=2)

    def test_check_lonlat_to_eastingnorthing_identity(self):
        for lon in np.linspace(-180, 180, 100):
            for lat in np.linspace(-85, 85, 100):
                easting, northing = Tiles.lon_lat_to_easting_northing(lon, lat)
                new_lon, new_lat = Tiles.easting_northing_to_lon_lat(easting, northing)
                self.assertAlmostEqual(lon, new_lon, places=2)
                self.assertAlmostEqual(lat, new_lat, places=2)

    def test_check_eastingnorthing_to_lonlat_identity(self):
        for easting in np.linspace(-20037508.34, 20037508.34, 100):
            for northing in np.linspace(-20037508.34, 20037508.34, 100):
                lon, lat = Tiles.easting_northing_to_lon_lat(easting, northing)
                new_easting, new_northing = Tiles.lon_lat_to_easting_northing(lon, lat)
                self.assertAlmostEqual(easting, new_easting, places=2)
                self.assertAlmostEqual(northing, new_northing, places=2)

    def check_array_type_preserved(self, constructor, array_type, check):
        lons, lats = np.meshgrid(
            np.linspace(-180, 180, 100), np.linspace(-85, 85, 100)
        )
        lons = lons.flatten()
        lats = lats.flatten()

        array_lons = constructor(lons)
        array_lats = constructor(lats)

        self.assertIsInstance(array_lons, array_type)
        self.assertIsInstance(array_lats, array_type)

        eastings, northings = Tiles.lon_lat_to_easting_northing(
            array_lons, array_lats
        )
        self.assertIsInstance(eastings, array_type)
        self.assertIsInstance(northings, array_type)

        new_lons, new_lats = Tiles.easting_northing_to_lon_lat(
            eastings, northings
        )
        self.assertIsInstance(new_lons, array_type)
        self.assertIsInstance(new_lats, array_type)

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
                a, b, check_exact=False, check_less_precise=True,
            )
        )


