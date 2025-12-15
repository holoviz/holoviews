import numpy as np
import pytest

from holoviews.element import Curve, Tiles
from holoviews.plotting.plotly.util import PLOTLY_MAP
from holoviews.testing import assert_data_equal

from .test_plot import TestPlotlyPlot


class TestCurvePlot(TestPlotlyPlot):

    def test_curve_state(self):
        curve = Curve([1, 2, 3])
        state = self._get_plot_state(curve)
        assert_data_equal(state['data'][0]['x'], np.array([0, 1, 2]))
        assert_data_equal(state['data'][0]['y'], np.array([1, 2, 3]))
        assert state['data'][0]['mode'] == 'lines'
        assert state['layout']['xaxis']['range'] == [0, 2]
        assert state['layout']['yaxis']['range'] == [1, 3]

    def test_curve_inverted(self):
        curve = Curve([1, 2, 3]).opts(invert_axes=True)
        state = self._get_plot_state(curve)
        assert_data_equal(state['data'][0]['x'], np.array([1, 2, 3]))
        assert_data_equal(state['data'][0]['y'], np.array([0, 1, 2]))
        assert state['data'][0]['mode'] == 'lines'
        assert state['layout']['xaxis']['range'] == [1, 3]
        assert state['layout']['yaxis']['range'] == [0, 2]
        assert state['layout']['xaxis']['title']['text'] == 'y'
        assert state['layout']['yaxis']['title']['text'] == 'x'

    def test_curve_interpolation(self):
        curve = Curve([1, 2, 3]).opts(interpolation='steps-mid')
        state = self._get_plot_state(curve)
        assert_data_equal(state['data'][0]['x'], np.array([0., 0.5, 0.5, 1.5, 1.5, 2.]))
        assert_data_equal(state['data'][0]['y'], np.array([1, 1, 2, 2, 3, 3]))

    def test_curve_color(self):
        curve = Curve([1, 2, 3]).opts(color='red')
        state = self._get_plot_state(curve)
        assert state['data'][0]['line']['color'] == 'red'

    def test_curve_color_mapping_error(self):
        curve = Curve([1, 2, 3]).opts(color='x')
        with pytest.raises(ValueError):  # noqa: PT011
            self._get_plot_state(curve)

    def test_curve_dash(self):
        curve = Curve([1, 2, 3]).opts(dash='dash')
        state = self._get_plot_state(curve)
        assert state['data'][0]['line']['dash'] == 'dash'

    def test_curve_line_width(self):
        curve = Curve([1, 2, 3]).opts(line_width=5)
        state = self._get_plot_state(curve)
        assert state['data'][0]['line']['width'] == 5

    def test_visible(self):
        element = Curve([1, 2, 3]).opts(visible=False)
        state = self._get_plot_state(element)
        assert state['data'][0]['visible'] is False


class TestMapboxCurvePlot(TestPlotlyPlot):

    def setup_method(self):
        super().setup_method()

        # Precompute coordinates
        self.xs = [3000000, 2000000, 1000000]
        self.ys = [-3000000, -2000000, -1000000]
        self.x_range = (-5000000, 4000000)
        self.x_center = sum(self.x_range) / 2.0
        self.y_range = (-3000000, 2000000)
        self.y_center = sum(self.y_range) / 2.0
        self.lon_centers, self.lat_centers = Tiles.easting_northing_to_lon_lat(
            [self.x_center], [self.y_center]
        )
        self.lon_center, self.lat_center = self.lon_centers[0], self.lat_centers[0]
        self.lons, self.lats = Tiles.easting_northing_to_lon_lat(self.xs, self.ys)

    def test_curve_state(self):
        curve = Tiles("") * Curve((self.xs, self.ys)).redim.range(
            x=self.x_range, y=self.y_range
        )
        state = self._get_plot_state(curve)
        assert_data_equal(state['data'][1]['lon'], self.lons)
        assert_data_equal(state['data'][1]['lat'], self.lats)
        assert state['data'][1]['mode'] == 'lines'
        assert state['layout'][PLOTLY_MAP]['center'] == {
                'lat': self.lat_center, 'lon': self.lon_center
            }

    def test_curve_inverted(self):
        curve = Tiles("") * Curve([1, 2, 3]).opts(invert_axes=True)
        with pytest.raises(ValueError, match="invert_axes"):
            self._get_plot_state(curve)

    def test_curve_interpolation(self):
        from holoviews.operation import interpolate_curve
        interp_xs = np.array([0., 0.5, 0.5, 1.5, 1.5, 2.])
        interp_curve = interpolate_curve(Curve(self.ys), interpolation='steps-mid')
        interp_ys = interp_curve.dimension_values("y")
        interp_lons, interp_lats = Tiles.easting_northing_to_lon_lat(interp_xs, interp_ys)

        curve = Tiles("") * Curve(self.ys).opts(interpolation='steps-mid')
        state = self._get_plot_state(curve)
        assert_data_equal(state['data'][1]['lat'], interp_lats)
        assert_data_equal(state['data'][1]['lon'], interp_lons)

    def test_curve_color(self):
        curve = Tiles("") * Curve([1, 2, 3]).opts(color='red')
        state = self._get_plot_state(curve)
        assert state['data'][1]['line']['color'] == 'red'

    def test_curve_color_mapping_error(self):
        curve = Tiles("") * Curve([1, 2, 3]).opts(color='x')
        with pytest.raises(ValueError):  # noqa: PT011
            self._get_plot_state(curve)

    def test_curve_dash(self):
        curve = Tiles("") * Curve([1, 2, 3]).opts(dash='dash')
        with pytest.raises(ValueError, match="dash"):
            self._get_plot_state(curve)

    def test_curve_line_width(self):
        curve = Tiles("") * Curve([1, 2, 3]).opts(line_width=5)
        state = self._get_plot_state(curve)
        assert state['data'][1]['line']['width'] == 5

    def test_visible(self):
        element = Tiles("") * Curve([1, 2, 3]).opts(visible=False)
        state = self._get_plot_state(element)
        assert state['data'][1]['visible'] is False
