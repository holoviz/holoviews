import numpy as np

from holoviews.element import Scatter, Tiles
from holoviews.plotting.plotly.util import PLOTLY_MAP, PLOTLY_SCATTERMAP
from holoviews.testing import assert_data_equal

from .test_plot import TestPlotlyPlot


class TestScatterPlot(TestPlotlyPlot):

    def test_scatter_state(self):
        scatter = Scatter([3, 2, 1])
        state = self._get_plot_state(scatter)
        assert state['data'][0]['type'] == 'scatter'
        assert_data_equal(state['data'][0]['y'], np.array([3, 2, 1]))
        assert state['data'][0]['mode'] == 'markers'
        assert state['layout']['yaxis']['range'] == [1, 3]

    def test_scatter_inverted(self):
        scatter = Scatter([1, 2, 3]).opts(invert_axes=True)
        state = self._get_plot_state(scatter)
        assert_data_equal(state['data'][0]['x'], np.array([1, 2, 3]))
        assert_data_equal(state['data'][0]['y'], np.array([0, 1, 2]))
        assert state['data'][0]['mode'] == 'markers'
        assert state['layout']['xaxis']['range'] == [1, 3]
        assert state['layout']['yaxis']['range'] == [0, 2]
        assert state['layout']['xaxis']['title']['text'] == 'y'
        assert state['layout']['yaxis']['title']['text'] == 'x'

    def test_scatter_color_mapped(self):
        scatter = Scatter([3, 2, 1]).opts(color='x')
        state = self._get_plot_state(scatter)
        assert_data_equal(state['data'][0]['marker']['color'], np.array([0, 1, 2]))
        assert state['data'][0]['marker']['cmin'] == 0
        assert state['data'][0]['marker']['cmax'] == 2

    def test_scatter_size(self):
        scatter = Scatter([3, 2, 1]).opts(size='y')
        state = self._get_plot_state(scatter)
        assert_data_equal(state['data'][0]['marker']['size'], np.array([3, 2, 1]))

    def test_scatter_colors(self):
        scatter = Scatter([
            (0, 1, 'red'), (1, 2, 'green'), (2, 3, 'blue')
        ], vdims=['y', 'color']).opts(color='color')
        state = self._get_plot_state(scatter)
        assert np.array_equal(state['data'][0]['marker']['color'],
                                        np.array(['red', 'green', 'blue'])) is True

    def test_scatter_categorical_color(self):
        scatter = Scatter([
            (0, 1, 'A'), (1, 2, 'B'), (2, 3, 'C')
        ], vdims=['y', 'category']).opts(color='category')
        state = self._get_plot_state(scatter)

        np.testing.assert_array_equal(state['data'][0]['marker']['color'], [0, 1, 2])
        assert 'colorscale' in state['data'][0]['marker']
        assert state['data'][0]['marker']['cmin'] == 0
        assert state['data'][0]['marker']['cmax'] == 2

    def test_scatter_markers(self):
        scatter = Scatter([
            (0, 1, 'square'), (1, 2, 'circle'), (2, 3, 'triangle-up')
        ], vdims=['y', 'marker']).opts(marker='marker')
        state = self._get_plot_state(scatter)
        assert np.array_equal(state['data'][0]['marker']['symbol'],
                                        np.array(['square', 'circle', 'triangle-up'])) is True

    def test_scatter_selectedpoints(self):
        scatter = Scatter([
            (0, 1,), (1, 2), (2, 3)
        ]).opts(selectedpoints=[1, 2])
        state = self._get_plot_state(scatter)
        assert state['data'][0]['selectedpoints'] == [1, 2]

    def test_visible(self):
        element = Scatter([3, 2, 1]).opts(visible=False)
        state = self._get_plot_state(element)
        assert state['data'][0]['visible'] is False


class TestMapboxScatterPlot(TestPlotlyPlot):
    def test_scatter_state(self):
        # Precompute coordinates
        xs = [3000000, 2000000, 1000000]
        ys = [-3000000, -2000000, -1000000]
        x_range = (-5000000, 4000000)
        x_center = sum(x_range) / 2.0
        y_range = (-3000000, 2000000)
        y_center = sum(y_range) / 2.0
        lon_centers, lat_centers = Tiles.easting_northing_to_lon_lat([x_center], [y_center])
        lon_center, lat_center = lon_centers[0], lat_centers[0]
        lons, lats = Tiles.easting_northing_to_lon_lat(xs, ys)

        scatter = Tiles('') * Scatter((xs, ys)).redim.range(x=x_range, y=y_range)
        state = self._get_plot_state(scatter)
        assert state['data'][1]['type'] == PLOTLY_SCATTERMAP
        assert_data_equal(state['data'][1]['lon'], lons)
        assert_data_equal(state['data'][1]['lat'], lats)
        assert state['data'][1]['mode'] == 'markers'
        assert state['layout'][PLOTLY_MAP]['center'] == {'lat': lat_center, 'lon': lon_center}

        # There xaxis and yaxis should not be in the layout
        assert 'xaxis' not in state['layout']
        assert 'yaxis' not in state['layout']

    def test_scatter_color_mapped(self):
        scatter = Tiles('') * Scatter([3, 2, 1]).opts(color='x')
        state = self._get_plot_state(scatter)
        assert_data_equal(state['data'][1]['marker']['color'], np.array([0, 1, 2]))
        assert state['data'][1]['marker']['cmin'] == 0
        assert state['data'][1]['marker']['cmax'] == 2

    def test_scatter_size(self):
        # size values should not go through meters-to-lnglat conversion
        scatter = Tiles('') * Scatter([3, 2, 1]).opts(size='y')
        state = self._get_plot_state(scatter)
        assert_data_equal(state['data'][1]['marker']['size'], np.array([3, 2, 1]))

    def test_scatter_colors(self):
        scatter = Tiles('') * Scatter([
            (0, 1, 'red'), (1, 2, 'green'), (2, 3, 'blue')
        ], vdims=['y', 'color']).opts(color='color')
        state = self._get_plot_state(scatter)
        assert np.array_equal(state['data'][1]['marker']['color'],
                                        np.array(['red', 'green', 'blue'])) is True

    def test_scatter_categorical_color(self):
        scatter = Tiles('') * Scatter([
            (0, 1, 'A'), (1, 2, 'B'), (2, 3, 'C')
        ], vdims=['y', 'category']).opts(color='category')
        state = self._get_plot_state(scatter)

        np.testing.assert_array_equal(state['data'][1]['marker']['color'], [0, 1, 2])
        assert 'colorscale' in state['data'][1]['marker']
        assert state['data'][1]['marker']['cmin'] == 0
        assert state['data'][1]['marker']['cmax'] == 2


    def test_scatter_markers(self):
        scatter = Tiles('') * Scatter([
            (0, 1, 'square'), (1, 2, 'circle'), (2, 3, 'triangle-up')
        ], vdims=['y', 'marker']).opts(marker='marker')
        state = self._get_plot_state(scatter)
        assert np.array_equal(
                state['data'][1]['marker']['symbol'],
                np.array(['square', 'circle', 'triangle-up'])) is True

    def test_scatter_selectedpoints(self):
        scatter = Tiles('') * Scatter([
            (0, 1,), (1, 2), (2, 3)
        ]).opts(selectedpoints=[1, 2])
        state = self._get_plot_state(scatter)
        assert state['data'][1]['selectedpoints'] == [1, 2]

    def test_visible(self):
        element = Tiles('') * Scatter([3, 2, 1]).opts(visible=False)
        state = self._get_plot_state(element)
        assert state['data'][1]['visible'] is False
