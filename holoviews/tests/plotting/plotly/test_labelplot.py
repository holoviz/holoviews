import numpy as np
import pytest

from holoviews.core.options import Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Labels, Tiles
from holoviews.plotting.plotly.util import PLOTLY_MAP
from holoviews.testing import assert_data_equal

from .test_plot import TestPlotlyPlot


class TestLabelsPlot(TestPlotlyPlot):

    def test_labels_state(self):
        labels = Labels([(0, 3, 0), (1, 2, 1), (2, 1, 1)])
        state = self._get_plot_state(labels)
        assert_data_equal(state['data'][0]['x'], np.array([0, 1, 2]))
        assert_data_equal(state['data'][0]['y'], np.array([3, 2, 1]))
        assert state['data'][0]['text'] == ['0', '1', '1']
        assert state['data'][0]['mode'] == 'text'
        assert state['layout']['xaxis']['range'] == [0, 2]
        assert state['layout']['yaxis']['range'] == [1, 3]
        assert state['layout']['xaxis']['title']['text'] == 'x'
        assert state['layout']['yaxis']['title']['text'] == 'y'

    def test_labels_inverted(self):
        labels = Labels([(0, 3, 0), (1, 2, 1), (2, 1, 1)]).opts(invert_axes=True)
        state = self._get_plot_state(labels)
        assert_data_equal(state['data'][0]['x'], np.array([3, 2, 1]))
        assert_data_equal(state['data'][0]['y'], np.array([0, 1, 2]))
        assert state['data'][0]['text'] == ['0', '1', '1']
        assert state['data'][0]['mode'] == 'text'
        assert state['layout']['xaxis']['range'] == [1, 3]
        assert state['layout']['yaxis']['range'] == [0, 2]
        assert state['layout']['xaxis']['title']['text'] == 'y'
        assert state['layout']['yaxis']['title']['text'] == 'x'

    def test_labels_size(self):
        labels = Labels([(0, 3, 0), (0, 2, 1), (0, 1, 1)]).opts(size='y')
        state = self._get_plot_state(labels)
        assert_data_equal(state['data'][0]['textfont']['size'], np.array([3, 2, 1]))

    def test_labels_xoffset(self):
        labels = Labels([(0, 3, 0), (1, 2, 1), (2, 1, 1)]).opts(xoffset=0.5)
        state = self._get_plot_state(labels)
        assert_data_equal(state['data'][0]['x'], np.array([0.5, 1.5, 2.5]))

    def test_labels_yoffset(self):
        labels = Labels([(0, 3, 0), (1, 2, 1), (2, 1, 1)]).opts(yoffset=0.5)
        state = self._get_plot_state(labels)
        assert_data_equal(state['data'][0]['y'], np.array([3.5, 2.5, 1.5]))

    def test_visible(self):
        element = Labels([(0, 3, 0), (1, 2, 1), (2, 1, 1)]).opts(visible=False)
        state = self._get_plot_state(element)
        assert state['data'][0]['visible'] is False


class TestMapboxLabelsPlot(TestPlotlyPlot):

    def setUp(self):
        super().setUp()

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

    def test_labels_state(self):
        labels = Tiles("") * Labels([
            (self.xs[0], self.ys[0], 'A'),
            (self.xs[1], self.ys[1], 'B'),
            (self.xs[2], self.ys[2], 'C')
        ]).redim.range(
            x=self.x_range, y=self.y_range
        )
        state = self._get_plot_state(labels)
        assert_data_equal(state['data'][1]['lon'], self.lons)
        assert_data_equal(state['data'][1]['lat'], self.lats)
        assert state['data'][1]['text'] == ['A', 'B', 'C']
        assert state['data'][1]['mode'] == 'text'
        assert state['layout'][PLOTLY_MAP]['center'] == {
                'lat': self.lat_center, 'lon': self.lon_center
            }

    def test_labels_inverted(self):
        labels = Tiles("") * Labels([(0, 3, 0), (1, 2, 1), (2, 1, 1)]).opts(
            invert_axes=True
        )
        with pytest.raises(ValueError, match="invert_axes"):
            self._get_plot_state(labels)

    def test_labels_size(self):
        labels = Tiles("") * Labels([(0, 3, 0), (0, 2, 1), (0, 1, 1)]).opts(size=23)
        state = self._get_plot_state(labels)
        assert state['data'][1]['textfont']['size'] == 23

    def test_labels_xoffset(self):
        offset = 10000
        labels = Tiles("") * Labels([
            (self.xs[0], self.ys[0], 'A'),
            (self.xs[1], self.ys[1], 'B'),
            (self.xs[2], self.ys[2], 'C')
        ]).opts(xoffset=offset)

        state = self._get_plot_state(labels)
        lons, lats = Tiles.easting_northing_to_lon_lat(np.array(self.xs) + offset, self.ys)
        assert_data_equal(state['data'][1]['lon'], lons)
        assert_data_equal(state['data'][1]['lat'], lats)

    def test_labels_yoffset(self):
        offset = 20000
        labels = Tiles("") * Labels([
            (self.xs[0], self.ys[0], 'A'),
            (self.xs[1], self.ys[1], 'B'),
            (self.xs[2], self.ys[2], 'C')
        ]).opts(yoffset=offset)
        state = self._get_plot_state(labels)
        lons, lats = Tiles.easting_northing_to_lon_lat(self.xs, np.array(self.ys) + offset)
        assert_data_equal(state['data'][1]['lon'], lons)
        assert_data_equal(state['data'][1]['lat'], lats)

    def test_visible(self):
        element = Tiles("") * Labels([(0, 3, 0), (1, 2, 1), (2, 1, 1)]).opts(visible=False)
        state = self._get_plot_state(element)
        assert state['data'][1]['visible'] is False

    def test_labels_text_color_cycle(self):
        hm = HoloMap(
            {i: Labels([
                (0, 0 + i, "Label 1"),
                (1, 1 + i, "Label 2")
            ]) for i in range(3)}
        ).overlay()
        assert isinstance(hm[0].opts["color"], Cycle)
