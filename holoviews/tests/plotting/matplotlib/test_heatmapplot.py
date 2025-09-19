import numpy as np
import pandas as pd

from holoviews.element import HeatMap, Image

from .test_plot import MPL_GE_3_8_0, TestMPLPlot, mpl_renderer


class TestHeatMapPlot(TestMPLPlot):

    def test_heatmap_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]])
        hm = HeatMap(Image(arr)).opts(invert_axes=True)
        plot = mpl_renderer.get_plot(hm)
        artist = plot.handles['artist']
        if MPL_GE_3_8_0:
            np.testing.assert_equal(artist.get_array().data, arr.T[::-1])
        else:
            np.testing.assert_equal(artist.get_array().data, arr.T[::-1].flatten())

    def test_heatmap_extents(self):
        hmap = HeatMap([('A', 50, 1), ('B', 2, 2), ('C', 50, 1)])
        plot = mpl_renderer.get_plot(hmap)
        assert plot.get_extents(hmap, {}) == (-.5, -22, 2.5, 74)

    def test_heatmap_invert_xaxis(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).opts(invert_xaxis=True)
        plot = mpl_renderer.get_plot(hmap)
        array = plot.handles['artist'].get_array()
        if MPL_GE_3_8_0:
            expected = np.array([[1, np.inf], [np.inf, 2]])
        else:
            expected = np.array([1, np.inf, np.inf, 2])
        masked = np.ma.array(expected, mask=np.logical_not(np.isfinite(expected)))
        np.testing.assert_equal(array, masked)

    def test_heatmap_invert_yaxis(self):
        hmap = HeatMap([('A',1, 1), ('B', 2, 2)]).opts(invert_yaxis=True)
        plot = mpl_renderer.get_plot(hmap)
        array = plot.handles['artist'].get_array()
        expected = np.array([1, np.inf, np.inf, 2])
        if MPL_GE_3_8_0:
            expected = np.array([[1, np.inf], [np.inf, 2]])
        else:
            expected = np.array([1, np.inf, np.inf, 2])
        masked = np.ma.array(expected, mask=np.logical_not(np.isfinite(expected)))
        np.testing.assert_equal(array, masked)

    def test_heatmap_categorical_yaxis_label_order(self):
        data = pd.DataFrame({
            'depth_class': ['Shallow', 'Shallow', 'Intermediate', 'Intermediate', 'Deep', 'Deep'],
            'mag_class': ['Light', 'Strong', 'Light', 'Strong', 'Light', 'Strong'],
            'count': [100, 10, 50, 5, 20, 2]
        })
        data['depth_class'] = pd.Categorical(
            data['depth_class'],
            categories=['Shallow', 'Intermediate', 'Deep'],
            ordered=True
        )
        data['mag_class'] = pd.Categorical(
            data['mag_class'],
            categories=['Light', 'Strong'],
            ordered=True
        )

        hmap = HeatMap(data, ['mag_class', 'depth_class']).aggregate(function=np.mean)
        plot = mpl_renderer.get_plot(hmap)

        # Get y-tick labels from the plot
        _, _, axis_kwargs = plot.get_data(hmap, {}, {})
        yticks = axis_kwargs['yticks']
        ylabels = [label for _, label in yticks]

        expected_order = ['Deep', 'Intermediate', 'Shallow']
        assert ylabels == expected_order
