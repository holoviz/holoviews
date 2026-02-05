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

    def test_heatmap_categorical_factors_preserve_appearance_and_Z_edges(self):
        """Categorical x/y should preserve first-seen order."""
        data = pd.DataFrame(
            {
                "X": ["L", "N", "O", "L", "N", "L", "N", "M"],
                "Y": ["C", "C", "C", "B", "B", "A", "A", "A"],
                "count": [301, 37, 2, 212, 8, 34, 1, 1],
            }
        )
        # Categorical/object dtypes: appearance order should be preserved
        hmap = HeatMap(data, ["X", "Y"]).aggregate(function=np.mean)
        agg = hmap.aggregate(function=np.mean).gridded
        xdim, ydim = agg.dimensions(label=True)[:2]

        Z = agg.dimension_values(2, flat=False)
        Z = np.ma.array(Z, mask=np.logical_not(np.isfinite(Z)))

        # Expected factors: first-seen order in input rows
        expected_x = ["L", "N", "O", "M"]
        expected_y = ["A", "B", "C"]

        assert list(agg.dimension_values(xdim, False)) == expected_x
        assert list(agg.dimension_values(ydim, False)) == expected_y

        # Expected Z edges match first and last y rows
        assert Z[0].tolist() == [34.0, 1.0, None, 1.0]
        assert Z[-1].tolist() == [301.0, 37.0, 2.0, None]

    def test_heatmap_categorical_factors_preserve_appearance_with_inverted_input(self):
        """Changing first-seen order in the input should change factors and Z edges accordingly."""
        inv_data = pd.DataFrame(
            {
                "X": ["M", "N", "L", "N", "L", "O", "N", "L"],
                "Y": ["A", "A", "A", "B", "B", "C", "C", "C"],
                "count": [1, 1, 34, 8, 212, 2, 37, 301],
            }
        )

        hmap = HeatMap(inv_data, ["X", "Y"]).aggregate(
            function=np.mean
        )
        agg = hmap.aggregate(function=np.mean).gridded
        xdim, ydim = agg.dimensions(label=True)[:2]

        Z = agg.dimension_values(2, flat=False)
        Z = np.ma.array(Z, mask=np.logical_not(np.isfinite(Z)))

        # Expected factors: first-seen order in input rows
        expected_x = ["M", "N", "L", "O"]
        expected_y = ["C", "B", "A"]

        assert list(agg.dimension_values(xdim, False)) == expected_x
        assert list(agg.dimension_values(ydim, False)) == expected_y

        # Expected Z edges match first and last y rows
        assert Z[0].tolist() == [301.0, 37.0, 2.0, None]
        assert Z[-1].tolist() == [34.0, 1.0, None, 1.0]

    def test_heatmap_numeric_axes_sorted_and_Z_edges(self):
        """Numeric axes should be sorted ascending; verify factors and Z edges."""
        df = pd.DataFrame(
            {
                "x": [2, 1, 3, 1],
                "y": [200, 100, 300, 100],
                "val": [ 20, 10, 30, 11],
            }
        )
        hmap = HeatMap(df, ["x", "y"]).aggregate(function=np.mean)
        agg = hmap.aggregate(function=np.mean).gridded
        xdim, ydim = agg.dimensions(label=True)[:2]

        Z = agg.dimension_values(2, flat=False)
        Z = np.ma.array(Z, mask=np.logical_not(np.isfinite(Z)))

        # Numeric factors sorted ascending
        assert list(agg.dimension_values(xdim, False)) == [1, 2, 3]
        assert list(agg.dimension_values(ydim, False)) == [100, 200, 300]

        # First row corresponds to min y=100; last row to max y=300
        # Columns correspond to x=[1,2,3]
        np.testing.assert_equal( Z[0], np.array([10.5, np.nan, np.nan]) )  # mean of (10,11)
        np.testing.assert_equal(Z[-1], np.array([np.nan, np.nan, 30]))
