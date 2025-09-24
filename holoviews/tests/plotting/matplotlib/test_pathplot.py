import numpy as np
import pytest

from holoviews.core import NdOverlay
from holoviews.core.options import AbbreviatedException
from holoviews.core.spaces import HoloMap
from holoviews.element import Contours, Path, Polygons

from .test_plot import TestMPLPlot, mpl_renderer


class TestPathPlot(TestMPLPlot):

    def test_path_continuously_varying_color_op(self):
        xs = [1, 2, 3, 4]
        ys = xs[::-1]
        color = [998, 999, 998, 994]
        data = {'x': xs, 'y': ys, 'color': color}
        levels = [0, 38, 73, 95, 110, 130, 156, 999]
        colors = ['#5ebaff', '#00faf4', '#ffffcc', '#ffe775', '#ffc140', '#ff8f20', '#ff6060']
        path = Path([data], vdims='color').opts(
            color='color', color_levels=levels, cmap=colors)
        plot = mpl_renderer.get_plot(path)
        artist = plot.handles['artist']
        self.assertEqual(np.asarray(artist.get_array()), np.array(color))
        self.assertEqual(artist.get_clim(), (994, 999))

    def test_path_continuously_varying_alpha_op(self):
        xs = [1, 2, 3, 4]
        ys = xs[::-1]
        alpha = [0.1, 0.7, 0.3, 0.2]
        data = {'x': xs, 'y': ys, 'alpha': alpha}
        path = Path([data], vdims='alpha').opts(alpha='alpha')
        msg = 'ValueError: Mapping a dimension to the "alpha" style'
        with pytest.raises(AbbreviatedException, match=msg):
            mpl_renderer.get_plot(path)

    def test_path_continuously_varying_line_width_op(self):
        xs = [1, 2, 3, 4]
        ys = xs[::-1]
        line_width = [1, 7, 3, 2]
        data = {'x': xs, 'y': ys, 'line_width': line_width}
        path = Path([data], vdims='line_width').opts(linewidth='line_width')
        plot = mpl_renderer.get_plot(path)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_linewidths(), line_width)

    def test_path_continuously_varying_line_width_op_update(self):
        xs = [1, 2, 3, 4]
        ys = xs[::-1]
        path = HoloMap({
            0: Path([{'x': xs, 'y': ys, 'line_width': [1, 7, 3, 2]}], vdims='line_width'),
            1: Path([{'x': xs, 'y': ys, 'line_width': [3, 8, 2, 3]}], vdims='line_width')
        }).opts(linewidth='line_width')
        plot = mpl_renderer.get_plot(path)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_linewidths(), [1, 7, 3, 2])
        plot.update((1,))
        self.assertEqual(artist.get_linewidths(), [3, 8, 2, 3])

    def test_path_style_mapped_scalar_color_segments_array_and_cmap(self):
        # Five paths, scalar 'c' per geometry; style-mapped color
        n_pts = 3
        data = [
            {'x': np.arange(n_pts) + x,
             'y': np.arange(n_pts) + 1,
             'c': x * 10}
            for x in range(5)
        ]
        path = Path(data, vdims=['c']).opts(color='c', cmap='viridis', colorbar=True)
        plot = mpl_renderer.get_plot(path)
        artist = plot.handles['artist']

        # Must have a colormap and a data array
        assert artist.get_cmap() is not None
        arr = artist.get_array()
        assert arr is not None

        # Expect 5 * (3-1) = 10 segment values
        assert len(arr) == 10
        # Value range should reflect our scalar-per-geometry inputs (0-40)
        assert float(arr.min()) == 0.0
        assert float(arr.max()) == 40.0

    def test_path_style_mapped_per_vertex_color_segments_array(self):
        # Five paths, per-vertex 'c' values; style-mapped color
        n_pts = 7
        data = [
            {'x': np.arange(n_pts) + x,
             'y': np.arange(n_pts) + 1,
             'c': np.full(n_pts, x * 10)}  # per-vertex constant per geometry
            for x in range(5)
        ]
        path = Path(data, vdims=['c']).opts(color='c', cmap='viridis', colorbar=True)
        plot = mpl_renderer.get_plot(path)
        artist = plot.handles['artist']

        arr = artist.get_array()
        assert arr is not None
        # Expect 5 * (7-1) = 30 segment values
        assert len(arr) == 30
        # Range again spans 0-40 for these inputs
        assert float(arr.min()) == 0.0
        assert float(arr.max()) == 40.0

    def test_path_color_mapping(self):
        n_pts = 3
        data = [
            {'x': np.arange(n_pts) + x,
             'y': np.arange(n_pts) + 1,
             'c': x * 10}
            for x in range(5)
        ]
        path = Path(data, vdims=["c"]).opts(color="c", cmap="viridis", colorbar=True)

        plot = mpl_renderer.get_plot(path)
        artist = plot.handles['artist']

        # Check that the color array has the expected values
        arr = artist.get_array()
        expected = np.array([0, 0, 10, 10, 20, 20, 30, 30, 40, 40])
        np.testing.assert_array_equal(arr, expected)

        # Check that colormap is properly set
        assert artist.get_cmap() is not None
        # Check color limits span the full range
        assert artist.get_clim() == (0, 40)


class TestPolygonPlot(TestMPLPlot):

    def test_polygons_colored(self):
        polygons = NdOverlay({j: Polygons([[(i**j, i, j) for i in range(10)]], vdims='Value')
                              for j in range(5)})
        plot = mpl_renderer.get_plot(polygons)
        for j, splot in enumerate(plot.subplots.values()):
            artist = splot.handles['artist']
            self.assertEqual(np.asarray(artist.get_array()), np.array([j]))
            self.assertEqual(artist.get_clim(), (0, 4))

    def test_polygon_with_hole_plot(self):
        xs = [1, 2, 3]
        ys = [2, 0, 7]
        holes = [[[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]]]
        poly = Polygons([{'x': xs, 'y': ys, 'holes': holes}])
        plot = mpl_renderer.get_plot(poly)
        artist = plot.handles['artist']
        paths = artist.get_paths()
        self.assertEqual(len(paths), 1)
        path = paths[0]
        self.assertEqual(path.vertices, np.array([
            (1, 2), (2, 0), (3, 7), (1, 2), (1.5, 2), (2, 3), (1.6, 1.6),
            (1.5, 2), (2.1, 4.5), (2.5, 5), (2.3, 3.5), (2.1, 4.5)])
        )
        self.assertEqual(path.codes, np.array([1, 2, 2, 79, 1, 2, 2, 79, 1, 2, 2, 79]))

    def test_multi_polygon_hole_plot(self):
        xs = [1, 2, 3, np.nan, 3, 7, 6]
        ys = [2, 0, 7, np.nan, 2, 5, 7]
        holes = [
            [[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]],
            []
        ]
        poly = Polygons([{'x': xs, 'y': ys, 'holes': holes, 'value': 1}], vdims=['value'])
        plot = mpl_renderer.get_plot(poly)
        artist = plot.handles['artist']
        self.assertEqual(np.asarray(artist.get_array()), np.array([1, 1]))
        paths = artist.get_paths()
        self.assertEqual(len(paths), 2)
        path = paths[0]
        self.assertEqual(path.vertices, np.array([
            (1, 2), (2, 0), (3, 7), (1, 2), (1.5, 2), (2, 3), (1.6, 1.6),
            (1.5, 2), (2.1, 4.5), (2.5, 5), (2.3, 3.5), (2.1, 4.5)])
        )
        self.assertEqual(path.codes, np.array([1, 2, 2, 79, 1, 2, 2, 79, 1, 2, 2, 79]))
        path2 = paths[1]
        self.assertEqual(path2.vertices, np.array([(3, 2), (7, 5), (6, 7), (3, 2)]))
        self.assertEqual(path2.codes, np.array([1, 2, 2, 79]))

    def test_polygons_color_op(self):
        polygons = Polygons([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 'green'},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 'red'}
        ], vdims='color').opts(color='color')
        plot = mpl_renderer.get_plot(polygons)
        artist = plot.handles['artist']
        colors = np.array([[0. , 0.501961, 0. , 1. ],
                           [1. , 0. , 0. , 1. ]])
        self.assertEqual(artist.get_facecolors(), colors)

    def test_polygons_color_op_update(self):
        polygons = HoloMap({
            0: Polygons([
                {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 'green'},
                {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 'red'}
            ], vdims='color'),
            1: Polygons([
                {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 'blue'},
                {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 'green'}
            ], vdims='color'),
        }).opts(color='color')
        plot = mpl_renderer.get_plot(polygons)
        artist = plot.handles['artist']
        colors = np.array([[0, 0.501961, 0, 1],
                           [1, 0, 0, 1]])
        self.assertEqual(artist.get_facecolors(), colors)
        plot.update((1,))
        colors = np.array([[0, 0, 1, 1],
                           [0, 0.501961, 0, 1]])
        self.assertEqual(artist.get_facecolors(), colors)

    def test_polygons_linear_color_op(self):
        polygons = Polygons([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 7},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 3}
        ], vdims='color').opts(color='color')
        plot = mpl_renderer.get_plot(polygons)
        artist = plot.handles['artist']
        self.assertEqual(np.asarray(artist.get_array()), np.array([7, 3]))
        self.assertEqual(artist.get_clim(), (3, 7))

    def test_polygons_linear_color_op_update(self):
        polygons = HoloMap({
            0: Polygons([
                {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 7},
                {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 3}
            ], vdims='color'),
            1: Polygons([
                {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 2},
                {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 5}
            ], vdims='color'),
        }).opts(color='color', framewise=True)
        plot = mpl_renderer.get_plot(polygons)
        artist = plot.handles['artist']
        self.assertEqual(np.asarray(artist.get_array()), np.array([7, 3]))
        self.assertEqual(artist.get_clim(), (3, 7))
        plot.update((1,))
        self.assertEqual(np.asarray(artist.get_array()), np.array([2, 5]))
        self.assertEqual(artist.get_clim(), (2, 5))

    def test_polygons_categorical_color_op(self):
        polygons = Polygons([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 'b'},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 'a'}
        ], vdims='color').opts(color='color')
        plot = mpl_renderer.get_plot(polygons)
        artist = plot.handles['artist']
        self.assertEqual(np.asarray(artist.get_array()), np.array([0, 1]))
        self.assertEqual(artist.get_clim(), (0, 1))

    def test_polygons_alpha_op(self):
        polygons = Polygons([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'alpha': 0.7},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'alpha': 0.3}
        ], vdims='alpha').opts(alpha='alpha')
        msg = 'ValueError: Mapping a dimension to the "alpha" style'
        with pytest.raises(AbbreviatedException, match=msg):
            mpl_renderer.get_plot(polygons)

    def test_polygons_line_width_op(self):
        polygons = Polygons([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'line_width': 7},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'line_width': 3}
        ], vdims='line_width').opts(linewidth='line_width')
        plot = mpl_renderer.get_plot(polygons)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_linewidths(), [7, 3])



class TestContoursPlot(TestMPLPlot):

    def test_contours_categorical_color(self):
        path = Contours([{('x', 'y'): np.random.rand(10, 2), 'z': cat}
                     for cat in ('B', 'A', 'B')],
                    vdims='z').opts(color_index='z')
        plot = mpl_renderer.get_plot(path)
        artist = plot.handles['artist']
        self.assertEqual(np.asarray(artist.get_array()), np.array([0, 1, 0]))
        self.assertEqual(artist.get_clim(), (0, 1))

    def test_contours_color_op(self):
        contours = Contours([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 'green'},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 'red'}
        ], vdims='color').opts(color='color')
        plot = mpl_renderer.get_plot(contours)
        artist = plot.handles['artist']
        colors = np.array([[0. , 0.501961, 0. , 1. ],
                           [1. , 0. , 0. , 1. ]])
        self.assertEqual(artist.get_edgecolors(), colors)

    def test_contours_color_op_update(self):
        contours = HoloMap({
            0: Contours([
                {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 'green'},
                {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 'red'}
            ], vdims='color'),
            1: Contours([
                {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 'blue'},
                {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 'green'}
            ], vdims='color'),
        }).opts(color='color')
        plot = mpl_renderer.get_plot(contours)
        artist = plot.handles['artist']
        colors = np.array([[0, 0.501961, 0, 1],
                           [1, 0, 0, 1]])
        self.assertEqual(artist.get_edgecolors(), colors)
        plot.update((1,))
        colors = np.array([[0, 0, 1, 1],
                           [0, 0.501961, 0, 1]])
        self.assertEqual(artist.get_edgecolors(), colors)

    def test_contours_linear_color_op(self):
        contours = Contours([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 7},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 3}
        ], vdims='color').opts(color='color')
        plot = mpl_renderer.get_plot(contours)
        artist = plot.handles['artist']
        self.assertEqual(np.asarray(artist.get_array()), np.array([7, 3]))
        self.assertEqual(artist.get_clim(), (3, 7))

    def test_contours_linear_color_op_update(self):
        contours = HoloMap({
            0: Contours([
                {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 7},
                {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 3}
            ], vdims='color'),
            1: Contours([
                {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 2},
                {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 5}
            ], vdims='color'),
        }).opts(color='color', framewise=True)
        plot = mpl_renderer.get_plot(contours)
        artist = plot.handles['artist']
        self.assertEqual(np.asarray(artist.get_array()), np.array([7, 3]))
        self.assertEqual(artist.get_clim(), (3, 7))
        plot.update((1,))
        self.assertEqual(np.asarray(artist.get_array()), np.array([2, 5]))
        self.assertEqual(artist.get_clim(), (2, 5))

    def test_contours_categorical_color_op(self):
        contours = Contours([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 'b'},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 'a'}
        ], vdims='color').opts(color='color')
        plot = mpl_renderer.get_plot(contours)
        artist = plot.handles['artist']
        self.assertEqual(np.asarray(artist.get_array()), np.array([0, 1]))
        self.assertEqual(artist.get_clim(), (0, 1))

    def test_contours_alpha_op(self):
        contours = Contours([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'alpha': 0.7},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'alpha': 0.3}
        ], vdims='alpha').opts(alpha='alpha')
        msg = 'ValueError: Mapping a dimension to the "alpha" style'
        with pytest.raises(AbbreviatedException, match=msg):
            mpl_renderer.get_plot(contours)

    def test_contours_line_width_op(self):
        contours = Contours([
            {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'line_width': 7},
            {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'line_width': 3}
        ], vdims='line_width').opts(linewidth='line_width')
        plot = mpl_renderer.get_plot(contours)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_linewidths(), [7, 3])

    def test_contours_line_width_op_update(self):
        contours = HoloMap({
            0: Contours([
                {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'line_width': 7},
                {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'line_width': 3}
            ], vdims='line_width'),
            1: Contours([
                {('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'line_width': 2},
                {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'line_width': 5}
            ], vdims='line_width'),
        }).opts(linewidth='line_width', framewise=True)
        plot = mpl_renderer.get_plot(contours)
        artist = plot.handles['artist']
        self.assertEqual(artist.get_linewidths(), [7, 3])
        plot.update((1,))
        self.assertEqual(artist.get_linewidths(), [2, 5])
