from unittest import SkipTest

import numpy as np

from holoviews.core import Dimension
from holoviews.element import HexTiles
from holoviews.plotting.bokeh.hex_tiles import hex_binning
from holoviews.plotting.bokeh.util import bokeh_version

from .testplot import TestBokehPlot, bokeh_renderer


class TestHexTilesOperation(TestBokehPlot):

    def setUp(self):
        super(TestHexTilesOperation, self).setUp()
        if bokeh_version < '0.12.15':
            raise SkipTest("Bokeh >= 0.12.15 required to test HexTiles operation.")

    def test_hex_tiles_count_aggregation(self):
        tiles = HexTiles([(0, 0), (0.5, 0.5), (-0.5, -0.5), (-0.4, -0.4)])
        binned = hex_binning(tiles, gridsize=3)
        expected = HexTiles([(0, 0, 1), (2, -1, 1), (-2, 1, 2)],
                            kdims=[Dimension('x', range=(-0.5, 0.5)),
                                   Dimension('y', range=(-0.5, 0.5))],
                            vdims='Count')
        self.assertEqual(binned, expected)


    def test_hex_tiles_sum_value_aggregation(self):
        tiles = HexTiles([(0, 0, 1), (0.5, 0.5, 2), (-0.5, -0.5, 3), (-0.4, -0.4, 4)], vdims='z')
        binned = hex_binning(tiles, gridsize=3, aggregator=np.sum)
        expected = HexTiles([(0, 0, 1), (2, -1, 2), (-2, 1, 7)],
                            kdims=[Dimension('x', range=(-0.5, 0.5)),
                                   Dimension('y', range=(-0.5, 0.5))],
                            vdims='z')
        self.assertEqual(binned, expected)



class TestHexTilesPlot(TestBokehPlot):

    def setUp(self):
        super(TestHexTilesPlot, self).setUp()
        if bokeh_version < '0.12.15':
            raise SkipTest("Bokeh >= 0.12.15 required to test HexTilesPlot.")

    def test_hex_tiles_empty(self):
        tiles = HexTiles([])
        plot = list(bokeh_renderer.get_plot(tiles).subplots.values())[0]
        self.assertEqual(plot.handles['source'].data, {'q': [], 'r': []})

    def test_hex_tiles_only_nans(self):
        tiles = HexTiles([(np.NaN, 0), (1, np.NaN)])
        plot = list(bokeh_renderer.get_plot(tiles).subplots.values())[0]
        self.assertEqual(plot.handles['source'].data, {'q': [], 'r': []})

    def test_hex_tiles_zero_min_count(self):
        tiles = HexTiles([(0, 0), (0.5, 0.5), (-0.5, -0.5), (-0.4, -0.4)]).options(min_count=0)
        plot = list(bokeh_renderer.get_plot(tiles).subplots.values())[0]
        cmapper = plot.handles['color_mapper']
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(plot.state.background_fill_color, cmapper.palette[0])

    def test_hex_tiles_gridsize_tuple(self):
        tiles = HexTiles([(0, 0), (0.5, 0.5), (-0.5, -0.5), (-0.4, -0.4)]).options(gridsize=(5, 10))
        plot = list(bokeh_renderer.get_plot(tiles).subplots.values())[0]
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.size, 0.066666666666666666)
        self.assertEqual(glyph.aspect_scale, 0.5)

    def test_hex_tiles_gridsize_tuple_flat_orientation(self):
        tiles = HexTiles([(0, 0), (0.5, 0.5), (-0.5, -0.5), (-0.4, -0.4)]).options(gridsize=(5, 10), orientation='flat')
        plot = list(bokeh_renderer.get_plot(tiles).subplots.values())[0]
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.size, 0.13333333333333333)
        self.assertEqual(glyph.aspect_scale, 0.5)

    def test_hex_tiles_scale(self):
        tiles = HexTiles([(0, 0), (0.5, 0.5), (-0.5, -0.5), (-0.4, -0.4)]).options(size_index=2, gridsize=3)
        plot = list(bokeh_renderer.get_plot(tiles).subplots.values())[0]
        source = plot.handles['source']
        self.assertEqual(source.data['scale'], np.array([0.45, 0.45, 0.9]))

    def test_hex_tiles_scale_all_equal(self):
        tiles = HexTiles([(0, 0), (0.5, 0.5), (-0.5, -0.5), (-0.4, -0.4)]).options(size_index=2)
        plot = list(bokeh_renderer.get_plot(tiles).subplots.values())[0]
        source = plot.handles['source']
        self.assertEqual(source.data['scale'], np.array([0.9, 0.9, 0.9, 0.9]))

    def test_hex_tiles_hover_count(self):
        tiles = HexTiles([(0, 0), (0.5, 0.5), (-0.5, -0.5), (-0.4, -0.4)]).options(tools=['hover'])
        plot = list(bokeh_renderer.get_plot(tiles).subplots.values())[0]
        dims, opts = plot._hover_opts(tiles)
        self.assertEqual(dims, [Dimension('Count')])
        self.assertEqual(opts, {})

    def test_hex_tiles_hover_weighted(self):
        tiles = HexTiles([(0, 0, 0.1), (0.5, 0.5, 0.2), (-0.5, -0.5, 0.3)], vdims='z').options(aggregator=np.mean)
        plot = list(bokeh_renderer.get_plot(tiles).subplots.values())[0]
        dims, opts = plot._hover_opts(tiles)
        self.assertEqual(dims, [Dimension('z')])
        self.assertEqual(opts, {})
