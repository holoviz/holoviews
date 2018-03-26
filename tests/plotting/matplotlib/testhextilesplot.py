import numpy as np

from holoviews.element import HexTiles

from .testplot import TestMPLPlot, mpl_renderer


class TestHexTilesPlot(TestMPLPlot):

    def test_hex_tiles_empty(self):
        tiles = HexTiles([])
        mpl_renderer.get_plot(tiles)

    def test_hex_tiles_opts(self):
        tiles = HexTiles([(0, 0), (0.5, 0.5), (-0.5, -0.5), (-0.4, -0.4)])
        plot = mpl_renderer.get_plot(tiles)
        args, style, _ = plot.get_data(tiles, {}, {})
        self.assertEqual(args[0], tiles['x'])
        self.assertEqual(args[1], tiles['y'])
        self.assertEqual(args[2], np.ones(4))
        self.assertEqual(style, {'reduce_C_function': np.sum,
                                 'vmin': None, 'vmax': None,
                                 'xscale': 'linear', 'yscale': 'linear',
                                 'gridsize': 50, 'mincnt': None,
                                 'cmap': None})
