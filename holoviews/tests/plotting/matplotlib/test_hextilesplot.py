import numpy as np

from holoviews.element import HexTiles
from holoviews.testing import assert_data_equal

from .test_plot import TestMPLPlot, mpl_renderer


class TestHexTilesPlot(TestMPLPlot):

    def test_hex_tiles_empty(self):
        tiles = HexTiles([])
        mpl_renderer.get_plot(tiles)

    def test_hex_tiles_opts(self):
        from holoviews.plotting.util import process_cmap
        tiles = HexTiles([(0, 0), (0.5, 0.5), (-0.5, -0.5), (-0.4, -0.4)])
        plot = mpl_renderer.get_plot(tiles)
        args, style, _ = plot.get_data(tiles, {}, {})
        assert_data_equal(args[0], tiles['x'])
        assert_data_equal(args[1], tiles['y'])
        assert_data_equal(args[2], np.ones(4))
        cmap = style.pop('cmap')
        assert style == {'reduce_C_function': np.sum,
                                 'vmin': None, 'vmax': None,
                                 'xscale': 'linear', 'yscale': 'linear',
                                 'gridsize': 50, 'mincnt': None}
        assert cmap.colors == process_cmap('viridis')
