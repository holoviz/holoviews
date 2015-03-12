from hashlib import sha256
import numpy as np

from holoviews import HoloMap, Store
from holoviews.element import Image
from holoviews.element.comparison import ComparisonTestCase

from nose.plugins.attrib import attr

def digest_data(data):
    hashfn = sha256()
    hashfn.update(data)
    return hashfn.hexdigest()


@attr(optional=1)
class MPLPlotRendererTest(ComparisonTestCase):
    """
    Note if not possible to compare the hashes of SVG and WebM formats
    as the hashes are not stable across exports.
    """

    def setUp(self):
        self.basename = 'no-file'
        self.image1 = Image(np.array([[0,1],[2,3]]), label='Image1')
        self.image2 = Image(np.array([[1,0],[4,-2]]), label='Image2')
        self.map1 = HoloMap({1:self.image1, 2:self.image2}, label='TestMap')

        self.renderer = Store.PlotRenderer.instance()

    def test_simple_export_png1(self):
        data = self.renderer(self.image1, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         'cb32c2cef6a451de84fb7bce53f301338c55d9b08d96c978fbdc89f3e4df967d')

    def test_simple_export_png2(self):
        data = self.renderer(self.image2, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         'ba937c2b3ce837e81afac6cf3d8c0fd2f32cd7c0ab220b2d036ed5452de8136d')

    def test_simple_export_gif(self):
        data = self.renderer(self.map1, fmt='gif')[0]
        self.assertEqual(digest_data(data),
                         'ab1286fa02e964b2bbd216874eb3264eb5dee33b4389ff53c436621194241277')

    def test_simple_export_png1_double_size(self):
        data = self.renderer.instance(size=200)(self.image1, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         '510f16e3ba1b30b9b91255fa3f7c3952e59bec0ee0636b2a8ce439a3700f08f5')

    def test_simple_export_gif_double_size(self):
        data = self.renderer.instance(size=200)(self.map1, fmt='gif')[0]
        self.assertEqual(digest_data(data),
                         '62fe0be75691322737207459148c90fed02e9376451f0b610404ce62fc683b15')

    def test_simple_export_gif_half_fps(self):
        data = self.renderer.instance(fps=10)(self.map1, fmt='gif', )[0]
        self.assertEqual(digest_data(data),
                         'fb38af88cf54bdb974e5615b89c0549421c6e0a517c0aa68dda982cf7d4875a6')

    def test_simple_export_png2_double_size(self):
        data = self.renderer.instance(size=200)(self.image2, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         '04ce50b930e9007ab6f910f0a0c0fa408174e7a1668287e92b59eb9cee5e6b4c')



if __name__ == "__main__":
    import sys
    import nose
    nose.runmodule(argv=[sys.argv[0], "--logging-level", "ERROR"])
