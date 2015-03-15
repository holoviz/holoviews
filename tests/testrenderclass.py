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

        self.renderer = Store.renderer.instance()

    def test_simple_export_png1(self):
        data = self.renderer(self.image1, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         '11a5a9ca23200b4cb32914779fc25d701d7c2fe074e729dc05eaad46d160a192')

    def test_simple_export_png2(self):
        data = self.renderer(self.image2, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         '0ea1a4fbc69999363560a0f70a03da22fc7bd8698808714975430c931be61865')

    def test_simple_export_gif(self):
        data = self.renderer(self.map1, fmt='gif')[0]
        self.assertEqual(digest_data(data),
                         '4057abe74cb508f6666da39b0dfd117665270bd85f8fc3257b583e2d00527dc5')

    def test_simple_export_png1_double_size(self):
        data = self.renderer.instance(size=200)(self.image1, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         '7f99e5dd9d1b946b95c85aef1bae62c1b293799fc192391f0e6445a36b2599b3')

    def test_simple_export_gif_double_size(self):
        data = self.renderer.instance(size=200)(self.map1, fmt='gif')[0]
        self.assertEqual(digest_data(data),
                         '40f99783a29c3eeb172c8f035c810f32c7bf294a8cbcf4ea5c665e0640d76119')

    def test_simple_export_gif_half_fps(self):
        data = self.renderer.instance(fps=10)(self.map1, fmt='gif', )[0]
        self.assertEqual(digest_data(data),
                         '461f94c6c9dc85c8cb594f949a091ead0602710fc1bf224a1bd7f664712bf5b0')

    def test_simple_export_png2_double_size(self):
        data = self.renderer.instance(size=200)(self.image2, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         'fdfde1c69bd11038529ce18e18862b45818dae742a6165f476ed8a0080adf138')



if __name__ == "__main__":
    import sys
    import nose
    nose.runmodule(argv=[sys.argv[0], "--logging-level", "ERROR"])
