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

    def test_simple_export_gif(self):
        data = self.renderer(self.map1, fmt='gif')[0]
        self.assertEqual(digest_data(data),
                         '913fdc6158d798a6553f5e63a65b45cc4ddbd55390f4ebb643114197f20e14f2')

    def test_simple_export_gif_double_size(self):
        data = self.renderer.instance(size=200)(self.map1, fmt='gif')[0]
        self.assertEqual(digest_data(data),
                         'aa8ba7b98fb20791a4564f8760753ebaba52ef2f0c9433fc3a7e4de9d8f51696')

    def test_simple_export_gif_half_fps(self):
        data = self.renderer.instance(fps=10)(self.map1, fmt='gif', )[0]
        self.assertEqual(digest_data(data),
                         '607fdc655296776af7e1930ae5aab1a875ba38ae20e9c2bc29da587e91fa7d17')

    def test_simple_export_png1(self):
        data = self.renderer(self.image1, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         'bed67977226262bda7e45d6d9a0172fe6a098b3b89909b954942e945a7b4aa8a')

    def test_simple_export_png1_double_size(self):
        data = self.renderer.instance(size=200)(self.image1, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         '1ff3d7dcaf7f59452a00dea0aa194f5c7f08a74a314bd727ad7c38f233ad2b7d')

    def test_simple_export_png2(self):
        data = self.renderer(self.image2, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         '10b4eb2d044e7a4f5da4a66e1ca32f356f20ae15d19d20ec2354359439921ac3')

    def test_simple_export_png2_double_size(self):
        data = self.renderer.instance(size=200)(self.image2, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         '9a249e03f6da99afc1fee826acc1a96a76448d571bdebaf31a8d3483327bcee9')


if __name__ == "__main__":
    import sys
    import nose
    nose.runmodule(argv=[sys.argv[0], "--logging-level", "ERROR"])
