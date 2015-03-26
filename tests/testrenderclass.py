from hashlib import sha256
from unittest import SkipTest
import numpy as np

from holoviews import HoloMap, Store
from holoviews.element import Image
from holoviews.element.comparison import ComparisonTestCase

from nose.plugins.attrib import attr

try:
    # Standardize backend due to random inconsistencies
    from matplotlib import pyplot
    pyplot.switch_backend('agg')
except:
    pyplot = None

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
        if pyplot is None:
            raise SkipTest("Matplotlib required to test widgets")

        self.basename = 'no-file'
        self.image1 = Image(np.array([[0,1],[2,3]]), label='Image1')
        self.image2 = Image(np.array([[1,0],[4,-2]]), label='Image2')
        self.map1 = HoloMap({1:self.image1, 2:self.image2}, label='TestMap')

        self.renderer = Store.renderer.instance()

    def test_simple_export_gif(self):
        data = self.renderer(self.map1, fmt='gif')[0]
        self.assertEqual(digest_data(data),
                         'dd61926210f18b527cfa0ee29d47179f24e06325c1f9abde52b84a58217a9844')

    def test_simple_export_gif_double_size(self):
        data = self.renderer.instance(size=200)(self.map1, fmt='gif')[0]
        self.assertEqual(digest_data(data),
                         '3d5921cebd6601e524fbd620ce7d6f607ec74415cc8af08b1b82b3fa10a5f874')

    def test_simple_export_gif_half_fps(self):
        data = self.renderer.instance(fps=10)(self.map1, fmt='gif', )[0]
        self.assertEqual(digest_data(data),
                         'f71b5aa9e001e22d502e6c4ad0e9fc4aea5c04cb2d1a68f2f8a4296e563107e1')

    def test_simple_export_png1(self):
        data = self.renderer(self.image1, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         'ed1930b4d25870b0630e030e607bb6e487952e576ebd0a1a7232ca72ef7cf50c')

    def test_simple_export_png1_double_size(self):
        data = self.renderer.instance(size=200)(self.image1, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         '10ad6d8969aa12d51e17c30244044264bab566c4bb5a2e900ace937fa48c2105')

    def test_simple_export_png2(self):
        data = self.renderer(self.image2, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         '16bb9117d5cc0959167264a9183179641318f5f8ae269ccdab66df9b2b5b40d7')

    def test_simple_export_png2_double_size(self):
        data = self.renderer.instance(size=200)(self.image2, fmt='png')[0]
        self.assertEqual(digest_data(data),
                         '1fa233a601bc7942031e434c20253a8551639bd8cf574440ea9b4485a185c2a1')


if __name__ == "__main__":
    import sys
    import nose
    nose.runmodule(argv=[sys.argv[0], "--logging-level", "ERROR"])
