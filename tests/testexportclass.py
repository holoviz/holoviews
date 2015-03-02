import sys
from hashlib import sha256
import numpy as np
from io import BytesIO

from holoviews import HoloMap
from holoviews.element import Image
from holoviews.plotting import Export
from holoviews.core.options import SaveOptions
from holoviews.element.comparison import ComparisonTestCase

from nose.plugins.attrib import attr

def digest_data(data):
    hashfn = sha256()
    hashfn.update(data)
    return hashfn.hexdigest()


Export.capture_mode = 1 # Use mode 2 to see the saved files, 1 otherwise (capture only)

# If capture_mode is 2, output will be in directory: 2000_01_01-00_00_00
Export.save_options = SaveOptions(time=(0,0,0,0,0,0,0,0,0),
                                  directory = '{timestamp}',
                                  timestamp_format = "%Y_%m_%d-%H_%M_%S")

@attr(optional=1)
class ExportTest(ComparisonTestCase):
    """
    Note if not possible to compare the hashes of SVG and WebM formats
    as the hashes are not stable across exports.
    """

    def setUp(self):
        self.image1 = Image(np.array([[0,1],[2,3]]), label='Image1')
        self.image2 = Image(np.array([[1,0],[4,-2]]), label='Image2')
        self.map1 = HoloMap({1:self.image1, 2:self.image2}, label='TestMap')

    def test_simple_export_png1(self):
        Export.save(self.image1, fmt='png')
        self.assertEqual(digest_data(Export.captured_data),
                         '31a9af4a57492b79814a9922a573ca550358d3cce21933fc4cfa5de490037416')

    def test_simple_export_png2(self):
        Export.save(self.image2, fmt='png')
        self.assertEqual(digest_data(Export.captured_data),
                         '9eed2badfc9ff203b6f34e22e3d97845dff588a502eb0e82befb30f88584028b')

    def test_simple_export_gif(self):
        Export.save(self.map1, fmt='gif')
        self.assertEqual(digest_data(Export.captured_data),
                         '44fee52c887a6a3a4116cf6d9a0a0f4451849e1aaef1d03a9a1a6c098681771b')

    def test_simple_export_png1_double_size(self):
        Export.save(self.image1, fmt='png', size=200)
        self.assertEqual(digest_data(Export.captured_data),
                         '577a2f8eb63cd5189a536895bd4f8f2d358b901ee808d2b2c976dffac14e27f4')

    def test_simple_export_gif_double_size(self):
        Export.save(self.map1, fmt='gif', size=200)
        self.assertEqual(digest_data(Export.captured_data),
                         'e73cb85ecf8876eef4bd22caf86685e97b7f6d990da16d93a36cc215a0c4072f')

    def test_simple_export_gif_half_fps(self):
        Export.save(self.map1, fmt='gif', fps=10)
        self.assertEqual(digest_data(Export.captured_data),
                         '1f329070e9d2c70f1fc6e4254ff3c346887079c2d09e951d1742e81f3d7815ba')

    def test_simple_export_png1_double_size(self):
        Export.save(self.image1, fmt='png', size=200)
        self.assertEqual(digest_data(Export.captured_data),
                         '577a2f8eb63cd5189a536895bd4f8f2d358b901ee808d2b2c976dffac14e27f4')



if __name__ == "__main__":
    import sys
    import nose
    nose.runmodule(argv=[sys.argv[0], "--logging-level", "ERROR"])
