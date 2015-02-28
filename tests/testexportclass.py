import sys
from hashlib import sha256
import numpy as np
from io import BytesIO

from holoviews.element import Image
from holoviews.plotting import Export
from holoviews.core.options import SaveOptions
from holoviews.element.comparison import ComparisonTestCase


def digest_data(data):
    hashfn = sha256()
    hashfn.update(data)
    return hashfn.hexdigest()


Export.capture_mode = 1 # Use mode 2 to see the saved files, 1 otherwise (capture only)

# If capture_mode is 2, output will be in directory: 2000_01_01-00_00_00
Export.save_options = SaveOptions(time=(0,0,0,0,0,0,0,0,0),
                                  directory = '{timestamp}',
                                  timestamp_format = "%Y_%m_%d-%H_%M_%S")

class ExportTest(ComparisonTestCase):
    """
    Note if not possible to compare the hashes of SVG and WebM formats
    as the hashes are not stable across exports.
    """

    def setUp(self):
        self.image1 = Image(np.array([[0,1],[2,3]]))
        self.image2 = Image(np.array([[1,0],[4,-2]]))

    def test_simple_export_png1(self):
        Export.save(self.image1, fmt='png')
        self.assertEqual(digest_data(Export.captured_data),
                         'a2daf8b589508d3d305f66b2b0c3c606396f0e5feb3c7eff60880bc3f1157c59')

    def test_simple_export_png2(self):
        Export.save(self.image2, fmt='png')
        self.assertEqual(digest_data(Export.captured_data),
                        'e47fca35d069cdbf5ab0f3337c408ed7ed3e366baba7ec650abb8711394e8b1d')



if __name__ == "__main__":
    import sys
    import nose
    nose.runmodule(argv=[sys.argv[0], "--logging-level", "ERROR"])
