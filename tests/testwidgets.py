"""
Test cases for the HTML/JavaScript scrubber and widgets.
"""
import re
from hashlib import sha256
from unittest import SkipTest
import numpy as np

try:
    from holoviews.ipython import IPTestCase
    from holoviews.ipython.widgets import ScrubberWidget, SelectionWidget
    # Standardize backend due to random inconsistencies
    from matplotlib import pyplot
    pyplot.switch_backend('agg')
except:
    raise SkipTest("Matplotlib required to test widgets")

from holoviews import Image, HoloMap
from holoviews.plotting import RasterPlot

def digest_data(data):
    hashfn = sha256()
    hashfn.update(data.encode('utf-16'))
    return hashfn.hexdigest()


prefixes =  ['anim', '_anim_slider', '_anim_img',
             '_anim_loop_select', 'textInput', '_anim_widget', 'valMap']
filters  = [re.compile('{p}[a-f0-9]+'.format(p=p)) for p in prefixes]
filters += [re.compile('new Animation\([a-z0-9_, "]+\)')]
filters += [re.compile('new NDSlider\([a-z0-9_, "]+')]

def normalize(data):
    for f in filters:
        data = re.sub(f, '[CLEARED]', data)
    # Hack around inconsistencies in jinja between Python 2 and 3
    return data.replace('0.0', '0').replace('1.0', '1')

class TestWidgets(IPTestCase):

    def setUp(self):
        super(TestWidgets, self).setUp()
        im1 = Image(np.array([[1,2],[3,4]]))
        im2 = Image(np.array([[1,2],[3,5]]))
        holomap = HoloMap(initial_items=[(0,im1), (1,im2)], key_dimensions=['test'])
        self.plot1 = RasterPlot(im1)
        self.plot2 = RasterPlot(holomap)

    def tearDown(self):
        super(TestWidgets, self).tearDown()

    def test_scrubber_widget_1(self):
        html = normalize(ScrubberWidget(self.plot1)())
        self.assertEqual(digest_data(html), '7d308d9094ef188b34e69690b3ddc6234c66ca9b61fbdeb16015f0cc3dc688ce')

    def test_selection_widget_1(self):
        html = normalize(SelectionWidget(self.plot1)())
        self.assertEqual(digest_data(html), '6c30f5e2b11562d82aa43205221f8e870c3f5a524f39c47520b15a8a416968c9')

    def test_scrubber_widget_2(self):
        html = normalize(ScrubberWidget(self.plot2)())
        self.assertEqual(digest_data(html), '15009ab21807dcd9b2a934c16f03c64047d7a2799a2cd354860cda825b01f049')

    def test_selection_widget_2(self):
        html = normalize(SelectionWidget(self.plot2)())
        self.assertEqual(digest_data(html), '69d073f94cc22c8086453a607a2fd71841ed9ed34e7b32978e230216f41ed257')
