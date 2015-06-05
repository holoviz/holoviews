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
        holomap = HoloMap(initial_items=[(0,im1), (1,im2)], kdims=['test'])
        self.plot1 = RasterPlot(im1)
        self.plot2 = RasterPlot(holomap)

    def tearDown(self):
        super(TestWidgets, self).tearDown()

    def test_scrubber_widget_1(self):
        html = normalize(ScrubberWidget(self.plot1)())
        self.assertEqual(digest_data(html), '007089a2f85fdd8fee3d079dd68e79ec3f03fb10aec552a499b6c8af09d1f3e0')

    def test_selection_widget_1(self):
        html = normalize(SelectionWidget(self.plot1)())
        self.assertEqual(digest_data(html), '8ebe145a62f14cf3ae8c5134362c81d31fef993aa0a98d593af8ea8900e0cbeb')

    def test_scrubber_widget_2(self):
        html = normalize(ScrubberWidget(self.plot2)())
        self.assertEqual(digest_data(html), '515758cdb7a74080327f8c63d9e81d39afd5fe6a86d9af3be7f136b34ff7b50d')

    def test_selection_widget_2(self):
        html = normalize(SelectionWidget(self.plot2)())
        self.assertEqual(digest_data(html), '5391e6b62b73e6347b451581d364af94a5cf32f6c5ce3a1e822590c6aa61c208')
