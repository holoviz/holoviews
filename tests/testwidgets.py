"""
Test cases for the HTML/JavaScript scrubber and widgets.
"""
import re
from hashlib import sha256
from unittest import SkipTest
import numpy as np

try:
    from holoviews.ipython import IPTestCase
    from holoviews.plotting.mpl.widgets import ScrubberWidget, SelectionWidget
    # Standardize backend due to random inconsistencies
    from matplotlib import pyplot
    pyplot.switch_backend('agg')
except:
    raise SkipTest("Matplotlib required to test widgets")

from holoviews import Image, HoloMap
from holoviews.plotting.mpl import RasterPlot

def digest_data(data):
    hashfn = sha256()
    hashfn.update(data.encode('utf-16'))
    return hashfn.hexdigest()


prefixes =  ['anim', '_anim_slider', '_anim_img',
             '_anim_loop_select', 'textInput', '_anim_widget', 'valMap']
filters  = [re.compile('{p}[a-f0-9]+'.format(p=p)) for p in prefixes]
filters += [re.compile('new ScrubberWidget\([a-z0-9_, "]+')]
filters += [re.compile('new SelectionWidget\([a-z0-9_, "]+')]

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
        html = normalize(ScrubberWidget(self.plot1, display_options={'figure_format': 'png'})())
        self.assertEqual(digest_data(html), 'b74ffdc42d67a75e1e4aeb54166ccd1bfaa2695428ae7413871b47ed5fc87a4c')

    def test_selection_widget_1(self):
        html = normalize(SelectionWidget(self.plot1, display_options={'figure_format': 'png'})())
        self.assertEqual(digest_data(html), 'ea5f9640210e5512c3ef541cdac621ac7ab053dab04c975db4914cec2a57ab35')

    def test_scrubber_widget_2(self):
        html = normalize(ScrubberWidget(self.plot2, display_options={'figure_format': 'png'})())
        self.assertEqual(digest_data(html), '162d3ee2cea1c859ab2844ee1a2022d24d2a9075ddf223bb0d6c46adadddc14b')

    def test_selection_widget_2(self):
        html = normalize(SelectionWidget(self.plot2, display_options={'figure_format': 'png'})())
        self.assertEqual(digest_data(html), '00a47cbd1330123e6e1e95e101e183b51c67b2bff8bc4f14f7c4b5d1e5edfbf5')
