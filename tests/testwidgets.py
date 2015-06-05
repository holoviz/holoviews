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
        self.assertEqual(digest_data(html), 'edcef95594bad9dda088cf883ee09f89331ddc2439c9b7751b589a125bc919ba')

    def test_selection_widget_1(self):
        html = normalize(SelectionWidget(self.plot1)())
        self.assertEqual(digest_data(html), 'a906b151cfd9b5b46323720f5061f3ddd6ca3da6f68e1c471e0c445502f79735')

    def test_scrubber_widget_2(self):
        html = normalize(ScrubberWidget(self.plot2)())
        self.assertEqual(digest_data(html), '1fd80b4b3a5eb0ab5888f8aab509373cfa2c6b96c94601e56744c339c8664981')

    def test_selection_widget_2(self):
        html = normalize(SelectionWidget(self.plot2)())
        self.assertEqual(digest_data(html), '4c436d665892c52d8d40e9c7102188604274e7f3fcc56803a73af312e5743a42')
