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
        self.assertEqual(digest_data(html), '999137bba4702862425da5f89d564d70c0ac0b68fd73923f1a51042014ca5796')

    def test_scrubber_widget_2(self):
        html = normalize(ScrubberWidget(self.plot2, display_options={'figure_format': 'png'})())
        self.assertEqual(digest_data(html), '2d0f8440e675ba6a2be31560472858e824aa82715cca9aa440a54e0aa1cb2e42')

    def test_selection_widget_1(self):
        html = normalize(SelectionWidget(self.plot1, display_options={'figure_format': 'png'})())
        self.assertEqual(digest_data(html), '5aafbe259f25ba4f7ad288e87b674840d86ab45585c6b1c3df837562c5c5b346')

    def test_selection_widget_2(self):
        html = normalize(SelectionWidget(self.plot2, display_options={'figure_format': 'png'})())
        self.assertEqual(digest_data(html), '64b6949745b63176fdca668089fcc3252068111675eea6174770eb2d98b7cae8')
