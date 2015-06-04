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
        self.assertEqual(digest_data(html), 'b3b5a2c1b797e7382a6d56c507483ecc5415a3268c27d7eae072ac0167acb50f')

    def test_selection_widget_1(self):
        html = normalize(SelectionWidget(self.plot1)())
        self.assertEqual(digest_data(html), 'bb4fe3165ad845e763542d4cd6deac5f18a663b734bc82ceef77eed46f57e386')

    def test_scrubber_widget_2(self):
        html = normalize(ScrubberWidget(self.plot2)())
        self.assertEqual(digest_data(html), '623d3ab77b1ac49a429f1e8736064ff46219f17245c2caeea8bc3e9acba8efb3')

    def test_selection_widget_2(self):
        html = normalize(SelectionWidget(self.plot2)())
        self.assertEqual(digest_data(html), '63e0fbb650b6c66ff738c969d0e3fade1fcaf417449feacc7b076d00ddb9e078')
