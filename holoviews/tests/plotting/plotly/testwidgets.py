from unittest import SkipTest
import numpy as np

from holoviews.core import HoloMap, Store
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase

try:
    import holoviews.plotting.plotly # noqa (Activate backend)
    plotly_renderer = Store.renderers['plotly']
except:
    plotly_renderer = None


class TestSelectionWidget(ComparisonTestCase):

    def setUp(self):
        if plotly_renderer is None:
            raise SkipTest("Plotly required to test plotly widgets")

    def test_holomap_slider_unsorted_datetime_values_initialization(self):
        hmap = HoloMap([(np.datetime64(10005, 'D'), Curve([1, 2, 3])),
                        (np.datetime64(10000, 'D'), Curve([1, 2, 4]))], sort=False)
        widgets = plotly_renderer.get_widget(hmap, 'widgets')
        widgets()
        self.assertEqual(widgets.plot.current_key, (np.datetime64(10000, 'D'),))
        self.assertEqual(widgets.plot.current_frame, hmap[np.datetime64(10000, 'D')])
