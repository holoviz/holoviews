import json
import datetime as dt

import numpy as np

from holoviews.core import Dimension, DynamicMap, HoloMap
from holoviews.element import Image, Curve, VLine

from .testplot import TestMPLPlot, mpl_renderer


class TestSelectionWidget(TestMPLPlot):

    def test_dynamic_nonoverlap(self):
        kdims = [Dimension('File', range=(0.01, 1)),
                 Dimension('SliceDimension', range=(0.01, 1)),
                 Dimension('Coordinates', range=(0.01, 1))]
        dmap1 = DynamicMap(lambda x, y, z: Image(np.random.rand(10,10)), kdims=kdims)
        dmap2 = DynamicMap(lambda x: Curve(np.random.rand(10,2))*VLine(x),
                           kdims=kdims[:1])
        mpl_renderer.get_widget(dmap1 + dmap2, 'selection')

    def test_dynamic_values_partial_overlap(self):
        kdims = [Dimension('File', range=(0.01, 1)),
                 Dimension('SliceDimension', values=['x', 'y', 'z']),
                 Dimension('Coordinates', range=(0.01, 1))]
        dmap1 = DynamicMap(lambda x, y, z: Image(np.random.rand(10,10)), kdims=kdims)
        dmap2 = DynamicMap(lambda x: Curve(np.random.rand(10,2))*VLine(x),
                           kdims=kdims[:1])
        mpl_renderer.get_widget(dmap1 + dmap2, 'selection')

    def test_holomap_datetime_widgets(self):
        hmap = HoloMap({np.datetime64(dt.datetime(2017, 1, i)): Curve([i]) for i in range(1, 3)})
        widgets = mpl_renderer.get_widget(hmap, 'widgets')

        key_data = json.loads(widgets.get_key_data())
        widget_data, dimensions, init_vals = widgets.get_widgets()

        expected = {
            'type': 'slider',
            'vals': "['2017-01-01T00:00:00.000000000', '2017-01-02T00:00:00.000000000']",
            'labels': "['2017-01-01T00:00:00.000000000', '2017-01-02T00:00:00.000000000']",
            'step': 1, 'default': 0, 'next_vals': '{}', 'next_dim': None,
            'init_val': '2017-01-01T00:00:00.000000000', 'visible': True,
            'dim': 'Default', 'dim_label': 'Default', 'dim_idx': 0,
            'visibility': '', 'value': np.datetime64(dt.datetime(2017, 1, 1))
        }
        self.assertEqual(widget_data[0], expected)

        expected_keys = {"('2017-01-01T00:00:00.000000000',)": 0,
                         "('2017-01-02T00:00:00.000000000',)": 1}
        self.assertEqual(key_data, expected_keys)

