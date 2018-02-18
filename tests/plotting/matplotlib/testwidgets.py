from nose.plugins.attrib import attr

import numpy as np

from holoviews.core import Dimension, DynamicMap
from holoviews.element import Image, Curve, VLine

from .testplot import TestMPLPlot, mpl_renderer


class TestSelectionWidget(TestMPLPlot):

    @attr(optional=1) # Requires jinja2
    def test_dynamic_nonoverlap(self):
        kdims = [Dimension('File', range=(0.01, 1)),
                 Dimension('SliceDimension', range=(0.01, 1)),
                 Dimension('Coordinates', range=(0.01, 1))]
        dmap1 = DynamicMap(lambda x, y, z: Image(np.random.rand(10,10)), kdims=kdims)
        dmap2 = DynamicMap(lambda x: Curve(np.random.rand(10,2))*VLine(x),
                           kdims=kdims[:1])
        mpl_renderer.get_widget(dmap1 + dmap2, 'selection')

    @attr(optional=1) # Requires jinja2
    def test_dynamic_values_partial_overlap(self):
        kdims = [Dimension('File', range=(0.01, 1)),
                 Dimension('SliceDimension', values=['x', 'y', 'z']),
                 Dimension('Coordinates', range=(0.01, 1))]
        dmap1 = DynamicMap(lambda x, y, z: Image(np.random.rand(10,10)), kdims=kdims)
        dmap2 = DynamicMap(lambda x: Curve(np.random.rand(10,2))*VLine(x),
                           kdims=kdims[:1])
        mpl_renderer.get_widget(dmap1 + dmap2, 'selection')
