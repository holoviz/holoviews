"""
Example app demonstrating how to use the HoloViews API to generate
a bokeh app with complex interactivity. Uses a Selection1D stream
to compute the mean y-value of the current selection.
"""

import numpy as np
import holoviews as hv
import holoviews.plotting.bokeh
from holoviews.streams import Selection1D

renderer = hv.Store.renderers['bokeh']
hv.Store.options(backend='bokeh').Points = hv.Options('plot', tools=['box_select'])

data = np.random.multivariate_normal((0, 0), [[1, 0.1], [0.1, 1]], (1000,))
points = hv.Points(data)
sel = Selection1D(source=points)
mean_sel = hv.DynamicMap(lambda index: hv.HLine(points['y'][index].mean()
                                                if index else -10),
                         kdims=[], streams=[sel])

doc = renderer.app((points * mean_sel))
doc.title = 'HoloViews Selection Stream'
