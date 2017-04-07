import numpy as np
import holoviews as hv
import holoviews.plotting.bokeh
from holoviews.streams import Selection1D

hv.Store.current_backend = 'bokeh'
renderer = hv.Store.renderers['bokeh'].instance(mode='server')
hv.Store.options(backend='bokeh').Points = hv.Options('plot', tools=['box_select'])

data = np.random.multivariate_normal((0, 0), [[1, 0.1], [0.1, 1]], (1000,))
points = hv.Points(data)
sel = Selection1D(source=points)
mean_sel = hv.DynamicMap(lambda index: hv.HLine(points['y'][index].mean()
                                                if index else -10),
                         kdims=[], streams=[sel])
doc,_ = renderer((points * mean_sel))
doc.title = 'HoloViews Selection Stream'
