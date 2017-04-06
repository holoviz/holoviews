# -*- coding: utf-8 -*-
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import (
    ColumnDataSource, HoverTool, SingleIntervalTicker, Slider, Button, Label,
    CategoricalColorMapper,
)
import holoviews as hv
import holoviews.plotting.bokeh

renderer = hv.Store.renderers['bokeh']

start = 0
end = 10

hmap = hv.HoloMap({i: hv.Image(np.random.rand(10,10)) for i in range(start, end+1)})
plot = renderer.get_plot(hmap)
plot.update(0)

def animate_update():
    year = slider.value + 1
    if year > end:
        year = start
    slider.value = year

def slider_update(attrname, old, new):
    plot.update(slider.value)

slider = Slider(start=start, end=end, value=0, step=1, title="Year")
slider.on_change('value', slider_update)

def animate():
    if button.label == '► Play':
        button.label = '❚❚ Pause'
        curdoc().add_periodic_callback(animate_update, 200)
    else:
        button.label = '► Play'
        curdoc().remove_periodic_callback(animate_update)

button = Button(label='► Play', width=60)
button.on_click(animate)

layout = layout([
    [plot.state],
    [slider, button],
], sizing_mode='fixed')

curdoc().add_root(layout)
