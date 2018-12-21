"""
Gapminder demo demonstrating how to combine to extend a HoloViews plot
with custom bokeh widgets to deploy an app.
"""

import pandas as pd
import numpy as np
import holoviews as hv

from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import Slider, Button
from bokeh.sampledata import gapminder
from holoviews import dim, opts

renderer = hv.renderer('bokeh')

# Declare dataset
panel = pd.Panel({'Fertility': gapminder.fertility,
                  'Population': gapminder.population,
                  'Life expectancy': gapminder.life_expectancy})
gapminder_df = panel.to_frame().reset_index().rename(columns={'minor': 'Year'})
gapminder_df = gapminder_df.merge(gapminder.regions.reset_index(), on='Country')
gapminder_df['Country'] = gapminder_df['Country'].astype('str')
gapminder_df['Group'] = gapminder_df['Group'].astype('str')
gapminder_df.Year = gapminder_df.Year.astype('f')
ds = hv.Dataset(gapminder_df)

# Apply dimension labels and ranges
kdims = ['Fertility', 'Life expectancy']
vdims = ['Country', 'Population', 'Group']
dimensions = {
    'Fertility' : dict(label='Children per woman (total fertility)', range=(0, 10)),
    'Life expectancy': dict(label='Life expectancy at birth (years)', range=(15, 100)),
    'Population': ('population', 'Population')
}

# Create Points plotting fertility vs life expectancy indexed by Year
gapminder_ds = ds.redim(**dimensions).to(hv.Points, kdims, vdims, 'Year')

# Define annotations
text = gapminder_ds.clone({yr: hv.Text(1.2, 25, str(int(yr)), fontsize=30)
                           for yr in gapminder_ds.keys()})

# Define options
# Combine Points and Text
hvgapminder = (gapminder_ds * text).opts(
    opts.Points(alpha=0.6, color='Group', cmap='Set1', line_color='black', 
                size=np.sqrt(dim('Population'))*0.005, width=1000, height=600,
                tools=['hover'], title='Gapminder Demo'),
    opts.Text(text_font_size='52pt', text_color='lightgray'))


# Define custom widgets
def animate_update():
    year = slider.value + 1
    if year > end:
        year = start
    slider.value = year

# Update the holoviews plot by calling update with the new year.
def slider_update(attrname, old, new):
    hvplot.update((new,))

callback_id = None

def animate():
    global callback_id
    if button.label == '► Play':
        button.label = '❚❚ Pause'
        callback_id = doc.add_periodic_callback(animate_update, 200)
    else:
        button.label = '► Play'
        doc.remove_periodic_callback(callback_id)

start, end = ds.range('Year')
slider = Slider(start=start, end=end, value=start, step=1, title="Year")
slider.on_change('value', slider_update)

button = Button(label='► Play', width=60)
button.on_click(animate)

# Get HoloViews plot and attach document
doc = curdoc()
hvplot = renderer.get_plot(hvgapminder, doc)
hvplot.update((1964,))

# Make a bokeh layout and add it as the Document root
plot = layout([[hvplot.state], [slider, button]], sizing_mode='fixed')
doc.add_root(plot)
