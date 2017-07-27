"""
An example demonstrating how to put together a crossfilter app based
on the Auto MPG dataset. Demonstrates how to dynamically generate
bokeh plots using the HoloViews API and replacing the bokeh plot
based on the current widget selections.
"""
import holoviews as hv

from bokeh.layouts import row, widgetbox
from bokeh.models import Select
from bokeh.plotting import curdoc
from bokeh.sampledata.autompg import autompg

df = autompg.copy()

SIZES = list(range(6, 22, 3))
ORIGINS = ['North America', 'Europe', 'Asia']

# data cleanup
df.cyl = [str(x) for x in df.cyl]
df.origin = [ORIGINS[x-1] for x in df.origin]

df['year'] = [str(x) for x in df.yr]
del df['yr']

df['mfr'] = [x.split()[0] for x in df.name]
df.loc[df.mfr=='chevy', 'mfr'] = 'chevrolet'
df.loc[df.mfr=='chevroelt', 'mfr'] = 'chevrolet'
df.loc[df.mfr=='maxda', 'mfr'] = 'mazda'
df.loc[df.mfr=='mercedes-benz', 'mfr'] = 'mercedes'
df.loc[df.mfr=='toyouta', 'mfr'] = 'toyota'
df.loc[df.mfr=='vokswagen', 'mfr'] = 'volkswagen'
df.loc[df.mfr=='vw', 'mfr'] = 'volkswagen'
del df['name']

columns = sorted(df.columns)
discrete = [x for x in columns if df[x].dtype == object]
continuous = [x for x in columns if x not in discrete]
quantileable = [x for x in continuous if len(df[x].unique()) > 20]

renderer = hv.renderer('bokeh')
options = hv.Store.options(backend='bokeh')
options.Points = hv.Options('plot', width=800, height=600, size_index=None,)
options.Points = hv.Options('style', cmap='rainbow', line_color='black')

def create_figure():
    label = "%s vs %s" % (x.value.title(), y.value.title())
    kdims = [x.value, y.value]

    opts, style = {}, {}
    opts['color_index'] = color.value if color.value != 'None' else None
    if size.value != 'None':
        opts['size_index'] = size.value
        opts['scaling_factor'] = (1./df[size.value].max())*200
    points = hv.Points(df, kdims=kdims, label=label).opts(plot=opts, style=style)
    return renderer.get_plot(points).state

def update(attr, old, new):
    layout.children[1] = create_figure()

x = Select(title='X-Axis', value='mpg', options=quantileable)
x.on_change('value', update)

y = Select(title='Y-Axis', value='hp', options=quantileable)
y.on_change('value', update)

size = Select(title='Size', value='None', options=['None'] + quantileable)
size.on_change('value', update)

color = Select(title='Color', value='None', options=['None'] + quantileable)
color.on_change('value', update)

controls = widgetbox([x, y, color, size], width=200)
layout = row(controls, create_figure())

curdoc().add_root(layout)
curdoc().title = "Crossfilter"
