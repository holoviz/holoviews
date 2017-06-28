"""
Bokeh app example using datashader for rasterizing a large dataset and
geoviews for reprojecting coordinate systems.

This example requires the 1.7GB nyc_taxi.csv dataset which you can
obtain by following the instructions for 'nyc_taxi' at:

  https://github.com/bokeh/datashader/blob/master/examples/README.md

Once this CSV is placed in a data/ subfolder, you can run this app with:

  bokeh serve --show nytaxi_hover.py

"""
import holoviews as hv
import geoviews as gv
import dask.dataframe as dd
import cartopy.crs as ccrs

from holoviews.operation.datashader import aggregate

hv.extension('bokeh')

# Set plot and style options
hv.util.opts('Image [width=800 height=400 shared_axes=False logz=True] {+axiswise} ')
hv.util.opts("HLine VLine (color='white' line_width=1) Layout [shared_axes=False] ")
hv.util.opts("Curve [xaxis=None yaxis=None show_grid=False, show_frame=False] (color='orangered') {+framewise}")

# Read the CSV file
df = dd.read_csv('data/nyc_taxi.csv',usecols=['pickup_x', 'pickup_y'])
df = df.persist()

# Reproject points from Mercator to PlateCarree (latitude/longitude)
points = gv.Points(df, kdims=['pickup_x', 'pickup_y'], vdims=[], crs=ccrs.GOOGLE_MERCATOR)
projected = gv.operation.project_points(points, projection=ccrs.PlateCarree())
projected = projected.redim(pickup_x='lon', pickup_y='lat')

# Use datashader to rasterize and linked streams for interactivity
agg = aggregate(projected, link_inputs=True, x_sampling=0.0001, y_sampling=0.0001)
pointerx = hv.streams.PointerX(x=-74, source=projected)
pointery = hv.streams.PointerY(y=40.8,  source=projected)
vline = hv.DynamicMap(lambda x: hv.VLine(x), streams=[pointerx])
hline = hv.DynamicMap(lambda y: hv.HLine(y), streams=[pointery])

sampled = hv.util.Dynamic(agg, operation=lambda obj, x: obj.sample(lon=x),
                          streams=[pointerx], link_inputs=False)

hvobj = ((agg * hline * vline) << sampled.opts(plot={'Curve': dict(width=100)}))

# Obtain Bokeh document and set the title
doc = hv.renderer('bokeh').server_doc(hvobj)
doc.title = 'NYC Taxi Crosshair'
