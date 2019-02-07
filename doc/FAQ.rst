FAQ
===

Here is a list of questions we have either been asked by users or
potential pitfalls we hope to help users avoid:

**Q: How should I use HoloViews as a short qualified import?**

**A:** We recommend importing HoloViews using ``import holoviews as hv``.


**Q: How do I specify axis labels?**

**A:** Axes are labeled with the label of the corresponding Dimension,
which for a Pandas dataframe will default to the name of that column.
If you want to define your own specific label to display for a
dimension, you can provide a tuple containing the column name and your
preferred label for it. For instance, if the column is named `x_col`,
you can make the label 'X Label' using:

.. code:: python

  curve = hv.Curve(df, ('x_col', 'X Label'), 'y_col')

This is the recommended way to specify labels in a declarative way,
which will persist when applying operations to your data. You can also
change the labels later, even after the object has been defined, by
passing arguments (or an unpacked dictionary) to ``redim.label``:

.. code:: python

  curve = hv.Curve(df, 'x_col', 'y_col')
  curve = curve.redim.label(x_col='X Label', y_col='Label for Y')

To override a label for plotting it is also possible to use the
`xlabel` and `ylabel` plot options:

.. code:: python

  curve = hv.Curve(df, 'x_col', 'y_col')
  curve = curve.options(xlabel='X Label', ylabel='Label for Y')

**Q: How do I adjust the x/y/z axis bounds (matplotlib's xlim, ylim)?**

**A:** Pass an unpacked dictionary containing the kdims/vdims' names
as keys and a tuple of the bounds as values into ``redim.range``.

This constrains the bounds of x_col to `(0, max(x_col))`.

.. code:: python

  curve = hv.Curve(df, 'x_col', 'y_col')
  curve = curve.redim.range(x_col=(0, None))

As in the discussion of labels above, this approach allows you to declaratively associate ranges
with the dimensions of your data in a way that will persist even if
you apply operations to the object. This same method is applicable to
adjust the limits of the colormapping range (i.e. ``clim``).

To override the range specifically for plotting it is also possible to
set the ``xlim`` and ``ylim`` plot options:

.. code:: python

  hv.Curve(df, 'x_col', 'y_col').options(xlim=(0, None), ylim=(0, 10))

This approach allows you to customize objects easily as a final step, but note that the values won't be applied to the underlying data, and thus won't be inherited if this object is subsequently used in an operation or data selection command.


**Q: How do I control the auto-ranging/normalization of axis limits across frames in a HoloMap or objects in a Layout?**

**A:** Where feasible, HoloViews defaults to normalizing axis ranges
across all objects that are presented together, so that they can be
compared directly. If you don't want objects that share a dimension to
be normalized together in your layout, you can change the ``axiswise``
normalization option to True, making each object be normalized
independently:

.. code:: python

    your_layout.options(axiswise=True)

Similarly, if you have a HoloMap composed of multiple frames in an
animation or controlled with widgets, you can make each frame be
normalized independently by changing ``framewise`` to True:

.. code:: python

    your_holomap.options(framewise=True)


**Q: Why doesn't my DynamicMap respect the ``framewise=False`` option for axis normalization across frames?**

**A:** Unfortunately, HoloViews has no way of knowing the axis ranges
of objects that might be returned by future calls to a DynamicMap's
callback function, and so there is no way for it to fully implement
``framewise=False`` normalization (even though such normalization
is the default in HoloViews). Thus, as a special case, a DynamicMap
(whether created specifically or as the return value of various
operations that accept a ``dynamic=True`` argument) will by default
compute its ranges *using data from the first frame only*. If that is not
the behavior you want, you can either set ``framewise=True`` on it to enable
normalization on every frame independently, or you can manually
determine the appropriate axis range yourself and set that, e.g. with
``.redim.range()`` as described above.


**Q: The default figure size is so tiny! How do I enlarge it?**

**A:** Depending on the selected backend...

.. code:: python

  # for matplotlib:
  hv_obj = hv_obj.options(fig_size=500)

  # for bokeh:
  hv_obj = hv_obj.options(width=1000, height=500)


**Q: How do I get a legend on my overlay figure?**

**A:** Legends are generated in two different ways, depending on the
``Overlay`` type you are using. When using ``*`` to generate a normal ``Overlay``,
the legends are generated from the labels of the Elements.
Alternatively, you can construct an ``NdOverlay``, where the key dimensions
and values will become part of the legend. The
`Dimensioned Containers <user_guide/Dimensioned_Containers.html>`_ user guide
shows an example of an ``NdOverlay`` in action.


**Q: How do I export a figure?**

**A:** The easiest way to save a figure is the ``hv.save`` utility,
which allows saving plots in different formats depending on what is
supported by the selected backend:

.. code:: python

  # Using bokeh
  hv.save(obj, 'plot.html', backend='bokeh')

  # Using matplotlib
  hv.save(obj, 'plot.svg', backend='matplotlib')

Note that the backend is optional and will default to the currently
activated backend (i.e. ``hv.Store.current_backend``).


**Q: Can I export and customize a bokeh or matplotlib figure directly?**

**A:** Sometimes it is useful to customize a plot further using the
underlying plotting API used to render it. The ``hv.render`` method
returns the rendered representation of a holoviews object as bokeh or
matplotlib figure:

.. code:: python

  # Using bokeh
  p = hv.render(obj, backend='bokeh')

  # Using matplotlib
  fig = hv.render(obj, backend='matplotlib')

Note that the backend is optional and will default to the currently
activated backend (i.e. ``hv.Store.current_backend``).

If the main reason you want access to the object is to somehow customize it before it is
plotted, instead consider that it is possible to write so called ``hooks``:

.. code:: python

  def hook(plot, element):
    # The bokeh/matplotlib figure
    plot.state

	# A dictionary of handles on plot subobjects, e.g. in matplotlib
	# artist, axis, legend and in bokeh x_range, y_range, glyph, cds etc.
	plot.handles

  hv.Curve(df, 'x_col', 'y_col').options(hooks=[hook])

These hooks can modify the backend specific representation, e.g. the
matplotlib figure, before it is displayed, allowing arbitrary customizations to be
applied which are not implemented or exposed by HoloViews itself.


**Q: What if I need to do more complex customization supported by the backend but not exposed in HoloViews?**

**A:** If you need to, you can easily access the underlying Bokeh or
Matplotlib figure and then use Bokeh or Matplotlib's API directly on
that object. For instance, if you want to force Bokeh to use a
fixed list of tick labels for a HoloViews object ``h``, you can
grab the corresponding Bokeh figure ``b``, edit it to your heart's
content as a Bokeh figure, and then show it as for any other Bokeh
figure:

.. code:: python

  import holoviews as hv
  hv.extension('bokeh')
  h = hv.Curve([1,2,7], 'x_col', 'y_col')

  from bokeh.io import show
  from bokeh.models.tickers import FixedTicker

  b = hv.render(h)
  b.axis[0].ticker = FixedTicker(ticks=list(range(0, 10)))
  show(b)

Once you debug a modification like this manually as above, you'll probably
want to set it up to apply automatically whenever a Bokeh plot is generated
for that HoloViews object:

.. code:: python

  import holoviews as hv
  from bokeh.models.tickers import FixedTicker
  hv.extension('bokeh')

  def update_axis(plot, element):
      b = plot.state
      b.axis[0].ticker = FixedTicker(ticks=list(range(0, 10)))

  h = hv.Curve([1,2,7], 'x_col', 'y_col')
  h = h.options(hooks=[update_axis])
  h

Here, you've wrapped your Bokeh-API calls into a function, then
supplied that to HoloViews so that it can be run automatically
whenever object ``h`` is viewed.


**Q: Can I avoid generating extremely large HTML files when exporting my notebook?**

**A:** It is very easy to visualize large volumes of data with
HoloMaps, and all available display data is embedded in the HTML
snapshot when sliders are used so that the result can be viewed
without using a Python server process. It is therefore worth being
aware of file size when authoring a notebook or web page to be
published on the web. Useful tricks to reduce file size of HoloMaps
include:

* Reducing the figure size.
* Selecting fewer frames for display (e.g selecting a smaller number
  of keys in any displayed ``HoloMap`` object)
* Displaying your data in a more highly compressed format such as
  ``webm``, ``mp4`` or animated ``gif``, while being aware that those
  formats may introduce visible artifacts.
* When using bokeh use lower precision dtypes (e.g. float16 vs. float64)
* Replace figures with lots of data with images prerendered
  by `datashade() <user_guide/Large_Data.html>`_.

It is also possible to generate web pages that do not actually include
all of the data shown, by specifying a ``DynamicMap`` as described
`Live Data <user_guide/Live_Data.html>`_ rather than a HoloMap. The
DynamicMap will request data only as needed, and so requires a Python
server to be running alongside the viewable web page. Such pages are
more difficult to share by email or on web sites, but much more feasible
for large datasets.


**Q: I wish to use special characters in my title, but then attribute access becomes confusing.**

**A:** The title format ``"{label} {group} {dimensions}"`` is simply a default
that you can override. If you want to use a lot of special characters
in your titles, you can pick simple ``group`` and ``label`` strings
that let you refer to the object easily in the code, and then you can
set the plot title directly, using the plot option
``title="my new title"``.

You can also use 2-tuples when specifying ``group`` and ``label`` where
the first item is the short name used for attribute access and the
second name is the long descriptive name used in the title.


**Q: Help! I don't know how to index into my object!**

**A:** In any Python session, you can look at ``print(obj)`` to see
the structure of ``obj``. For
an explanation of how this information helps you index into your
object, see our `Composing Elements <user_guide/Composing_Elements.html>`_
user guide.


**Q: How do I create a Layout or Overlay object from an arbitrary list?**

**A:** You can supply a list of ``elements`` directly to the ``Layout`` and
``Overlay`` constructors. For instance, you can use
``hv.Layout(elements)`` or ``hv.Overlay(elements)``.


**Q: How do I provide keyword arguments for items with spaces?**

**A:** If your column names have spaces, you may predefine a dictionary
using curly braces and unpack it.

.. code:: python

  bounds = {'x col': (0, None), 'z col': (None, 10)}
  curve = hv.Curve(df, 'x col', ['y col', 'z col'])
  curve = curve.redim.range(**bounds)


**Q: How do I plot data without storing it first as a pandas/xarray object?**

**A:** HoloViews typically uses pandas and xarray objects in its examples,
but it can accept standard Python data structures as well.
Whatever data type is used, it needs to be provided to the first
argument of the Element as *a single object*, so if you are using a
pair of lists, be sure to pass them as a tuple, not as two separate
arguments.


**Q: Help! How do I find out the options for customizing the appearance of my object?**

**A:** If you are in the IPython/Jupyter Notebook you can use the cell magic
``%%output info=True`` at the top of your code cell. This will
present the available style and plotting options for that object.

The same information is also available in any Python session using
``hv.help(obj)``. For more information on customizing the display
of an object, see our `Customizing Plots <user_guide/Customizing_Plots.html>`_
user guide.


**Q: Why are my .options(), .relabel(), .redim(), and similar settings not having any effect?**

**A:** By default, HoloViews object methods like .options and
.redim return a *copy* of your object,
rather than modifying your original object. In HoloViews,
making a copy of the object is cheap, because only the metadata
is copied, not the data, and returning a copy makes it simple
to work with a variety of differently customized versions of any given
object. You can use ``.opts()`` or pass ``clone=False`` to
``.options()`` if you wish to modify the object in place, or you can
just reassign the new object to the old name (as in ``e =
e.relabel("New Label")``).


**Q: Why isn't my %%opts cell magic being applied to my HoloViews object?**

**A:** %%opts is convenient because it tab-completes, but it can be confusing
because of the "magic" way that it works. Specifically, if you use it at
the top of a Jupyter notebook cell, the indicated options will be applied
to the return value of that cell, if it's a HoloViews object. So, if you
want a given object to get customized, you need to make sure it is
returned from the cell, or the options won't ever be applied, and you
should only access it after it has been returned, or the options won't
*yet* have been applied. For instance, if you use `renderer.save()`
to export an object and only then return that object as the output of
a cell, the exported object won't have the options applied, because
they don't get applied until the object is returned
(during IPython's "display hooks" processing). So to make sure that
options get applied, (a) return the object from a cell, and then (b)
access it (e.g. for exporting) after the object has been returned.
To avoid confusion, you may prefer to use .options() directly on the
object to ensure that the options have been applied before exporting.
Example code below:

.. code:: python

  %%opts Curve [width=1000]
  # preceding cell
  curve = hv.Curve([1, 2, 3])
  # next cell
  hv.renderer('bokeh').save(curve, 'example_curve')


**Q: My output looks different from what is shown on the website**

**A:** HoloViews is organized as data structures that have
corresponding plotting code implemented in different plotting-library
backends, and each library will have differences in behavior.
Moreover, the same library can give different results depending on its
own internal options and versions. For instance, Matplotlib supports
a variety of internal plotting backends, and these can have
inconsistent output. HoloViews will not switch Matplotlib backends for
you, but when using Matplotlib we strongly recommend selecting the
'agg' backend for consistency:

.. code:: python

  from matplotlib import pyplot
  pyplot.switch_backend('agg')

You can generally set options explicitly to make the output more
consistent across HoloViews backends, but in general HoloViews tries
to use each backend's defaults where possible.


**Q: Why do my HoloViews and GeoViews objects work fine separately but are mismatched when overlaid?**

**A:** GeoViews works precisely the same as HoloViews, except that
GeoViews is aware of geographic projections. If you take an
``hv.Points`` object in lon,lat coordinates and overlay it on a
GeoViews map in Web Mercator, the HoloViews object will be in
entirely the wrong coordinate system, with the HoloViews object all
appearing at one tiny spot on the globe. If you declare the same
object as ``gv.Points``, then GeoViews will (a) assume it is in
lon,lat coordinates (which HoloViews cannot assume, as it knows
nothing of geography), and (b) convert it into the coordinates
needed for display (e.g. Web Mercator). So, just make sure that
anything with geographic coordinates is defined as a GeoViews object,
and make sure to declare the coordinates (``crs=...``) if the data is
in anything other than lon,lat.


**Q: Where have my custom styles gone after unpickling my object?**

**A:** HoloViews objects are designed to pickle and unpickle your core
data only, if you use Python's ``pickle.load`` and
``pickle.dump``. Because custom options are kept separate from
your data, you need to use the corresponding methods ``Store.dump`` and
``Store.load`` if you also want to save and restore per-object
customization. You can import ``Store`` from the main namespace with
``from holoviews import Store``.


**Q: Why are the sizing options so different between the Matplotlib and Bokeh backends?**

**A:** The way plot sizes are computed is handled in radically
different ways by these backends, with Matplotlib building plots 'inside
out' (from plot components with their own sizes) and Bokeh building
them 'outside in' (fitting plot components into a given overall size).
Thus there is not currently any way to specify sizes in a way that is
comparable between the two backends.


**Q: Why don't you let me pass** *matplotlib_option* **as a style through to matplotlib?**

**A:** We have selected a subset of default allowable style options
that are most commonly useful in order to hide the more arcane
matplotlib options. If you do need such an option to be passed to
the plotting system, you are welcome to declare that this is allowed.
For instance, say you may want the ``'filternorm'`` option to be passed
to matplotlib's ``imshow`` command when displaying an ``Image``
element:

.. code:: python

  import holoviews as hv
  from holoviews import Store

  hv.extension('matplotlib')
  Store.add_style_opts(hv.Image, ['filternorm'], backend='matplotlib')

Now you can freely use ``'filternorm'`` in ``.options()`` and in the
``%opts`` line/cell magic, including tab-completion!


**Q: What I want to change is about how HoloViews works, not about the underlying backend. Is that possible?**

**A:** Sure, if you need more customization and configurability than is
possible with either HoloViews options or with extra backend-specific
code as above, then you can always subclass the plotting class used
for a HoloViews element and modify any of its behavior. You can also
add your own Element types, which need corresponding plotting classes
before they will be viewable in a given backend. The resulting objects
will still interact normally with other HoloViews objects (e.g. in
Layout or Overlay configurations).
