FAQ
===

Here is a list of questions we have either been asked by users or
potential pitfalls we hope to help users avoid:

**Q: How do I adjust the x/y/z axis bounds (matplotlib's xlim, ylim)?**

**A:** Pass an unpacked dictionary containing the kdims/vdims' names as
keys and a tuple of the bounds as values into obj.redim.range().

This constrains the bounds of x_col to (0, max(x_col)).

.. code:: python

  curve = hv.Curve(df, 'x_col', 'y_col')
  curve = curve.redim.range(x_col=(0, None), z_col=(None, 10))

If your column names have spaces, you may predefine a dictionary
and unpack it in obj.redim.range(). This same method is applicable to
adjust the range of a color bar.

.. code:: python

  bounds = {'x col': (0, None), 'z col': (None, 10)}
  curve = hv.Curve(df, 'x col', ['y col', 'z col']).options(color_index='z col')
  curve = curve.redim.range(**bounds)

**Q: How do I export a figure?**

**A:** Create a renderer object by passing a backend (matplotlib / bokeh)
and pass the object and name of file without any suffix into the .save method.

.. code:: python

  backend = 'bokeh'
  renderer = hv.renderer(backend)
  renderer.save(obj, 'name_of_file')

Just note that if have used Jupyter magic (%%opts) to customize your plot
and you try to export it, the customizations will not be retained in the export.

.. code:: python

  %%opts Curve [width=1000]
  curve = hv.Curve([1, 2, 3])
  hv.renderer('bokeh').save(curve, 'example_curve')  # width will not be saved

You'll have to pass the keyword arguments into .options().

.. code:: python

  curve = hv.Curve([1, 2, 3]).options(width=1000)
  hv.renderer('bokeh').save(curve, 'example_curve')

**Q: How do I provide axes labels?**

**A:** Pass a tuple containing the column name and label.

This will relabel 'x_col' to 'X Label'
.. code:: python
  curve = hv.Curve(df, ('x_col', 'X Label'), 'y_col')

You may also label after the fact by passing an unpacked dictionary
to .redim.label()
.. code:: python
  curve = hv.Curve(df, 'x_col', 'y_col')
  curve = curve.redim.label(x_col='X Label', **{'y_col': 'Label for Y'})

**Q: How can I access all the options that aren't exposed in HoloViews,
but are available in the backend?**

**A:** There are two approaches you can take.

The first is converting HoloViews objects as bokeh/matplotlib figures,
and then continuing to work on those figures natively in the selected backend.

.. code:: python
  backend = 'matplotlib'
  hv_obj = hv.Curve(df, 'x_col', 'y_col')
  fig = hv.renderer(backend).get_plot(hv_obj).state
  # this is just a demonstration; you can directly relabel in HoloViews
  fig.axes[0].set_xlabel('X Label')

The second is through finalize_hooks (bokeh) / final_hooks (matplotlib)
which helps retain a HoloViews object.

.. code:: python
   def relabel(plot, element):
       # this is for demonstration purposes
       # use the .redim.label() method instead!
       fig = plot.state
       fig.axes[0].set_xlabel('X Label')

  backend = 'matplotlib'
  hv_obj = hv.Curve(df, 'x_col', 'y_col')
  hv_obj = hv_obj.options(final_hooks=[relabel])

**Q: The default figure size is so tiny! How do I enlarge it?**

**A:** Depending on the selected backend...

.. code:: python
    # for matplotlib:
    hv_obj = hv_obj.options(fig_size=500)

    # for bokeh:
    hv_obj = hv_obj.options(width=1000, height=500)

**Q: How do I plot data without storing it first as a pandas dataframe?**

**A:** Add an extra level of parentheses like this:

.. code:: python
  x = [1, 2, 3]
  y = [4, 5, 6]
  curve = hv.Curve((x, y))

**Q: Can I use HoloViews without IPython/Jupyter?**

**A:** Yes! The IPython/Jupyter notebook support makes a lot of tasks easier, and
helps keep your data objects separate from the customization options,
but everything available in IPython can also be done directly from
Python.  For instance, since HoloViews 1.3.0 you can render an object
directly to disk, with custom options, like this:

.. code:: python

  import holoviews as hv
  renderer = hv.renderer('matplotlib').instance(fig='svg', holomap='gif')
  renderer.save(my_object, 'example_I', style=dict(Image={'cmap':'jet'}))

This process is described in detail in the
`Customizing Plots <user_guide/Customizing_Plots.html>`_ user guide.
Of course, notebook-specific functionality like capturing the data in
notebook cells or saving cleared notebooks is only for IPython/Jupyter.

**Q: How should I use HoloViews as a short qualified import?**

**A:** We recommend importing HoloViews using ``import holoviews as hv``.

**Q: My output looks different from what is shown on the website**

**A:** HoloViews is organized as data structures that have
corresponding plotting code implemented in different plotting-library
backends, and each library will have differences in behavior.
Moreover, the same library can give different results depending on its
own internal options and versions.  For instance, Matplotlib supports
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

**Q: Help! I don't know how to index into my object!**

**A:**  In any Python session, you can look at ``print(obj)``. For
an explanation of how this information helps you index into your
object, see our `Composing Elements <user_guides/Composing_Elements.html>`_
user guide.

**Q: Help! How do I find out the options for customizing the
appearance of my object?**

**A:** If you are in the IPython/Jupyter Notebook you can use the cell magic
``%%output info=True`` at the top of your code cell. This will
present the available style and plotting options for that object.

The same information is also available in any Python session using
``hv.help(obj)``. For more information on customizing the display
of an object, see our `Customizing Plots <user_guides/Customizing_Plots.html>`_
user guide.

**Q: Why don't you let me pass** *matplotlib_option* **as a style
through to matplotlib?**

**A:** We have selected a subset of default allowable style options
that are most commonly useful in order to hide the more arcane
matplotlib options. If you do need such an option to be passed to
the plotting system, you are welcome to declare that this is allowed.
For instance, say you may want the ``'filternorm'`` option to be passed
to matplotlib's ``imshow`` command when displaying an ``Image``
element:

.. code:: python

  from holoviews import Store
  Store.add_style_opts(Image, ['filternorm'], backend='matplotlib')

Now you can freely use ``'filternorm'`` in the ``%opts`` line/cell
magic, including tab-completion!

**Q: I still can't tweak my figure in exactly the way I want. What can I do?**

The parameters provided by HoloViews should normally cover the most
common plotting options needed.  In case you need further control, you
can always subclass any HoloViews object and modify any of its
behavior, and the object will still normally interact with other
HoloViews objects (e.g. in Layout or Overlay configurations).

**Q: How do I get a legend on my overlay figure?**

**A:** Legends are generated in two different ways, depending on the
``Overlay`` type you are using. When using ``*`` to generate a normal ``Overlay``,
the legends are generated from the labels of the Elements.
Alternatively, you can construct an ``NdOverlay``, where the key dimensions
and values will become part of the legend. The
`Dimensioned Containers <user_guides/Dimensioned_Containers.html>`_ user guide
shows an example of an ``NdOverlay`` in action.

**Q: I wish to use special characters in my title, but then attribute
access becomes confusing.**

**A:** The title format ``"{label} {group} {dimensions}"`` is simply a default
that you can override. If you want to use a lot of special characters
in your titles, you can pick simple ``group`` and ``label`` strings
that let you refer to the object easily in the code, and then you can
set the plot title directly, using the plot option
``title_format="my new title"``.

You can also use 2-tuples when specifying ``group`` and ``label`` where
the first item is the short name used for attribute access and the second name is the long descriptive name used in the title.

**Q: Where have my custom styles gone after unpickling my object?**

**A:** HoloViews objects are designed to pickle and unpickle your core
data only, if you use Python's ``pickle.load`` and
``pickle.dump``. Because custom options are kept separate from
your data, you need to use the corresponding methods ``Store.dump`` and
``Store.load`` if you also want to save and restore per-object
customization. You can import ``Store`` from the main namespace with
``from holoviews import Store``.

**Q: Can I avoid generating extremely large HTML files when exporting
my notebook?**

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

It is also possible to generate web pages that do not actually include
all of the data shown, by specifying a `DynamicMap`` as described in
`Live Data <user_guides/Live_Data.html>`_ rather than a HoloMap.  The
DynamicMap will request data only as needed, and so requires a Python
server to be running alongside the viewable web page.  Such pages are
more difficult to share by email or on web sites, but much more feasible
for large datasets.

**Q: How do I create a Layout or Overlay object from an arbitrary list?**

You can supply a list of ``elements`` directly to the ``Layout`` and
``Overlay`` constructors. For instance, you can use
``hv.Layout(elements)`` or ``hv.Overlay(elements)``.
