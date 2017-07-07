FAQ
===

Here is a list of questions we have either been asked by users or
potential pitfalls we hope to help users avoid:


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

We recommend importing HoloViews using ``import holoviews as hv``.

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

**A:** The title default of ``"{label} {group}"`` is simply a default
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
