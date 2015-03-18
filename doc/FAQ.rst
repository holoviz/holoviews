FAQ
===

Here is a list of questions we have either been asked by users or
potential pitfalls we hope to help users avoid:


**Q: Can I use HoloViews without IPython?**

**A:** Yes! The IPython support makes a lot of tasks easier, and
helps keep your data objects separate from the customization options,
but everything available in IPython can also be done directly from
Python.  For instance, you can render an object directly to disk, with
custom options, like this:

.. code:: python

  from holoviews import Store
  renderer = Store.renderer.instance(fig='svg', holomap='gif')
  renderer.save(my_object, 'example_I', style=dict(Image={'cmap':'jet'}))

This process is described in detail in the 
`Options tutorial <Tutorials/Options>`_, and some more information is 
on `this wiki page
<https://github.com/ioam/holoviews/wiki/HoloViews-without-IPython>`_.
Of course, notebook-specific functionality like capturing the data in
notebook cells or saving cleared notebooks is only for IPython.


**Q: My output looks different from what is shown on the website**

**A:** Matplotlib supports its own backends and they can have 
inconsistent output. HoloViews will not switch the backend for
you, but we recommend selecting the 'agg' backend in general:

.. code:: python

  from matplotlib import pyplot as plt
  plt.switch_backend('agg')


**Q: Help! I don't know how to index into my object!**

**A:** If you are in the IPython Notebook you can use the cell magic
``%%output fig='repr' holomap='repr'`` at the top of your code cell.

In a regular Python session, you can look at ``print repr(obj)``. For
an explanation of how this information helps you index into your
object, see our `Composing Data tutorial <Tutorials/Composing_Data>`_.


**Q: Help! How do I find out the options for customizing the
appearance of my object?**

**A:** If you are in the IPython Notebook you can use the cell magic
``%%output info=True`` at the top of your code cell. This will
present the available style and plotting options for that object.

The same information is also available using
``holoviews.help(obj, visualization=True)``. For more
information on customizing the display of an object,
see our `Options Tutorial <Tutorials/Options>`_.


**Q: Why don't you let me pass** *matplotlib_option* **as a style
through to matplotlib?**

**A:** We have selected a subset of default allowable style options
that are most commonly useful in order to hide the more arcane
matplotlib options. If you do need to such an option to be passed to
the plotting system, you are welcome to declare that this is allowed.
For instance, say you may want the ``'filternom'`` option to be passed
to matplotlib's ``imshow`` command when displaying an ``Image``
element:

.. code:: python

  from holoviews import Store
  Store.add_style_opts(Image, ['filternorm'])

Now you can freely use ``'filternorm'`` in the ``%opts`` line/cell
magic, including tab-completion!

**Q: I still can't tweak my figure in exactly the way I want. What can I do?**

HoloViews is designed to be as flexible as possible and should always
support all the common visualization options. We intend to add a new 
tutorial soon to explain how you can continue to 
`tweak and extend HoloViews <https://github.com/ioam/holoviews/issues/19>`_.

**Q: How do I get a legend on my figure?**

**A:** Legends are generated in two different ways depending on the
Overlay type you are using. When using ``*`` to generate an ``Overlay``,
the legends are generated from the label of the Elements.
Alternatively, you can construct an ``NdOverlay``, where the key_dimensions
and values will become part of the legend. An example of an ``NdOverlay``
in action may be `viewed here <Tutorials/Containers.html#NdOverlay>`_.


**Q: I wish to use special characters in my title but then attribute
access becomes confusing.**

**A:** The title default of ``"{label} {group}"`` is simply a default
that you can override. If you want to use a lot of special characters
in your titles, you can pick simple ``group`` and ``label`` strings
that let you refer to the object easily in the code, and then you can
set the title directly, using the plot option
``title_format="my new title"``.


**Q: Where have my custom styles gone after unpickling my object?**

**A:** HoloViews objects are designed to pickle and unpickle your core
data, if you use Python's ``pickle.load`` and
``pickle.dump``. However, as custom options are kept separate from
your data, you need to use the corresponding methods ``Store.dump`` and
``Store.load`` if you want to save and restore per-object
customization. You can import ``Store`` from the main namespace with
``from holoviews import Store``.


**Q: Can I avoid generating extremely large HTML files when exporting
my notebook?**

**A:** It is very easy to visualize large volumes of data with
HoloViews, and all available display data is embedded in the HTML
snapshot when sliders are used. It is therefore worth being aware of
file size when authoring a notebook to be made make public on the
web. Useful tricks to reduce file size include:

* Reducing the figure size.
* Selecting fewer frames for display (e.g selecting a smaller number
  of keys in any displayed ``HoloMap`` object)
* Displaying your data in a more highly compressed format such as
  ``webm``, ``mp4`` or animated ``gif``, while being aware that those
  formats may introduce visible artifacts.

