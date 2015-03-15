FAQ
===

Here is a list of questions we have either been asked by users or potential pitfalls we hope to help users avoid:

**Q: Can I use HoloViews without IPython?**

**A:** Yes! All the basic syntax for constructing, slicing and indexing our objects work in standard Python. The key differences are how you customize the options associated with your object and how you export your object to disk. You can do both at once as follows:

```python
from holoviews import Store
renderer = Store.renderer.instance(fig='svg', holomap='gif')
renderer.save(my_object, 'example_I', style=dict(Image={'cmap':'jet'}))
```

More information is available on `this wiki page <https://github.com/ioam/holoviews/wiki/HoloViews-without-IPython>`_ and in this tutorial [LINK]. You may also use the ``holoviews.archive`` object directly (although it won't be able to automatically capture your data or export notebook snapshots).

**Q: Help! I don't know how to index into my object!**

**A:** If you are in the IPython Notebook you can use the cell magic ``%%output info=True`` at the top of your code cell.

In a regular Python session, you can look at ``print repr(obj)``. For an explanation of how this helps you index your object `see our tutorial <https://ioam.github.io/holoviews/Tutorials/Composing_Data.html>`_.

**Q: Help! How do I find out the options for customizing the appearance of my object?**

**A:**  If you are in the IPython Notebook you can use the cell magic ``%%output info=True`` at the top of your code cell. This will present the available style and plotting options.

You can also view all this information available using ``Store.info(obj)``. For more information on customizing the display of an object, see our `Options Tutorial [NO LINK]`.

**Q: Why don't you let me pass** *matplotlib_option* **as a style through to matplotlib?**

**A:** We have selected a subset of default allowable style options that are most commonly useful in order to hide the more arcane matplotlib options. If you need to such an option to be passed to the plotting system, you can declare this intent. For instance, say you may want the ``'filternom'`` option to be passed to matplotlib's ``imshow`` command when displaying an ``Image`` element:

```python
from holoviews import Store
Store.add_style_opts(Image, ['filternorm'])
```

Now you can freely use ``'filternorm'``in the ``%opts`` line/cell magic, including tab-completion!


**Q: How do I get a legend on my figure?**

**A:** Overlaying is a very general operation designed to work across different Element type. In order to get a legend, use an the ``NdOverlay`` class instead. An example of an ``NdOverlay`` in action may be `viewed here <https://ioam.github.io/holoviews/Tutorials/Containers.html#NdOverlay>`_

**Q: Where have my custom styles gone after unpickling my object?**

**A:** HoloViews objects are designed to pickle and unpickle your core data using regular ``pickle.load`` and ``pickle.dump``. However, as custom options are kept separate from your data, you need to use ``Store.dump`` and ``Store.load`` to save and restore per-object customization. You can import ``Store`` from the main namespace with ``from holoviews import Store``.

**Q: Can I avoid generating extremely large HTML files when exporting my notebook?**

**A:** It is very easy to visualize large volumes of data with HoloViews and all available display data is embedded in the HTML snapshot when sliders are used. It is therefore worth being aware of file size when authoring a notebook to be made make public on the web. Useful tricks to reduce file size include:

* Reducing the figure size.
* Selecting fewer frames for display (e.g selecting a smaller number of keys in any displayed ``HoloMap`` object)
* Displaying your data in a more highly compressed format such as ``webm``, ``mp4`` or ``gif``.
