.. HoloViews documentation master file

HoloViews
____________

**Stop plotting your data - annotate your data and let it visualize itself.**

HoloViews is an `open-source <https://github.com/ioam/holoviews/blob/master/LICENSE.txt>`_ Python library designed to make data analysis and visualization seamless and simple.  With HoloViews, you can usually express what you want to do in very few lines of code, letting you focus on what you are trying to explore and convey, not on the process of plotting.

For examples, check out the thumbnails below and the other items in the `Gallery <gallery>`_ of demos and apps and the `Reference Gallery <reference>`_ that shows every HoloViews component.  Be sure to look at the code, not just the pictures, to appreciate how easy it is to create such plots yourself!

The `Getting-Started <getting_started>`_ guide explains the basic concepts and how to start using HoloViews, and is the recommended way to understand how everything works.

The `User Guide <user_guide>`_ goes more deeply into key concepts from HoloViews, when you are ready for further study.

The `API <Reference_Manual>`_ is the definitive guide to each HoloViews object, but the same information is available more conveniently via the `hv.help()` command and tab completion in the Jupyter notebook.

If you have any `issues <https://github.com/ioam/holoviews/issues>`_ or wish to `contribute code <https://help.github.com/articles/about-pull-requests>`_, you can visit our `GitHub site <https://github.com/ioam/holoviews>`_ or chat with the developers on `gitter <https://gitter.im/ioam/holoviews>`_.

.. raw:: html
  :file: latest_news.html


.. raw:: html

   <div>
   <div >
     <a href="http://holoviews.org/gallery/demos/bokeh/iris_splom_example.html">
       <img src="http://holoviews.org/_images/iris_splom_example.png" width='20%'>    </img> </a>
     <a href="http://holoviews.org/getting_started/Gridded_Datasets.html">
       <img src="http://assets.holoviews.org/collage/cells.png" width='22%'> </img>  </a>
     <a href="http://holoviews.org/gallery/demos/bokeh/scatter_economic.html">
       <img src="http://holoviews.org/_images/scatter_economic.png" width='43%'> </img>    </a>
   </div>

   <div >
     <a href="http://holoviews.org/gallery/demos/bokeh/square_limit.html">
       <img src="http://holoviews.org/_images/square_limit.png" width='20%'> </a>
     <a href="http://holoviews.org/gallery/demos/bokeh/bars_economic.html">
       <img src="http://holoviews.org/_images/bars_economic.png" width='20%'> </a>
     <a href="http://holoviews.org/gallery/demos/bokeh/texas_choropleth_example.html">
       <img src="http://holoviews.org/_images/texas_choropleth_example.png"    width='20%'> </a>
     <a href="http://holoviews.org/gallery/demos/bokeh/verhulst_mandelbrot.html">
       <img src="http://holoviews.org/_images/verhulst_mandelbrot.png" width='20%'>    </a>
   </div>
   <div >
       <a href="http://holoviews.org/gallery/demos/bokeh/dropdown_economic.html">
         <img src="http://assets.holoviews.org/collage/dropdown.gif" width='31%'> </a>
       <a href="http://holoviews.org/gallery/demos/bokeh/dragon_curve.html">
         <img src="http://assets.holoviews.org/collage/dragon_fractal.gif" width='26%'> </a>
       <a href="http://holoviews.org/gallery/apps/bokeh/nytaxi_hover.html">
         <img src="http://assets.holoviews.org/collage/ny_datashader.gif" width='31%'> </a>
   </div>
   </div>


Installation
____________

|CondaPkg|_ |PyPI|_ |License|_ |Coveralls|_


HoloViews works with `Python 2.7 and Python 3 <https://travis-ci.org/ioam/holoviews>`_ on Linux, Windows, or Mac, and provides optional extensions for working with the `Jupyter/IPython Notebook <http://jupyter.org>`_.

The recommended way to install HoloViews is using the `conda <http://conda.pydata.org/docs/>`_ command provided by `Anaconda <http://docs.continuum.io/anaconda/install>`_ or `Miniconda <http://conda.pydata.org/miniconda.html>`_::

  conda install -c ioam holoviews bokeh

This command will install the typical packages most useful with HoloViews, though HoloViews itself
directly depends only on `Numpy <http://numpy.org>`_ and `Param <http://ioam.github.com/param/>`_.
Additional installation and configuration options are described in the
`user guide <user_guide/Installing_and_Configuring.html>`_.


Usage
-----

Once you've installed HoloViews, you can get a copy of all the examples shown on this website::

  holoviews --install-examples
  cd holoviews-examples

And then you can launch Jupyter Notebook to explore them::

  jupyter notebook --NotebookApp.iopub_data_rate_limit=100000000

(Increasing the rate limit in this way is `required for the current 5.0 Jupyter version <user_guide/Installing_and_Configuring.html>`_, but should not be needed in later Jupyter releases.)


.. |PyPI| image:: https://img.shields.io/pypi/v/holoviews.svg
.. _PyPI: https://pypi.python.org/pypi/holoviews

.. |CondaPkg| image:: https://anaconda.org/ioam/holoviews/badges/installer/conda.svg
.. _CondaPkg: https://anaconda.org/ioam/holoviews

.. |License| image:: https://img.shields.io/pypi/l/holoviews.svg
.. _License: https://github.com/ioam/holoviews/blob/master/LICENSE.txt

.. |Coveralls| image:: https://img.shields.io/coveralls/ioam/holoviews.svg
.. _Coveralls: https://coveralls.io/r/ioam/holoviews

.. toctree::
   :titlesonly:
   :hidden:
   :maxdepth: 2

   Home <self>
   Getting Started <getting_started/index>
   User Guide <user_guide/index>
   Reference Gallery <reference/index>
   Showcase <showcase/index>
   API <Reference_Manual/index>
   FAQ
   Github source <http://github.com/ioam/holoviews>
   About <about>
