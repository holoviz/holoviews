.. HoloViews documentation master file

.. raw:: html

  <h1><img src="_static/logo_horizontal.png" style="width: 30%;"></h1>

**Stop plotting your data - annotate your data and let it visualize itself.**

.. raw:: html

  <div style="display: flex">
    <div style="width: 70%">

HoloViews is an `open-source <https://github.com/holoviz/holoviews/blob/master/LICENSE.txt>`_ Python library designed to make data analysis and visualization seamless and simple.  With HoloViews, you can usually express what you want to do in very few lines of code, letting you focus on what you are trying to explore and convey, not on the process of plotting.

For examples, check out the thumbnails below and the other items in the `Gallery <gallery>`_ of demos and apps and the `Reference Gallery <reference>`_ that shows every HoloViews component.  Be sure to look at the code, not just the pictures, to appreciate how easy it is to create such plots yourself!

The `Getting-Started <getting_started>`_ guide explains the basic concepts and how to start using HoloViews, and is the recommended way to understand how everything works.

The `User Guide <user_guide>`_ goes more deeply into key concepts from HoloViews, when you are ready for further study.

The `API <reference_manual>`_ is the definitive guide to each HoloViews object, but the same information is available more conveniently via the `hv.help()` command and tab completion in the Jupyter notebook.

If you have any `issues <https://github.com/holoviz/holoviews/issues>`_ or wish to `contribute code <https://help.github.com/articles/about-pull-requests>`_, you can visit our `GitHub site <https://github.com/holoviz/holoviews>`_ or file a topic on the `HoloViz Discourse <https://discourse.holoviz.org/>`_.

.. raw:: html

  </div>

.. raw:: html
  :file: latest_news.html

.. raw:: html

  </div>
  <hr width='100%'></hr>

.. raw:: html

   <div>
   <div >
     <a href="https://holoviews.org/gallery/demos/bokeh/iris_splom_example.html">
       <img src="https://holoviews.org/_images/iris_splom_example.png" width='24%'>    </img> </a>
     <a href="https://holoviews.org/getting_started/Gridded_Datasets.html">
       <img src="https://assets.holoviews.org/collage/cells.png" width='27%'> </img>  </a>
     <a href="https://holoviews.org/gallery/demos/bokeh/scatter_economic.html">
       <img src="https://holoviews.org/_images/scatter_economic.png" width='47%'> </img>    </a>
   </div>

   <div >
     <a href="https://holoviews.org/gallery/demos/bokeh/square_limit.html">
       <img src="https://holoviews.org/_images/square_limit.png" width='24%'> </a>
     <a href="https://holoviews.org/gallery/demos/bokeh/bars_economic.html">
       <img src="https://holoviews.org/_images/bars_economic.png" width='24%'> </a>
     <a href="https://holoviews.org/gallery/demos/bokeh/texas_choropleth_example.html">
       <img src="https://holoviews.org/_images/texas_choropleth_example.png"    width='24%'> </a>
     <a href="https://holoviews.org/gallery/demos/bokeh/verhulst_mandelbrot.html">
       <img src="https://holoviews.org/_images/verhulst_mandelbrot.png" width='24%'>    </a>
   </div>
   <div >
       <a href="https://holoviews.org/gallery/demos/bokeh/dropdown_economic.html">
         <img src="https://assets.holoviews.org/collage/dropdown.gif" width='33%'> </a>
       <a href="https://holoviews.org/gallery/demos/bokeh/dragon_curve.html">
         <img src="https://assets.holoviews.org/collage/dragon_fractal.gif" width='30%'> </a>
       <a href="https://holoviews.org/gallery/apps/bokeh/nytaxi_hover.html">
         <img src="https://assets.holoviews.org/collage/ny_datashader.gif" width='33%'> </a>
   </div>
   </div>


Installation
____________

|CondaPkg|_ |PyPI|_ |License|_ |Coveralls|_


HoloViews works with `Python 2.7 and Python 3 <https://github.com/holoviz/holoviews/actions?query=workflow%3Apytest>`_ on Linux, Windows, or Mac, and works seamlessly with `Jupyter Notebook and JupyterLab <https://jupyter.org>`_.

The recommended way to install HoloViews is using the `conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_ command provided by `Anaconda <https://docs.anaconda.com/anaconda/install/>`_ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_::

  conda install -c pyviz holoviews bokeh

This command will install the typical packages most useful with HoloViews, though HoloViews itself
directly depends only on `Numpy <https://numpy.org>`_, `Pandas <https://pandas.pydata.org>`_ and `Param <https://param.holoviz.org/>`_.

Additional installation and configuration options are described in the
`user guide <user_guide/Installing_and_Configuring.html>`_.

Additional methods of installation, including different ways to use
``pip`` can be found in the `installation guide <install.html>`_.

Usage
-----

Once you've installed HoloViews, you can get a copy of all the examples shown on this website::

  holoviews --install-examples
  cd holoviews-examples

Now you can launch Jupyter Notebook or JupyterLab to explore them::

  jupyter notebook

  jupyter lab

If you are working with a JupyterLab version <2.0 you will also need the PyViz JupyterLab
extension::

  jupyter labextension install @pyviz/jupyterlab_pyviz

For more details on installing and configuring HoloViews see `the installing and configuring guide <user_guide/Installing_and_Configuring.html>`_.

After you have successfully installed and configured HoloViews, please see `Getting Started <getting_started/index.html>`_.


.. |PyPI| image:: https://img.shields.io/pypi/v/holoviews.svg
.. _PyPI: https://pypi.python.org/pypi/holoviews

.. |CondaPkg| image:: https://anaconda.org/pyviz/holoviews/badges/installer/conda.svg
.. _CondaPkg: https://anaconda.org/pyviz/holoviews

.. |License| image:: https://img.shields.io/pypi/l/holoviews.svg
.. _License: https://github.com/holoviz/holoviews/blob/master/LICENSE.txt

.. |Coveralls| image:: https://img.shields.io/coveralls/holoviz/holoviews.svg
.. _Coveralls: https://coveralls.io/r/holoviz/holoviews

.. toctree::
   :titlesonly:
   :hidden:
   :maxdepth: 2

   Home <self>
   Getting Started <getting_started/index>
   User Guide <user_guide/index>
   Reference Gallery <reference/index>
   Releases <releases>
   API <reference_manual/index>
   FAQ
   Roadmap <roadmap>
   Github source <https://github.com/holoviz/holoviews>
   About <about>
