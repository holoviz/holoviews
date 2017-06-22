.. HoloViews documentation master file

.. raw:: html
  :file: latest_news.html

Introduction
____________

**Stop plotting your data - annotate your data and let it visualize itself.**

.. raw:: html

   <div>
   <div ><img src="http://assets.holoviews.org/collage/iris.png" width='20%'>
                    <img src="http://assets.holoviews.org/collage/cells.png" width='22%'>
                    <img src="http://assets.holoviews.org/collage/scatter_example.png"
                    width='43%'></div>
   <div ><img src="http://assets.holoviews.org/collage/square_limit.png"
         width='20%'>
                    <img src="http://assets.holoviews.org/collage/bars_example.png"
                    width='20%'>
                    <img src="http://assets.holoviews.org/collage/texas.png" width='20%'>
                    <img src="http://assets.holoviews.org/collage/mandelbrot.png"
                    width='20%'></div>
   <div><img src="http://assets.holoviews.org/collage/dropdown.gif"
   width='31%'>
                    <img src="http://assets.holoviews.org/collage/dragon_fractal.gif"
                    width='26%'>
                    <img src="http://assets.holoviews.org/collage/ny_datashader.gif"
                    width='31%'></div>
   </div>
     
------------

Installation
____________

HoloViews works with Python `2.7 and Python 3
<https://travis-ci.org/ioam/holoviews>`_.  HoloViews is pure Python, but
it also provides optional extensions enabled with
``hv.notebook_extension()`` above that make it integrate well with
`Jupyter/IPython Notebook <http://jupyter.org>`_ versions 3, 4 and 5.

The quickest and easiest way to get the latest version of all the
recommended packages for working with HoloViews on Linux, Windows, or
Mac systems is via the
`conda <http://conda.pydata.org/docs/>`_ command provided by 
the `Anaconda <http://docs.continuum.io/anaconda/install>`_ or
`Miniconda <http://conda.pydata.org/miniconda.html>`_ scientific
Python distributions::

  conda install -c ioam holoviews bokeh

See our `installation page <install.html>`_ if you need other options,
including `pip <https://pip.pypa.io/en/stable/installing>`_
installations, additional packages, development versions, and minimal
installations.  Minimal installations include only `Param
<http://ioam.github.com/param/>`_ and `Numpy <http://numpy.org>`_ as
dependencies, neither of which has any required dependencies, making
it simple to generate HoloViews objects from within your own code.


|PyPI|_ |Conda|_ |License|_  |Coveralls|_


Usage
-----

Once you've installed HoloViews, you can get started by launching
Jupyter Notebook::

  jupyter notebook

Now you can download the `tutorial notebooks`_.  unzip them somewhere
Jupyter Notebook can find them, and then open the Homepage.ipynb
tutorial or any of the others in the Notebook.  Enjoy exploring your
data!

Note: When running HoloViews in Jupyter Notebook 5.0 a data rate limit
was introduced which severely limits the output that HoloViews can
display.  This limit will be removed again in the upcoming 5.1
release, in the meantime you can raise the limit manually by
overriding the default ``iopub_data_rate_limit``::

   jupyter notebook --NotebookApp.iopub_data_rate_limit=100000000

Alternatively you can set a higher default in the user configuration file
in ``~/.jupyter/jupyter_notebook_config.py``, by adding::

   c.NotebookApp.iopub_data_rate_limit=100000000

If the configuration file does not exist generate one first using::

   jupyter notebook --generate-config


------------

Contributors
____________

HoloViews is developed by `Jean-Luc R. Stevens <https://github.com/jlstevens>`_
and `Philipp Rudiger <https://github.com/philippjfr>`_,
in collaboration with `James A. Bednar <http://homepages.inf.ed.ac.uk/jbednar>`_,
with support from `Continuum Analytics <https://continuum.io>`_.

HoloViews is completely `open source
<https://github.com/ioam/holoviews>`_, available under a BSD license
freely for both commercial and non-commercial use.  HoloViews is 
designed to be easily extensible, and contributions from
users are welcome and encouraged.  In particular, HoloViews components
can be combined in an infinite number of ways, and although the
tutorials are tested continuously, it is impossible
for us to test all other conceivable combinations.  Thus we welcome `bug
reports and feature requests <https://github.com/ioam/holoviews/issues>`_, 
particularly if they come with test cases showing how to reproduce the bugs and 
`pull requests <http://yangsu.github.io/pull-request-tutorial/>`_
showing how to fix the bug or implement the feature!

.. |PyPI| image:: https://img.shields.io/pypi/v/holoviews.svg
.. _PyPI: https://pypi.python.org/pypi/holoviews

.. |Conda| image:: https://anaconda.org/ioam/holoviews/badges/installer/conda.svg
.. _Conda: https://anaconda.org/ioam/holoviews

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
   Features <features>
   Tutorials <Tutorials/index>
   Examples <Examples/index>
   Reference Manual <Reference_Manual/index>
   FAQ
   Github source <http://github.com/ioam/holoviews>
   About <about>
