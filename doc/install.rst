Installing HoloViews
====================

The quickest and easiest way to get the latest version of all the
recommended packages for working with HoloViews on Linux, Windows, or
Mac systems is via the
`conda <http://conda.pydata.org/docs/>`_ command provided by 
the
`Anaconda <http://docs.continuum.io/anaconda/install>`_ or
`Miniconda <http://conda.pydata.org/miniconda.html>`_ scientific
Python distributions::

  conda install -c ioam holoviews bokeh

This recommended installation includes the default `Matplotlib
<http://matplotlib.org>`_ plotting library backend, the
more interactive `Bokeh <http://bokeh.pydata.org>`_ plotting library
backend, and the `Jupyter/IPython Notebook <http://jupyter.org>`_.

A similar set of packages can be installed using ``pip``, if that
command is available on your system::

  pip install 'holoviews[recommended]'

``pip`` also supports other installation options, including a minimal
install of only the packages necessary to generate and manipulate
HoloViews objects without visualization::

  pip install holoviews

This minimal install includes only the two required libraries `Param
<http://ioam.github.com/param/>`_ and `Numpy <http://numpy.org>`_,
neither of which has any required dependencies, which makes it very
easy to integrate HoloViews into your workflow or as part of another
project.

Alternatively, you can ask ``pip`` to install a larger set of
packages that provide additional functionality in HoloViews::

  pip install 'holoviews[extras]'

This option installs all the required and recommended packages,
including the `pandas <http://pandas.pydata.org/>`_ and `Seaborn
<http://stanford.edu/~mwaskom/software/seaborn/>`_ libraries.

Lastly, to get *everything*, including `cyordereddict
<https://pypi.python.org/pypi/cyordereddict>`_ to enable optional
speed optimizations and `nose <https://pypi.python.org/pypi/nose/>`_
for running unit tests, you can use::

  pip install 'holoviews[all]'

Between releases, development snapshots are made available on conda and
can be installed using::

  conda install -c ioam/label/dev holoviews

To get the very latest development version you can clone our git
repositories::

  git clone git://github.com/ioam/param.git
  git clone git://github.com/ioam/holoviews.git

Once you've installed HoloViews, you can get started by launching
Jupyter Notebook::

  jupyter notebook

Now you can download the `tutorial notebooks`_.  unzip them somewhere
Jupyter Notebook can find them, and then open the Homepage.ipynb
tutorial or any of the others in the Notebook.  Enjoy exploring your
data!

