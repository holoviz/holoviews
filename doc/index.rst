.. holoviews documentation master file, created by
   sphinx-quickstart on Wed May 14 14:25:57 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

holoviews
=========

**Composable, declarative data structures for building even complex visualizations easily.**

Introduction
____________

Holoviews is a library that makes analyzing and visualizing scientific
or engineering data much simpler, more intuitive, and more reproducible.
Holoviews is based on `Matplotlib <http://http://matplotlib.org/>`_ and
`IPython Notebook <http://ipython.org/notebook/>`_, making the
combination of these two packages vastly more usable and powerful.

First, Holoviews provides a set of completely general sparse
hierarchical data structures for incrementally collecting results,
images, etc. from measurements or simulations.  It then defines a set
of fully customisable Matplotlib-based visualisations for any data
available in that data structure. The data can be sliced, selected,
combined, re-sorted, sampled, etc. very easily, and whatever you come
up with will just display itself with no further work from you.  For
instance, if what you have selected turns out to be a 2D array, it
will display as an image, but if it is 3D or 4D, it would be an
animation (automatically), and if you then sliced the 2D array along
the x axis you'd get a line plot (since you've reduced 2D to 1D).
I.e., the data just displays itself, in whatever form it is.  With
Holoviews, you can see precisely what you are interested in exploring,
without time spent on writing or maintaining plotting code.  Check out
how it works in our `Tutorials`_!

Although this functionality can be utilized without IPython Notebook,
it is most powerful when combined with the notebook interface.  As
shown in the `Tutorials`_, the notebook allows you to interleave code,
output, and graphics easily.  With holoviews, you can just put a
minimum of code in the notebook (typically one or two lines per
visualization), specifying what you would like to see rather than the
details of how it should be plotted.  This makes the IPython Notebook
a practical solution for both exploratory research (since viewing
nearly anything just takes a line or two of code) and for long-term
reproducibility of the work (because both the line or two of code and
the resulting figure are preserved in the notebook file forever).

Without Holoviews, notebooks can become filled with long and detailed
listings of Matplotlib-based code, which obscures the meaning of what
is being plotted and makes it much harder to explore the results.
This code is also locked into very specific types of visualizations,
which again make it harder to explore different aspects of the
results.  With Holoviews, nearly everything is very easily
visualizable as-is, with customization needed only when you want to
change specific aspects of each figure.  See the `Tutorials`_ for
detailed examples, and then start enjoying working with your data!


Installation
____________

holoviews requires `Param <http://ioam.github.com/param/>`_,
`Matplotlib <http://http://matplotlib.org/>`_ and is designed to work
with `IPython Notebook <http://ipython.org/notebook/>`_.  These
dependencies can be installed using your operating system's package
manager, or by using pip:

::

pip install matplotlib 'ipython[notebook]'
pip install --user https://github.com/ioam/param/archive/master.zip

You can then obtain the latest version of holoviews by cloning the git
repository::

   git clone git://github.com/ioam/holoviews.git

|BuildStatus|_


Contributors
____________

`Jean-Luc Stevens <https://github.com/jlstevens>`_

`Philipp Rudiger <https://github.com/philippjfr>`_

`James A. Bednar <https://github.com/jbednar>`_

.. |BuildStatus| image:: https://travis-ci.org/ioam/holoviews.svg?branch=master
.. _BuildStatus: https://travis-ci.org/ioam/holoviews
.. _Tutorials: Tutorials/
