.. HoloViews documentation master file

.. raw:: html
  :file: latest_news.html

Introduction
____________

**Stop plotting your data - annotate your data and let it visualize itself.**


..
   # Code used to generate mandelbrot.npy, for reference
   from numpy import *
   import pylab

   def mandelbrot( h,w, maxit=200 ):
           y,x = ogrid[ -1.4:1.4:h*1j, -2:0.8:w*1j ]
           c = x+y*1j
           z = c
           divtime = maxit + zeros(z.shape, dtype=int)
           for i in xrange(maxit):
                   z  = z**2 + c
                   diverge = z*conj(z) > 2**2
                   div_now = diverge & (divtime==maxit)
                   divtime[div_now] = i
                   z[diverge] = 2
           return divtime
   # Wait a long time..., then normalize
   arr = mandelbrot(4000,4000, maxit=2000)[400:800, 2500:2900]


..

.. notebook:: holoviews Homepage.ipynb

------------

Installation
____________

HoloViews is compatible with Python `2.7, 3.3, and 3.4 <https://travis-ci.org/ioam/holoviews>`_.

HoloViews requires `Param <http://ioam.github.com/param/>`_ and `Numpy
<http://numpy.org>`_, neither of which has any required dependencies,
and so it should be very easy to integrate HoloViews into your
workflow or as part of another project.

For a minimal install, you can obtain HoloViews along with the latest
public releases of its core dependencies (`Param
<http://ioam.github.com/param/>`_ and `Numpy <http://numpy.org>`_)
using pip::

  pip install holoviews


For plotting, HoloViews uses `Matplotlib <http://matplotlib.org/>`_,
as its default backend, which most scientists and engineers
using Python will already have installed.  HoloViews is pure Python,
but it also provides optional extensions enabled with ``%load_ext``
above that make it integrate well with `IPython Notebook
<http://ipython.org/notebook>`_ 2 and 3.

The quickest and easiest way to get the latest version of all the
packages recommended for working with HoloViews in the IPython
Notebook is via `conda <http://conda.pydata.org/docs/>`_ (e.g using
`miniconda <http://conda.pydata.org/miniconda.html>`_)::

  conda install -c ioam holoviews

Alternatively, you can also use pip::

  pip install 'holoviews[recommended]'

This will install Matplotlib and IPython Notebook if they are not
already available as part of a scientific Python distribution such as
`Anaconda <http://continuum.io/downloads>`_ (particularly convenient
on systems shipped without pip, such as Windows or Mac).

We also support the following install option::

  pip install 'holoviews[extras]'

In addition to the required and recommended packages, this command also
installs the optional `mpld3 <http://mpld3.github.io/>`_, `pandas
<http://pandas.pydata.org/>`_ and `Seaborn
<http://stanford.edu/~mwaskom/software/seaborn/index.html>`_
libraries.

Lastly, to get everything including `cyordereddict
<https://pypi.python.org/pypi/cyordereddict>`_ to enable optional
speed optimizations and `nose <https://pypi.python.org/pypi/nose/>`_
for running unit tests, you can use::

  pip install 'holoviews[all]'

To get the latest development version you can instead clone our git
repositories::

  git clone git://github.com/ioam/param.git
  git clone git://github.com/ioam/holoviews.git

Once you've installed HoloViews, you can get started by launching
IPython Notebook::

  ipython notebook

Now you can download the `tutorial notebooks`_.  unzip them somewhere
IPython Notebook can find them, and then open the Homepage.ipynb
tutorial or any of the others in the Notebook.  Enjoy exploring your
data!

|PyPI|_ |License|_  |Coveralls|_

------------

Contributors
____________

HoloViews was developed by `Jean-Luc R. Stevens <https://github.com/jlstevens>`_
and `Philipp Rudiger <https://github.com/philippjfr>`_,
in collaboration with `James A. Bednar <http://homepages.inf.ed.ac.uk/jbednar>`_.

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

.. |License| image:: https://img.shields.io/pypi/l/holoviews.svg
.. _License: https://github.com/ioam/holoviews/blob/master/LICENSE.txt

.. |Coveralls| image:: https://img.shields.io/coveralls/ioam/holoviews.svg
.. _Coveralls: https://coveralls.io/r/ioam/holoviews

.. toctree::
   :titlesonly:
   :hidden:
   :maxdepth: 2

   Home <self>
   Features <features>
   Tutorials <Tutorials/index>
   Examples <Examples/index>
   Reference Manual <Reference_Manual/index>
   FAQ
   Github source <http://github.com/ioam/holoviews>
   About <about>
