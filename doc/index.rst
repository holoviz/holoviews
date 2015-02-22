.. holoviews documentation master file

Introduction
____________

**Composable, declarative data structures for building even complex visualizations easily.**

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
   # Wait a long while..then normalize
   arr = mandelbrot(4000,4000, maxit=2000)[400:800, 2500:2900]

..
  # Code needed to run the index.ipynb tutorial (add as two new cells at the top)
  
  cd ..

  import holoviews
  %load_ext holoviews.ipython
  %view holomap='widgets'
  import numpy

.. notebook:: holoviews index.ipynb


Installation
____________

HoloViews is compatible with Python `2.7, 3.3, and 3.4. <https://travis-ci.org/ioam/holoviews>`_.

HoloViews requires `Param <http://ioam.github.com/param/>`_ and
`Numpy <http://numpy.org>`_, neither of which has any required dependencies,
and so it should be very easy to integrate HoloViews into your
workflow or as part of another project.

For plotting, HoloViews requires `Matplotlib <http://http://matplotlib.org/>`_,
which most scientists and engineers using Python will already have
installed.  HoloViews is designed to work well with `IPython Notebook
<http://ipython.org/notebook/>`_ 2 and 3, although it can also be used
separately. 

Param, Matplotlib, and IPython Notebook can be installed using your
operating system's package manager, or by using pip::

  pip install matplotlib 'ipython[notebook]'
  pip install --user https://github.com/ioam/param/archive/master.zip

You can then obtain the latest version of holoviews by cloning the git
repository::

  git clone git://github.com/ioam/holoviews.git

|BuildStatus|_


Contributors
____________

HoloViews was developed by `Jean-Luc R. Stevens <https://github.com/jlstevens>`_
and `Philipp Rudiger <https://github.com/philippjfr>`_,
in collaboration with `James A. Bednar <https://github.com/jbednar>`_.

HoloViews is completely open source, available under a BSD license
freely for both commercial and non-commercial use.  Contributions from
users are welcome and encouraged.  In particular, HoloViews components
can be combined in an infinite number of ways, and so it is impossible
for us to test all conceivable combinations.  Thus we welcome 
`bug reports <https://github.com/ioam/holoviews/issues>`_,
particularly if they come with 
`pull requests <http://yangsu.github.io/pull-request-tutorial/>`_ 
showing how to fix the bug!

.. |BuildStatus| image:: https://travis-ci.org/ioam/holoviews.svg?branch=master
.. _BuildStatus: https://travis-ci.org/ioam/holoviews

