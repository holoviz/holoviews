|BuildStatus|_ |holoviewsDocs|_ |PyPI|_ |License|_  |Coveralls|_ |Downloads|_ |Gitter|_

holoviews
=========

**Stop plotting your data - annotate your data and let it visualize itself.**

.. image:: http://assets.holoviews.org/demo.gif
   :target: http://www.holoviews.org

HoloViews requires `Param <http://ioam.github.com/param/>`_ and
`Numpy <http://www.numpy.org/>`_ and is designed to work 
together with `Matplotlib <http://matplotlib.org/>`_ and 
`IPython Notebook <http://ipython.org/notebook/>`_.  

Clone holoviews directly from GitHub with::

   git clone git://github.com/ioam/holoviews.git
   
Please visit `our website <http://ioam.github.com/holoviews/>`_ for
official releases, installation instructions, documentation, and examples.

For general discussion, we have a `gitter channel <https://gitter.im/ioam/holoviews>`_.
In addition
we have a `wiki page <https://github.com/ioam/holoviews/wiki/Experimental-Features>`_
describing current work-in-progress and experimental features. If you find any bugs or 
have any feature suggestions please file a GitHub Issue or submit a pull request.

Features
--------

**Overview**

* Lets you build data structures that both contain and visualize your data.
* Includes a rich `library of composable elements <https://ioam.github.io/holoviews/Tutorials/Elements>`_ that can be overlaid, nested and positioned with ease.
* Supports `rapid data exploration <https://ioam.github.io/holoviews/Tutorials/Exploring_Data>`_ that naturally develops into a `fully reproducible workflow <Tutorials/Exporting>`_.
* You can create complex animated or interactive visualizations with minimal code.
* Rich semantics for `indexing and slicing of data in arbitrarily high-dimensional spaces <https://ioam.github.io/holoviews/Tutorials/Transforming_Data>`_.
* Every parameter of every object includes easy-to-access documentation.
* All features `available in vanilla Python 2 or 3 <https://ioam.github.io/holoviews/Tutorials/Options>`_, with minimal dependencies.

**Support for maintainable, reproducible research**
  
* Supports a truly reproducible workflow by minimizing the code needed for analysis and visualization.
* Already used in a variety of research projects, from conception to final publication.
* All HoloViews objects can be pickled and unpickled.
* Provides comparison utilities for testing, so you know when your results have changed and why.
* Core data structures only depend on the numpy and param libraries.
* Provides `export and archival facilities <https://ioam.github.io/holoviews/Tutorials/Exporting>`_ for keeping track of your work throughout the lifetime of a project.

**Analysis and data access features**

* Allows you to annotate your data with dimensions, units, labels and data ranges.
* Easily `slice and access <https://ioam.github.io/holoviews/Tutorials/Transforming_Data>`_ regions of your data, no matter how high the dimensionality.
* Apply any suitable function to collapse your data or reduce dimensionality.
* Helpful textual representation to inform you how every level of your data may be accessed.
* Includes small library of common operations for any scientific or engineering data.
* Highly extensible: add new operations to easily apply the data transformations you need.

**Visualization features**

* Useful default settings make it easy to inspect data, with minimal code.
* Powerful normalization system to make understanding your data across plots easy.
* Build `complex animations or interactive visualizations in seconds  <https://ioam.github.io/holoviews/Tutorials/Exploring_Data>`_ instead of hours or days.
* Refine the visualization of your data interactively and incrementally.
* Separation of concerns: all visualization settings are kept separate from your data objects.
* Support for interactive tooltips/panning/zooming, via the optional mpld3 backend.

**IPython Notebook support**

* Support for both IPython 2 and 3.
* Automatic tab-completion everywhere.
* Exportable sliders and scrubber widgets.
* Automatic display of animated formats in the notebook or for export, including gif, webm, and mp4.
* Useful IPython magics for configuring global display options and for customizing objects.
* `Automatic archival and export of notebooks <https://ioam.github.io/holoviews/Tutorials/Exporting>`_, including extracting figures as SVG, generating a static HTML copy of your results for reference, and storing your optional metadata like version control information.

**Integration with third-party libraries**  

* Flexible interface to both the `pandas and Seaborn libraries <https://ioam.github.io/holoviews/Tutorials/Pandas_Seaborn>`_
* Immediately visualize pandas data as any HoloViews object.
* Seamlessly combine and animate your Seaborn plots in HoloViews rich, compositional data-structures.
   

.. |PyPI| image:: https://img.shields.io/pypi/v/holoviews.svg
.. _PyPI: https://pypi.python.org/pypi/holoviews

.. |License| image:: https://img.shields.io/pypi/l/holoviews.svg
.. _License: https://github.com/ioam/holoviews/blob/master/LICENSE.txt

.. |Coveralls| image:: https://img.shields.io/coveralls/ioam/holoviews.svg
.. _Coveralls: https://coveralls.io/r/ioam/holoviews

.. |BuildStatus| image:: https://travis-ci.org/ioam/holoviews.svg?branch=master
.. _BuildStatus: https://travis-ci.org/ioam/holoviews

.. |holoviewsDocs| image:: http://buildbot.holoviews.org:8010/png?builder=website
.. _holoviewsDocs: http://buildbot.holoviews.org:8010/waterfall

.. |Downloads| image:: https://img.shields.io/pypi/dm/holoviews.svg
.. _Downloads: https://pypi.python.org/pypi/holoviews

.. |Gitter| image:: https://badges.gitter.im/Join%20Chat.svg
.. _Gitter: https://gitter.im/ioam/holoviews?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
