*******************
HoloViews Tutorials
*******************

The HoloViews tutorials are the best way to learn what HoloViews can
do and how to use it.  The web site has static copies of each
tutorial, but it is much more effective to install HoloViews and try
it out live for yourself, using the notebook files in
"doc/Tutorials/*.ipynb" in the HoloViews distribution.

Introductory Tutorials
----------------------

These explanatory tutorials are meant to be viewed and worked through
in this order:

* `Showcase: <Showcase>`_
  Brief demonstration of what HoloViews can do for you and your data.

* `Introduction: <Introduction>`_
  How to use HoloViews -- basic concepts and getting started.

* `Exploring Data: <Exploring_Data>`_
  How to use HoloViews containers to flexibly hold all your data
  ready for selecting, sampling, slicing, viewing, combining, and
  animating.

* `Transforming Data: <Transforming_Data>`_
  How to transform data within HoloViews containers, e.g. by
  collapsing across dimensions. (Coming soon!)


Supplementary Tutorials
------------------------

There are additional tutorials detailing other features of HoloViews:

* `Options: <Options>`_
  Listing and changing the many options that control how HoloViews
  visualizes your objects, from Python or IPython.

* `Exporting: <Exporting>`_
  How to save HoloViews output for use in reports and publications,
  as part of a reproducible yet interactive scientific workflow.

* `Continuous Coordinates: <Continuous_Coordinates>`_
  Support for continuous coordinates offered by HoloViews, for use
  with real-world data.

* `Composing Data: <Composing_Data>`_
  Complete example of the full range of hierarchical, multidimensional
  discrete and continuous data structures provided by HoloViews.

* `Pandas and Seaborn: <Pandas_Seaborn>`_
  Using HoloViews with the external Pandas and Seaborn libraries.


Reference Notebooks
-------------------

At any point, you can look through these comprehensive but less
explanatory overview tutorials that simply list each of the objects
available and show examples:

* `Elements: <Elements>`_
  Overview and examples of all HoloViews element types, the atomic items
  that can be combined together.

* `Containers: <Containers>`_
  Overview and examples of all the HoloViews container types.

For more detailed (but less readable!) information on any component
described in these tutorials, please refer to the `Reference Manual
<../Reference_Manual>`_.


External Examples
-----------------

Finally, here are some real-world examples of HoloViews being used:

* `Interactive plots with mpld3: <http://philippjfr.com/blog/interactive-plots-with-holoviews-and-mpld3/>`_
  Example usage of the mpld3 rendering backend allowing you to
  generate D3.js-based plots with interactive widgets.

* `HoloViews for machine learning: Kaggle competition for EEG
  classification <http://philippjfr.com/blog/kaggle-bci-challenge-visualizing-eeg-data-in-holoviews/>`_

* `ImaGen library <http:https://github.com/ioam/imagen>`_:  Generates
  HoloViews Image and RGB patterns from mathematical functions.

* Neural simulator tutorials using HoloViews: `Topographica <http://topographica.org/Tutorials/>`_.
  (also see publications list, e.g. 
  `Gee (2014) <http://homepages.inf.ed.ac.uk/jbednar/papers/gee.ms14.pdf>`_)


.. toctree::
   :maxdepth: 2
   :hidden:

   Showcase
   Introduction
   Exploring Data <Exploring_Data>
   Options <Options>
   Transforming Data <Transforming_Data>
   Exporting
   Elements
   Containers
   Pandas and Seaborn <Pandas_Seaborn>
   Continuous Coordinates <Continuous_Coordinates>
   Composing Data <Composing_Data>
