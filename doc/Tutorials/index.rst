*******************
HoloViews Tutorials
*******************

The HoloViews tutorials are the best way to learn what HoloViews can do
and how to use it.  The web site has static copies of each tutorial, but
you may also try out live copies on `mybinder.org
<http://mybinder.org/repo/ioam/holoviews-contrib>`_ where you can also
explore many other examples in our `contrib repository
<https://github.com/ioam/holoviews-contrib>`_. Lastly, for the most
responsive experience, you can install HoloViews and try out the
`tutorial notebooks`_ for yourself.


Introductory Tutorials
----------------------

These explanatory tutorials are meant to be viewed and worked through
in this order:

* `Showcase: <Showcase.html>`_
  Brief demonstration of what HoloViews can do for you and your data.

* `Introduction: <Introduction.html>`_
  How to use HoloViews -- basic concepts and getting started.

* `Exploring Data: <Exploring_Data.html>`_
  How to use HoloViews containers to flexibly hold all your data
  ready for selecting, sampling, slicing, viewing, combining, and
  animating.

* `Sampling Data: <Sampling_Data.html>`_
  How to select data in multiple dimensions, returning a specific
  (potentially lower dimensional) region of the available space.

* `Columnar Data: <Columnar_Data.html>`_
  How to work with table-like data, including options for storing the
  data, and how to apply operations to transform the data into 
  complex visualizations easily.

* `Dynamic Map: <Dynamic_Map.html>`_
  How to work with datasets larger than the available memory by
  computing elements on-the-fly. Using DynamicMap you can immediately
  begin exploring huge volumes of data while keeping interaction
  responsive and without running out of memory.

Supplementary Tutorials
------------------------

There are additional tutorials detailing other features of HoloViews:

* `Options: <Options.html>`_
  Listing and changing the many options that control how HoloViews
  visualizes your objects, from Python or IPython.

* `Exporting: <Exporting.html>`_
  How to save HoloViews output for use in reports and publications,
  as part of a reproducible yet interactive scientific workflow.

* `Continuous Coordinates: <Continuous_Coordinates.html>`_
  How to use continuous coordinates to work with real-world data or
  smooth functions.

* `Composing Data: <Composing_Data.html>`_
  Complete example of the full range of hierarchical, multidimensional
  discrete and continuous data structures supported by HoloViews.

* `Bokeh Backend: <Bokeh_Backend.html>`_
  Additional interactivity available via the
  `Bokeh <http://bokeh.pydata.org>`_ backend, such as interactive zooming,
  panning, and selection linked automatically between plots.

* `Pandas and Seaborn: <Pandas_Seaborn.html>`_
  Specialized visualizations provided by pandas and seaborn.


Reference Notebooks
-------------------

At any point, you can look through these comprehensive but less
explanatory overview tutorials.  For each of the HoloViews components
available, these tutorials show how to create it, how the objects are
plotted by default, and show how to list and change all of the
visualization options for that object type:

* `Elements: <Elements.html>`_
  Overview and examples of all HoloViews element types, the atomic items
  that can be combined together, available for either the
  `Matplotlib <Elements.html>`_ or `Bokeh <Bokeh_Elements.html>`_ plotting
  library backends. 

* `Containers: <Containers.html>`_
  Overview and examples of all the HoloViews container types.

For more detailed (but less readable!) information on any component
described in these tutorials, please refer to the `Reference Manual
<../Reference_Manual.html>`_. For further notebooks demonstrating how to
extend HoloViews and apply it to real world data see the `Examples
<../Examples.html>`_ page.

.. toctree::
   :maxdepth: 2
   :hidden:

   Showcase
   Introduction
   Exploring Data <Exploring_Data>
   Sampling Data <Sampling_Data>
   Columnar Data <Columnar_Data>
   Dynamic Map <Dynamic_Map>
   Options
   Exporting
   Continuous Coordinates <Continuous_Coordinates>
   Composing Data <Composing_Data>
   Bokeh Backend <Bokeh_Backend>
   Pandas and Seaborn <Pandas_Seaborn>
   Matplotlib Elements <Elements>
   Bokeh Elements <Bokeh_Elements>
   Containers
