********************
API Reference Manual
********************

To learn how to use HoloViews, we recommend that people start with the
`Getting Started guide`_, and then use the online help features from within
IPython Notebook to learn more.  Each option should have corresponding
tab-completion and help provided, and this is usually enough to
determine how to use the HoloViews components.

Even so, for a truly comprehensive reference to every object in
HoloViews, we provide this API documentation. The API reference manual
is generated directly from documentation and declarations in the
source code, and is thus often much more verbose than necessary,
because many little-used yet often-duplicated methods are listed for
each class.  Still, the reference for a given component does provide a
comprehensive listing of all attributes and methods, inherited or
otherwise, which is can be difficult to obtain from the source code
directly.

--------

Module Structure
________________

HoloViews subpackages
---------------------

`annotators`_
 Helper functions and classes to annotate visual elements.
`core`_
 Base classes implementing the core data structures of HoloViews.
`core.data`_
 Data Interface classes allowing HoloViews to work with different types of data.
`element`_
 Elements that form the basis of more complex visualizations.
`ipython`_
 Interface to IPython notebook, including magics and display hooks.
`operation`_
 Operations applied to transform existing Elements or other data structures.
`plotting`_
 Base plotting classes and utilities.
`plotting.bokeh`_
 Bokeh plotting classes and utilities.
`plotting.mpl`_
 Matplotlib plotting classes and utilities.
`plotting.plotly`_
 Plotly plotting classes and utilities.
`selection`_
 Helper functions to apply linked brushing and selections
`streams`_
 Stream classes to provide interactivity for DynamicMap
`util`_
 High-level utilities

.. toctree::
   :maxdepth: 2
   :hidden:

   annotators <holoviews.annotators>
   core <holoviews.core>
   core.data <holoviews.core.data>
   element <holoviews.element>
   interface <holoviews.interface>
   ipython <holoviews.ipython>
   operation <holoviews.operation>
   plotting <holoviews.plotting>
   plotting.bokeh <holoviews.plotting.bokeh>
   plotting.matplotlib <holoviews.plotting.matplotlib>
   plotting.plotly <holoviews.plotting.plotly>
   selection <holoviews.selection>
   streams <holoviews.streams>
   util <holoviews.util>

.. _User Manual: ../User_Manual/index.html
.. _Getting Started guide: ../getting_started/index.html

.. _annotators: holoviews.annotators.html
.. _core: holoviews.core.html
.. _core.data: holoviews.core.data.html
.. _element: holoviews.element.html
.. _ipython: holoviews.ipython.html
.. _operation: holoviews.operation.html
.. _plotting: holoviews.plotting.html
.. _plotting.bokeh: holoviews.plotting.bokeh.html
.. _plotting.mpl: holoviews.plotting.mpl.html
.. _plotting.plotly: holoviews.plotting.plotly.html
.. _selection: holoviews.selection.html
.. _streams: holoviews.streams.html
.. _util: holoviews.util.html
