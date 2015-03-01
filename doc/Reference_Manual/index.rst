****************
Reference Manual
****************

To learn how to use HoloViews, we recommend that people start with the
`Tutorials`_, and then use the online help features from within
IPython Notebook to learn more.  Each option should have corresponding
tab-completion and help provided, and this is usually enough to
determine how to use the HoloViews components.

Even so, for a truly comprehensive reference to every object in
HoloViews, we provide this reference manual. The reference manual is
generated directly from documentation and declarations in the source
code, and is thus often much more verbose than necessary, because many
little-used yet often-duplicated methods are listed for each class.
Still, the reference for a given component does provide a
comprehensive listing of all attributes and methods, inherited or
otherwise, which is can be difficult to obtain from the source code
directly.

--------

Module Structure
________________

`An overview of all modules within all subpackages of HoloViews. <holoviews.html>`_


HoloViews subpackages
---------------------

`core`_
 Base classes implementing the core data structures of HoloViews.
`element`_
 Elements that form the basis of more complex visualizations.
`operation`_
 Operations applied to transform existing Elements or other data structures.
`plotting`_
 Matplotlib-based implementations of plotting for each component type.
`ipython`_
 Interface to IPython notebook, including magics and display hooks.
`interface`_
 Interfaces to optional external libraries (e.g. Pandas and Seaborn), as
 well as collecting and collating data from an external source.
`testing`_
 TestComparison classes to compare different component objects.

.. toctree::
   :maxdepth: 2
   :hidden:

   core <holoviews.core>
   element <holoviews.element>
   interface <holoviews.interface>
   ipython <holoviews.ipython>
   operation <holoviews.operation>
   plotting <holoviews.plotting>

.. _User Manual: ../User_Manual/index.html
.. _Tutorials: ../Tutorials/index.html
.. _external dependencies: ../Downloads/dependencies.html
.. _main reference manual page: hierarchy.html

.. _core: holoviews.core.html
.. _element: holoviews.element.html
.. _interface: holoviews.interface.html
.. _ipython: holoviews.ipython.html
.. _operation: holoviews.operation.html
.. _plotting: holoviews.plotting.html
.. _testing: holoviews.html#module-holoviews.testing
