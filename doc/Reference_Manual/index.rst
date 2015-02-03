****************
Reference Manual
****************

This reference manual contains detailed documentation of each
component making up holoviews, assuming that the user is already
familiar with basic holoviews usage. (See the `Tutorials`_ for such
an introduction.) The reference manual is
generated directly from documentation and declarations in the source
code, and is often much more verbose than necessary, because many
little-used yet often-duplicated methods are listed for each class.
Still, the reference for a given component does provide a
comprehensive listing of all attributes and methods, inherited or
otherwise, which is difficult to obtain from the source code
directly and is not covered by the User Manual or Tutorials.

holoviews package
_________________

`An overview of the holoviews package. <holoviews.html>`_

holoviews modules
-----------------

`core`_
 Base classes implementing the core data structures of HoloViews.
`interface`_
 Defines interfaces to external libraries including Pandas and Seaborn,
 but also to collect and collate data from an external source.
`ipython`_
 Defines the IPython notebook magics and display hooks.
`operation`_
 Defines ElementOperation and MapOperation classes used to manipulate and
 transform existing Views and Stacks.
`plotting`_
 Defines the plotting classes generating matplotlib figures.
`styles`_
 Defines default styles for View objects and allows selecting different
 mplstyles. 
`testing`_
 Defines TestComparison classes to compare different View objects.
`view`_
 Defines the basic View classes forming the atomic display units.


.. _User Manual: ../User_Manual/index.html
.. _Tutorials: ../Tutorials/index.html
.. _external dependencies: ../Downloads/dependencies.html
.. _main reference manual page: hierarchy.html

.. _core: holoviews.core.html
.. _interface: holoviews.interface.html
.. _ipython: holoviews.ipython.html
.. _operation: holoviews.operation.html
.. _plotting: holoviews.plotting.html
.. _styles: holoviews.styles.html
.. _testing: holoviews.html#module-holoviews.testing
.. _view: holoviews.view.html
