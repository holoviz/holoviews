****************
Reference Manual
****************

This reference manual contains detailed documentation of each
component making up DataViews, assuming that the user is already
familiar with basic DataViews usage. (See the `Tutorials`_ for such
an introduction.) The reference manual is
generated directly from documentation and declarations in the source
code, and is often much more verbose than necessary, because many
little-used yet often-duplicated methods are listed for each class.
Still, the reference for a given component does provide a
comprehensive listing of all attributes and methods, inherited or
otherwise, which is difficult to obtain from the source code
directly and is not covered by the User Manual or Tutorials.

DataViews package
_________________

`An overview of the DataViews package. <dataviews.html>`_

DataViews modules
-----------------

`collector`_
 Defines classes to collect and collate data.
`dataviews`_
 View classes associated with arbitrary dimensions such as Curves
 and Annotations.
`interface`_
 Defines interfaces to other visualization libraries including
 Pandas and Seaborn.
`ipython`_
 Defines the IPython notebook magics and display hooks.
`ndmapping`_
 Defines the Dimension and NdMapping classes, which implement
 n-dimensional data containers.
`operation`_
 Defines ViewOperation and StackOperation classes used to manipulate and
 transform existing Views and Stacks.
`options`_
 Defines options which control plotting and style configuration.
`plotting`_
 Defines the plotting classes generating matplotlib figures.
`sheetviews`_
 View classes associated with 2D sheets and projections from a 2D sheet.
`styles`_
 Defines default styles for View objects and allows selecting different
 mplstyles. 
`testing`_
 Defines TestComparison classes to compare different View objects.
`views`_
 Defines the basic View classes forming the atomic display units.


.. _User Manual: ../User_Manual/index.html
.. _Tutorials: ../Tutorials/index.html
.. _external dependencies: ../Downloads/dependencies.html
.. _main reference manual page: hierarchy.html

.. _collector: dataviews.html#module-dataviews.collector
.. _dataviews: dataviews.html#module-dataviews.dataviews
.. _interface: dataviews.interface.html
.. _ipython: dataviews.ipython.html
.. _ndmapping: dataviews.html#module-dataviews.ndmapping
.. _sheetviews: dataviews.sheetviews.html
.. _operation: dataviews.html#module-dataviews.operation
.. _options: dataviews.html#module-dataviews.options
.. _plotting: dataviews.plotting.html
.. _styles: dataviews.styles.html
.. _testing: dataviews.html#module-dataviews.ndmapping
.. _views: dataviews.html#module-dataviews.views
