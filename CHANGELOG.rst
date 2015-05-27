Version 1.2.0
-------------

Since the last release we closed over 20 issues and have made 334
commits, adding a ton of functionality and fixing a large range of
bugs in the process.

In this release we received some excellent feedback from our users,
which has been greatly appreciated and has helped us address a wide
range of problems.

Highlights/Features:

* Added new ``ErrorBars`` Element (f2b276b)
* Added ``Empty`` pseudo-Element to define empty placeholders in
  Layouts (35bac9f1d)
* Added support for changing font sizes easily (0f54bea)
* Support for holoviews.rc file (79076c8)
* Many major speed optimizations for working with and plotting
  HoloViews data structures (fe87b4c, 7578c51, 5876fe6, 8863333)
* Support for ``GridSpace`` with inner axes (93295c8)
* New ``aspect_weight`` and ``tight`` Layout plot options for more
  customizability of Layout arrangements (4b1f03d, e6a76b7)
* Added ``bgcolor`` plot option to easily set axis background color
  (92eb95c)
* Improved widget layout (f51af02)
* New ``OutputMagic`` css option to style html output (9d42dc2)
* Experimental support for PDF output (1e8a59b)
* Added support for 3D interactivity with nbagg (781bc25)
* Added ability to support deprecated plot options in %%opts magic.
* Added ``DrawPlot`` simplifying the implementation of custom plots
  (38e9d44)

API changes:

* ``Path`` and ``Histogram`` support new constructors (7138ef4, 03b5d38)
* New depth argument on the relabel method (f89b89f)
* Interface to Pandas improved (1a7cd3d)
* Removed ``xlim``, ``ylim`` and ``zlim`` to eliminate redundancy.
* Renaming of various plot and style options including:

  * ``figure_*`` to ``fig_*``
  * ``vertical_spacing`` and ``horizontal_spacing`` to ``vspace`` and ``hspace`` respectively
  * Deprecation of confusing ``origin`` style option on RasterPlot
* ``Overlay.__getitem__`` no longer supports integer indexing (use ``get`` method instead)

Important bug fixes:

* Important fixes to inheritance in the options system (d34a931, 71c1f3a7)
* Fixes to the select method (df839bea5)
* Fixes to normalization system (c3ef40b)
* Fixes to ``Raster`` and ``Image`` extents, ``__getitem__`` and sampling.
* Fixed bug with disappearing adjoined plots (2360972)
* Fixed plot ordering of overlaid elements across a ``HoloMap`` (c4f1685)


Version 1.1.0
-------------

Highlights:

* Support for nbagg as a backend (09eab4f1)
* New .hvz file format for saving HoloViews objects (bfd5f7af)
* New ``Polygon`` element type (d1ec8ec8)
* Greatly improved Unicode support throughout, including support for
  unicode characters in Python 3 attribute names (609a8454)
* Regular SelectionWidget now supports live rendering (eb5bf8b6)
* Supports a list of objects in Layout and Overlay constructors (5ba1866e)
* Polar projections now supported (3801b76e)

API changes (not backward compatible):

* ``xlim``, ``ylim``, ``zlim``, ``xlabel``, ``ylabel`` and ``zlabel``
  have been deprecated (081d4123)
* Plotting options ``show_xaxis`` and ``show_yaxis`` renamed to
  ``xaxis`` and ``yaxis``, respectively (13393f2a).
* Deprecated IPySelectionWidget (f59c34c0)

In addition to the above improvements, many miscellaneous bug fixes
were made.


Version 1.0.1
-------------

Minor release addressing bugs and issues with 1.0.0.

Highlights:

* New separate Pandas Tutorial (8455abc3)
* Silenced warnings when loading the IPython extension in IPython 3 (aaa6861b)
* Added more useful installation options via ``setup.py`` (72ece4db)
* Improvements and bug-fixes for the ``%%opts`` magic tab-completion (e0ad7108)
* ``DFrame`` now supports standard constructor for pandas dataframes (983825c5)
* ``Tables`` are now correctly formatted using the appropriate ``Dimension`` formatter (588bc2a3)
* Support for unlimited alphabetical subfigure labelling (e039d00b)
* Miscellaneous bug fixes, including Python 3 compatibility improvements.


Version 1.0.0
-------------

First public release available on GitHub and PyPI.
