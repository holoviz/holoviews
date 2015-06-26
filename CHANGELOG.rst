Version 1.3.0
-------------

Since the last release we closed over 34 issues and have made 380
commits mostly focused on fixing bugs, cleaning up the API and
working extensively on the plotting and rendering system to
ensure HoloViews is fully backend independent.

We'd again like to thank our growing user base for all their input,
which has helped us in making the API more understandable and
fixing a number of important bugs.

Highlights/Features:

* Allowed display of data structures which do not match the
  recommended nesting hierarchy (67b28f3, fbd89c3).
* Dimensions now sanitized for ``.select``, ``.sample`` and
  ``.reduce`` calls (6685633, 00b5a66).
* Added ``holoviews.ipython.display`` function to render (and display)
  any HoloViews object, useful for IPython interact widgets (0fa49cd).
* Table column widths now adapt to cell contents (be90a54).
* Defaulting to matplotlib ticking behavior (62e1e58).
* Allowed specifying fixed figure sizes to matplotlib via
  ``fig_inches`` tuples using (width, None) and (None, height) formats
  (632facd).
* Constructors of ``Chart``, ``Path`` and ``Histogram`` classes now support
  additional data formats (2297375).
* ``ScrubberWidget`` now supports all figure formats (c317db4).
* Allowed customizing legend positions on ``Bars`` Elements (5a12882).
* Support for multiple colorbars on one axis (aac7b92).
* ``.reindex`` on ``NdElement`` types now support converting between
  key and value dimensions allowing more powerful conversions. (03ac3ce)
* Improved support for casting between ``Element`` types (cdaab4e, b2ad91b,
  ce7fe2d, 865b4d5).
* The ``%%opts`` cell magic may now be used multiple times in the same
  cell (2a77fd0)
* Matplotlib rcParams can now be set correctly per figure (751210f).
* Improved ``OptionTree`` repr which now works with eval (2f824c1).
* Refactor of rendering system and IPython extension to allow easy
  swapping of plotting backend (#141)
* Large plotting optimization by computing tight ``bbox_inches`` once
  (e34e339).
* Widgets now cache frames in the DOM, avoiding flickering in some
  browsers and make use of jinja2 template inheritance. (fc7dd2b)
  

API Changes

* Renamed key_dimensions and value_dimensions to kdims and vdims
  respectively, while providing backward compatibility for passing
  and accessing the long names (8feb7d2).
* Combined x/y/zticker plot options into x/y/zticks parameters which
  now accept an explicit number of ticks, an explicit list of tick
  positions (and labels), and a matplotlib tick locator.
* Changed backend options in %output magic, ``nbagg`` and ``d3`` are
  now modes of the matplotlib backend and can be selected with
  ``backend='matplotlib:nbagg'`` and ``backend='matplotlib:mpld3'``
  respectively. The 'd3' and 'nbagg' options remain supported but will
  be deprecated in future.
* Customizations should no longer be applied directly to ``Store.options``;  
  the ``Store.options(backend='matplotlib')`` object should be
  customized instead.  There is no longer a need to call the
  deprecated ``Store.register_plots`` method.
  
  
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
