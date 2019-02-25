Releases
========

Announcements
-------------

* `Version 1.10 <http://blog.pyviz.org/release_1.10.html>`__ (April 11, 2018)


Release notes
-------------

Version 1.11
~~~~~~~~~~~~

Version 1.11.3
**************

This is the last micro-release in the 1.11 series providing a number
of important fixes. Many thanks to our users for reporting these
issues and @poplarShift and @henriqueribeiro for contributing a number
of crucial fixes.

Bug fixes:

* All unused Options objects are now garbage collected fixing the last
  memory leak (`#3438 <https://github.com/pyviz/holoviews/pull/3438>`_)
* Ensured updating of size on matplotlib charts does not error
  (`#3442 <https://github.com/pyviz/holoviews/pull/3442>`_)
* Fix casting of datetimes on dask dataframes
  (`#3460 <https://github.com/pyviz/holoviews/pull/3460>`_)
* Ensure that calling redim does not break streams and links
  (`#3478 <https://github.com/pyviz/holoviews/pull/3478>`_)
* Ensure that matplotlib polygon plots close the edge path
  (`#3477 <https://github.com/pyviz/holoviews/pull/3477>`_)
* Fixed bokeh ArrowPlot error handling colorbars
  (`#3476 <https://github.com/pyviz/holoviews/pull/3476>`_)
* Fixed bug in angle conversion on the VectorField if invert_axes
  (`#3488 <https://github.com/pyviz/holoviews/pull/3488>`_)
* Ensure that all non-Annotation elements support empty constructors
  (`#3511 <https://github.com/pyviz/holoviews/pull/3511>`_)
* Fixed bug handling out-of-bounds errors when using tap events on
  datetime axis
  (`#3519 <https://github.com/pyviz/holoviews/pull/3519>`_)

Enhancements:


* Apply Labels element offset using a bokeh transform allowing Labels
  element to share data with original data
  (`#3445 <https://github.com/pyviz/holoviews/pull/3445>`_)
* Allow using datetimes in xlim/ylim/zlim
  (`#3491 <https://github.com/pyviz/holoviews/pull/3491>`_)
* Optimized rendering of TriMesh wireframes
  (`#3495 <https://github.com/pyviz/holoviews/pull/3495>`_)
* Add support for datetime formatting when hovering on Image/Raster
  (`#3520 <https://github.com/pyviz/holoviews/pull/3520>`_)
* Added Tiles element from GeoViews
  (`#3515 <https://github.com/pyviz/holoviews/pull/3515>`_)


Version 1.11.2
**************

This is a minor bug fix release with a number of minor but important
bug fixes. Special thanks to @darynwhite for his contributions.

Bug fixes:

* Fixed persisting options during clone on Overlay
  (`#3435 <https://github.com/pyviz/aholoviews/pull/3435>`_)
* Ensure cftime datetimes are displayed as a slider
  (`#3413 <https://github.com/pyviz/holoviews/pull/3413>`_)
* Fixed timestamp selections on streams
  (`#3427 <https://github.com/pyviz/holoviews/pull/3427>`_)
* Compatibility with pandas 0.24.0 release
  (`#3433 <https://github.com/pyviz/holoviews/pull/3433>`_)

Enhancements:

* Allow defining hook on backend load
  (`#3429 <https://github.com/pyviz/holoviews/pull/3429>`_)
* Improvements for handling graph attributes in `Graph.from_networkx`
  (``#3432 <https://github.com/pyviz/holoviews/pull/3432>`_)


Version 1.11.1
**************

This is a minor bug fix release with a number of important bug fixes,
enhancements and updates to the documentation. Special thanks to
Andrew Huang (@ahuang11), @garibarba and @Safrone for their
contributions.

Bug fixes:

* Fixed bug plotting adjoined histograms in matplotlib
  (`#3377 <https://github.com/pyviz/holoviews/pull/3377>`_)
* Fixed bug updating bokeh RGB alpha value
  (`#3371 <https://github.com/pyviz/holoviews/pull/3371>`_)
* Handled issue when colorbar limits were equal in bokeh
  (`#3382 <https://github.com/pyviz/holoviews/pull/3382>`_)
* Fixed bugs plotting empty Violin and BoxWhisker elements
  (`#3397 <https://github.com/pyviz/holoviews/pull/3397>`_,
  `#3405 <https://github.com/pyviz/holoviews/pull/3405>`_)
* Fixed handling of characters that have no uppercase on Layout and
  Overlay objects
  ((`#3403 <https://github.com/pyviz/holoviews/pull/3403>`_)
* Fixed bug updating Polygon plots in bokeh
  (`#3409 <https://github.com/pyviz/holoviews/pull/3409>`_)

Enhancements:

* Provide control over gridlines ticker and mirrored axis ticker by
  default (`#3398 <https://github.com/pyviz/holoviews/pull/3377>`_)
* Enabled colorbars on CompositePlot classes such as Graphs, Chords
  etc. (`#3397 <https://github.com/pyviz/holoviews/pull/3396>`_)
* Ensure that xarray backend retains dimension metadata when casting
  element (`#3401 <https://github.com/pyviz/holoviews/pull/3401>`_)
* Consistently support clim options
  (`#3382 <https://github.com/pyviz/holoviews/pull/3382>`_)

Documentation:

* Completed updates from .options to .opts API in the documentation
  (`#3364 <(https://github.com/pyviz/holoviews/pull/3364>`_,
  `#3367 <(https://github.com/pyviz/holoviews/pull/3367>`_)

Version 1.11.0
**************

This is a major release containing a large number of features and API
improvements. Specifically this release was devoted to improving the
general usability and accessibility of the HoloViews API and
deprecating parts of the API in anticipation for the 2.0 release.
To enable deprecation warnings for these deprecations set:

.. code-block::

   hv.config.future_deprecations = True


The largest updates to the API relate to the options system which is now
more consistent, has better validation and better supports notebook
users without requiring IPython magics. The new ``dim`` transform
generalizes the mapping from data dimensions to visual dimensions,
greatly increasing the expressive power of the options system. Please
consult the updated user guides for more information.

Special thanks for the contributions by Andrew Huang (@ahuang11),
Julia Signell (@jsignell), Jon Mease (@jonmmease), and Zachary Barry
(@zbarry).

Features:

* Generalized support for style mapping using ``dim`` transforms
  (`2152 <https://github.com/pyviz/holoviews/pull/2152>`_)
* Added alternative to opts magic with tab-completion
  (`#3173 <https://github.com/pyviz/holoviews/pull/3173>`_)
* Added support for Polygons with holes and improved contours
  operation (`#3092 <https://github.com/pyviz/holoviews/pull/3092>`_)
* Added support for Links to express complex interactivity in JS
  (`#2832 <https://github.com/pyviz/holoviews/pull/2832>`_)
* Plotly improvements including support for plotly 3.0
  (`#3194 <https://github.com/pyviz/holoviews/pull/3194>`_), improved
  support for containers
  (`#3255 <https://github.com/pyviz/holoviews/pull/3255>`_) and support
  for more elements
  (`#3256 <https://github.com/pyviz/holoviews/pull/3256>`_)
* Support for automatically padding plots using new ``padding`` option
  (`#2293 <https://github.com/pyviz/holoviews/pull/2293>`_)
* Added ``xlim``\ /\ ``ylim`` plot options to simplify setting axis ranges
  (`#2293 <https://github.com/pyviz/holoviews/pull/2293>`_)
* Added ``xlabel``\ /\ ``ylabel`` plot options to simplify overriding axis
  labels (`#2833 <https://github.com/pyviz/holoviews/issues/2833>`_)
* Added ``xformatter``\ /\ ``yformatter`` plot options to easily override tick
  formatter (`#3042 <https://github.com/pyviz/holoviews/pull/3042>`_)
* Added ``active_tools`` options to allow defining tools to activate on
  bokeh plot initialization
  (`#3251 <https://github.com/pyviz/holoviews/pull/3251>`_)
* Added ``FreehandDraw`` stream to allow freehand drawing on bokeh plots
  (`#2937 <https://github.com/pyviz/holoviews/pull/2937>`_)
* Added support for ``cftime`` types for dates which are not supported
  by standard datetimes and calendars
  (`#2728 <https://github.com/pyviz/holoviews/pull/2728>`_)
* Added top-level ``save`` and ``render`` functions to simplify exporting
  plots (`#3134 <https://github.com/pyviz/holoviews/pull/3134>`_)
* Added support for updating Bokeh bokeh legends
  (`#3139 <https://github.com/pyviz/holoviews/pull/3139>`_)
* Added support for indicating directed graphs with arrows
  (`#2521 <https://github.com/pyviz/holoviews/issues/2521>`_)

Enhancements:

* Improved import times
  (`#3055 <https://github.com/pyviz/holoviews/pull/3055>`_)
* Adopted Google style docstring and documented most core methods and
  classes (`#3128 <https://github.com/pyviz/holoviews/pull/3128>`_

Bug fixes:

* GIF rendering fixed under Windows
  (`#3151 <https://github.com/pyviz/holoviews/issues/3151>`_)
* Fixes for hover on Path elements in bokeh
  (`#2472 <https://github.com/pyviz/holoviews/issues/2427>`_,
  `#2872 <https://github.com/pyviz/holoviews/issues/2872>`_)
* Fixes for handling TriMesh value dimensions on rasterization
  (`#3050 <https://github.com/pyviz/holoviews/pull/3050>`_)

Deprecations:

* ``finalize_hooks`` renamed to ``hooks``
  (`#3134 <https://github.com/pyviz/holoviews/pull/3134>`_)
* All ``*_index`` and related options are now deprecated including
  ``color_index``, ``size_index``, ``scaling_method``, ``scaling_factor``,
  ``size_fn`` (`#2152 <https://github.com/pyviz/holoviews/pull/2152>`_)
* Bars ``group_index``, ``category_index`` and ``stack_index`` are deprecated in
  favor of stacked option
  (`#2828 <https://github.com/pyviz/holoviews/issues/2828>`_)
* iris interface was moved to GeoViews
  (`#3054 <https://github.com/pyviz/holoviews/pull/3054>`_)
* Top-level namespace was cleaned up
  (`#2224 <https://github.com/pyviz/holoviews/pull/2224>`_)
* ``ElementOpration``, ``Layout.display`` and ``mdims`` argument to ``.to``
  now fully removed
  (`#3128 <https://github.com/pyviz/holoviews/pull/3128>`_)
* ``Element.mapping``, ``ItemTable.values``, ``Element.table``,
  ``HoloMap.split_overlays``, ``ViewableTree.from_values``,
  ``ViewableTree.regroup`` and ``Element.collapse_data`` methods now
  marked for deprecation
  (`#3128 <https://github.com/pyviz/holoviews/pull/3128>`_)


Version 1.10
~~~~~~~~~~~~

Version 1.10.8
**************

This a likely the last hotfix release in the 1.10.x series containing
fixes for compatibility with bokeh 1.0 and matplotlib 3.0. It also
contains a wide array of fixes contributed and reported by users:

Special thanks for the contributions by Andrew Huang (@ahuang11),
Julia Signell (@jsignell), and Zachary Barry (@zbarry).

Enhancements:

- Add support for labels, choord, hextiles and area in .to interface
  (`#2924 <https://github.com/pyviz/holoviews/pull/2924>`_)
- Allow defining default bokeh themes as strings on Renderer
  (`#2972 <https://github.com/pyviz/holoviews/pull/2972>`_)
- Allow specifying fontsize for categorical axis ticks in bokeh
  (`#3047 <https://github.com/pyviz/holoviews/pull/3047>`_)
- Allow hiding toolbar without disabling tools
  (`#3074 <https://github.com/pyviz/holoviews/pull/3074>`_)
- Allow specifying explicit colormapping on non-categorical data
  (`#3071 <https://github.com/pyviz/holoviews/pull/3071>`_)
- Support for displaying xarray without explicit coordinates
  (`#2968 <https://github.com/pyviz/holoviews/pull/2968>`_)

Fixes:

- Allow dictionary data to reference values which are not dimensions
  (`#2855 <https://github.com/pyviz/holoviews/pull/2855>`_,
  `#2859 <https://github.com/pyviz/holoviews/pull/2859>`_)
- Fixes for zero and non-finite ranges in datashader operation
  (`#2860 <https://github.com/pyviz/holoviews/pull/2860>`_,
  `#2863 <https://github.com/pyviz/holoviews/pull/2863>`_,
  `#2869 <https://github.com/pyviz/holoviews/pull/2869>`_)
- Fixes for CDSStream and drawing tools on bokeh server
  (`#2915 <https://github.com/pyviz/holoviews/pull/2915>`_)
- Fixed issues with nans, datetimes and streaming on Area and Spread
  elements (`#2951 <https://github.com/pyviz/holoviews/pull/2951>`_,
  `c55b044 <https://github.com/pyviz/holoviews/commit/c55b044>`_)
- General fixes for datetime handling
  (`#3005 <https://github.com/pyviz/holoviews/pull/3005>`_,
  `#3045 <https://github.com/pyviz/holoviews/pull/3045>`_,
  `#3075 <https://github.com/pyviz/holoviews/pull/3074>`_)
- Fixed handling of curvilinear and datetime coordinates on QuadMesh
  (`#3017 <https://github.com/pyviz/holoviews/pull/3017>`_,
  `#3081 <https://github.com/pyviz/holoviews/pull/3081>`_)
- Fixed issue when inverting a shared axis in bokeh
  (`#3083 <https://github.com/pyviz/holoviews/pull/3083>`_)
- Fixed formatting of values in HoloMap widgets
  (`#2954 <https://github.com/pyviz/holoviews/pull/2954>`_)
- Fixed setting fontsize for z-axis label
  (`#2967 <https://github.com/pyviz/holoviews/pull/2967>`_)

Compatibility:

- Suppress warnings about rcParams in matplotlib 3.0
  (`#3013 <https://github.com/pyviz/holoviews/pull/3013>`_,
  `#3058 <https://github.com/pyviz/holoviews/pull/3058>`_,
  `#3104 <https://github.com/pyviz/holoviews/pull/3104>`_)
- Fixed incompatibility with Python <=3.5
  (`#3073 <https://github.com/pyviz/holoviews/pull/3073>`_)
- Fixed incompatibility with bokeh >=1.0
  (`#3051 <https://github.com/pyviz/holoviews/pull/3051>`_)

Documentation:

- Completely overhauled the FAQ
  (`#2928 <https://github.com/pyviz/holoviews/pull/2928>`_,
  `#2941 <https://github.com/pyviz/holoviews/pull/2941>`_,
  `#2959 <https://github.com/pyviz/holoviews/pull/2959>`_,
  `#3025 <https://github.com/pyviz/holoviews/pull/3025>`_)


Version 1.10.7
**************

This a very minor hotfix release mostly containing fixes for datashader
aggregation of empty datasets:

Fixes:

- Fix datashader aggregation of empty and zero-range data
  (`#2860 <https://github.com/pyviz/holoviews/pull/2860>`_,
  `#2863 <https://github.com/pyviz/holoviews/pull/2863>`_)
- Disable validation for additional, non-referenced keys in the
  DictInterface (`#2860 <https://github.com/pyviz/holoviews/pull/2860>`_)
- Fixed frame lookup for non-overlapping dimensions
  (`#2861 <https://github.com/pyviz/holoviews/pull/2861>`_)
- Fixed ticks on log Colorbar if low value <= 0
  (`#2865 <https://github.com/pyviz/holoviews/pull/2865>`_)

Version 1.10.6
**************

This another minor bug fix release in the 1.10 series and likely the
last one before the upcoming 1.11 release. In addition to some important
fixes relating to datashading and the handling of dask data, this
release includes a number of enhancements and fixes.

Enhancements:

- Added the ability to specify color intervals using the color_levels
  plot options (`#2797 <https://github.com/pyviz/holoviews/pull/2797>`_)
- Allow defining port and multiple websocket origins on BokehRenderer.app
  (`#2801 <https://github.com/pyviz/holoviews/pull/2801>`_)
- Support for datetimes in Curve step interpolation
  (`#2757 <https://github.com/pyviz/holoviews/pull/2757>`_)
- Add ability to mute legend by default
  (`#2831 <https://github.com/pyviz/holoviews/pull/2831>`_)
- Implemented ability to collapse and concatenate gridded data
  (`#2762 <https://github.com/pyviz/holoviews/pull/2762>`_)
- Add support for cumulative histogram and explicit bins
  (`#2812 <https://github.com/pyviz/holoviews/pull/2812>`_)

Fixes:

- Dataset discovers multi-indexes on dask dataframes
  (`#2789 <https://github.com/pyviz/holoviews/pull/2789>`_)
- Fixes for datashading NdOverlays with datetime axis and data with
  zero range (`#2829 <https://github.com/pyviz/holoviews/pull/2829>`_,
  `#2842 <https://github.com/pyviz/holoviews/pull/2842>`_)

Version 1.10.5
**************

This is a minor bug fix release containing a mixture of small
enhancements, a number of important fixes and improved compatibility
with pandas 0.23.

Enhancements:

- Graph.from_networkx now extracts node and edge attributes from
  networkx graphs
  (`#2714 <https://github.com/pyviz/holoviews/pull/2714>`_)
- Added throttling support to scrubber widget
  (`#2748 <https://github.com/pyviz/holoviews/pull/2748>`_)
- histogram operation now works on datetimes
  (`#2719 <https://github.com/pyviz/holoviews/pull/2719>`_)
- Legends on NdOverlay containing overlays now supported
  (`#2755 <https://github.com/pyviz/holoviews/pull/2755>`_)
- Dataframe indexes may now be referenced in ``.to`` conversion
  (`#2739 <https://github.com/pyviz/holoviews/pull/2739>`_)
- Reindexing a gridded Dataset without arguments now behaves
  consistently with NdMapping types and drops scalar dimensions making
  it simpler to drop dimensions after selecting
  (`#2746 <https://github.com/pyviz/holoviews/pull/2746>`_)

Fixes:

- Various fixes for QuadMesh support including support for contours,
  nan coordinates and inverted coordinates
  (`#2691 <https://github.com/pyviz/holoviews/pull/2691),
  [#2702 <https://github.com/pyviz/holoviews/pull/2702),
  [#2771 <https://github.com/pyviz/holoviews/pull/2771>`_)
- Fixed bugs laying out complex layouts in bokeh
  (`#2740 <https://github.com/pyviz/holoviews/pull/2740>`_)
- Fix for adding value dimensions to an xarray dataset
  (`#2761 <https://github.com/pyviz/holoviews/pull/2761>`_)

Version 1.10.4
**************

This is a minor bug fix release including a number of crucial fixes
for issues reported by our users.

Enhancement:

- Allow setting alpha on Image/RGB/HSV and Raster types in bokeh
  (`#2680 <https://github.com/pyviz/holoviews/pull/2680>`_)

Fixes:

- Fixed bug running display multiple times in one cell
  (`#2677 <https://github.com/pyviz/holoviews/pull/2677>`_)
- Avoid sending hover data unless explicitly requested
  (`#2681 <https://github.com/pyviz/holoviews/pull/2681>`_)
- Fixed bug slicing xarray with tuples
  (`#2674 <https://github.com/pyviz/holoviews/pull/2674>`_)


Version 1.10.3
**************

This is a minor bug fix release including a number of crucial fixes for
issues reported by our users.

Enhancement:

-  The dimensions of elements may now be changed allowing updates to
   axis labels and table column headers
   (`#2666 <https://github.com/pyviz/holoviews/pull/2666>`__)

Fixes:

-  Fix for ``labelled`` plot option
   (`#2643 <https://github.com/pyviz/holoviews/pull/2643>`__)
-  Optimized initialization of dynamic plots specifying a large
   parameter space
   (`#2646 <https://github.com/pyviz/holoviews/pull/2646>`__)
-  Fixed unicode and reversed axis slicing issues in XArrayInterface
   (`#2658 <https://github.com/pyviz/holoviews/issues/2658>`__,
   `#2653 <https://github.com/pyviz/holoviews/pull/2653>`__)
-  Fixed widget sorting issues when applying dynamic groupby
   (`#2641 <https://github.com/pyviz/holoviews/issues/2641>`__)

API:

-  The PlotReset reset parameter was renamed to resetting to avoid clash
   with a method
   (`#2665 <https://github.com/pyviz/holoviews/pull/2665>`__)
-  PolyDraw tool data parameter now always indexed with 'xs' and 'ys'
   keys for consistency
   (`#2650 <https://github.com/pyviz/holoviews/issues/2650>`__)

Version 1.10.2
**************

This is a minor bug fix release with a number of small fixes for
features and regressions introduced in 1.10:

Enhancement:

-  Exposed Image hover functionality for upcoming bokeh 0.12.16 release
   (`#2625 <https://github.com/pyviz/holoviews/pull/2625>`__)

Fixes:

-  Minor fixes for newly introduced elements and plots including Chord
   (`#2581 <https://github.com/pyviz/holoviews/issues/2581>`__) and
   RadialHeatMap
   (`#2610 <https://github.com/pyviz/holoviews/issues/2610>`__
-  Fixes for .options method including resolving style and plot option
   clashes (`#2411 <https://github.com/pyviz/holoviews/issues/2411>`__)
   and calling it without arguments
   (`#2630 <https://github.com/pyviz/holoviews/pull/2630>`__)
-  Fixes for IPython display function
   (`#2587 <https://github.com/pyviz/holoviews/issues/2587>`__) and
   display\_formats
   (`#2592 <https://github.com/pyviz/holoviews/issues/2592>`__)

Deprecations:

-  BoxWhisker and Bars ``width`` bokeh style options and Arrow
   matplotlib ``fontsize`` option are deprecated
   (`#2411 <https://github.com/pyviz/holoviews/issues/2411>`__)

Version 1.10.1
**************

This is a minor bug fix release with a number of fixes for regressions
and minor bugs introduced in the 1.10.0 release:

Fixes:

-  Fixed static HTML export of notebooks
   (`#2574 <https://github.com/pyviz/holoviews/pull/2574>`__)
-  Ensured Chord element allows recurrent edges
   (`#2583 <https://github.com/pyviz/holoviews/pull/2583>`__)
-  Restored behavior for inferring key dimensions order from XArray
   Dataset (`#2579 <https://github.com/pyviz/holoviews/pull/2579>`__)
-  Fixed Selection1D stream on bokeh server after changes in bokeh
   0.12.15 (`#2586 <https://github.com/pyviz/holoviews/pull/2586>`__)

Version 1.10.0
**************

This is a major release with a large number of new features and bug
fixes, as well as a small number of API changes. Many thanks to the
numerous users who filed bug reports, tested development versions, and
contributed a number of new features and bug fixes, including special
thanks to @mansenfranzen, @ea42gh, @drs251 and @jakirkham.

JupyterLab support:

-  Full compatibility with JupyterLab when installing the
   jupyterlab\_holoviews extension
   (`#687 <https://github.com/pyviz/holoviews/issues/687>`__)

New components:

-  Added |Sankey|_ element to plot directed flow graphs
   (`#1123 <https://github.com/pyviz/holoviews/issues/1123>`__)
-  Added |TriMesh|_ element
   and datashading operation to plot small and large irregular meshes
   (`#2143 <https://github.com/pyviz/holoviews/issues/2143>`__)
-  Added a |Chord|_ element
   to draw flow graphs between different nodes
   (`#2137 <https://github.com/pyviz/holoviews/issues/2137>`__,
   `#2143 <https://github.com/pyviz/holoviews/pull/2143>`__)
-  Added |HexTiles|_ element
   to plot data binned into a hexagonal grid
   (`#1141 <https://github.com/pyviz/holoviews/issues/1141>`__)
-  Added |Labels|_ element
   to plot a large number of text labels at once (as data rather than as
   annotations)
   (`#1837 <https://github.com/pyviz/holoviews/issues/1837>`__)
-  Added |Div|_ element
   to add arbitrary HTML elements to a Bokeh layout
   (`#2221 <https://github.com/pyviz/holoviews/issues/2221>`__)
-  Added |PointDraw|_, |PolyDraw|_, |BoxEdit|_ and |PolyEdit|_
   streams to allow drawing, editing, and annotating glyphs on a Bokeh
   plot, and syncing the resulting data to Python
   (`#2268 <https://github.com/pyviz/holoviews/issues/2459>`__)

Features:

-  Added |radial HeatMap|_  option to allow plotting heatmaps with a cyclic x-axis
   (`#2139 <https://github.com/pyviz/holoviews/pull/2139>`__)
-  All elements now support declaring bin edges as well as centers
   allowing ``Histogram`` and ``QuadMesh`` to become first class
   ``Dataset`` types
   (`#547 <https://github.com/pyviz/holoviews/issues/547>`__)
-  When using widgets, their initial or default value can now be set via
   the ``Dimension.default`` parameter
   (`#704 <https://github.com/pyviz/holoviews/issues/704>`__)
-  n-dimensional Dask arrays are now supported directly via the gridded
   dictionary data interface
   (`#2305 <https://github.com/pyviz/holoviews/pull/2305>`__)
-  Added new `Styling
   Plots <http://holoviews.org/user_guide/Styling_Plots.html>`__ and
   `Colormaps <http://holoviews.org/user_guide/Colormaps.html>`__ user
   guides, including new functionality for working with colormaps.

Enhancements:

-  Improvements to exceptions
   (`#1127 <https://github.com/pyviz/holoviews/issues/1127>`__)
-  Toolbar position and merging (via a new ``merge_toolbar`` option) can
   now be controlled for Layout and Grid plots
   (`#1977 <https://github.com/pyviz/holoviews/issues/1977>`__)
-  Bokeh themes can now be applied at the renderer level
   (`#1861 <https://github.com/pyviz/holoviews/issues/1861>`__)
-  Dataframe and Series index can now be referenced by name when
   constructing an element
   (`#2000 <https://github.com/pyviz/holoviews/issues/2000>`__)
-  Option-setting methods such as ``.opts``, ``.options`` and
   ``hv.opts`` now allow specifying the backend instead of defaulting to
   the current backend
   (`#1801 <https://github.com/pyviz/holoviews/issues/1801>`__)
-  Handled API changes in streamz 0.3.0 in Buffer stream
   (`#2409 <https://github.com/pyviz/holoviews/issues/2409>`__)
-  Supported GIF output on windows using new Matplotlib pillow animation
   support (`#385 <https://github.com/pyviz/holoviews/issues/385>`__)
-  Provided simplified interface to ``rasterize`` most element types
   using datashader
   (`#2465 <https://github.com/pyviz/holoviews/pull/2465>`__)
-  ``Bivariate`` element now support ``levels`` as a plot option
   (`#2099 <https://github.com/pyviz/holoviews/issues/2099>`__)
-  ``NdLayout`` and ``GridSpace`` now consistently support ``*`` overlay
   operation (`#2075 <https://github.com/pyviz/holoviews/issues/2075>`__)
-  The Bokeh backend no longer has a hard dependency on Matplotlib
   (`#829 <https://github.com/pyviz/holoviews/issues/829>`__)
-  ``DynamicMap`` may now return (``Nd``)\ ``Overlay`` with varying
   number of elements
   (`#1388 <https://github.com/pyviz/holoviews/issues/1388>`__)
-  In the notebook, deleting or re-executing a cell will now delete the
   plot and clean up any attached streams
   (`#2141 <https://github.com/pyviz/holoviews/issues/2141>`__)
-  Added ``color_levels`` plot option to set discrete number of levels
   during colormapping
   (`#2483 <https://github.com/pyviz/holoviews/pull/2483>`__)
-  Expanded the `Large
   Data <http://holoviews.org/user_guide/Large_Data.html>`__ user guide
   to show examples of all Element and Container types supported for
   datashading and give performance guidelines.

Fixes:

-  ``Layout`` and ``Overlay`` objects no longer create lower-case nodes
   on attribute access
   (`#2331 <https://github.com/pyviz/holoviews/pull/2331>`__)
-  ``Dimension.step`` now correctly respects both integer and float
   steps (`#1707 <https://github.com/pyviz/holoviews/issues/1707>`__)
-  Fixed timezone issues when using linked streams on datetime axes
   (`#2459 <https://github.com/pyviz/holoviews/issues/2459>`__)

Changes affecting backwards compatibility:

-  Image elements now expect and validate regular sampling
   (`#1869 <https://github.com/pyviz/holoviews/issues/1869>`__); for
   genuinely irregularly sampled data QuadMesh should be used.
-  Tabular elements will no longer default to use ``ArrayInterface``,
   instead preferring pandas and dictionary data formats
   (`#1236 <https://github.com/pyviz/holoviews/issues/1236>`__)
-  ``Cycle``/``Palette`` values are no longer zipped together; instead
   they now cycle independently
   (`#2333 <https://github.com/pyviz/holoviews/pull/2333>`__)
-  The default color ``Cycle`` was expanded to provide more unique
   colors (`#2483 <https://github.com/pyviz/holoviews/pull/2483>`__)
-  Categorical colormapping was made consistent across backends,
   changing the behavior of categorical Matplotlib colormaps
   (`#2483 <https://github.com/pyviz/holoviews/pull/2483>`__)
-  Disabled auto-indexable property of the Dataset baseclass, i.e. if a
   single column is supplied no integer index column is added
   automatically
   (`#2522 <https://github.com/pyviz/holoviews/pull/2522>`__)


Version 1.9
~~~~~~~~~~~


Version 1.9.5
*************

This release includes a very small number of minor bugfixes and a new
feature to simplify setting options in python:

Enhancements:

-  Added .options method for simplified options setting.
   (`#2306 <https://github.com/pyviz/holoviews/pull/2306>`__)

Fixes:

-  Allow plotting bytes datausing the Bokeh backend in python3
   (`#2357 <https://github.com/pyviz/holoviews/pull/2357>`__)
-  Allow .range to work on data with heterogeneous types in Python 3
   (`#2345 <https://github.com/pyviz/holoviews/pull/2345>`__)
-  Fixed bug streaming data containing datetimes using bokeh>-0.12.14
   (`#2383 <https://github.com/pyviz/holoviews/pull/2383>`__)

Version 1.9.4
*************

This release contains a small number of important bug fixes:

-  Compatibility with recent versions of Dask and pandas
   (`#2329 <https://github.com/pyviz/holoviews/pull/2329>`__)
-  Fixed bug referencing columns containing non-alphanumeric characters
   in Bokeh Tables
   (`#2336 <https://github.com/pyviz/holoviews/pull/2336>`__)
-  Fixed issue in regrid operation
   (`2337 <https://github.com/pyviz/holoviews/pull/2337>`__)
-  Fixed issue when using datetimes with datashader when processing
   ranges (`#2344 <https://github.com/pyviz/holoviews/pull/2344>`__)

Version 1.9.3
*************

This release contains a number of important bug fixes and minor
enhancements.

Particular thanks to @jbampton, @ea42gh, @laleph, and @drs251 for a
number of fixes and improvements to the documentation.

Enhancements:

-  Optimized rendering of stream based OverlayPlots
   (`#2253 <https://github.com/pyviz/holoviews/pull/2253>`__)
-  Added ``merge_toolbars`` and ``toolbar`` options to control toolbars
   on ``Layout`` and Grid plots
   (`#2289 <https://github.com/pyviz/holoviews/pull/2289>`__)
-  Optimized rendering of ``VectorField``
   (`#2314 <https://github.com/pyviz/holoviews/pull/2289>`__)
-  Improvements to documentation
   (`#2198 <https://github.com/pyviz/holoviews/pull/2198>`__,
   `#2220 <https://github.com/pyviz/holoviews/pull/2220>`__,
   `#2233 <https://github.com/pyviz/holoviews/pull/2233>`__,
   `#2235 <https://github.com/pyviz/holoviews/pull/2235>`__,
   `#2316 <https://github.com/pyviz/holoviews/pull/2316>`__)
-  Improved Bokeh ``Table`` formatting
   (`#2267 <https://github.com/pyviz/holoviews/pull/2267>`__)
-  Added support for handling datetime.date types
   (`#2267 <https://github.com/pyviz/holoviews/pull/2267>`__)
-  Add support for pre- and post-process hooks on operations
   (`#2246 <https://github.com/pyviz/holoviews/pull/2246>`__,
   `#2334 <https://github.com/pyviz/holoviews/pull/2334>`__)

Fixes:

-  Fix for Bokeh server widgets
   (`#2218 <https://github.com/pyviz/holoviews/pull/2218>`__)
-  Fix using event based streams on Bokeh server
   (`#2239 <https://github.com/pyviz/holoviews/pull/2239>`__,
   `#2256 <https://github.com/pyviz/holoviews/pull/2256>`__)
-  Switched to drawing ``Distribution``, ``Area`` and ``Spread`` using
   patch glyphs in Bokeh fixing legends
   (`#2225 <https://github.com/pyviz/holoviews/pull/2225>`__)
-  Fixed categorical coloring of ``Polygons``/``Path`` elements in
   Matplotlib (`#2259 <https://github.com/pyviz/holoviews/pull/2259>`__)
-  Fixed bug computing categorical datashader aggregates
   (`#2295 <https://github.com/pyviz/holoviews/pull/2295>`__)
-  Allow using ``Empty`` object in ``AdjointLayout``
   (`#2275 <https://github.com/pyviz/holoviews/pull/2275>`__)

API Changes:

-  Renamed ``Trisurface`` to ``TriSurface`` for future consistency
   (`#2219 <https://github.com/pyviz/holoviews/pull/2219>`__)

Version 1.9.2
*************

This release is a minor bug fix release patching various issues which
were found in the 1.9.1 release.

Enhancements:

-  Improved the Graph element, optimizing the constructor and adding
   support for defining a ``edge_color_index``
   (`#2145 <https://github.com/pyviz/holoviews/pull/2145>`__)
-  Added support for adding jitter to Bokeh Scatter and Points plots
   (`e56208 <https://github.com/pyviz/holoviews/commit/e56208e1eb6e1e4af67b6a3ffbb5a925bfc37e14>`__)

Fixes:

-  Ensure dimensions, group and label are inherited when casting Image
   to QuadMesh (`#2144 <https://github.com/pyviz/holoviews/pull/2144>`__)
-  Handle compatibility for Bokeh version >- 0.12.11
   (`#2159 <https://github.com/pyviz/holoviews/pull/2159>`__)
-  Fixed broken Bokeh ArrowPlot
   (`#2172 <https://github.com/pyviz/holoviews/pull/2172>`__)
-  Fixed Pointer based streams on datetime axes
   (`#2179 <https://github.com/pyviz/holoviews/pull/2179>`__)
-  Allow constructing and plotting of empty Distribution and Bivariate
   elements (`#2190 <https://github.com/pyviz/holoviews/pull/2190>`__)
-  Added support for hover info on Bokeh BoxWhisker plots
   (`#2187 <https://github.com/pyviz/holoviews/pull/2187>`__)
-  Fixed bug attaching streams to (Nd)Overlay types
   (`#2194 <https://github.com/pyviz/holoviews/pull/2194>`__)

Version 1.9.1
*************

This release is a minor bug fix release patching various issues which
were found in the 1.9.0 release.

Enhancements:

-  Exposed min\_alpha parameter on datashader shade and datashade
   operations (`#2109 <https://github.com/pyviz/holoviews/pull/2109>`__)

Fixes:

-  Fixed broken Bokeh server linked stream throttling
   (`#2112 <https://github.com/pyviz/holoviews/pull/2112>`__)
-  Fixed bug in Bokeh callbacks preventing linked streams using Bokeh's
   on\_event callbacks from working
   (`#2112 <https://github.com/pyviz/holoviews/pull/2112>`__)
-  Fixed insufficient validation issue for Image and bugs when applying
   regrid operation to xarray based Images
   (`#2117 <https://github.com/pyviz/holoviews/pull/2117>`__)
-  Fixed handling of dimensions and empty elements in univariate\_kde
   and bivariate\_kde operations
   (`#2103 <https://github.com/pyviz/holoviews/pull/2103>`__)

Version 1.9.0
*************

This release includes a large number of long awaited features,
improvements and bug fixes, including streaming and graph support,
binary transfer of Bokeh data, fast Image/RGB regridding, first-class
statistics elements and a complete overhaul of the geometry elements.

Particular thanks to all users and contributers who have reported issues
and submitted pull requests.

Features:

-  The kdim and vdim keyword arguments are now positional making the
   declaration of elements less verbose (e.g. Scatter(data, 'x', 'y'))
   (`#1946 <https://github.com/pyviz/holoviews/pull/1946>`__)
-  Added Graph, Nodes, and EdgePaths elements adding support for
   plotting network graphs
   (`#1829 <https://github.com/pyviz/holoviews/pull/1829>`__)
-  Added datashader based regrid operation for fast Image and RGB
   regridding (`#1773 <https://github.com/pyviz/holoviews/pull/1773>`__)
-  Added support for binary transport when plotting with Bokeh,
   providing huge speedups for dynamic plots
   (`#1894 <https://github.com/pyviz/holoviews/pull/1894>`__,
   `#1896 <https://github.com/pyviz/holoviews/pull/1896>`__)
-  Added Pipe and Buffer streams for streaming data support
   (`#2011 <https://github.com/pyviz/holoviews/pull/2011>`__)
-  Add support for datetime axes on Image, RGB and when applying
   datashading and regridding operations
   (`#2023 <https://github.com/pyviz/holoviews/pull/2023>`__)
-  Added Distribution and Bivariate as first class elements which can be
   plotted with Matplotlib and Bokeh without depending on seaborn
   (`#1985 <https://github.com/pyviz/holoviews/pull/1985>`__)
-  Completely overhauled support for plotting geometries with Path,
   Contours and Polygons elements including support for coloring
   individual segments and paths by value
   (`#1991 <https://github.com/pyviz/holoviews/pull/1991>`__)

Enhancements:

-  Add support for adjoining all elements on Matplotlib plots
   (`#1033 <https://github.com/pyviz/holoviews/pull/1033>`__)
-  Improved exception handling for data interfaces
   (`#2041 <https://github.com/pyviz/holoviews/pull/2041>`__)
-  Add groupby argument to histogram operation
   (`#1725 <https://github.com/pyviz/holoviews/pull/1725>`__)
-  Add support for reverse sort on Dataset elements
   (`#1843 <https://github.com/pyviz/holoviews/pull/1843>`__)
-  Added support for invert\_x/yaxis on all elements
   (`#1872 <https://github.com/pyviz/holoviews/pull/1872>`__,
   `#1919 <https://github.com/pyviz/holoviews/pull/1919>`__)

Fixes:

-  Fixed a bug in Matplotlib causing the first frame in gif and mp4
   getting stuck
   (`#1922 <https://github.com/pyviz/holoviews/pull/1922>`__)
-  Fixed various issues with support for new nested categorical axes in
   Bokeh (`#1933 <https://github.com/pyviz/holoviews/pull/1933>`__)
-  A large range of other bug fixes too long to list here.

Changes affecting backwards compatibility:

-  The contours operation no longer overlays the contours on top of the
   supplied Image by default and returns a single Contours/Polygons
   rather than an NdOverlay of them
   (`#1991 <https://github.com/pyviz/holoviews/pull/1991>`__)
-  The values of the Distribution element should now be defined as a key
   dimension (`#1985 <https://github.com/pyviz/holoviews/pull/1985>`__)
-  The seaborn interface was removed in its entirety being replaced by
   first class support for statistics elements such as Distribution and
   Bivariate (`#1985 <https://github.com/pyviz/holoviews/pull/1985>`__)
-  Since kdims and vdims can now be passed as positional arguments the
   bounds argument on Image is no longer positional
   (`#1946 <https://github.com/pyviz/holoviews/pull/1946>`__).
-  The datashade and shade cmap was reverted back to blue due to issues
   with the fire cmap against a white background.
   (`#2078 <https://github.com/pyviz/holoviews/pull/2078>`__)
-  Dropped all support for Bokeh versions older than 0.12.10
-  histogram operation now returns Histogram elements with less generic
   value dimension and customizable label
   (`#1836 <https://github.com/pyviz/holoviews/pull/1836>`__)

Version 1.8
~~~~~~~~~~~

Version 1.8.4
*************

This bugfix release includes a number of critical fixes for compatiblity
with Bokeh 0.12.9 along with various other bug fixes. Many thanks to our
users for various detailed bug reports, feedback and contributions.

Fixes:

-  Fixes to register BoundsXY stream.
   (`#1826 <https://github.com/pyviz/holoviews/pull/1826>`__)
-  Fix for Bounds streams on Bokeh server.
   (`#1883 <https://github.com/pyviz/holoviews/pull/1883>`__)
-  Compatibility with Matplotlib 2.1
   (`#1842 <https://github.com/pyviz/holoviews/pull/1842>`__)
-  Fixed bug in scrubber widget and support for scrubbing discrete
   DynamicMaps (`#1832 <https://github.com/pyviz/holoviews/pull/1832>`__)
-  Various fixes for compatibility with Bokeh 0.12.9
   (`#1849 <https://github.com/pyviz/holoviews/pull/1849>`__,
   `#1866 <https://github.com/pyviz/holoviews/pull/1886>`__)
-  Fixes for setting QuadMesh ranges.
   (`#1876 <https://github.com/pyviz/holoviews/pull/1876>`__)
-  Fixes for inverting Image/RGB/Raster axes in Bokeh.
   (`#1872 <https://github.com/pyviz/holoviews/pull/1872>`__)

Version 1.8.3
*************

This bugfix release fixes a number of minor issues identified since the
last release:

Features:

-  Add support for setting the Bokeh sizing\_mode as a plot option
   (`#1813 <https://github.com/pyviz/holoviews/pull/1813>`__)

Fixes:

-  Handle StopIteration on DynamicMap correctly.
   (`#1792 <https://github.com/pyviz/holoviews/pull/1792>`__)
-  Fix bug with linked streams on empty source element
   (`#1725 <https://github.com/pyviz/holoviews/pull/1806>`__)
-  Compatibility with latest datashader 0.6.0 release
   (`#1773 <https://github.com/pyviz/holoviews/pull/1773>`__)
-  Fixed missing HTML closing tag in extension
   (`#1797 <https://github.com/pyviz/holoviews/issues/1797>`__,
   `#1809 <https://github.com/pyviz/holoviews/pull/1809>`__)
-  Various fixes and improvements for documentation
   (`#1664 <https://github.com/pyviz/holoviews/pull/1664>`__,
   `#1796 <https://github.com/pyviz/holoviews/pull/1796>`__)

Version 1.8.2
*************

This bugfix release addresses a number of minor issues identified since
the 1.8.1 release:

Feature:

-  Added support for groupby to histogram operation.
   (`#1725 <https://github.com/pyviz/holoviews/pull/1725>`__)

Fixes:

-  Fixed problem with HTML export due to new extension logos.
   (`#1778 <https://github.com/pyviz/holoviews/pull/1778>`__)
-  Replaced deprecated ``__call__`` usage with opts method throughout
   codebase. (`#1759 <https://github.com/pyviz/holoviews/pull/1759>`__,
   `#1763 <https://github.com/pyviz/holoviews/pull/1763>`__,
   `#1779 <https://github.com/pyviz/holoviews/pull/1779>`__)
-  Fixed pip installation.
   (`#1782 <https://github.com/pyviz/holoviews/pull/1782>`__)
-  Fixed miscellaneous bugs
   (`#1724 <https://github.com/pyviz/holoviews/pull/1724>`__,
   `#1739 <https://github.com/pyviz/holoviews/pull/1739>`__,
   `#1711 <https://github.com/pyviz/holoviews/pull/1711>`__)

Version 1.8.1
*************

This bugfix release addresses a number of minor issues identified since
the 1.8 release:

Feature:

-  All enabled plotting extension logos now shown
   (`#1694 <https://github.com/pyviz/holoviews/pull/1694>`__)

Fixes:

-  Updated search ordering when looking for holoviews.rc
   (`#1700 <https://github.com/pyviz/holoviews/pull/1700>`__)
-  Fixed lower bound inclusivity bug when no upper bound supplied
   (`#1686 <https://github.com/pyviz/holoviews/pull/1686>`__)
-  Raise SkipRendering error when plotting nested layouts
   (`#1687 <https://github.com/pyviz/holoviews/pull/1687>`__)
-  Added safety margin for grid axis constraint issue
   (`#1695 <https://github.com/pyviz/holoviews/pull/1685>`__)
-  Fixed bug when using +framewise
   (`#1685 <https://github.com/pyviz/holoviews/pull/1685>`__)
-  Fixed handling of Spacer models in sparse grid
   (`#1682 <https://github.com/pyviz/holoviews/pull/>`__)
-  Renamed Bounds to BoundsXY for consistency
   (`#1672 <https://github.com/pyviz/holoviews/pull/1672>`__)
-  Fixed Bokeh log axes with axis lower bound <-0
   (`#1691 <https://github.com/pyviz/holoviews/pull/1691>`__)
-  Set default datashader cmap to fire
   (`#1697 <https://github.com/pyviz/holoviews/pull/1697>`__)
-  Set SpikesPlot color index to None by default
   (`#1671 <https://github.com/pyviz/holoviews/pull/1671>`__)
-  Documentation fixes
   (`#1662 <https://github.com/pyviz/holoviews/pull/1662>`__,
   `#1665 <https://github.com/pyviz/holoviews/pull/1665>`__,
   `#1690 <https://github.com/pyviz/holoviews/pull/1690>`__,
   `#1692 <https://github.com/pyviz/holoviews/pull/1692>`__,
   `#1658 <https://github.com/pyviz/holoviews/pull/1658>`__)

Version 1.8.0
*************

This release includes a complete and long awaited overhaul of the
HoloViews documentation and website, with a new gallery, getting-started
section, and logo. In the process, we have also improved and made small
fixes to all of the major new functionality that appeared in 1.7.0 but
was not properly documented until now. We want to thank all our old and
new contributors for providing feedback, bug reports, and pull requests.

Major features:

-  Completely overhauled the documentation and website
   (`#1384 <https://github.com/pyviz/holoviews/pull/1384>`__,
   `#1473 <https://github.com/pyviz/holoviews/pull/1473>`__,
   `#1476 <https://github.com/pyviz/holoviews/pull/1476>`__,
   `#1473 <https://github.com/pyviz/holoviews/pull/1473>`__,
   `#1537 <https://github.com/pyviz/holoviews/pull/1537>`__,
   `#1585 <https://github.com/pyviz/holoviews/pull/1585>`__,
   `#1628 <https://github.com/pyviz/holoviews/pull/1628>`__,
   `#1636 <https://github.com/pyviz/holoviews/pull/1636>`__)
-  Replaced dependency on bkcharts with new Bokeh bar plot
   (`#1416 <https://github.com/pyviz/holoviews/pull/1416>`__) and Bokeh
   BoxWhisker plot
   (`#1604 <https://github.com/pyviz/holoviews/pull/1604>`__)
-  Added support for drawing the ``Arrow`` annotation in Bokeh
   (`#1608 <https://github.com/pyviz/holoviews/pull/1608>`__)
-  Added periodic method DynamicMap to schedule recurring events
   (`#1429 <https://github.com/pyviz/holoviews/pull/1429>`__)
-  Cleaned up the API for deploying to Bokeh server
   (`#1444 <https://github.com/pyviz/holoviews/pull/1444>`__,
   `#1469 <https://github.com/pyviz/holoviews/pull/1469>`__,
   `#1486 <https://github.com/pyviz/holoviews/pull/1486>`__)
-  Validation of invalid backend specific options
   (`#1465 <https://github.com/pyviz/holoviews/pull/1465>`__)
-  Added utilities and entry points to convert notebooks to scripts
   including magics
   (`#1491 <https://github.com/pyviz/holoviews/pull/1491>`__)
-  Added support for rendering to png in Bokeh backend
   (`#1493 <https://github.com/pyviz/holoviews/pull/1493>`__)
-  Made Matplotlib and Bokeh styling more consistent and dropped custom
   Matplotlib rc file
   (`#1518 <https://github.com/pyviz/holoviews/pull/1518>`__)
-  Added ``iloc`` and ``ndloc`` method to allow integer based indexing
   on tabular and gridded datasets
   (`#1435 <https://github.com/pyviz/holoviews/pull/1435>`__)
-  Added option to restore case sensitive completion order by setting
   ``hv.extension.case_sensitive_completion-True`` in python or via
   holoviews.rc file
   (`#1613 <https://github.com/pyviz/holoviews/pull/1613>`__)

Other new features and improvements:

-  Optimized datashading of ``NdOverlay``
   (`#1430 <https://github.com/pyviz/holoviews/pull/1430>`__)
-  Expose last ``DynamicMap`` args and kwargs on Callable
   (`#1453 <https://github.com/pyviz/holoviews/pull/1453>`__)
-  Allow colormapping ``Contours`` Element
   (`#1499 <https://github.com/pyviz/holoviews/pull/1499>`__)
-  Add support for fixed ticks with labels in Bokeh backend
   (`#1503 <https://github.com/pyviz/holoviews/pull/1503>`__)
-  Added a ``clim`` parameter to datashade controlling the color range
   (`#1508 <https://github.com/pyviz/holoviews/pull/1508>`__)
-  Add support for wrapping xarray DataArrays containing Dask arrays
   (`#1512 <https://github.com/pyviz/holoviews/pull/1512>`__)
-  Added support for aggregating to target ``Image`` dimensions in
   datashader ``aggregate`` operation
   (`#1513 <https://github.com/pyviz/holoviews/pull/1513>`__)
-  Added top-level hv.extension and ``hv.renderer`` utilities
   (`#1517 <https://github.com/pyviz/holoviews/pull/1517>`__)
-  Added support for ``Splines`` defining multiple cubic splines in
   Bokeh (`#1529 <https://github.com/pyviz/holoviews/pull/1529>`__)
-  Add support for redim.label to quickly define dimension labels
   (`#1541 <https://github.com/pyviz/holoviews/pull/1541>`__)
-  Add ``BoundsX`` and ``BoundsY`` streams
   (`#1554 <https://github.com/pyviz/holoviews/pull/1554>`__)
-  Added support for adjoining empty plots
   (`#1561 <https://github.com/pyviz/holoviews/pull/1561>`__)
-  Handle zero-values correctly when using ``logz`` colormapping option
   in Matplotlib
   (`#1576 <https://github.com/pyviz/holoviews/pull/1576>`__)
-  Define a number of ``Cycle`` and ``Palette`` defaults across backends
   (`#1605 <https://github.com/pyviz/holoviews/pull/1605>`__)
-  Many other small improvements and fixes
   (`#1399 <https://github.com/pyviz/holoviews/pull/1399>`__,
   `#1400 <https://github.com/pyviz/holoviews/pull/1400>`__,
   `#1405 <https://github.com/pyviz/holoviews/pull/1405>`__,
   `#1412 <https://github.com/pyviz/holoviews/pull/1412>`__,
   `#1413 <https://github.com/pyviz/holoviews/pull/1413>`__,
   `#1418 <https://github.com/pyviz/holoviews/pull/1418>`__,
   `#1439 <https://github.com/pyviz/holoviews/pull/1439>`__,
   `#1442 <https://github.com/pyviz/holoviews/pull/1442>`__,
   `#1443 <https://github.com/pyviz/holoviews/pull/1443>`__,
   `#1467 <https://github.com/pyviz/holoviews/pull/1467>`__,
   `#1485 <https://github.com/pyviz/holoviews/pull/1485>`__,
   `#1505 <https://github.com/pyviz/holoviews/pull/1505>`__,
   `#1493 <https://github.com/pyviz/holoviews/pull/1493>`__,
   `#1509 <https://github.com/pyviz/holoviews/pull/1509>`__,
   `#1524 <https://github.com/pyviz/holoviews/pull/1524>`__,
   `#1543 <https://github.com/pyviz/holoviews/pull/1543>`__,
   `#1547 <https://github.com/pyviz/holoviews/pull/1547>`__,
   `#1560 <https://github.com/pyviz/holoviews/pull/1560>`__,
   `#1603 <https://github.com/pyviz/holoviews/pull/1603>`__)

Changes affecting backwards compatibility:

-  Renamed ``ElementOperation`` to ``Operation``
   (`#1421 <https://github.com/pyviz/holoviews/pull/1421>`__)
-  Removed ``stack_area`` operation in favor of ``Area.stack``
   classmethod (`#1515 <https://github.com/pyviz/holoviews/pull/1515>`__)
-  Removed all mpld3 support
   (`#1516 <https://github.com/pyviz/holoviews/pull/1516>`__)
-  Added ``opts`` method on all types, replacing the now-deprecated
   ``__call__`` syntax to set options
   (`#1589 <https://github.com/pyviz/holoviews/pull/1589>`__)
-  Styling changes for both Matplotlib and Bokeh, which can be reverted
   for a notebook with the ``config`` option of ``hv.extension``. For
   instance, ``hv.extension('bokeh', config-dict(style_17-True))``
   (`#1518 <https://github.com/pyviz/holoviews/pull/1518>`__)

Version 1.7
~~~~~~~~~~~
   
Version 1.7.0
*************

This version is a major new release incorporating seven months of work
involving several hundred PRs and over 1700 commits. Highlights include
extensive new support for easily building highly interactive
`Bokeh <http://bokeh.pydata.org>`__ plots, support for using
`datashader <https://github.com/bokeh/datashader>`__-based plots for
working with large datasets, support for rendering images interactively
but outside of the notebook, better error handling, and support for
Matplotlib 2.0 and Bokeh 0.12.5. The PRs linked below serve as initial
documentation for these features, and full documentation will be added
in the run-up to HoloViews 2.0.

Major features and improvements:

-  Interactive Streams API (PR
   `#832 <https://github.com/pyviz/holoviews/pull/832>`__,
   `#838 <https://github.com/pyviz/holoviews/pull/838>`__,
   `#842 <https://github.com/pyviz/holoviews/pull/842>`__,
   `#844 <https://github.com/pyviz/holoviews/pull/844>`__,
   `#845 <https://github.com/pyviz/holoviews/pull/845>`__,
   `#846 <https://github.com/pyviz/holoviews/pull/846>`__,
   `#858 <https://github.com/pyviz/holoviews/pull/858>`__,
   `#860 <https://github.com/pyviz/holoviews/pull/860>`__,
   `#889 <https://github.com/pyviz/holoviews/pull/889>`__,
   `#904 <https://github.com/pyviz/holoviews/pull/904>`__,
   `#913 <https://github.com/pyviz/holoviews/pull/913>`__,
   `#933 <https://github.com/pyviz/holoviews/pull/933>`__,
   `#962 <https://github.com/pyviz/holoviews/pull/962>`__,
   `#964 <https://github.com/pyviz/holoviews/pull/964>`__,
   `#1094 <https://github.com/pyviz/holoviews/pull/1094>`__,
   `#1256 <https://github.com/pyviz/holoviews/pull/1256>`__,
   `#1274 <https://github.com/pyviz/holoviews/pull/1274>`__,
   `#1297 <https://github.com/pyviz/holoviews/pull/1297>`__,
   `#1301 <https://github.com/pyviz/holoviews/pull/1301>`__,
   `#1303 <https://github.com/pyviz/holoviews/pull/1303>`__).
-  Dynamic Callable API (PR
   `#951 <https://github.com/pyviz/holoviews/pull/951>`__,
   `#1103 <https://github.com/pyviz/holoviews/pull/1103>`__,
   `#1029 <https://github.com/pyviz/holoviews/pull/1029>`__,
   `#968 <https://github.com/pyviz/holoviews/pull/968>`__,
   `#935 <https://github.com/pyviz/holoviews/pull/935>`__,
   `#1063 <https://github.com/pyviz/holoviews/pull/1063>`__,
   `#1260 <https://github.com/pyviz/holoviews/pull/1260>`__).
-  Simpler and more powerful DynamicMap (PR
   `#1238 <https://github.com/pyviz/holoviews/pull/1238>`__,
   `#1240 <https://github.com/pyviz/holoviews/pull/1240>`__,
   `#1243 <https://github.com/pyviz/holoviews/pull/1243>`__,
   `#1257 <https://github.com/pyviz/holoviews/pull/1257>`__,
   `#1267 <https://github.com/pyviz/holoviews/pull/1267>`__,
   `#1302 <https://github.com/pyviz/holoviews/pull/1302>`__,
   `#1304 <https://github.com/pyviz/holoviews/pull/1304>`__,
   `#1305 <https://github.com/pyviz/holoviews/pull/1305>`__).
-  Fully general support for Bokeh events (PR
   `#892 <https://github.com/pyviz/holoviews/pull/892>`__,
   `#1148 <https://github.com/pyviz/holoviews/pull/1148>`__,
   `#1235 <https://github.com/pyviz/holoviews/pull/1235>`__).
-  Datashader operations (PR
   `#894 <https://github.com/pyviz/holoviews/pull/894>`__,
   `#907 <https://github.com/pyviz/holoviews/pull/907>`__,
   `#963 <https://github.com/pyviz/holoviews/pull/963>`__,
   `#1125 <https://github.com/pyviz/holoviews/pull/1125>`__,
   `#1281 <https://github.com/pyviz/holoviews/pull/1281>`__,
   `#1306 <https://github.com/pyviz/holoviews/pull/1306>`__).
-  Support for Bokeh apps and Bokeh Server (PR
   `#959 <https://github.com/pyviz/holoviews/pull/959>`__,
   `#1283 <https://github.com/pyviz/holoviews/pull/1283>`__).
-  Working with renderers interactively outside the notebook (PR
   `#1214 <https://github.com/pyviz/holoviews/pull/1214>`__).
-  Support for Matplotlib 2.0 (PR
   `#867 <https://github.com/pyviz/holoviews/pull/867>`__,
   `#868 <https://github.com/pyviz/holoviews/pull/868>`__,
   `#1131 <https://github.com/pyviz/holoviews/pull/1131>`__,
   `#1264 <https://github.com/pyviz/holoviews/pull/1264>`__,
   `#1266 <https://github.com/pyviz/holoviews/pull/1266>`__).
-  Support for Bokeh 0.12.2, 0.12.3, 0.12.4, and 0.12.5 (PR
   `#899 <https://github.com/pyviz/holoviews/pull/899>`__,
   `#900 <https://github.com/pyviz/holoviews/pull/900>`__,
   `#1007 <https://github.com/pyviz/holoviews/pull/1007>`__,
   `#1036 <https://github.com/pyviz/holoviews/pull/1036>`__,
   `#1116 <https://github.com/pyviz/holoviews/pull/1116>`__).
-  Many new features for the Bokeh backend: widgets editable (PR
   `#1247 <https://github.com/pyviz/holoviews/pull/1247>`__), selection
   colors and interactive legends (PR
   `#1220 <https://github.com/pyviz/holoviews/pull/1220>`__), GridSpace
   axes (PR `#1150 <https://github.com/pyviz/holoviews/pull/1150>`__),
   categorical axes and colormapping (PR
   `#1089 <https://github.com/pyviz/holoviews/pull/1089>`__,
   `#1137 <https://github.com/pyviz/holoviews/pull/1137>`__), computing
   plot size (PR
   `#1140 <https://github.com/pyviz/holoviews/pull/1140>`__), GridSpaces
   inside Layouts (PR
   `#1104 <https://github.com/pyviz/holoviews/pull/1104>`__), Layout/Grid
   titles (PR `#1017 <https://github.com/pyviz/holoviews/pull/1017>`__),
   histogram with live colormapping (PR
   `#928 <https://github.com/pyviz/holoviews/pull/928>`__), colorbars (PR
   `#861 <https://github.com/pyviz/holoviews/pull/861>`__),
   finalize\_hooks (PR
   `#1040 <https://github.com/pyviz/holoviews/pull/1040>`__), labelled
   and show\_frame options (PR
   `#863 <https://github.com/pyviz/holoviews/pull/863>`__,
   `#1013 <https://github.com/pyviz/holoviews/pull/1013>`__), styling
   hover glyphs (PR
   `#1286 <https://github.com/pyviz/holoviews/pull/1286>`__), hiding
   legends on BarPlot (PR
   `#837 <https://github.com/pyviz/holoviews/pull/837>`__), VectorField
   plot (PR `#1196 <https://github.com/pyviz/holoviews/pull/1196>`__),
   Histograms now have same color cycle as mpl
   (`#1008 <https://github.com/pyviz/holoviews/pull/1008>`__).
-  Implemented convenience redim methods to easily set dimension ranges,
   values etc. (PR
   `#1302 <https://github.com/pyviz/holoviews/pull/1302>`__)
-  Made methods on and operations applied to DynamicMap lazy
   (`#422 <https://github.com/pyviz/holoviews/pull/422>`__,
   `#588 <https://github.com/pyviz/holoviews/pull/588>`__,
   `#1188 <https://github.com/pyviz/holoviews/pull/1188>`__,
   `#1240 <https://github.com/pyviz/holoviews/pull/1240>`__,
   `#1227 <https://github.com/pyviz/holoviews/pull/1227>`__)
-  Improved documentation (PR
   `#936 <https://github.com/pyviz/holoviews/pull/936>`__,
   `#1070 <https://github.com/pyviz/holoviews/pull/1070>`__,
   `#1242 <https://github.com/pyviz/holoviews/pull/1242>`__,
   `#1273 <https://github.com/pyviz/holoviews/pull/1273>`__,
   `#1280 <https://github.com/pyviz/holoviews/pull/1280>`__).
-  Improved error handling (PR
   `#906 <https://github.com/pyviz/holoviews/pull/906>`__,
   `#932 <https://github.com/pyviz/holoviews/pull/932>`__,
   `#939 <https://github.com/pyviz/holoviews/pull/939>`__,
   `#949 <https://github.com/pyviz/holoviews/pull/949>`__,
   `#1011 <https://github.com/pyviz/holoviews/pull/1011>`__,
   `#1290 <https://github.com/pyviz/holoviews/pull/1290>`__,
   `#1262 <https://github.com/pyviz/holoviews/pull/1262>`__,
   `#1295 <https://github.com/pyviz/holoviews/pull/1295>`__), including
   re-enabling option system keyword validation (PR
   `#1277 <https://github.com/pyviz/holoviews/pull/1277>`__).
-  Improved testing (PR
   `#834 <https://github.com/pyviz/holoviews/pull/834>`__,
   `#871 <https://github.com/pyviz/holoviews/pull/871>`__,
   `#881 <https://github.com/pyviz/holoviews/pull/881>`__,
   `#941 <https://github.com/pyviz/holoviews/pull/941>`__,
   `#1117 <https://github.com/pyviz/holoviews/pull/1117>`__,
   `#1153 <https://github.com/pyviz/holoviews/pull/1153>`__,
   `#1171 <https://github.com/pyviz/holoviews/pull/1171>`__,
   `#1207 <https://github.com/pyviz/holoviews/pull/1207>`__,
   `#1246 <https://github.com/pyviz/holoviews/pull/1246>`__,
   `#1259 <https://github.com/pyviz/holoviews/pull/1259>`__,
   `#1287 <https://github.com/pyviz/holoviews/pull/1287>`__).

Other new features and improvements:

-  Operations for timeseries (PR
   `#1172 <https://github.com/pyviz/holoviews/pull/1172>`__),
   downsample\_columns (PR
   `#903 <https://github.com/pyviz/holoviews/pull/903>`__),
   interpolate\_curve (PR
   `#1097 <https://github.com/pyviz/holoviews/pull/1097>`__), and stacked
   area (PR `#1193 <https://github.com/pyviz/holoviews/pull/1193>`__).
-  Dataset types can be declared as empty by passing an empty list (PR
   `#1355 <https://github.com/pyviz/holoviews/pull/1355>`__)
-  Plot or style options for Curve interpolation (PR
   `#1097 <https://github.com/pyviz/holoviews/pull/1097>`__), transposing
   layouts (PR `#1100 <https://github.com/pyviz/holoviews/pull/1100>`__),
   multiple paths (PR
   `#997 <https://github.com/pyviz/holoviews/pull/997>`__), and norm for
   ColorbarPlot (PR
   `#957 <https://github.com/pyviz/holoviews/pull/957>`__).
-  Improved options inheritance for more intuitive behavior (PR
   `#1275 <https://github.com/pyviz/holoviews/pull/1275>`__).
-  Image interface providing similar functionality for Image and
   non-Image types (making GridImage obsolete) (PR
   `#994 <https://github.com/pyviz/holoviews/pull/994>`__).
-  Dask data interface (PR
   `#974 <https://github.com/pyviz/holoviews/pull/974>`__,
   `#991 <https://github.com/pyviz/holoviews/pull/991>`__).
-  xarray aggregate/reduce (PR
   `#1192 <https://github.com/pyviz/holoviews/pull/1192>`__).
-  Indicate color clipping and control clipping colors (PR
   `#686 <https://github.com/pyviz/holoviews/pull/686>`__).
-  Better datetime handling (PR
   `#1098 <https://github.com/pyviz/holoviews/pull/1098>`__).
-  Gridmatrix diagonal types (PR
   `#1194 <https://github.com/pyviz/holoviews/pull/1194>`__,
   `#1027 <https://github.com/pyviz/holoviews/pull/1027>`__).
-  log option for histogram operation (PR
   `#929 <https://github.com/pyviz/holoviews/pull/929>`__).
-  Perceptually uniform fire colormap (PR
   `#943 <https://github.com/pyviz/holoviews/pull/943>`__).
-  Support for adjoining overlays (PR
   `#1213 <https://github.com/pyviz/holoviews/pull/1213>`__).
-  coloring weighted average in SideHistogram (PR
   `#1087 <https://github.com/pyviz/holoviews/pull/1087>`__).
-  HeatMap allows displaying multiple values on hover (PR
   `#849 <https://github.com/pyviz/holoviews/pull/849>`__).
-  Allow casting Image to QuadMesh (PR
   `#1282 <https://github.com/pyviz/holoviews/pull/1282>`__).
-  Unused columns are now preserved in gridded groupby (PR
   `#1154 <https://github.com/pyviz/holoviews/pull/1154>`__).
-  Optimizations and fixes for constructing Layout/Overlay types (PR
   `#952 <https://github.com/pyviz/holoviews/pull/952>`__).
-  DynamicMap fixes (PR
   `#848 <https://github.com/pyviz/holoviews/pull/848>`__,
   `#883 <https://github.com/pyviz/holoviews/pull/883>`__,
   `#911 <https://github.com/pyviz/holoviews/pull/911>`__,
   `#922 <https://github.com/pyviz/holoviews/pull/922>`__,
   `#923 <https://github.com/pyviz/holoviews/pull/923>`__,
   `#927 <https://github.com/pyviz/holoviews/pull/927>`__,
   `#944 <https://github.com/pyviz/holoviews/pull/944>`__,
   `#1170 <https://github.com/pyviz/holoviews/pull/1170>`__,
   `#1227 <https://github.com/pyviz/holoviews/pull/1227>`__,
   `#1270 <https://github.com/pyviz/holoviews/pull/1270>`__).
-  Bokeh-backend fixes including handling of empty frames
   (`#835 <https://github.com/pyviz/holoviews/pull/835>`__), faster
   updates (`#905 <https://github.com/pyviz/holoviews/pull/905>`__),
   hover tool fixes
   (`#1004 <https://github.com/pyviz/holoviews/pull/1004>`__,
   `#1178 <https://github.com/pyviz/holoviews/pull/1178>`__,
   `#1092 <https://github.com/pyviz/holoviews/pull/1092>`__,
   `#1250 <https://github.com/pyviz/holoviews/pull/1250>`__) and many
   more (PR `#537 <https://github.com/pyviz/holoviews/pull/537>`__,
   `#851 <https://github.com/pyviz/holoviews/pull/851>`__,
   `#852 <https://github.com/pyviz/holoviews/pull/852>`__,
   `#854 <https://github.com/pyviz/holoviews/pull/854>`__,
   `#880 <https://github.com/pyviz/holoviews/pull/880>`__,
   `#896 <https://github.com/pyviz/holoviews/pull/896>`__,
   `#898 <https://github.com/pyviz/holoviews/pull/898>`__,
   `#921 <https://github.com/pyviz/holoviews/pull/921>`__,
   `#934 <https://github.com/pyviz/holoviews/pull/934>`__,
   `#1004 <https://github.com/pyviz/holoviews/pull/1004>`__,
   `#1010 <https://github.com/pyviz/holoviews/pull/1010>`__,
   `#1014 <https://github.com/pyviz/holoviews/pull/1014>`__,
   `#1030 <https://github.com/pyviz/holoviews/pull/1030>`__,
   `#1069 <https://github.com/pyviz/holoviews/pull/1069>`__,
   `#1072 <https://github.com/pyviz/holoviews/pull/1072>`__,
   `#1085 <https://github.com/pyviz/holoviews/pull/1085>`__,
   `#1157 <https://github.com/pyviz/holoviews/pull/1157>`__,
   `#1086 <https://github.com/pyviz/holoviews/pull/1086>`__,
   `#1169 <https://github.com/pyviz/holoviews/pull/1169>`__,
   `#1195 <https://github.com/pyviz/holoviews/pull/1195>`__,
   `#1263 <https://github.com/pyviz/holoviews/pull/1263>`__).
-  Matplotlib-backend fixes and improvements (PR
   `#864 <https://github.com/pyviz/holoviews/pull/864>`__,
   `#873 <https://github.com/pyviz/holoviews/pull/873>`__,
   `#954 <https://github.com/pyviz/holoviews/pull/954>`__,
   `#1037 <https://github.com/pyviz/holoviews/pull/1037>`__,
   `#1068 <https://github.com/pyviz/holoviews/pull/1068>`__,
   `#1128 <https://github.com/pyviz/holoviews/pull/1128>`__,
   `#1132 <https://github.com/pyviz/holoviews/pull/1132>`__,
   `#1143 <https://github.com/pyviz/holoviews/pull/1143>`__,
   `#1163 <https://github.com/pyviz/holoviews/pull/1163>`__,
   `#1209 <https://github.com/pyviz/holoviews/pull/1209>`__,
   `#1211 <https://github.com/pyviz/holoviews/pull/1211>`__,
   `#1225 <https://github.com/pyviz/holoviews/pull/1225>`__,
   `#1269 <https://github.com/pyviz/holoviews/pull/1269>`__,
   `#1300 <https://github.com/pyviz/holoviews/pull/1300>`__).
-  Many other small improvements and fixes (PR
   `#830 <https://github.com/pyviz/holoviews/pull/830>`__,
   `#840 <https://github.com/pyviz/holoviews/pull/840>`__,
   `#841 <https://github.com/pyviz/holoviews/pull/841>`__,
   `#850 <https://github.com/pyviz/holoviews/pull/850>`__,
   `#855 <https://github.com/pyviz/holoviews/pull/855>`__,
   `#856 <https://github.com/pyviz/holoviews/pull/856>`__,
   `#859 <https://github.com/pyviz/holoviews/pull/859>`__,
   `#865 <https://github.com/pyviz/holoviews/pull/865>`__,
   `#893 <https://github.com/pyviz/holoviews/pull/893>`__,
   `#897 <https://github.com/pyviz/holoviews/pull/897>`__,
   `#902 <https://github.com/pyviz/holoviews/pull/902>`__,
   `#912 <https://github.com/pyviz/holoviews/pull/912>`__,
   `#916 <https://github.com/pyviz/holoviews/pull/916>`__,
   `#925 <https://github.com/pyviz/holoviews/pull/925>`__,
   `#938 <https://github.com/pyviz/holoviews/pull/938>`__,
   `#940 <https://github.com/pyviz/holoviews/pull/940>`__,
   `#948 <https://github.com/pyviz/holoviews/pull/948>`__,
   `#950 <https://github.com/pyviz/holoviews/pull/950>`__,
   `#955 <https://github.com/pyviz/holoviews/pull/955>`__,
   `#956 <https://github.com/pyviz/holoviews/pull/956>`__,
   `#967 <https://github.com/pyviz/holoviews/pull/967>`__,
   `#970 <https://github.com/pyviz/holoviews/pull/970>`__,
   `#972 <https://github.com/pyviz/holoviews/pull/972>`__,
   `#973 <https://github.com/pyviz/holoviews/pull/973>`__,
   `#981 <https://github.com/pyviz/holoviews/pull/981>`__,
   `#992 <https://github.com/pyviz/holoviews/pull/992>`__,
   `#998 <https://github.com/pyviz/holoviews/pull/998>`__,
   `#1009 <https://github.com/pyviz/holoviews/pull/1009>`__,
   `#1012 <https://github.com/pyviz/holoviews/pull/1012>`__,
   `#1016 <https://github.com/pyviz/holoviews/pull/1016>`__,
   `#1023 <https://github.com/pyviz/holoviews/pull/1023>`__,
   `#1034 <https://github.com/pyviz/holoviews/pull/1034>`__,
   `#1043 <https://github.com/pyviz/holoviews/pull/1043>`__,
   `#1045 <https://github.com/pyviz/holoviews/pull/1045>`__,
   `#1046 <https://github.com/pyviz/holoviews/pull/1046>`__,
   `#1048 <https://github.com/pyviz/holoviews/pull/1048>`__,
   `#1050 <https://github.com/pyviz/holoviews/pull/1050>`__,
   `#1051 <https://github.com/pyviz/holoviews/pull/1051>`__,
   `#1054 <https://github.com/pyviz/holoviews/pull/1054>`__,
   `#1060 <https://github.com/pyviz/holoviews/pull/1060>`__,
   `#1062 <https://github.com/pyviz/holoviews/pull/1062>`__,
   `#1074 <https://github.com/pyviz/holoviews/pull/1074>`__,
   `#1082 <https://github.com/pyviz/holoviews/pull/1082>`__,
   `#1084 <https://github.com/pyviz/holoviews/pull/1084>`__,
   `#1088 <https://github.com/pyviz/holoviews/pull/1088>`__,
   `#1093 <https://github.com/pyviz/holoviews/pull/1093>`__,
   `#1099 <https://github.com/pyviz/holoviews/pull/1099>`__,
   `#1115 <https://github.com/pyviz/holoviews/pull/1115>`__,
   `#1119 <https://github.com/pyviz/holoviews/pull/1119>`__,
   `#1121 <https://github.com/pyviz/holoviews/pull/1121>`__,
   `#1130 <https://github.com/pyviz/holoviews/pull/1130>`__,
   `#1133 <https://github.com/pyviz/holoviews/pull/1133>`__,
   `#1151 <https://github.com/pyviz/holoviews/pull/1151>`__,
   `#1152 <https://github.com/pyviz/holoviews/pull/1152>`__,
   `#1155 <https://github.com/pyviz/holoviews/pull/1155>`__,
   `#1156 <https://github.com/pyviz/holoviews/pull/1156>`__,
   `#1158 <https://github.com/pyviz/holoviews/pull/1158>`__,
   `#1162 <https://github.com/pyviz/holoviews/pull/1162>`__,
   `#1164 <https://github.com/pyviz/holoviews/pull/1164>`__,
   `#1174 <https://github.com/pyviz/holoviews/pull/1174>`__,
   `#1175 <https://github.com/pyviz/holoviews/pull/1175>`__,
   `#1180 <https://github.com/pyviz/holoviews/pull/1180>`__,
   `#1187 <https://github.com/pyviz/holoviews/pull/1187>`__,
   `#1197 <https://github.com/pyviz/holoviews/pull/1197>`__,
   `#1202 <https://github.com/pyviz/holoviews/pull/1202>`__,
   `#1205 <https://github.com/pyviz/holoviews/pull/1205>`__,
   `#1206 <https://github.com/pyviz/holoviews/pull/1206>`__,
   `#1210 <https://github.com/pyviz/holoviews/pull/1210>`__,
   `#1217 <https://github.com/pyviz/holoviews/pull/1217>`__,
   `#1219 <https://github.com/pyviz/holoviews/pull/1219>`__,
   `#1228 <https://github.com/pyviz/holoviews/pull/1228>`__,
   `#1232 <https://github.com/pyviz/holoviews/pull/1232>`__,
   `#1241 <https://github.com/pyviz/holoviews/pull/1241>`__,
   `#1244 <https://github.com/pyviz/holoviews/pull/1244>`__,
   `#1245 <https://github.com/pyviz/holoviews/pull/1245>`__,
   `#1249 <https://github.com/pyviz/holoviews/pull/1249>`__,
   `#1254 <https://github.com/pyviz/holoviews/pull/1254>`__,
   `#1255 <https://github.com/pyviz/holoviews/pull/1255>`__,
   `#1271 <https://github.com/pyviz/holoviews/pull/1271>`__,
   `#1276 <https://github.com/pyviz/holoviews/pull/1276>`__,
   `#1278 <https://github.com/pyviz/holoviews/pull/1278>`__,
   `#1285 <https://github.com/pyviz/holoviews/pull/1285>`__,
   `#1288 <https://github.com/pyviz/holoviews/pull/1288>`__,
   `#1289 <https://github.com/pyviz/holoviews/pull/1289>`__).

Changes affecting backwards compatibility:

-  Automatic coloring and sizing on Points now disabled (PR
   `#748 <https://github.com/pyviz/holoviews/pull/748>`__).
-  Deprecated max\_branches output magic option (PR
   `#1293 <https://github.com/pyviz/holoviews/pull/1293>`__).
-  Deprecated GridImage (PR
   `#1292 <https://github.com/pyviz/holoviews/pull/1292>`__,
   `#1223 <https://github.com/pyviz/holoviews/pull/1223>`__).
-  Deprecated NdElement (PR
   `#1191 <https://github.com/pyviz/holoviews/pull/1191>`__).
-  Deprecated DFrame conversion methods (PR
   `#1065 <https://github.com/pyviz/holoviews/pull/1065>`__).
-  Banner text removed from notebook\_extension() (PR
   `#1231 <https://github.com/pyviz/holoviews/pull/1231>`__,
   `#1291 <https://github.com/pyviz/holoviews/pull/1291>`__).
-  Bokeh's Matplotlib compatibility module removed (PR
   `#1239 <https://github.com/pyviz/holoviews/pull/1239>`__).
-  ls as Matplotlib linestyle alias dropped (PR
   `#1203 <https://github.com/pyviz/holoviews/pull/1203>`__).
-  mdims argument of conversion interface renamed to groupby (PR
   `#1066 <https://github.com/pyviz/holoviews/pull/1066>`__).
-  Replaced global alias state with Dimension.label
   (`#1083 <https://github.com/pyviz/holoviews/pull/1083>`__).
-  DynamicMap only update ranges when set to framewise
-  Deprecated DynamicMap sampled, bounded, open and generator modes
   (`#969 <https://github.com/pyviz/holoviews/pull/969>`__,
   `#1305 <https://github.com/pyviz/holoviews/pull/1305>`__)
-  Layout.display method is now deprecated
   (`#1026 <https://github.com/pyviz/holoviews/pull/1026>`__)
-  Layout fix for Matplotlib figures with non-square aspects introduced
   in 1.6.2 (PR `#826 <https://github.com/pyviz/holoviews/pull/826>`__),
   now enabled by default.

Version 1.6
~~~~~~~~~~~
   
Version 1.6.2
*************

Bug fix release with various fixes for gridded data backends and
optimizations for Bokeh.

-  Optimized Bokeh event messaging, reducing the average json payload by
   30-50% (PR `#807 <https://github.com/pyviz/holoviews/pull/807>`__).
-  Fixes for correctly handling NdOverlay types returned by DynamicMaps
   (PR `#814 <https://github.com/pyviz/holoviews/pull/814>`__).
-  Added support for datetime64 handling in Matplotlib and support for
   datetime formatters on Dimension.type\_formatters (PR
   `#816 <https://github.com/pyviz/holoviews/pull/816>`__).
-  Fixed handling of constant dimensions when slicing xarray datasets
   (PR `#817 <https://github.com/pyviz/holoviews/pull/817>`__).
-  Fixed support for passing custom dimensions to iris Datasets (PR
   `#818 <https://github.com/pyviz/holoviews/pull/818>`__).
-  Fixed support for add\_dimension on xarray interface (PR
   `#820 <https://github.com/pyviz/holoviews/pull/820>`__).
-  Improved extents computation on Matplotlib SpreadPlot (PR
   `#821 <https://github.com/pyviz/holoviews/pull/821>`__).
-  Bokeh backend avoids sending data for static frames and empty events
   (PR `#822 <https://github.com/pyviz/holoviews/pull/822>`__).
-  Added major layout fix for figures with non-square aspects, reducing
   the amount of unnecessary whitespace (PR
   `#826 <https://github.com/pyviz/holoviews/pull/826>`__). Disabled by
   default until 1.7 release but can be enabled with:

.. code:: python

    from holoviews.plotting.mpl import LayoutPlot
    LayoutPlot.v17_layout_format - True
    LayoutPlot.vspace - 0.3

Version 1.6.1
*************

Bug fix release following the 1.6 major release with major bug fixes for
the grid data interfaces and improvements to the options system.

-  Ensured that style options incompatible with active backend are
   ignored (PR `#802 <https://github.com/pyviz/holoviews/pull/802>`__).
-  Added support for placing legends outside the plot area in Bokeh (PR
   `#801 <https://github.com/pyviz/holoviews/pull/801>`__).
-  Fix to ensure Bokeh backend does not depend on pandas (PR
   `#792 <https://github.com/pyviz/holoviews/pull/792>`__).
-  Fixed option system to ensure correct inheritance when redefining
   options (PR `#796 <https://github.com/pyviz/holoviews/pull/796>`__).
-  Major refactor and fixes for the grid based data backends (iris,
   xarray and arrays with coordinates) ensuring the data is oriented and
   transposed correctly (PR
   `#794 <https://github.com/pyviz/holoviews/pull/794>`__).

Version 1.6.0
*************

A major release with an optional new data interface based on xarray,
support for batching Bokeh plots for huge increases in performance,
support for Bokeh 0.12 and various other fixes and improvements.

Features and improvements:

-  Made VectorFieldPlot more general with support for independent
   coloring and scaling (PR
   `#701 <https://github.com/pyviz/holoviews/pull/701>`__).
-  Iris interface now allows tuple and dict formats in the constructor
   (PR `#709 <https://github.com/pyviz/holoviews/pull/709>`__.
-  Added support for dynamic groupby on all data interfaces (PR
   `#711 <https://github.com/pyviz/holoviews/pull/711>`__).
-  Added an xarray data interface (PR
   `#713 <https://github.com/pyviz/holoviews/pull/713>`__).
-  Added the redim method to all Dimensioned objects making it easy to
   quickly change dimension names and attributes on nested objects
   `#715 <https://github.com/pyviz/holoviews/pull/715>`__).
-  Added support for batching plots (PR
   `#715 <https://github.com/pyviz/holoviews/pull/717>`__).
-  Support for Bokeh 0.12 release (PR
   `#725 <https://github.com/pyviz/holoviews/pull/725>`__).
-  Added support for logz option on Bokeh Raster plots (PR
   `#729 <https://github.com/pyviz/holoviews/pull/729>`__).
-  Bokeh plots now support custom tick formatters specified via
   Dimension value\_format (PR
   `#728 <https://github.com/pyviz/holoviews/pull/728>`__).
Version 1.5
~~~~~~~~~~~
   
Version 1.5.0
*************

A major release with a large number of new features including new data
interfaces for grid based data, major improvements for DynamicMaps and a
large number of bug fixes.

Features and improvements:

-  Added a grid based data interface to explore n-dimensional gridded
   data easily (PR
   `#562 <https://github.com/pyviz/holoviews/pull/542>`__).
-  Added data interface based on `iris
   Cubes <http://scitools.org.uk/iris/docs/v1.9.2/index.html>`__ (PR
   `#624 <https://github.com/pyviz/holoviews/pull/624>`__).
-  Added support for dynamic operations and overlaying of DynamicMaps
   (PR `#588 <https://github.com/pyviz/holoviews/pull/588>`__).
-  Added support for applying groupby operations to DynamicMaps (PR
   `#667 <https://github.com/pyviz/holoviews/pull/667>`__).
-  Added dimension value formatting in widgets (PR
   `#562 <https://github.com/pyviz/holoviews/issues/562>`__).
-  Added support for indexing and slicing with a function (PR
   `#619 <https://github.com/pyviz/holoviews/pull/619>`__).
-  Improved throttling behavior on widgets (PR
   `#596 <https://github.com/pyviz/holoviews/pull/596>`__).
-  Major refactor of Matplotlib plotting classes to simplify
   implementing new Element plots (PR
   `#438 <https://github.com/pyviz/holoviews/pull/438>`__).
-  Added Renderer.last\_plot attribute to allow easily debugging or
   modifying the last displayed plot (PR
   `#538 <https://github.com/pyviz/holoviews/pull/538>`__).
-  Added Bokeh QuadMeshPlot (PR
   `#661 <https://github.com/pyviz/holoviews/pull/661>`__).

Bug fixes:

-  Fixed overlaying of 3D Element types (PR
   `#504 <https://github.com/pyviz/holoviews/pull/504>`__).
-  Fix for Bokeh hovertools with dimensions with special characters (PR
   `#524 <https://github.com/pyviz/holoviews/pull/524>`__).
-  Fixed bugs in seaborn Distribution Element (PR
   `#630 <https://github.com/pyviz/holoviews/pull/630>`__).
-  Fix for inverted Raster.reduce method (PR
   `#672 <https://github.com/pyviz/holoviews/pull/672>`__).
-  Fixed Store.add\_style\_opts method (PR
   `#587 <https://github.com/pyviz/holoviews/pull/587>`__).
-  Fixed bug preventing simultaneous logx and logy plot options (PR
   `#554 <https://github.com/pyviz/holoviews/pull/554>`__).

Backwards compatibility:

-  Renamed ``Columns`` type to ``Dataset`` (PR
   `#620 <https://github.com/pyviz/holoviews/issues/620>`__).

Version 1.4
~~~~~~~~~~~
   
Version 1.4.3
*************

A minor bugfix release to patch a number of small but important issues.

Fixes and improvements:

-  Added a `DynamicMap
   Tutorial <http://holoviews.org/Tutorials/Dynamic_Map.html>`__ to
   explain how to explore very large or continuous parameter spaces in
   HoloViews (`PR
   #470 <https://github.com/pyviz/holoviews/issues/470>`__).
-  Various fixes and improvements for DynamicMaps including slicing (`PR
   #488 <https://github.com/pyviz/holoviews/issues/488>`__) and
   validation (`PR
   #483 <https://github.com/pyviz/holoviews/issues/478>`__) and
   serialization (`PR
   #483 <https://github.com/pyviz/holoviews/issues/478>`__)
-  Widgets containing Matplotlib plots now display the first frame from
   cache providing at least the initial frame when exporting DynamicMaps
   (`PR #486 <https://github.com/pyviz/holoviews/issues/483>`__)
-  Fixed plotting Bokeh plots using widgets in live mode, after changes
   introduced in latest Bokeh version (commit
   `1b87c91e9 <https://github.com/pyviz/holoviews/commit/1b87c91e9e7cf35b267344ccd4a2fa91dd052890>`__).
-  Fixed issue in coloring Point/Scatter objects by values (`Issue
   #467 <https://github.com/pyviz/holoviews/issues/467>`__).

Backwards compatibility:

-  The behavior of the ``scaling_factor`` on Point and Scatter plots has
   changed now simply multiplying ``area`` or ``width`` (as defined by
   the ``scaling_method``). To disable scaling points by a dimension set
   ``size_index-None``.
-  Removed hooks to display 3D Elements using the ``BokehMPLRawWrapper``
   in Bokeh (`PR #477 <https://github.com/pyviz/holoviews/pull/477>`__)
-  Renamed the DynamicMap mode ``closed`` to ``bounded`` (`PR
   #477 <https://github.com/pyviz/holoviews/pull/485>`__)

Version 1.4.2
*************

Over the past month since the 1.4.1 release, we have improved our
infrastructure for building documentation, updated the main website and
made several additional usability improvements.

Documentation changes:

-  Major overhaul of website and notebook building making it much easier
   to test user contributions (`Issue
   #180 <https://github.com/pyviz/holoviews/issues/180>`__, `PR
   #429 <https://github.com/pyviz/holoviews/pull/429>`__)
-  Major rewrite of the documentation (`PR
   #401 <https://github.com/pyviz/holoviews/pull/401>`__, `PR
   #411 <https://github.com/pyviz/holoviews/pull/411>`__)
-  Added Columnar Data Tutorial and removed most of Pandas Conversions
   as it is now supported by the core.

Fixes and improvements:

-  Major improvement for grid based layouts with varying aspects (`PR
   #457 <https://github.com/pyviz/holoviews/pull/457>`__)
-  Fix for interleaving %matplotline inline and holoviews plots (`Issue
   #179 <https://github.com/pyviz/holoviews/issues/179>`__)
-  Matplotlib legend z-orders and updating fixed (`Issue
   #304 <https://github.com/pyviz/holoviews/issues/304>`__, `Issue
   #305 <https://github.com/pyviz/holoviews/issues/305>`__)
-  ``color_index`` and ``size_index`` plot options support specifying
   dimension by name (`Issue
   #391 <https://github.com/pyviz/holoviews/issues/391>`__)
-  Added ``Area`` Element type for drawing area under or between Curves.
   (`PR #427 <https://github.com/pyviz/holoviews/pull/427>`__)
-  Fixed issues where slicing would remove styles applied to an Element.
   (`Issue #423 <https://github.com/pyviz/holoviews/issues/423>`__, `PR
   #439 <https://github.com/pyviz/holoviews/pull/439>`__)
-  Updated the ``title_format`` plot option to support a
   ``{dimensions}`` formatter (`PR
   #436 <https://github.com/pyviz/holoviews/pull/436>`__)
-  Improvements to Renderer API to allow JS and CSS requirements for
   exporting standalone widgets (`PR
   #426 <https://github.com/pyviz/holoviews/pull/426>`__)
-  Compatibility with the latest Bokeh 0.11 release (`PR
   #393 <https://github.com/pyviz/holoviews/pull/393>`__)

Version 1.4.1
*************

Over the past two weeks since the 1.4 release, we have implemented
several important bug fixes and have made several usability
improvements.

New features:

-  Improved help system. It is now possible to recursively list all the
   applicable documentation for a composite object. In addition, the
   documentation may now be filtered using a regular expression pattern.
   (`PR #370 <https://github.com/pyviz/holoviews/pull/370>`__)
-  HoloViews now supports multiple active display hooks making it easier
   to use nbconvert. For instance, PNG data will be embedded in the
   notebook if the argument display\_formats-['html','png'] is supplied
   to the notebook\_extension. (`PR
   #355 <https://github.com/pyviz/holoviews/pull/355>`__)
-  Improvements to the display of DynamicMaps as well as many new
   improvements to the Bokeh backend including better VLines/HLines and
   support for the Bars element. (`PR
   #367 <https://github.com/pyviz/holoviews/pull/367>`__ , `PR
   #362 <https://github.com/pyviz/holoviews/pull/362>`__, `PR
   #339 <https://github.com/pyviz/holoviews/pull/339>`__).
-  New Spikes and BoxWhisker elements suitable for representing
   distributions as a sequence of lines or as a box-and-whisker plot.
   (`PR #346 <https://github.com/pyviz/holoviews/pull/346>`__, `PR
   #339 <https://github.com/pyviz/holoviews/pull/339>`__)
-  Improvements to the notebook\_extension. For instance, executing
   hv.notebook\_extension('bokeh') will now load BokehJS and
   automatically activate the Bokeh backend (if available).
-  Significant performance improvements when using the groupby operation
   on HoloMaps and when working with highly nested datastructures. (`PR
   #349 <https://github.com/pyviz/holoviews/pull/349>`__, `PR
   #359 <https://github.com/pyviz/holoviews/pull/359>`__)

Notable bug fixes:

-  DynamicMaps are now properly integrated into the style system and can
   be customized in the same way as HoloMaps. (`PR
   #368 <https://github.com/pyviz/holoviews/pull/368>`__)
-  Widgets now work correctly when unicode is used in the dimension
   labels and values (`PR
   #376 <https://github.com/pyviz/holoviews/pull/376>`__).

Version 1.4.0
*************

Over the past few months we have added several major new features and
with the help of our users have been able to address a number of bugs
and inconsistencies. We have closed 57 issues and added over 1100 new
commits.

Major new features:

-  Data API: The new data API brings an extensible system of to add new
   data interfaces to column based Element types. These interfaces allow
   applying powerful operations on the data independently of the data
   format. The currently supported datatypes include NumPy, pandas
   dataframes and a simple dictionary format. (`PR
   #284 <https://github.com/pyviz/holoviews/pull/284>`__)
-  Backend API: In this release we completely refactored the rendering,
   plotting and IPython display system to make it easy to add new
   plotting backends. Data may be styled and pickled for each backend
   independently and renderers now support exporting all plotting data
   including widgets as standalone HTML files or with separate JSON
   data.
-  Bokeh backend: The first new plotting backend added via the new
   backend API. Bokeh plots allow for much faster plotting and greater
   interactivity. Supports most Element types and layouts and provides
   facilities for sharing axes across plots and linked brushing across
   plots. (`PR #250 <https://github.com/pyviz/holoviews/pull/250>`__)
-  DynamicMap: The new DynamicMap class allows HoloMap data to be
   generated on-the-fly while running a Jupyter IPython notebook kernel.
   Allows visualization of unbounded data streams and smooth exploration
   of large continuous parameter spaces. (`PR
   #278 <https://github.com/pyviz/holoviews/pull/278>`__)

Other features:

-  Easy definition of custom aliases for group, label and Dimension
   names, allowing easier use of LaTeX.
-  New Trisurface and QuadMesh elements.
-  Widgets now allow expressing hierarchical relationships between
   dimensions.
-  Added GridMatrix container for heterogeneous Elements and gridmatrix
   operation to generate scatter matrix showing relationship between
   dimensions.
-  Filled contour regions can now be generated using the contours
   operation.
-  Consistent indexing semantics for all Elements and support for
   boolean indexing for Columns and NdMapping types.
-  New hv.notebook\_extension function offers a more flexible
   alternative to %load\_ext, e.g. for loading other extensions
   hv.notebook\_extension(bokeh-True).

Experimental features:

-  Bokeh callbacks allow adding interactivity by communicating between
   BokehJS tools and the IPython kernel, e.g. allowing downsampling
   based on the zoom level.

Notable bug fixes:

-  Major speedup rendering large HoloMaps (~ 2-3 times faster).
-  Colorbars now consistent for all plot configurations.
-  Style pickling now works correctly.

API Changes:

-  Dimension formatter parameter now deprecated in favor of
   value\_format.
-  Types of Chart and Table Element data now dependent on selected
   interface.
-  DFrame conversion interface deprecated in favor of Columns pandas
   interface.

Version 1.3
~~~~~~~~~~~

Version 1.3.2
*************

Minor bugfix release to address a small number of issues:

Features:

-  Added support for colorbars on Surface Element (1cd5281).
-  Added linewidth style option to SurfacePlot (9b6ccc5).

Bug fixes:

-  Fixed inversion inversion of y-range during sampling (6ff81bb).
-  Fixed overlaying of 3D elements (787d511).
-  Ensuring that underscore.js is loaded in widgets (f2f6378).
-  Fixed Python3 issue in Overlay.get (8ceabe3).

Version 1.3.1
*************

Minor bugfix release to address a number of issues that weren't caught
in time for the 1.3.0 release with the addition of a small number of
features:

Features:

-  Introduced new ``Spread`` element to plot errors and confidence
   intervals (30d3184).
-  ``ErrorBars`` and ``Spread`` elements now allow most Chart
   constructor types (f013deb).

Bug fixes:

-  Fixed unicode handling for dimension labels (061e9af).
-  Handling of invalid dimension label characters in widgets (a101b9e).
-  Fixed setting of fps option for MPLRenderer video output (c61b9df).
-  Fix for multiple and animated colorbars (5e1e4b5).
-  Fix to Chart slices starting or ending at zero (edd0039).

Version 1.3.0
*************

Since the last release we closed over 34 issues and have made 380
commits mostly focused on fixing bugs, cleaning up the API and working
extensively on the plotting and rendering system to ensure HoloViews is
fully backend independent.

We'd again like to thank our growing user base for all their input,
which has helped us in making the API more understandable and fixing a
number of important bugs.

Highlights/Features:

-  Allowed display of data structures which do not match the recommended
   nesting hierarchy (67b28f3, fbd89c3).
-  Dimensions now sanitized for ``.select``, ``.sample`` and ``.reduce``
   calls (6685633, 00b5a66).
-  Added ``holoviews.ipython.display`` function to render (and display)
   any HoloViews object, useful for IPython interact widgets (0fa49cd).
-  Table column widths now adapt to cell contents (be90a54).
-  Defaulting to Matplotlib ticking behavior (62e1e58).
-  Allowed specifying fixed figure sizes to Matplotlib via
   ``fig_inches`` tuples using (width, None) and (None, height) formats
   (632facd).
-  Constructors of ``Chart``, ``Path`` and ``Histogram`` classes now
   support additional data formats (2297375).
-  ``ScrubberWidget`` now supports all figure formats (c317db4).
-  Allowed customizing legend positions on ``Bars`` Elements (5a12882).
-  Support for multiple colorbars on one axis (aac7b92).
-  ``.reindex`` on ``NdElement`` types now support converting between
   key and value dimensions allowing more powerful conversions.
   (03ac3ce)
-  Improved support for casting between ``Element`` types (cdaab4e,
   b2ad91b, ce7fe2d, 865b4d5).
-  The ``%%opts`` cell magic may now be used multiple times in the same
   cell (2a77fd0)
-  Matplotlib rcParams can now be set correctly per figure (751210f).
-  Improved ``OptionTree`` repr which now works with eval (2f824c1).
-  Refactor of rendering system and IPython extension to allow easy
   swapping of plotting backend (#141)
-  Large plotting optimization by computing tight ``bbox_inches`` once
   (e34e339).
-  Widgets now cache frames in the DOM, avoiding flickering in some
   browsers and make use of jinja2 template inheritance. (fc7dd2b)
-  Calling a HoloViews object without arguments now clears any
   associated custom styles. (9e8c343)

API Changes

-  Renamed key\_dimensions and value\_dimensions to kdims and vdims
   respectively, while providing backward compatibility for passing and
   accessing the long names (8feb7d2).
-  Combined x/y/zticker plot options into x/y/zticks parameters which
   now accept an explicit number of ticks, an explicit list of tick
   positions (and labels), and a Matplotlib tick locator.
-  Changed backend options in %output magic, ``nbagg`` and ``d3`` are
   now modes of the Matplotlib backend and can be selected with
   ``backend-'matplotlib:nbagg'`` and ``backend-'matplotlib:mpld3'``
   respectively. The 'd3' and 'nbagg' options remain supported but will
   be deprecated in future.
-  Customizations should no longer be applied directly to
   ``Store.options``; the ``Store.options(backend-'matplotlib')`` object
   should be customized instead. There is no longer a need to call the
   deprecated ``Store.register_plots`` method.

Version 1.2
~~~~~~~~~~~
   
Version 1.2.0
*************

Since the last release we closed over 20 issues and have made 334
commits, adding a ton of functionality and fixing a large range of bugs
in the process.

In this release we received some excellent feedback from our users,
which has been greatly appreciated and has helped us address a wide
range of problems.

Highlights/Features:

-  Added new ``ErrorBars`` Element (f2b276b)
-  Added ``Empty`` pseudo-Element to define empty placeholders in
   Layouts (35bac9f1d)
-  Added support for changing font sizes easily (0f54bea)
-  Support for holoviews.rc file (79076c8)
-  Many major speed optimizations for working with and plotting
   HoloViews data structures (fe87b4c, 7578c51, 5876fe6, 8863333)
-  Support for ``GridSpace`` with inner axes (93295c8)
-  New ``aspect_weight`` and ``tight`` Layout plot options for more
   customizability of Layout arrangements (4b1f03d, e6a76b7)
-  Added ``bgcolor`` plot option to easily set axis background color
   (92eb95c)
-  Improved widget layout (f51af02)
-  New ``OutputMagic`` css option to style html output (9d42dc2)
-  Experimental support for PDF output (1e8a59b)
-  Added support for 3D interactivity with nbagg (781bc25)
-  Added ability to support deprecated plot options in %%opts magic.
-  Added ``DrawPlot`` simplifying the implementation of custom plots
   (38e9d44)

API changes:

-  ``Path`` and ``Histogram`` support new constructors (7138ef4,
   03b5d38)
-  New depth argument on the relabel method (f89b89f)
-  Interface to Pandas improved (1a7cd3d)
-  Removed ``xlim``, ``ylim`` and ``zlim`` to eliminate redundancy.
-  Renaming of various plot and style options including:

   -  ``figure_*`` to ``fig_*``
   -  ``vertical_spacing`` and ``horizontal_spacing`` to ``vspace`` and
      ``hspace`` respectively

   \* Deprecation of confusing ``origin`` style option on RasterPlot
-  ``Overlay.__getitem__`` no longer supports integer indexing (use
   ``get`` method instead)

Important bug fixes:

-  Important fixes to inheritance in the options system (d34a931,
   71c1f3a7)
-  Fixes to the select method (df839bea5)
-  Fixes to normalization system (c3ef40b)
-  Fixes to ``Raster`` and ``Image`` extents, ``__getitem__`` and
   sampling.
-  Fixed bug with disappearing adjoined plots (2360972)
-  Fixed plot ordering of overlaid elements across a ``HoloMap``
   (c4f1685)

Version 1.1
~~~~~~~~~~~

Version 1.1.0
*************

Highlights:

-  Support for nbagg as a backend (09eab4f1)
-  New .hvz file format for saving HoloViews objects (bfd5f7af)
-  New ``Polygon`` element type (d1ec8ec8)
-  Greatly improved Unicode support throughout, including support for
   unicode characters in Python 3 attribute names (609a8454)
-  Regular SelectionWidget now supports live rendering (eb5bf8b6)
-  Supports a list of objects in Layout and Overlay constructors
   (5ba1866e)
-  Polar projections now supported (3801b76e)

API changes (not backward compatible):

-  ``xlim``, ``ylim``, ``zlim``, ``xlabel``, ``ylabel`` and ``zlabel``
   have been deprecated (081d4123)
-  Plotting options ``show_xaxis`` and ``show_yaxis`` renamed to
   ``xaxis`` and ``yaxis``, respectively (13393f2a).
-  Deprecated IPySelectionWidget (f59c34c0)

In addition to the above improvements, many miscellaneous bug fixes were
made.

Version 1.0
~~~~~~~~~~~

Version 1.0.1
*************

Minor release addressing bugs and issues with 1.0.0.

Highlights:

-  New separate Pandas Tutorial (8455abc3)
-  Silenced warnings when loading the IPython extension in IPython 3
   (aaa6861b)
-  Added more useful installation options via ``setup.py`` (72ece4db)
-  Improvements and bug-fixes for the ``%%opts`` magic tab-completion
   (e0ad7108)
-  ``DFrame`` now supports standard constructor for pandas dataframes
   (983825c5)
-  ``Tables`` are now correctly formatted using the appropriate
   ``Dimension`` formatter (588bc2a3)
-  Support for unlimited alphabetical subfigure labelling (e039d00b)
-  Miscellaneous bug fixes, including Python 3 compatibility
   improvements.

Version 1.0.0
*************

First public release available on GitHub and PyPI.

.. Backticks and links don't play nicely together in RST

.. |Sankey| replace:: ``Sankey``
.. _Sankey: http://holoviews.org/reference/elements/bokeh/Sankey.html

.. |TriMesh| replace:: ``TriMesh``
.. _TriMesh: http://holoviews.org/reference/elements/bokeh/TriMesh.html

.. |Chord| replace:: ``Chord``
.. _Chord: http://holoviews.org/reference/elements/bokeh/Chord.html

.. |HexTiles| replace:: ``HexTiles``
.. _HexTiles: http://holoviews.org/reference/elements/bokeh/HexTiles.html

.. |Labels| replace:: ``Labels``
.. _Labels: http://holoviews.org/reference/elements/bokeh/Labels.html

.. |Div| replace:: ``Div``
.. _Div: http://holoviews.org/reference/elements/bokeh/Div.html

.. |PointDraw| replace:: ``PointDraw``
.. _PointDraw: http://holoviews.org/reference/streams/bokeh/PointDraw.html

.. |PolyDraw| replace:: ``PolyDraw``
.. _PolyDraw: http://holoviews.org/reference/streams/bokeh/PolyDraw.html

.. |BoxEdit| replace:: ``BoxEdit``
.. _BoxEdit: http://holoviews.org/reference/streams/bokeh/BoxEdit.html

.. |PolyEdit| replace:: ``PolyEdit``
.. _PolyEdit: http://holoviews.org/reference/streams/bokeh/PolyEdit.html

.. |radial HeatMap| replace:: radial ``HeatMap``
.. _radial HeatMap: http://holoviews.org/reference/elements/bokeh/RadialHeatMap.html
