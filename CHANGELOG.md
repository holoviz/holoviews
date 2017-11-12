Version 1.9.1
=============

This release is a minor bug fix release patching various issues
which were found in the 1.9.0 release.

Enhancements:

-   Exposed min_alpha parameter on datashader shade and datashade
    operations ([\#2109](https://github.com/ioam/holoviews/pull/2109))

Fixes:

-   Fixed broken bokeh server linked stream throttling
    ([\#2112](https://github.com/ioam/holoviews/pull/2112))
-   Fixed bug in bokeh callbacks preventing linked streams using
    bokeh's on_event callbacks from working
    ([\#2112](https://github.com/ioam/holoviews/pull/2112))
-   Fixed insufficient validation issue for Image and bugs when
    applying regrid operation to xarray based Images
    ([\2117](https://github.com/ioam/holoviews/pull/2117))
-   Fixed handling of dimensions and empty elements in univariate_kde
    and bivariate_kde operations
    ([\#2103](https://github.com/ioam/holoviews/pull/2103))
	
Version 1.9.0
=============

This release includes a large number of long awaited features,
improvements and bug fixes, including streaming and graph support,
binary transfer of bokeh data, fast Image/RGB regridding, first-class
statistics elements and a complete overhaul of the geometry elements.

Particular thanks to all users and contributers who have reported
issues and submitted pull requests.

Features:

-   The kdim and vdim keyword arguments are now positional making the
    declaration of elements less verbose (e.g. Scatter(data, 'x',
    'y')) ([\#1946](https://github.com/ioam/holoviews/pull/1946))
-   Added Graph, Nodes, and EdgePaths elements adding support for
    plotting network graphs
    ([\#1829](https://github.com/ioam/holoviews/pull/1829))
-   Added datashader based regrid operation for fast Image and RGB
    regridding ([\#1773](https://github.com/ioam/holoviews/pull/1773))
-   Added support for binary transport when plotting with bokeh,
    providing huge speedups for dynamic plots
    ([\#1894](https://github.com/ioam/holoviews/pull/1894),
    [\#1896](https://github.com/ioam/holoviews/pull/1896))
-   Added Pipe and Buffer streams for streaming data support
    ([\#2011](https://github.com/ioam/holoviews/pull/2011))
-   Add support for datetime axes on Image, RGB and when applying
    datashading and regridding operations
    ([\#2023](https://github.com/ioam/holoviews/pull/2023))
-   Added Distribution and Bivariate as first class elements which can
    be plotted with matplotlib and bokeh without depending on seaborn
    ([\#1985](https://github.com/ioam/holoviews/pull/1985))
-   Completely overhauled support for plotting geometries with Path,
    Contours and Polygons elements including support for coloring
    individual segments and paths by value
    ([\#1991](https://github.com/ioam/holoviews/pull/1991))

Enhancements:

-   Add support for adjoining all elements on matplotlib plots
    ([\#1033](https://github.com/ioam/holoviews/pull/1033))
-   Improved exception handling for data interfaces
    ([\#2041](https://github.com/ioam/holoviews/pull/2041))
-   Add groupby argument to histogram operation
    ([\#1725](https://github.com/ioam/holoviews/pull/1725))
-   Add support for reverse sort on Dataset elements
    ([\#1843](https://github.com/ioam/holoviews/pull/1843))
-   Added support for invert_x/yaxis on all elements
    ([\#1872](https://github.com/ioam/holoviews/pull/1872),
    [\#1919](https://github.com/ioam/holoviews/pull/1919))

Fixes:

-   Fixed a bug in matplotlib causing the first frame in gif and mp4
    getting stuck
    ([\#1922](https://github.com/ioam/holoviews/pull/1922))
-   Fixed various issues with support for new nested categorical axes
    in bokeh ([\#1933](https://github.com/ioam/holoviews/pull/1933))
-   A large range of other bug fixes too long to list here.

Changes affecting backwards compatibility:

-   The contours operation no longer overlays the contours on top of
    the supplied Image by default and returns a single
    Contours/Polygons rather than an NdOverlay of them
    ([\#1991](https://github.com/ioam/holoviews/pull/1991))
-   The values of the Distribution element should now be defined as a
    key dimension
    ([\#1985](https://github.com/ioam/holoviews/pull/1985))
-   The seaborn interface was removed in its entirety being replaced
    by first class support for statistics elements such as
    Distribution and Bivariate
    ([\#1985](https://github.com/ioam/holoviews/pull/1985))
-   Since kdims and vdims can now be passed as positional arguments
    the bounds argument on Image is no longer positional
    ([\#1946](https://github.com/ioam/holoviews/pull/1946)).
-   The datashade and shade cmap was reverted back to blue due to issues
    with the fire cmap against a white background.
    ([\#2078](https://github.com/ioam/holoviews/pull/2078))
-   Dropped all support for bokeh versions older than 0.12.10
-   histogram operation now returns Histogram elements with less
    generic value dimension and customizable label
    ([\#1836](https://github.com/ioam/holoviews/pull/1836))


Version 1.8.4
=============

This bugfix release includes a number of critical fixes for compatiblity
with bokeh 0.12.9 along with various other bug fixes. Many thanks to our
users for various detailed bug reports, feedback and contributions.

Fixes:

-   Fixes to register BoundsXY stream.
    ([\#1826](https://github.com/ioam/holoviews/pull/1826))
-   Fix for Bounds streams on bokeh server.
    ([\#1883](https://github.com/ioam/holoviews/pull/1883))
-   Compatibility with matplotlib 2.1
    ([\#1842](https://github.com/ioam/holoviews/pull/1842))
-   Fixed bug in scrubber widget and support for scrubbing discrete
    DynamicMaps ([\#1832](https://github.com/ioam/holoviews/pull/1832))
-   Various fixes for compatibility with bokeh 0.12.9
    ([\#1849](https://github.com/ioam/holoviews/pull/1849),
    [\#1866](https://github.com/ioam/holoviews/pull/1886))
-   Fixes for setting QuadMesh ranges.
    ([\#1876](https://github.com/ioam/holoviews/pull/1876))
-   Fixes for inverting Image/RGB/Raster axes in bokeh.
    ([\#1872](https://github.com/ioam/holoviews/pull/1872))


Version 1.8.3
=============

This bugfix release fixes a number of minor issues identified since the
last release:

Features:

-   Add support for setting the bokeh sizing_mode as a plot option
    ([\#1813](https://github.com/ioam/holoviews/pull/1813))

Fixes:

-   Handle StopIteration on DynamicMap correctly.
    ([\#1792](https://github.com/ioam/holoviews/pull/1792))
-   Fix bug with linked streams on empty source element
    ([\#1725](https://github.com/ioam/holoviews/pull/1806))
-   Compatibility with latest datashader 0.6.0 release
    ([\#1773](https://github.com/ioam/holoviews/pull/1773))
-   Fixed missing HTML closing tag in extension
    ([\#1797](https://github.com/ioam/holoviews/issues/1797),
     [\#1809](https://github.com/ioam/holoviews/pull/1809))
-   Various fixes and improvements for documentation
    ([\#1664](https://github.com/ioam/holoviews/pull/1664),
    [\#1796](https://github.com/ioam/holoviews/pull/1796))


Version 1.8.2
=============

This bugfix release addresses a number of minor issues identified since
the 1.8.1 release:

Feature:

-   Added support for groupby to histogram operation.
    ([\#1725](https://github.com/ioam/holoviews/pull/1725))

Fixes:

-   Fixed problem with HTML export due to new extension logos.
    ([\#1778](https://github.com/ioam/holoviews/pull/1778))
-   Replaced deprecated ``__call__`` usage with opts method throughout codebase.
    ([\#1759](https://github.com/ioam/holoviews/pull/1759),
    [\#1763](https://github.com/ioam/holoviews/pull/1763),
    [\#1779](https://github.com/ioam/holoviews/pull/1779))
-   Fixed pip installation.
    ([\#1782](https://github.com/ioam/holoviews/pull/1782))
-   Fixed miscellaneous bugs
   ([\#1724](https://github.com/ioam/holoviews/pull/1724),
   [\#1739](https://github.com/ioam/holoviews/pull/1739),
   [\#1711](https://github.com/ioam/holoviews/pull/1711))

Version 1.8.1
=============

This bugfix release addresses a number of minor issues identified since
the 1.8 release:

Feature:

-   All enabled plotting extension logos now shown
    ([\#1694](https://github.com/ioam/holoviews/pull/1694))

Fixes:

-   Updated search ordering when looking for holoviews.rc
    ([\#1700](https://github.com/ioam/holoviews/pull/1700))
-   Fixed lower bound inclusivity bug when no upper bound supplied
    ([\#1686](https://github.com/ioam/holoviews/pull/1686))
-   Raise SkipRendering error when plotting nested layouts
    ([\#1687](https://github.com/ioam/holoviews/pull/1687))
-   Added safety margin for grid axis constraint issue
    ([\#1695](https://github.com/ioam/holoviews/pull/1685))
-   Fixed bug when using +framewise
    ([\#1685](https://github.com/ioam/holoviews/pull/1685))
-   Fixed handling of Spacer models in sparse grid
    ([\#1682](https://github.com/ioam/holoviews/pull/))
-   Renamed Bounds to BoundsXY for consistency
    ([\#1672](https://github.com/ioam/holoviews/pull/1672))
-   Fixed bokeh log axes with axis lower bound &lt;=0
    ([\#1691](https://github.com/ioam/holoviews/pull/1691))
-   Set default datashader cmap to fire
    ([\#1697](https://github.com/ioam/holoviews/pull/1697))
-   Set SpikesPlot color index to None by default
    ([\#1671](https://github.com/ioam/holoviews/pull/1671))
-   Documentation fixes
    ([\#1662](https://github.com/ioam/holoviews/pull/1662),
    [\#1665](https://github.com/ioam/holoviews/pull/1665),
    [\#1690](https://github.com/ioam/holoviews/pull/1690),
    [\#1692](https://github.com/ioam/holoviews/pull/1692),
    [\#1658](https://github.com/ioam/holoviews/pull/1658))

Version 1.8.0
=============

This release includes a complete and long awaited overhaul of the
HoloViews documentation and website, with a new gallery, getting-started
section, and logo. In the process, we have also improved and made small
fixes to all of the major new functionality that appeared in 1.7.0 but
was not properly documented until now. We want to thank all our old and
new contributors for providing feedback, bug reports, and pull requests.

Major features:

-   Completely overhauled the documentation and website
    ([\#1384](https://github.com/ioam/holoviews/pull/1384),
    [\#1473](https://github.com/ioam/holoviews/pull/1473),
    [\#1476](https://github.com/ioam/holoviews/pull/1476),
    [\#1473](https://github.com/ioam/holoviews/pull/1473),
    [\#1537](https://github.com/ioam/holoviews/pull/1537),
    [\#1585](https://github.com/ioam/holoviews/pull/1585),
    [\#1628](https://github.com/ioam/holoviews/pull/1628),
    [\#1636](https://github.com/ioam/holoviews/pull/1636))
-   Replaced dependency on bkcharts with new Bokeh bar plot
    ([\#1416](https://github.com/ioam/holoviews/pull/1416)) and bokeh
    BoxWhisker plot
    ([\#1604](https://github.com/ioam/holoviews/pull/1604))
-   Added support for drawing the `Arrow` annotation in bokeh
    ([\#1608](https://github.com/ioam/holoviews/pull/1608))
-   Added periodic method DynamicMap to schedule recurring events
    ([\#1429](https://github.com/ioam/holoviews/pull/1429))
-   Cleaned up the API for deploying to bokeh server
    ([\#1444](https://github.com/ioam/holoviews/pull/1444),
    [\#1469](https://github.com/ioam/holoviews/pull/1469),
    [\#1486](https://github.com/ioam/holoviews/pull/1486))
-   Validation of invalid backend specific options
    ([\#1465](https://github.com/ioam/holoviews/pull/1465))
-   Added utilities and entry points to convert notebooks to scripts
    including magics
    ([\#1491](https://github.com/ioam/holoviews/pull/1491))
-   Added support for rendering to png in bokeh backend
    ([\#1493](https://github.com/ioam/holoviews/pull/1493))
-   Made matplotlib and bokeh styling more consistent and dropped custom
    matplotlib rc file
    ([\#1518](https://github.com/ioam/holoviews/pull/1518))
-   Added `iloc` and `ndloc` method to allow integer based indexing on
    tabular and gridded datasets
    ([\#1435](https://github.com/ioam/holoviews/pull/1435))
-   Added option to restore case sensitive completion order by setting
    `hv.extension.case_sensitive_completion=True` in python or via
    holoviews.rc file
    ([\#1613](https://github.com/ioam/holoviews/pull/1613))

Other new features and improvements:

-   Optimized datashading of `NdOverlay`
    ([\#1430](https://github.com/ioam/holoviews/pull/1430))
-   Expose last `DynamicMap` args and kwargs on Callable
    ([\#1453](https://github.com/ioam/holoviews/pull/1453))
-   Allow colormapping `Contours` Element
    ([\#1499](https://github.com/ioam/holoviews/pull/1499))
-   Add support for fixed ticks with labels in bokeh backend
    ([\#1503](https://github.com/ioam/holoviews/pull/1503))
-   Added a `clim` parameter to datashade controlling the color range
    ([\#1508](https://github.com/ioam/holoviews/pull/1508))
-   Add support for wrapping xarray DataArrays containing dask arrays
    ([\#1512](https://github.com/ioam/holoviews/pull/1512))
-   Added support for aggregating to target `Image` dimensions in
    datashader `aggregate` operation
    ([\#1513](https://github.com/ioam/holoviews/pull/1513))
-   Added top-level hv.extension and `hv.renderer` utilities
    ([\#1517](https://github.com/ioam/holoviews/pull/1517))
-   Added support for `Splines` defining multiple cubic splines in bokeh
    ([\#1529](https://github.com/ioam/holoviews/pull/1529))
-   Add support for redim.label to quickly define dimension labels
    ([\#1541](https://github.com/ioam/holoviews/pull/1541))
-   Add `BoundsX` and `BoundsY` streams
    ([\#1554](https://github.com/ioam/holoviews/pull/1554))
-   Added support for adjoining empty plots
    ([\#1561](https://github.com/ioam/holoviews/pull/1561))
-   Handle zero-values correctly when using `logz` colormapping option
    in matplotlib
    ([\#1576](https://github.com/ioam/holoviews/pull/1576))
-   Define a number of `Cycle` and `Palette` defaults across backends
    ([\#1605](https://github.com/ioam/holoviews/pull/1605))
-   Many other small improvements and fixes
    ([\#1399](https://github.com/ioam/holoviews/pull/1399),
    [\#1400](https://github.com/ioam/holoviews/pull/1400),
    [\#1405](https://github.com/ioam/holoviews/pull/1405),
    [\#1412](https://github.com/ioam/holoviews/pull/1412),
    [\#1413](https://github.com/ioam/holoviews/pull/1413),
    [\#1418](https://github.com/ioam/holoviews/pull/1418),
    [\#1439](https://github.com/ioam/holoviews/pull/1439),
    [\#1442](https://github.com/ioam/holoviews/pull/1442),
    [\#1443](https://github.com/ioam/holoviews/pull/1443),
    [\#1467](https://github.com/ioam/holoviews/pull/1467),
    [\#1485](https://github.com/ioam/holoviews/pull/1485),
    [\#1505](https://github.com/ioam/holoviews/pull/1505),
    [\#1493](https://github.com/ioam/holoviews/pull/1493),
    [\#1509](https://github.com/ioam/holoviews/pull/1509),
    [\#1524](https://github.com/ioam/holoviews/pull/1524),
    [\#1543](https://github.com/ioam/holoviews/pull/1543),
    [\#1547](https://github.com/ioam/holoviews/pull/1547),
    [\#1560](https://github.com/ioam/holoviews/pull/1560),
    [\#1603](https://github.com/ioam/holoviews/pull/1603))

Changes affecting backwards compatibility:

-   Renamed `ElementOperation` to `Operation`
    ([\#1421](https://github.com/ioam/holoviews/pull/1421))
-   Removed `stack_area` operation in favor of `Area.stack` classmethod
    ([\#1515](https://github.com/ioam/holoviews/pull/1515))
-   Removed all mpld3 support
    ([\#1516](https://github.com/ioam/holoviews/pull/1516))
-   Added `opts` method on all types, replacing the now-deprecated
    `__call__` syntax to set options
    ([\#1589](https://github.com/ioam/holoviews/pull/1589))
-   Styling changes for both matplotlib and bokeh, which can be reverted
    for a notebook with the `config` option of `hv.extension`. For
    instance, `hv.extension('bokeh', config=dict(style_17=True))`
    ([\#1518](https://github.com/ioam/holoviews/pull/1518))

Version 1.7.0
=============

This version is a major new release incorporating seven months of work
involving several hundred PRs and over 1700 commits. Highlights include
extensive new support for easily building highly interactive
[Bokeh](http://bokeh.pydata.org) plots, support for using
[datashader](https://github.com/bokeh/datashader)-based plots for
working with large datasets, support for rendering images interactively
but outside of the notebook, better error handling, and support for
Matplotlib 2.0 and Bokeh 0.12.5. The PRs linked below serve as initial
documentation for these features, and full documentation will be added
in the run-up to HoloViews 2.0.

Major features and improvements:

-   Interactive Streams API (PR
    [\#832](https://github.com/ioam/holoviews/pull/832),
    [\#838](https://github.com/ioam/holoviews/pull/838),
    [\#842](https://github.com/ioam/holoviews/pull/842),
    [\#844](https://github.com/ioam/holoviews/pull/844),
    [\#845](https://github.com/ioam/holoviews/pull/845),
    [\#846](https://github.com/ioam/holoviews/pull/846),
    [\#858](https://github.com/ioam/holoviews/pull/858),
    [\#860](https://github.com/ioam/holoviews/pull/860),
    [\#889](https://github.com/ioam/holoviews/pull/889),
    [\#904](https://github.com/ioam/holoviews/pull/904),
    [\#913](https://github.com/ioam/holoviews/pull/913),
    [\#933](https://github.com/ioam/holoviews/pull/933),
    [\#962](https://github.com/ioam/holoviews/pull/962),
    [\#964](https://github.com/ioam/holoviews/pull/964),
    [\#1094](https://github.com/ioam/holoviews/pull/1094),
    [\#1256](https://github.com/ioam/holoviews/pull/1256),
    [\#1274](https://github.com/ioam/holoviews/pull/1274),
    [\#1297](https://github.com/ioam/holoviews/pull/1297),
    [\#1301](https://github.com/ioam/holoviews/pull/1301),
    [\#1303](https://github.com/ioam/holoviews/pull/1303)).
-   Dynamic Callable API (PR
    [\#951](https://github.com/ioam/holoviews/pull/951),
    [\#1103](https://github.com/ioam/holoviews/pull/1103),
    [\#1029](https://github.com/ioam/holoviews/pull/1029),
    [\#968](https://github.com/ioam/holoviews/pull/968),
    [\#935](https://github.com/ioam/holoviews/pull/935),
    [\#1063](https://github.com/ioam/holoviews/pull/1063),
    [\#1260](https://github.com/ioam/holoviews/pull/1260)).
-   Simpler and more powerful DynamicMap (PR
    [\#1238](https://github.com/ioam/holoviews/pull/1238),
    [\#1240](https://github.com/ioam/holoviews/pull/1240),
    [\#1243](https://github.com/ioam/holoviews/pull/1243),
    [\#1257](https://github.com/ioam/holoviews/pull/1257),
    [\#1267](https://github.com/ioam/holoviews/pull/1267),
    [\#1302](https://github.com/ioam/holoviews/pull/1302),
    [\#1304](https://github.com/ioam/holoviews/pull/1304),
    [\#1305](https://github.com/ioam/holoviews/pull/1305)).
-   Fully general support for Bokeh events (PR
    [\#892](https://github.com/ioam/holoviews/pull/892),
    [\#1148](https://github.com/ioam/holoviews/pull/1148),
    [\#1235](https://github.com/ioam/holoviews/pull/1235)).
-   Datashader operations (PR
    [\#894](https://github.com/ioam/holoviews/pull/894),
    [\#907](https://github.com/ioam/holoviews/pull/907),
    [\#963](https://github.com/ioam/holoviews/pull/963),
    [\#1125](https://github.com/ioam/holoviews/pull/1125),
    [\#1281](https://github.com/ioam/holoviews/pull/1281),
    [\#1306](https://github.com/ioam/holoviews/pull/1306)).
-   Support for Bokeh apps and Bokeh Server (PR
    [\#959](https://github.com/ioam/holoviews/pull/959),
    [\#1283](https://github.com/ioam/holoviews/pull/1283)).
-   Working with renderers interactively outside the notebook (PR
    [\#1214](https://github.com/ioam/holoviews/pull/1214)).
-   Support for Matplotlib 2.0 (PR
    [\#867](https://github.com/ioam/holoviews/pull/867),
    [\#868](https://github.com/ioam/holoviews/pull/868),
    [\#1131](https://github.com/ioam/holoviews/pull/1131),
    [\#1264](https://github.com/ioam/holoviews/pull/1264),
    [\#1266](https://github.com/ioam/holoviews/pull/1266)).
-   Support for Bokeh 0.12.2, 0.12.3, 0.12.4, and 0.12.5 (PR
    [\#899](https://github.com/ioam/holoviews/pull/899),
    [\#900](https://github.com/ioam/holoviews/pull/900),
    [\#1007](https://github.com/ioam/holoviews/pull/1007),
    [\#1036](https://github.com/ioam/holoviews/pull/1036),
    [\#1116](https://github.com/ioam/holoviews/pull/1116)).
-   Many new features for the Bokeh backend: widgets editable (PR
    [\#1247](https://github.com/ioam/holoviews/pull/1247)), selection
    colors and interactive legends (PR
    [\#1220](https://github.com/ioam/holoviews/pull/1220)), GridSpace
    axes (PR [\#1150](https://github.com/ioam/holoviews/pull/1150)),
    categorical axes and colormapping (PR
    [\#1089](https://github.com/ioam/holoviews/pull/1089),
    [\#1137](https://github.com/ioam/holoviews/pull/1137)), computing
    plot size (PR
    [\#1140](https://github.com/ioam/holoviews/pull/1140)), GridSpaces
    inside Layouts (PR
    [\#1104](https://github.com/ioam/holoviews/pull/1104)), Layout/Grid
    titles (PR [\#1017](https://github.com/ioam/holoviews/pull/1017)),
    histogram with live colormapping (PR
    [\#928](https://github.com/ioam/holoviews/pull/928)), colorbars (PR
    [\#861](https://github.com/ioam/holoviews/pull/861)),
    finalize\_hooks (PR
    [\#1040](https://github.com/ioam/holoviews/pull/1040)), labelled and
    show\_frame options (PR
    [\#863](https://github.com/ioam/holoviews/pull/863),
    [\#1013](https://github.com/ioam/holoviews/pull/1013)), styling
    hover glyphs (PR
    [\#1286](https://github.com/ioam/holoviews/pull/1286)), hiding
    legends on BarPlot (PR
    [\#837](https://github.com/ioam/holoviews/pull/837)), VectorField
    plot (PR [\#1196](https://github.com/ioam/holoviews/pull/1196)),
    Histograms now have same color cycle as mpl
    ([\#1008](https://github.com/ioam/holoviews/pull/1008)).
-   Implemented convenience redim methods to easily set dimension
    ranges, values etc. (PR
    [\#1302](https://github.com/ioam/holoviews/pull/1302))
-   Made methods on and operations applied to DynamicMap lazy
    ([\#422](https://github.com/ioam/holoviews/pull/422),
    [\#588](https://github.com/ioam/holoviews/pull/588),
    [\#1188](https://github.com/ioam/holoviews/pull/1188),
    [\#1240](https://github.com/ioam/holoviews/pull/1240),
    [\#1227](https://github.com/ioam/holoviews/pull/1227))
-   Improved documentation (PR
    [\#936](https://github.com/ioam/holoviews/pull/936),
    [\#1070](https://github.com/ioam/holoviews/pull/1070),
    [\#1242](https://github.com/ioam/holoviews/pull/1242),
    [\#1273](https://github.com/ioam/holoviews/pull/1273),
    [\#1280](https://github.com/ioam/holoviews/pull/1280)).
-   Improved error handling (PR
    [\#906](https://github.com/ioam/holoviews/pull/906),
    [\#932](https://github.com/ioam/holoviews/pull/932),
    [\#939](https://github.com/ioam/holoviews/pull/939),
    [\#949](https://github.com/ioam/holoviews/pull/949),
    [\#1011](https://github.com/ioam/holoviews/pull/1011),
    [\#1290](https://github.com/ioam/holoviews/pull/1290),
    [\#1262](https://github.com/ioam/holoviews/pull/1262),
    [\#1295](https://github.com/ioam/holoviews/pull/1295)), including
    re-enabling option system keyword validation (PR
    [\#1277](https://github.com/ioam/holoviews/pull/1277)).
-   Improved testing (PR
    [\#834](https://github.com/ioam/holoviews/pull/834),
    [\#871](https://github.com/ioam/holoviews/pull/871),
    [\#881](https://github.com/ioam/holoviews/pull/881),
    [\#941](https://github.com/ioam/holoviews/pull/941),
    [\#1117](https://github.com/ioam/holoviews/pull/1117),
    [\#1153](https://github.com/ioam/holoviews/pull/1153),
    [\#1171](https://github.com/ioam/holoviews/pull/1171),
    [\#1207](https://github.com/ioam/holoviews/pull/1207),
    [\#1246](https://github.com/ioam/holoviews/pull/1246),
    [\#1259](https://github.com/ioam/holoviews/pull/1259),
    [\#1287](https://github.com/ioam/holoviews/pull/1287)).

Other new features and improvements:

-   Operations for timeseries (PR
    [\#1172](https://github.com/ioam/holoviews/pull/1172)),
    downsample\_columns (PR
    [\#903](https://github.com/ioam/holoviews/pull/903)),
    interpolate\_curve (PR
    [\#1097](https://github.com/ioam/holoviews/pull/1097)), and stacked
    area (PR [\#1193](https://github.com/ioam/holoviews/pull/1193)).
-   Dataset types can be declared as empty by passing an empty list (PR
    [\#1355](https://github.com/ioam/holoviews/pull/1355))
-   Plot or style options for Curve interpolation (PR
    [\#1097](https://github.com/ioam/holoviews/pull/1097)), transposing
    layouts (PR [\#1100](https://github.com/ioam/holoviews/pull/1100)),
    multiple paths (PR
    [\#997](https://github.com/ioam/holoviews/pull/997)), and norm for
    ColorbarPlot (PR
    [\#957](https://github.com/ioam/holoviews/pull/957)).
-   Improved options inheritance for more intuitive behavior (PR
    [\#1275](https://github.com/ioam/holoviews/pull/1275)).
-   Image interface providing similar functionality for Image and
    non-Image types (making GridImage obsolete) (PR
    [\#994](https://github.com/ioam/holoviews/pull/994)).
-   dask data interface (PR
    [\#974](https://github.com/ioam/holoviews/pull/974),
    [\#991](https://github.com/ioam/holoviews/pull/991)).
-   xarray aggregate/reduce (PR
    [\#1192](https://github.com/ioam/holoviews/pull/1192)).
-   Indicate color clipping and control clipping colors (PR
    [\#686](https://github.com/ioam/holoviews/pull/686)).
-   Better datetime handling (PR
    [\#1098](https://github.com/ioam/holoviews/pull/1098)).
-   Gridmatrix diagonal types (PR
    [\#1194](https://github.com/ioam/holoviews/pull/1194),
    [\#1027](https://github.com/ioam/holoviews/pull/1027)).
-   log option for histogram operation (PR
    [\#929](https://github.com/ioam/holoviews/pull/929)).
-   Perceptually uniform fire colormap (PR
    [\#943](https://github.com/ioam/holoviews/pull/943)).
-   Support for adjoining overlays (PR
    [\#1213](https://github.com/ioam/holoviews/pull/1213)).
-   coloring weighted average in SideHistogram (PR
    [\#1087](https://github.com/ioam/holoviews/pull/1087)).
-   HeatMap allows displaying multiple values on hover (PR
    [\#849](https://github.com/ioam/holoviews/pull/849)).
-   Allow casting Image to QuadMesh (PR
    [\#1282](https://github.com/ioam/holoviews/pull/1282)).
-   Unused columns are now preserved in gridded groupby (PR
    [\#1154](https://github.com/ioam/holoviews/pull/1154)).
-   Optimizations and fixes for constructing Layout/Overlay types (PR
    [\#952](https://github.com/ioam/holoviews/pull/952)).
-   DynamicMap fixes (PR
    [\#848](https://github.com/ioam/holoviews/pull/848),
    [\#883](https://github.com/ioam/holoviews/pull/883),
    [\#911](https://github.com/ioam/holoviews/pull/911),
    [\#922](https://github.com/ioam/holoviews/pull/922),
    [\#923](https://github.com/ioam/holoviews/pull/923),
    [\#927](https://github.com/ioam/holoviews/pull/927),
    [\#944](https://github.com/ioam/holoviews/pull/944),
    [\#1170](https://github.com/ioam/holoviews/pull/1170),
    [\#1227](https://github.com/ioam/holoviews/pull/1227),
    [\#1270](https://github.com/ioam/holoviews/pull/1270)).
-   Bokeh-backend fixes including handling of empty frames
    ([\#835](https://github.com/ioam/holoviews/pull/835)), faster
    updates ([\#905](https://github.com/ioam/holoviews/pull/905)), hover
    tool fixes ([\#1004](https://github.com/ioam/holoviews/pull/1004),
    [\#1178](https://github.com/ioam/holoviews/pull/1178),
    [\#1092](https://github.com/ioam/holoviews/pull/1092),
    [\#1250](https://github.com/ioam/holoviews/pull/1250)) and many more
    (PR [\#537](https://github.com/ioam/holoviews/pull/537),
    [\#851](https://github.com/ioam/holoviews/pull/851),
    [\#852](https://github.com/ioam/holoviews/pull/852),
    [\#854](https://github.com/ioam/holoviews/pull/854),
    [\#880](https://github.com/ioam/holoviews/pull/880),
    [\#896](https://github.com/ioam/holoviews/pull/896),
    [\#898](https://github.com/ioam/holoviews/pull/898),
    [\#921](https://github.com/ioam/holoviews/pull/921),
    [\#934](https://github.com/ioam/holoviews/pull/934),
    [\#1004](https://github.com/ioam/holoviews/pull/1004),
    [\#1010](https://github.com/ioam/holoviews/pull/1010),
    [\#1014](https://github.com/ioam/holoviews/pull/1014),
    [\#1030](https://github.com/ioam/holoviews/pull/1030),
    [\#1069](https://github.com/ioam/holoviews/pull/1069),
    [\#1072](https://github.com/ioam/holoviews/pull/1072),
    [\#1085](https://github.com/ioam/holoviews/pull/1085),
    [\#1157](https://github.com/ioam/holoviews/pull/1157),
    [\#1086](https://github.com/ioam/holoviews/pull/1086),
    [\#1169](https://github.com/ioam/holoviews/pull/1169),
    [\#1195](https://github.com/ioam/holoviews/pull/1195),
    [\#1263](https://github.com/ioam/holoviews/pull/1263)).
-   Matplotlib-backend fixes and improvements (PR
    [\#864](https://github.com/ioam/holoviews/pull/864),
    [\#873](https://github.com/ioam/holoviews/pull/873),
    [\#954](https://github.com/ioam/holoviews/pull/954),
    [\#1037](https://github.com/ioam/holoviews/pull/1037),
    [\#1068](https://github.com/ioam/holoviews/pull/1068),
    [\#1128](https://github.com/ioam/holoviews/pull/1128),
    [\#1132](https://github.com/ioam/holoviews/pull/1132),
    [\#1143](https://github.com/ioam/holoviews/pull/1143),
    [\#1163](https://github.com/ioam/holoviews/pull/1163),
    [\#1209](https://github.com/ioam/holoviews/pull/1209),
    [\#1211](https://github.com/ioam/holoviews/pull/1211),
    [\#1225](https://github.com/ioam/holoviews/pull/1225),
    [\#1269](https://github.com/ioam/holoviews/pull/1269),
    [\#1300](https://github.com/ioam/holoviews/pull/1300)).
-   Many other small improvements and fixes (PR
    [\#830](https://github.com/ioam/holoviews/pull/830),
    [\#840](https://github.com/ioam/holoviews/pull/840),
    [\#841](https://github.com/ioam/holoviews/pull/841),
    [\#850](https://github.com/ioam/holoviews/pull/850),
    [\#855](https://github.com/ioam/holoviews/pull/855),
    [\#856](https://github.com/ioam/holoviews/pull/856),
    [\#859](https://github.com/ioam/holoviews/pull/859),
    [\#865](https://github.com/ioam/holoviews/pull/865),
    [\#893](https://github.com/ioam/holoviews/pull/893),
    [\#897](https://github.com/ioam/holoviews/pull/897),
    [\#902](https://github.com/ioam/holoviews/pull/902),
    [\#912](https://github.com/ioam/holoviews/pull/912),
    [\#916](https://github.com/ioam/holoviews/pull/916),
    [\#925](https://github.com/ioam/holoviews/pull/925),
    [\#938](https://github.com/ioam/holoviews/pull/938),
    [\#940](https://github.com/ioam/holoviews/pull/940),
    [\#948](https://github.com/ioam/holoviews/pull/948),
    [\#950](https://github.com/ioam/holoviews/pull/950),
    [\#955](https://github.com/ioam/holoviews/pull/955),
    [\#956](https://github.com/ioam/holoviews/pull/956),
    [\#967](https://github.com/ioam/holoviews/pull/967),
    [\#970](https://github.com/ioam/holoviews/pull/970),
    [\#972](https://github.com/ioam/holoviews/pull/972),
    [\#973](https://github.com/ioam/holoviews/pull/973),
    [\#981](https://github.com/ioam/holoviews/pull/981),
    [\#992](https://github.com/ioam/holoviews/pull/992),
    [\#998](https://github.com/ioam/holoviews/pull/998),
    [\#1009](https://github.com/ioam/holoviews/pull/1009),
    [\#1012](https://github.com/ioam/holoviews/pull/1012),
    [\#1016](https://github.com/ioam/holoviews/pull/1016),
    [\#1023](https://github.com/ioam/holoviews/pull/1023),
    [\#1034](https://github.com/ioam/holoviews/pull/1034),
    [\#1043](https://github.com/ioam/holoviews/pull/1043),
    [\#1045](https://github.com/ioam/holoviews/pull/1045),
    [\#1046](https://github.com/ioam/holoviews/pull/1046),
    [\#1048](https://github.com/ioam/holoviews/pull/1048),
    [\#1050](https://github.com/ioam/holoviews/pull/1050),
    [\#1051](https://github.com/ioam/holoviews/pull/1051),
    [\#1054](https://github.com/ioam/holoviews/pull/1054),
    [\#1060](https://github.com/ioam/holoviews/pull/1060),
    [\#1062](https://github.com/ioam/holoviews/pull/1062),
    [\#1074](https://github.com/ioam/holoviews/pull/1074),
    [\#1082](https://github.com/ioam/holoviews/pull/1082),
    [\#1084](https://github.com/ioam/holoviews/pull/1084),
    [\#1088](https://github.com/ioam/holoviews/pull/1088),
    [\#1093](https://github.com/ioam/holoviews/pull/1093),
    [\#1099](https://github.com/ioam/holoviews/pull/1099),
    [\#1115](https://github.com/ioam/holoviews/pull/1115),
    [\#1119](https://github.com/ioam/holoviews/pull/1119),
    [\#1121](https://github.com/ioam/holoviews/pull/1121),
    [\#1130](https://github.com/ioam/holoviews/pull/1130),
    [\#1133](https://github.com/ioam/holoviews/pull/1133),
    [\#1151](https://github.com/ioam/holoviews/pull/1151),
    [\#1152](https://github.com/ioam/holoviews/pull/1152),
    [\#1155](https://github.com/ioam/holoviews/pull/1155),
    [\#1156](https://github.com/ioam/holoviews/pull/1156),
    [\#1158](https://github.com/ioam/holoviews/pull/1158),
    [\#1162](https://github.com/ioam/holoviews/pull/1162),
    [\#1164](https://github.com/ioam/holoviews/pull/1164),
    [\#1174](https://github.com/ioam/holoviews/pull/1174),
    [\#1175](https://github.com/ioam/holoviews/pull/1175),
    [\#1180](https://github.com/ioam/holoviews/pull/1180),
    [\#1187](https://github.com/ioam/holoviews/pull/1187),
    [\#1197](https://github.com/ioam/holoviews/pull/1197),
    [\#1202](https://github.com/ioam/holoviews/pull/1202),
    [\#1205](https://github.com/ioam/holoviews/pull/1205),
    [\#1206](https://github.com/ioam/holoviews/pull/1206),
    [\#1210](https://github.com/ioam/holoviews/pull/1210),
    [\#1217](https://github.com/ioam/holoviews/pull/1217),
    [\#1219](https://github.com/ioam/holoviews/pull/1219),
    [\#1228](https://github.com/ioam/holoviews/pull/1228),
    [\#1232](https://github.com/ioam/holoviews/pull/1232),
    [\#1241](https://github.com/ioam/holoviews/pull/1241),
    [\#1244](https://github.com/ioam/holoviews/pull/1244),
    [\#1245](https://github.com/ioam/holoviews/pull/1245),
    [\#1249](https://github.com/ioam/holoviews/pull/1249),
    [\#1254](https://github.com/ioam/holoviews/pull/1254),
    [\#1255](https://github.com/ioam/holoviews/pull/1255),
    [\#1271](https://github.com/ioam/holoviews/pull/1271),
    [\#1276](https://github.com/ioam/holoviews/pull/1276),
    [\#1278](https://github.com/ioam/holoviews/pull/1278),
    [\#1285](https://github.com/ioam/holoviews/pull/1285),
    [\#1288](https://github.com/ioam/holoviews/pull/1288),
    [\#1289](https://github.com/ioam/holoviews/pull/1289)).

Changes affecting backwards compatibility:

-   Automatic coloring and sizing on Points now disabled (PR
    [\#748](https://github.com/ioam/holoviews/pull/748)).
-   Deprecated max\_branches output magic option (PR
    [\#1293](https://github.com/ioam/holoviews/pull/1293)).
-   Deprecated GridImage (PR
    [\#1292](https://github.com/ioam/holoviews/pull/1292),
    [\#1223](https://github.com/ioam/holoviews/pull/1223)).
-   Deprecated NdElement (PR
    [\#1191](https://github.com/ioam/holoviews/pull/1191)).
-   Deprecated DFrame conversion methods (PR
    [\#1065](https://github.com/ioam/holoviews/pull/1065)).
-   Banner text removed from notebook\_extension() (PR
    [\#1231](https://github.com/ioam/holoviews/pull/1231),
    [\#1291](https://github.com/ioam/holoviews/pull/1291)).
-   Bokeh's matplotlib compatibility module removed (PR
    [\#1239](https://github.com/ioam/holoviews/pull/1239)).
-   ls as matplotlib linestyle alias dropped (PR
    [\#1203](https://github.com/ioam/holoviews/pull/1203)).
-   mdims argument of conversion interface renamed to groupby (PR
    [\#1066](https://github.com/ioam/holoviews/pull/1066)).
-   Replaced global alias state with Dimension.label
    ([\#1083](https://github.com/ioam/holoviews/pull/1083)).
-   DynamicMap only update ranges when set to framewise
-   Deprecated DynamicMap sampled, bounded, open and generator modes
    ([\#969](https://github.com/ioam/holoviews/pull/969),
    [\#1305](https://github.com/ioam/holoviews/pull/1305))
-   Layout.display method is now deprecated
    ([\#1026](https://github.com/ioam/holoviews/pull/1026))
-   Layout fix for matplotlib figures with non-square aspects introduced
    in 1.6.2 (PR [\#826](https://github.com/ioam/holoviews/pull/826)),
    now enabled by default.

Version 1.6.2
=============

Bug fix release with various fixes for gridded data backends and
optimizations for bokeh.

-   Optimized bokeh event messaging, reducing the average json payload
    by 30-50% (PR [\#807](https://github.com/ioam/holoviews/pull/807)).
-   Fixes for correctly handling NdOverlay types returned by DynamicMaps
    (PR [\#814](https://github.com/ioam/holoviews/pull/814)).
-   Added support for datetime64 handling in matplotlib and support for
    datetime formatters on Dimension.type\_formatters (PR
    [\#816](https://github.com/ioam/holoviews/pull/816)).
-   Fixed handling of constant dimensions when slicing xarray datasets
    (PR [\#817](https://github.com/ioam/holoviews/pull/817)).
-   Fixed support for passing custom dimensions to iris Datasets (PR
    [\#818](https://github.com/ioam/holoviews/pull/818)).
-   Fixed support for add\_dimension on xarray interface (PR
    [\#820](https://github.com/ioam/holoviews/pull/820)).
-   Improved extents computation on matplotlib SpreadPlot (PR
    [\#821](https://github.com/ioam/holoviews/pull/821)).
-   Bokeh backend avoids sending data for static frames and empty events
    (PR [\#822](https://github.com/ioam/holoviews/pull/822)).
-   Added major layout fix for figures with non-square aspects, reducing
    the amount of unnecessary whitespace (PR
    [\#826](https://github.com/ioam/holoviews/pull/826)). Disabled by
    default until 1.7 release but can be enabled with:

``` {.sourceCode .python}
from holoviews.plotting.mpl import LayoutPlot
LayoutPlot.v17_layout_format = True
LayoutPlot.vspace = 0.3
```

Version 1.6.1
=============

Bug fix release following the 1.6 major release with major bug fixes for
the grid data interfaces and improvements to the options system.

-   Ensured that style options incompatible with active backend are
    ignored (PR [\#802](https://github.com/ioam/holoviews/pull/802)).
-   Added support for placing legends outside the plot area in bokeh (PR
    [\#801](https://github.com/ioam/holoviews/pull/801)).
-   Fix to ensure bokeh backend does not depend on pandas (PR
    [\#792](https://github.com/ioam/holoviews/pull/792)).
-   Fixed option system to ensure correct inheritance when redefining
    options (PR [\#796](https://github.com/ioam/holoviews/pull/796)).
-   Major refactor and fixes for the grid based data backends (iris,
    xarray and arrays with coordinates) ensuring the data is oriented
    and transposed correctly (PR
    [\#794](https://github.com/ioam/holoviews/pull/794)).

Version 1.6
===========

A major release with an optional new data interface based on xarray,
support for batching bokeh plots for huge increases in performance,
support for bokeh 0.12 and various other fixes and improvements.

Features and improvements:

-   Made VectorFieldPlot more general with support for independent
    coloring and scaling (PR
    [\#701](https://github.com/ioam/holoviews/pull/701)).
-   Iris interface now allows tuple and dict formats in the constructor
    (PR [\#709](https://github.com/ioam/holoviews/pull/709).
-   Added support for dynamic groupby on all data interfaces (PR
    [\#711](https://github.com/ioam/holoviews/pull/711)).
-   Added an xarray data interface (PR
    [\#713](https://github.com/ioam/holoviews/pull/713)).
-   Added the redim method to all Dimensioned objects making it easy to
    quickly change dimension names and attributes on nested objects
    [\#715](https://github.com/ioam/holoviews/pull/715)).
-   Added support for batching plots (PR
    [\#715](https://github.com/ioam/holoviews/pull/717)).
-   Support for bokeh 0.12 release (PR
    [\#725](https://github.com/ioam/holoviews/pull/725)).
-   Added support for logz option on bokeh Raster plots (PR
    [\#729](https://github.com/ioam/holoviews/pull/729)).
-   Bokeh plots now support custom tick formatters specified via
    Dimension value\_format (PR
    [\#728](https://github.com/ioam/holoviews/pull/728)).

Version 1.5
===========

A major release with a large number of new features including new data
interfaces for grid based data, major improvements for DynamicMaps and a
large number of bug fixes.

Features and improvements:

-   Added a grid based data interface to explore n-dimensional gridded
    data easily (PR
    [\#562](https://github.com/ioam/holoviews/pull/542)).
-   Added data interface based on [iris
    Cubes](http://scitools.org.uk/iris/docs/v1.9.2/index.html) (PR
    [\#624](https://github.com/ioam/holoviews/pull/624)).
-   Added support for dynamic operations and overlaying of DynamicMaps
    (PR [\#588](https://github.com/ioam/holoviews/pull/588)).
-   Added support for applying groupby operations to DynamicMaps (PR
    [\#667](https://github.com/ioam/holoviews/pull/667)).
-   Added dimension value formatting in widgets (PR
    [\#562](https://github.com/ioam/holoviews/issues/562)).
-   Added support for indexing and slicing with a function (PR
    [\#619](https://github.com/ioam/holoviews/pull/619)).
-   Improved throttling behavior on widgets (PR
    [\#596](https://github.com/ioam/holoviews/pull/596)).
-   Major refactor of matplotlib plotting classes to simplify
    implementing new Element plots (PR
    [\#438](https://github.com/ioam/holoviews/pull/438)).
-   Added Renderer.last\_plot attribute to allow easily debugging or
    modifying the last displayed plot (PR
    [\#538](https://github.com/ioam/holoviews/pull/538)).
-   Added bokeh QuadMeshPlot (PR
    [\#661](https://github.com/ioam/holoviews/pull/661)).

Bug fixes:

-   Fixed overlaying of 3D Element types (PR
    [\#504](https://github.com/ioam/holoviews/pull/504)).
-   Fix for bokeh hovertools with dimensions with special characters (PR
    [\#524](https://github.com/ioam/holoviews/pull/524)).
-   Fixed bugs in seaborn Distribution Element (PR
    [\#630](https://github.com/ioam/holoviews/pull/630)).
-   Fix for inverted Raster.reduce method (PR
    [\#672](https://github.com/ioam/holoviews/pull/672)).
-   Fixed Store.add\_style\_opts method (PR
    [\#587](https://github.com/ioam/holoviews/pull/587)).
-   Fixed bug preventing simultaneous logx and logy plot options (PR
    [\#554](https://github.com/ioam/holoviews/pull/554)).

Backwards compatibility:

-   Renamed `Columns` type to `Dataset` (PR
    [\#620](https://github.com/ioam/holoviews/issues/620)).

Version 1.4.3
=============

A minor bugfix release to patch a number of small but important issues.

Fixes and improvements:

-   Added a [DynamicMap
    Tutorial](http://holoviews.org/Tutorials/Dynamic_Map.html) to
    explain how to explore very large or continuous parameter spaces in
    HoloViews ([PR
    \#470](https://github.com/ioam/holoviews/issues/470)).
-   Various fixes and improvements for DynamicMaps including slicing
    ([PR \#488](https://github.com/ioam/holoviews/issues/488)) and
    validation ([PR
    \#483](https://github.com/ioam/holoviews/issues/478)) and
    serialization ([PR
    \#483](https://github.com/ioam/holoviews/issues/478))
-   Widgets containing matplotlib plots now display the first frame from
    cache providing at least the initial frame when exporting
    DynamicMaps ([PR
    \#486](https://github.com/ioam/holoviews/issues/483))
-   Fixed plotting bokeh plots using widgets in live mode, after changes
    introduced in latest bokeh version (commit
    [1b87c91e9](https://github.com/ioam/holoviews/commit/1b87c91e9e7cf35b267344ccd4a2fa91dd052890)).
-   Fixed issue in coloring Point/Scatter objects by values ([Issue
    \#467](https://github.com/ioam/holoviews/issues/467)).

Backwards compatibility:

-   The behavior of the `scaling_factor` on Point and Scatter plots has
    changed now simply multiplying `area` or `width` (as defined by the
    `scaling_method`). To disable scaling points by a dimension set
    `size_index=None`.
-   Removed hooks to display 3D Elements using the `BokehMPLRawWrapper`
    in bokeh ([PR \#477](https://github.com/ioam/holoviews/pull/477))
-   Renamed the DynamicMap mode `closed` to `bounded` ([PR
    \#477](https://github.com/ioam/holoviews/pull/485))

Version 1.4.2
=============

Over the past month since the 1.4.1 release, we have improved our
infrastructure for building documentation, updated the main website and
made several additional usability improvements.

Documentation changes:

-   Major overhaul of website and notebook building making it much
    easier to test user contributions ([Issue
    \#180](https://github.com/ioam/holoviews/issues/180), [PR
    \#429](https://github.com/ioam/holoviews/pull/429))
-   Major rewrite of the documentation ([PR
    \#401](https://github.com/ioam/holoviews/pull/401), [PR
    \#411](https://github.com/ioam/holoviews/pull/411))
-   Added Columnar Data Tutorial and removed most of Pandas Conversions
    as it is now supported by the core.

Fixes and improvements:

-   Major improvement for grid based layouts with varying aspects ([PR
    \#457](https://github.com/ioam/holoviews/pull/457))
-   Fix for interleaving %matplotline inline and holoviews plots ([Issue
    \#179](https://github.com/ioam/holoviews/issues/179))
-   Matplotlib legend z-orders and updating fixed ([Issue
    \#304](https://github.com/ioam/holoviews/issues/304), [Issue
    \#305](https://github.com/ioam/holoviews/issues/305))
-   `color_index` and `size_index` plot options support specifying
    dimension by name ([Issue
    \#391](https://github.com/ioam/holoviews/issues/391))
-   Added `Area` Element type for drawing area under or between Curves.
    ([PR \#427](https://github.com/ioam/holoviews/pull/427))
-   Fixed issues where slicing would remove styles applied to
    an Element. ([Issue
    \#423](https://github.com/ioam/holoviews/issues/423), [PR
    \#439](https://github.com/ioam/holoviews/pull/439))
-   Updated the `title_format` plot option to support a `{dimensions}`
    formatter ([PR \#436](https://github.com/ioam/holoviews/pull/436))
-   Improvements to Renderer API to allow JS and CSS requirements for
    exporting standalone widgets ([PR
    \#426](https://github.com/ioam/holoviews/pull/426))
-   Compatibility with the latest Bokeh 0.11 release ([PR
    \#393](https://github.com/ioam/holoviews/pull/393))

Version 1.4.1
=============

Over the past two weeks since the 1.4 release, we have implemented
several important bug fixes and have made several usability
improvements.

New features:

-   Improved help system. It is now possible to recursively list all the
    applicable documentation for a composite object. In addition, the
    documentation may now be filtered using a regular
    expression pattern. ([PR
    \#370](https://github.com/ioam/holoviews/pull/370))
-   HoloViews now supports multiple active display hooks making it
    easier to use nbconvert. For instance, PNG data will be embedded in
    the notebook if the argument display\_formats=\['html','png'\] is
    supplied to the notebook\_extension. ([PR
    \#355](https://github.com/ioam/holoviews/pull/355))
-   Improvements to the display of DynamicMaps as well as many new
    improvements to the Bokeh backend including better VLines/HLines and
    support for the Bars element. ([PR
    \#367](https://github.com/ioam/holoviews/pull/367) , [PR
    \#362](https://github.com/ioam/holoviews/pull/362), [PR
    \#339](https://github.com/ioam/holoviews/pull/339)).
-   New Spikes and BoxWhisker elements suitable for representing
    distributions as a sequence of lines or as a box-and-whisker plot.
    ([PR \#346](https://github.com/ioam/holoviews/pull/346), [PR
    \#339](https://github.com/ioam/holoviews/pull/339))
-   Improvements to the notebook\_extension. For instance,
    executing hv.notebook\_extension('bokeh') will now load BokehJS and
    automatically activate the Bokeh backend (if available).
-   Significant performance improvements when using the groupby
    operation on HoloMaps and when working with highly
    nested datastructures. ([PR
    \#349](https://github.com/ioam/holoviews/pull/349), [PR
    \#359](https://github.com/ioam/holoviews/pull/359))

Notable bug fixes:

-   DynamicMaps are now properly integrated into the style system and
    can be customized in the same way as HoloMaps. ([PR
    \#368](https://github.com/ioam/holoviews/pull/368))
-   Widgets now work correctly when unicode is used in the dimension
    labels and values ([PR
    \#376](https://github.com/ioam/holoviews/pull/376)).

Version 1.4.0
=============

Over the past few months we have added several major new features and
with the help of our users have been able to address a number of bugs
and inconsistencies. We have closed 57 issues and added over 1100 new
commits.

Major new features:

-   Data API: The new data API brings an extensible system of to add new
    data interfaces to column based Element types. These interfaces
    allow applying powerful operations on the data independently of the
    data format. The currently supported datatypes include NumPy, pandas
    dataframes and a simple dictionary format. ([PR
    \#284](https://github.com/ioam/holoviews/pull/284))
-   Backend API: In this release we completely refactored the rendering,
    plotting and IPython display system to make it easy to add new
    plotting backends. Data may be styled and pickled for each backend
    independently and renderers now support exporting all plotting data
    including widgets as standalone HTML files or with separate
    JSON data.
-   Bokeh backend: The first new plotting backend added via the new
    backend API. Bokeh plots allow for much faster plotting and
    greater interactivity. Supports most Element types and layouts and
    provides facilities for sharing axes across plots and linked
    brushing across plots. ([PR
    \#250](https://github.com/ioam/holoviews/pull/250))
-   DynamicMap: The new DynamicMap class allows HoloMap data to be
    generated on-the-fly while running a Jupyter IPython
    notebook kernel. Allows visualization of unbounded data streams and
    smooth exploration of large continuous parameter spaces. ([PR
    \#278](https://github.com/ioam/holoviews/pull/278))

Other features:

-   Easy definition of custom aliases for group, label and Dimension
    names, allowing easier use of LaTeX.
-   New Trisurface and QuadMesh elements.
-   Widgets now allow expressing hierarchical relationships
    between dimensions.
-   Added GridMatrix container for heterogeneous Elements and gridmatrix
    operation to generate scatter matrix showing relationship
    between dimensions.
-   Filled contour regions can now be generated using the
    contours operation.
-   Consistent indexing semantics for all Elements and support for
    boolean indexing for Columns and NdMapping types.
-   New hv.notebook\_extension function offers a more flexible
    alternative to %load\_ext, e.g. for loading other
    extensions hv.notebook\_extension(bokeh=True).

Experimental features:

-   Bokeh callbacks allow adding interactivity by communicating between
    bokehJS tools and the IPython kernel, e.g. allowing downsampling
    based on the zoom level.

Notable bug fixes:

-   Major speedup rendering large HoloMaps (\~ 2-3 times faster).
-   Colorbars now consistent for all plot configurations.
-   Style pickling now works correctly.

API Changes:

-   Dimension formatter parameter now deprecated in favor
    of value\_format.
-   Types of Chart and Table Element data now dependent on
    selected interface.
-   DFrame conversion interface deprecated in favor of Columns
    pandas interface.

Version 1.3.2
=============

Minor bugfix release to address a small number of issues:

Features:

-   Added support for colorbars on Surface Element (1cd5281).
-   Added linewidth style option to SurfacePlot (9b6ccc5).

Bug fixes:

-   Fixed inversion inversion of y-range during sampling (6ff81bb).
-   Fixed overlaying of 3D elements (787d511).
-   Ensuring that underscore.js is loaded in widgets (f2f6378).
-   Fixed Python3 issue in Overlay.get (8ceabe3).

Version 1.3.1
=============

Minor bugfix release to address a number of issues that weren't caught
in time for the 1.3.0 release with the addition of a small number of
features:

Features:

-   Introduced new `Spread` element to plot errors and confidence
    intervals (30d3184).
-   `ErrorBars` and `Spread` elements now allow most Chart constructor
    types (f013deb).

Bug fixes:

-   Fixed unicode handling for dimension labels (061e9af).
-   Handling of invalid dimension label characters in widgets (a101b9e).
-   Fixed setting of fps option for MPLRenderer video output (c61b9df).
-   Fix for multiple and animated colorbars (5e1e4b5).
-   Fix to Chart slices starting or ending at zero (edd0039).

Version 1.3.0
=============

Since the last release we closed over 34 issues and have made 380
commits mostly focused on fixing bugs, cleaning up the API and working
extensively on the plotting and rendering system to ensure HoloViews is
fully backend independent.

We'd again like to thank our growing user base for all their input,
which has helped us in making the API more understandable and fixing a
number of important bugs.

Highlights/Features:

-   Allowed display of data structures which do not match the
    recommended nesting hierarchy (67b28f3, fbd89c3).
-   Dimensions now sanitized for `.select`, `.sample` and `.reduce`
    calls (6685633, 00b5a66).
-   Added `holoviews.ipython.display` function to render (and display)
    any HoloViews object, useful for IPython interact widgets (0fa49cd).
-   Table column widths now adapt to cell contents (be90a54).
-   Defaulting to matplotlib ticking behavior (62e1e58).
-   Allowed specifying fixed figure sizes to matplotlib via `fig_inches`
    tuples using (width, None) and (None, height) formats (632facd).
-   Constructors of `Chart`, `Path` and `Histogram` classes now support
    additional data formats (2297375).
-   `ScrubberWidget` now supports all figure formats (c317db4).
-   Allowed customizing legend positions on `Bars` Elements (5a12882).
-   Support for multiple colorbars on one axis (aac7b92).
-   `.reindex` on `NdElement` types now support converting between key
    and value dimensions allowing more powerful conversions. (03ac3ce)
-   Improved support for casting between `Element` types (cdaab4e,
    b2ad91b, ce7fe2d, 865b4d5).
-   The `%%opts` cell magic may now be used multiple times in the same
    cell (2a77fd0)
-   Matplotlib rcParams can now be set correctly per figure (751210f).
-   Improved `OptionTree` repr which now works with eval (2f824c1).
-   Refactor of rendering system and IPython extension to allow easy
    swapping of plotting backend (\#141)
-   Large plotting optimization by computing tight `bbox_inches`
    once (e34e339).
-   Widgets now cache frames in the DOM, avoiding flickering in some
    browsers and make use of jinja2 template inheritance. (fc7dd2b)
-   Calling a HoloViews object without arguments now clears any
    associated custom styles. (9e8c343)

API Changes

-   Renamed key\_dimensions and value\_dimensions to kdims and vdims
    respectively, while providing backward compatibility for passing and
    accessing the long names (8feb7d2).
-   Combined x/y/zticker plot options into x/y/zticks parameters which
    now accept an explicit number of ticks, an explicit list of tick
    positions (and labels), and a matplotlib tick locator.
-   Changed backend options in %output magic, `nbagg` and `d3` are now
    modes of the matplotlib backend and can be selected with
    `backend='matplotlib:nbagg'` and
    `backend='matplotlib:mpld3'` respectively. The 'd3' and 'nbagg'
    options remain supported but will be deprecated in future.
-   Customizations should no longer be applied directly to
    `Store.options`; the `Store.options(backend='matplotlib')` object
    should be customized instead. There is no longer a need to call the
    deprecated `Store.register_plots` method.

Version 1.2.0
=============

Since the last release we closed over 20 issues and have made 334
commits, adding a ton of functionality and fixing a large range of bugs
in the process.

In this release we received some excellent feedback from our users,
which has been greatly appreciated and has helped us address a wide
range of problems.

Highlights/Features:

-   Added new `ErrorBars` Element (f2b276b)
-   Added `Empty` pseudo-Element to define empty placeholders in
    Layouts (35bac9f1d)
-   Added support for changing font sizes easily (0f54bea)
-   Support for holoviews.rc file (79076c8)
-   Many major speed optimizations for working with and plotting
    HoloViews data structures (fe87b4c, 7578c51, 5876fe6, 8863333)
-   Support for `GridSpace` with inner axes (93295c8)
-   New `aspect_weight` and `tight` Layout plot options for more
    customizability of Layout arrangements (4b1f03d, e6a76b7)
-   Added `bgcolor` plot option to easily set axis background
    color (92eb95c)
-   Improved widget layout (f51af02)
-   New `OutputMagic` css option to style html output (9d42dc2)
-   Experimental support for PDF output (1e8a59b)
-   Added support for 3D interactivity with nbagg (781bc25)
-   Added ability to support deprecated plot options in %%opts magic.
-   Added `DrawPlot` simplifying the implementation of custom
    plots (38e9d44)

API changes:

-   `Path` and `Histogram` support new constructors (7138ef4, 03b5d38)
-   New depth argument on the relabel method (f89b89f)
-   Interface to Pandas improved (1a7cd3d)
-   Removed `xlim`, `ylim` and `zlim` to eliminate redundancy.
-   Renaming of various plot and style options including:

    -   `figure_*` to `fig_*`
    -   `vertical_spacing` and `horizontal_spacing` to `vspace` and
        `hspace` respectively

    \* Deprecation of confusing `origin` style option on RasterPlot
-   `Overlay.__getitem__` no longer supports integer indexing (use `get`
    method instead)

Important bug fixes:

-   Important fixes to inheritance in the options system
    (d34a931, 71c1f3a7)
-   Fixes to the select method (df839bea5)
-   Fixes to normalization system (c3ef40b)
-   Fixes to `Raster` and `Image` extents, `__getitem__` and sampling.
-   Fixed bug with disappearing adjoined plots (2360972)
-   Fixed plot ordering of overlaid elements across a
    `HoloMap` (c4f1685)

Version 1.1.0
=============

Highlights:

-   Support for nbagg as a backend (09eab4f1)
-   New .hvz file format for saving HoloViews objects (bfd5f7af)
-   New `Polygon` element type (d1ec8ec8)
-   Greatly improved Unicode support throughout, including support for
    unicode characters in Python 3 attribute names (609a8454)
-   Regular SelectionWidget now supports live rendering (eb5bf8b6)
-   Supports a list of objects in Layout and Overlay
    constructors (5ba1866e)
-   Polar projections now supported (3801b76e)

API changes (not backward compatible):

-   `xlim`, `ylim`, `zlim`, `xlabel`, `ylabel` and `zlabel` have been
    deprecated (081d4123)
-   Plotting options `show_xaxis` and `show_yaxis` renamed to `xaxis`
    and `yaxis`, respectively (13393f2a).
-   Deprecated IPySelectionWidget (f59c34c0)

In addition to the above improvements, many miscellaneous bug fixes were
made.

Version 1.0.1
=============

Minor release addressing bugs and issues with 1.0.0.

Highlights:

-   New separate Pandas Tutorial (8455abc3)
-   Silenced warnings when loading the IPython extension in IPython
    3 (aaa6861b)
-   Added more useful installation options via `setup.py` (72ece4db)
-   Improvements and bug-fixes for the `%%opts` magic
    tab-completion (e0ad7108)
-   `DFrame` now supports standard constructor for pandas
    dataframes (983825c5)
-   `Tables` are now correctly formatted using the appropriate
    `Dimension` formatter (588bc2a3)
-   Support for unlimited alphabetical subfigure labelling (e039d00b)
-   Miscellaneous bug fixes, including Python 3
    compatibility improvements.

Version 1.0.0
=============

First public release available on GitHub and PyPI.
