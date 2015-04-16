
Version 1.0.0
-------------

First public release available on GitHub and PyPI.

Version 1.0.1
-------------

Minor release address bugs and issues discovered shortly after 1.0.0.

Highlights:

* New separate Pandas Tutorial.  (8455abc3)
* Silenced warnings when loading the IPython extension in IPython 3. (aaa6861b)
* Added more useful installation options via ``setup.py``. (72ece4db)
* Improvements and bug-fixes the the ``%%opts`` magic tab-completion. (e0ad7108)
* ``DFrame`` now supports standard constructor for pandas dataframes. (983825c5)
* ``Tables`` are now correctly formatted using the appropriate ``Dimension`` formatter. (588bc2a3)
* Support for unlimited alphabetical subfigure labelling. (e039d00b)
* Miscellaneous bug fixes, including Python 3 compatibility improvements.


Version 1.1.0
-------------

Highlights

* Support for nbagg as a backend (09eab4f1)
* New .hvz file format for saving HoloViews objects (bfd5f7af)
* New ``Polygon`` element type (d1ec8ec8)
* Greatly improved Unicode support throughout including support for
  unicode characters in Python 3 attribute names (609a8454)
* Regular SelectionWidget now supports live rendering (eb5bf8b6).
* Support list of objects in Layout and Overlay constructor (5ba1866e)
* Polar projections now supported (3801b76e)

Backward incompatible changes:

* ``xlim``, ``ylim``, ``zlim``, ``xlabel``, ``ylabel`` and ``zlabel``
  have been deprecated (081d4123)
* Plotting options ``show_xaxis`` and ``show_yaxis`` renamed to
  ``xaxis`` and ``yaxis`` respectively (13393f2a).
* Deprecated IPySelectionWidget (f59c34c0)

In addition to the above improvements, many miscellaneous bug fixes
were made.
