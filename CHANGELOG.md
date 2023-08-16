Version 1.17.1
==============
**August 16, 2023**


This release contains a small number of important bug fixes and regressions — many thanks to @ianthomas23, @maximlt, @philippjfr, and @Hoxbro for their contributions.


Enhancements:

- Improve support for `hv.NdOverlay` and `hv.Overlay` in downsample1d ([#5856](https://github.com/holoviz/holoviews/pull/5856))
- Improve legend support for `hv.Layout` and add documentation for `sync_legends` and `show_legends` ([#5852](https://github.com/holoviz/holoviews/pull/5852))

Bug fixes:

- `RangeToolLink` now correctly reset to the predefined `boundsx` and `boundsy` values ([#5848](https://github.com/holoviz/holoviews/pull/5848))
- Fix regressions with Bokeh's axis ([#5838](https://github.com/holoviz/holoviews/pull/5838), [#5850](https://github.com/holoviz/holoviews/pull/5850), [#5851](https://github.com/holoviz/holoviews/pull/5851))

Compatibility:

- Pin Param version for Bokeh 2 ([#5844](https://github.com/holoviz/holoviews/pull/5844))
- Explicitly clear Matplotlib figure to remove warning about auto-removal of overlapping axes for Matplotlib 3.8 ([#5857](https://github.com/holoviz/holoviews/pull/5857))

Documentation:

- Set `autohide` example to `True` as in the hook ([#5832](https://github.com/holoviz/holoviews/pull/5832))

Maintenance:

- Add [OpenCollective](https://opencollective.com/holoviz) sponsor link on the repo page ([#5839](https://github.com/holoviz/holoviews/pull/5839))
- Update Ruff in pre-commit and report panel communication in `hv.show_versions` ([#5853](https://github.com/holoviz/holoviews/pull/5853))
- Cleanup imports ([#5846](https://github.com/holoviz/holoviews/pull/5846))


Version 1.17.0
==============
**July 24, 2023**

This release brings one of the most requested features - interactive twin-axis support! Another feature added in this release is the ability to easily set custom options on plot components with `backend_opts`, making it much easier to customize your plots. Datashaders `where` and `summary` aggregators are now supported, and `rasterize` now has a `selector` option, making it easy to get extra information about your rasterized data. Lastly, Bokeh figures with the same labels will synchronize muteness or visibility across different figures.

In addition, this release includes several enhancements and bug fixes.

Many thanks to the new contributors @alfredocarella and @RaulPL, as well as the returning contributors @ahuang11, @droumis, @jbednar, @jlstevens, @maximlt, @philippjfr, @TheoMathurin and @Hoxbro.

New features:

- Multi-yaxis support in the Bokeh backend ([#5621](https://github.com/holoviz/holoviews/pull/5621), [#5826](https://github.com/holoviz/holoviews/pull/5826), [#5827](https://github.com/holoviz/holoviews/pull/5827))
- Allow modifying the underlying Bokeh or Matplotlib figure, axes etc. using `backend_opts` ([#4463](https://github.com/holoviz/holoviews/pull/4463))
- Support Datashaders `where` and `summary` aggregators and add `selector` option to `rasterize` enabling instant hover inspection of value dimensions ([#5805](https://github.com/holoviz/holoviews/pull/5805))
- Synchronize muteness or visibility across Bokeh figures to support linked legends ([#5763](https://github.com/holoviz/holoviews/pull/5763))

Enhancements:

- Add option for initial ranges to RangeToolLink ([#5800](https://github.com/holoviz/holoviews/pull/5800))
- Allow resample's `pixel_ratio` to go below 1 ([#5813](https://github.com/holoviz/holoviews/pull/5813), [#5817](https://github.com/holoviz/holoviews/pull/5817))
Add the ability for `VectorField` to instantiate from UV coordinates ([#5797](https://github.com/holoviz/holoviews/pull/5797))
- Handle the `nodata` option for rasterized RGB image ([#5774](https://github.com/holoviz/holoviews/pull/5774))

Bug fixes:

- Fix bins option in the autompg_histogram demo ([#5750](https://github.com/holoviz/holoviews/pull/5750))
- Fix 0pt bug in safari ([#5755](https://github.com/holoviz/holoviews/pull/5755))
- Disable pan if `active_tools=[]` ([#5754](https://github.com/holoviz/holoviews/pull/5754))
- Better handling of inputs to `rasterize.instance()` ([#5767](https://github.com/holoviz/holoviews/pull/5767), [#5811](https://github.com/holoviz/holoviews/pull/5811))
- Fix class variable being used as instance `vdims` in `hv.RGB` ([#5773](https://github.com/holoviz/holoviews/pull/5773), [#5775](https://github.com/holoviz/holoviews/pull/5775))
- Improve notebook detection in VSCode and Google Colab ([#5792](https://github.com/holoviz/holoviews/pull/5792))
- Don't warn when running `rasterize` in VSCode ([#5780](https://github.com/holoviz/holoviews/pull/5780))
- Add `__init__` to `hv.Output` to not overwrite its parent signature ([#5799](https://github.com/holoviz/holoviews/pull/5799))
- Fix `XArrayInterface` crashing when input is an empty array ([#5809](https://github.com/holoviz/holoviews/pull/5809))
- Avoid setting `batched` before the class is initialized ([#5814](https://github.com/holoviz/holoviews/pull/5814))
- Fix aspect handling when plot size is still unknown ([#5808](https://github.com/holoviz/holoviews/pull/5808))
- Update callbacks to use Bokeh's `quad` instead of `quad` ([#5760](https://github.com/holoviz/holoviews/pull/5760))
- Update `hv.Image`/`hv.RGB` `invert_{x,y}axis` to work with Bokeh 3 ([#5796](https://github.com/holoviz/holoviews/pull/5796))
- `strip_magics` should also strip IPython line magic ([#5794](https://github.com/holoviz/holoviews/pull/5794))
- Fix `HoloMap.collapse` for `(Nd)Overlay` ([#5825](https://github.com/holoviz/holoviews/pull/5825))

Compatibility:

- Implement HEP1 - Drop support for Python 3.7 ([#5695](https://github.com/holoviz/holoviews/pull/5695))
- Replace deprecated `np.product` with `np.prod` ([#5787](https://github.com/holoviz/holoviews/pull/5787))
- Update `FileArchive` repr for Param 2.0 ([#5791](https://github.com/holoviz/holoviews/pull/5791))
- Deprecate functionality ([#5776](https://github.com/holoviz/holoviews/pull/5776))

Documentation:

- Fix typo in Getting Started section text ([#5759](https://github.com/holoviz/holoviews/pull/5759))
- Add sep keyword to `pd.read_csv` in documentation page ([#5798](https://github.com/holoviz/holoviews/pull/5798))

Maintenance:

- General maintenance ([#5758](https://github.com/holoviz/holoviews/pull/5758), [#5783](https://github.com/holoviz/holoviews/pull/5783), [#5802](https://github.com/holoviz/holoviews/pull/5802), [#5804](https://github.com/holoviz/holoviews/pull/5804), [#5806](https://github.com/holoviz/holoviews/pull/5806), [#5819](https://github.com/holoviz/holoviews/pull/5819))
- Correctly check the version for deprecation ([#5772](https://github.com/holoviz/holoviews/pull/5772))
- Update pre-commit and lint ([#5747](https://github.com/holoviz/holoviews/pull/5747), [#5768](https://github.com/holoviz/holoviews/pull/5768), [#5777](https://github.com/holoviz/holoviews/pull/5777))
- Setup infrastructure for UI tests and add first linked streams tests ([#5764](https://github.com/holoviz/holoviews/pull/5764))


Version 1.16.2
==============
**June 8, 2023**

This release includes a breaking change as notebooks will no longer be inlining as default. This change will reduce the size of the notebook files and, more importantly, address an upstream issue in Jupyter where running `hv.extension` would give an error when used in a notebook.

Critical bug fixes and compatibility:

- Correctly set `inline=False` when running `hv.extension` in a Jupyter Notebook ([#5748](https://github.com/holoviz/holoviews/pull/5748))
- Adding more Param 2 support ([#5717](https://github.com/holoviz/holoviews/pull/5717))

Enhancements:

- Speed up import time of Holoviews ([#5719](https://github.com/holoviz/holoviews/pull/5719))

Maintenance:

- Fix failing tests ([#5742](https://github.com/holoviz/holoviews/pull/5742))
- Misc. mainteance ([#5717](https://github.com/holoviz/holoviews/pull/5717))


Version 1.16.1
==============
**June 2, 2023**

This release contains a small number of important bug fixes and enhancements. Many thanks to @philippjfr and @Hoxbro.

This release includes a breaking change as notebooks will no longer be inlining as default. This change will reduce the size of the notebook files and, more importantly, address an upstream issue in Jupyter where running `hv.extension` would give an error when used in a notebook.


Critical bug fixes and compatibility:

- Add `enable_mathjax` and set it and inline to `False` ([#5729](https://github.com/holoviz/holoviews/pull/5729))
- Update to support Datashader on Python 3.11 ([#5720](https://github.com/holoviz/holoviews/pull/5720))

Enhancements:

- Add `show_versions` helper function ([#5725](https://github.com/holoviz/holoviews/pull/5725))
- Ignore known model warning in VS Code ([#5734](https://github.com/holoviz/holoviews/pull/5734))
- Add workaround for plots where the objects in a figure would overlap ([#5733](https://github.com/holoviz/holoviews/pull/5733))


Documentation:

- Upgrade to latest `nbsite` and `pydata-sphinx-theme` ([#5724](https://github.com/holoviz/holoviews/pull/5724), [#5735](https://github.com/holoviz/holoviews/pull/5735))


Version 1.16.0
==============
**May 9, 2023**

This release includes many new features, improvements, and bug fixes. Among the highlights are support for Bokeh 3.1 and Panel 1.0, both of which come with a wide range of new features and enhancements. Time series support has also been improved, with auto-ranging along one axis, a new downsample algorithm, and having WebGL enabled by default. In addition, the release includes various other enhancements and bug fixes.

We would like to thank the many users who contributed to this release by filing bug reports, providing new features, and bug fixes. We want to give a special shout-out to existing contributors @ianthomas23, @jlstevens, @jordansamuels, @maximlt, @philippjfr, @TheoMathurin, @Yura52, and @Hoxbro, as well as new contributors @GeoVizNow, @JRRudy1, @keewis, @michaelaye, and @wendrul.

This minor version will be the last to support Python 3.7. The next minor version will require Python 3.8 or higher.
In the next release, `holoviews.annotate` will start giving a deprecation warning about its future move to the new package [HoloNote](https://github.com/holoviz/holonote).


New features:

- Support for Bokeh 3.1 and Panel 1.0 ([#5388](https://github.com/holoviz/holoviews/pull/5388), [#5620](https://github.com/holoviz/holoviews/pull/5620), [#5640](https://github.com/holoviz/holoviews/pull/5640), [#5679](https://github.com/holoviz/holoviews/pull/5679), [#5683](https://github.com/holoviz/holoviews/pull/5683), [#5692](https://github.com/holoviz/holoviews/pull/5692), [#5703](https://github.com/holoviz/holoviews/pull/5703))
- Implement auto-ranging support along one axis ([#5554](https://github.com/holoviz/holoviews/pull/5554), [#5609](https://github.com/holoviz/holoviews/pull/5609))
- Add Largest Triangle Three Buckets (LTTB) as a downsample algorithm ([#5552](https://github.com/holoviz/holoviews/pull/5552))
- Enable WebGL by default ([#5708](https://github.com/holoviz/holoviews/pull/5708))

Enhancements:

- Improve `legend_cols` support for Bokeh 3 ([#5669](https://github.com/holoviz/holoviews/pull/5669))
- Add convenience `getter` interface to `opts` ([#5606](https://github.com/holoviz/holoviews/pull/5606))
- Ensure `.stack` works on areas with different `vdims` ([#5693](https://github.com/holoviz/holoviews/pull/5693))
- Add `muted` support to `PointPlot` like `hv.Scatter` ([#5705](https://github.com/holoviz/holoviews/pull/5705))
- Automatic detect `comms` without calling `pn.extension()` ([#5645](https://github.com/holoviz/holoviews/pull/5645))
- Add support for extra `Hovertool` variables in a Bokeh's `quadmesh` with 2D coordinates (with tests) ([#5638](https://github.com/holoviz/holoviews/pull/5638))
- Change `hv.Rectangles` to internally use Bokeh `Quad` and not `Rect` to support logarithmic axis in WebGL ([#5664](https://github.com/holoviz/holoviews/pull/5664), [#5702](https://github.com/holoviz/holoviews/pull/5702))


Bug fixes:

- Ensure `spatial_select` in non-zero indexed DataFrame is applied right ([#5625](https://github.com/holoviz/holoviews/pull/5625))
- Fix error handling for plotting class lookups on empty `DynamicMap`/`HoloMap` ([#5604](https://github.com/holoviz/holoviews/pull/5604))
- Fix `active_tools` to only be set for enabled tools ([#5616](https://github.com/holoviz/holoviews/pull/5616))
- Fix legend display when using categorical `datashade` on GPU ([#5631](https://github.com/holoviz/holoviews/pull/5631))
- Adding `GeoDataFrame` to `DataConversion` ([#5325](https://github.com/holoviz/holoviews/pull/5325))
- Don't emit warnings when the toolbar is disabled ([#5691](https://github.com/holoviz/holoviews/pull/5691))
- Don't try to find the closest match if the input is empty ([#5700](https://github.com/holoviz/holoviews/pull/5700))
- Only use `long_name` if it is a string ([#5646](https://github.com/holoviz/holoviews/pull/5646))
- Use Matplotlib's public API to list the colormaps ([#5598](https://github.com/holoviz/holoviews/pull/5598))

Compatibility:

- Add Param 2.0 support ([#5667](https://github.com/holoviz/holoviews/pull/5667), [#5641](https://github.com/holoviz/holoviews/pull/5641), [#5680](https://github.com/holoviz/holoviews/pull/5680), [#5704](https://github.com/holoviz/holoviews/pull/5704))
- Add Pandas 2.0 support ([#5662](https://github.com/holoviz/holoviews/pull/5662))
- Update `types.functiontype` to `collections.abc.callable` ([#5636](https://github.com/holoviz/holoviews/pull/5636))
- Improve error message for categorical data when used with `datashade` ([#5643](https://github.com/holoviz/holoviews/pull/5643))
- Don't disable Jedi completion by default ([#5701](https://github.com/holoviz/holoviews/pull/5701))


Documentation:

- Fix an incorrect number stated regarding available axis types ([#5623](https://github.com/holoviz/holoviews/pull/5623))
- Fix `BoundsY` example ([#5629](https://github.com/holoviz/holoviews/pull/5629))
- Fix formatting on FAQ ([#5630](https://github.com/holoviz/holoviews/pull/5630))
- Fix anchor links ([#5677](https://github.com/holoviz/holoviews/pull/5677))


Maintenance:

- Use `clean-notebook`, `codespell`, and `ruff` in `pre-commit` ([#5594](https://github.com/holoviz/holoviews/pull/5594), [#5627](https://github.com/holoviz/holoviews/pull/5627), [#5653](https://github.com/holoviz/holoviews/pull/5653))
- General maintenance ([#5607](https://github.com/holoviz/holoviews/pull/5607), [#5611](https://github.com/holoviz/holoviews/pull/5611), [#5612](https://github.com/holoviz/holoviews/pull/5612), [#5649](https://github.com/holoviz/holoviews/pull/5649))


Known issues:
- `BoxEditTool` is not yet supported with the new internals of `hv.Rectangle`.


Removals:

- Raise `DataError` for non-string column names in DataFrame ([#5654](https://github.com/holoviz/holoviews/pull/5654))
- Remove deprecated tile sources ([#5654](https://github.com/holoviz/holoviews/pull/5654))
- Removed support for `apply_groups` for old option groups signature ([#5497](https://github.com/holoviz/holoviews/pull/5497))


Version 1.15.4
==============
**January 16, 2023**

This release contains a small number of enhancements and important bug fixes.
Many thanks to our new contributors @mmorys, @jj-github-jj, and @sandhujasmine, but also our returning contributors @droumis, @jlstevens, @MarcSkovMadsen, @maximlt, @philippjfr, @stanwest, and @Hoxbro.


Enhancements:

- Make lasso select mask values using a Dask-compatible method ([#5568](https://github.com/holoviz/holoviews/pull/5568))
- Make plotly legend group unique ([#5570](https://github.com/holoviz/holoviews/pull/5570))
- Set pan and wheel_zoom as the default Bokeh active tools ([#5480](https://github.com/holoviz/holoviews/pull/5480))
- Enable rendering colorbars on bokeh `GraphPlot`s ([#5585](https://github.com/holoviz/holoviews/pull/5585))
- Add Plotly `Scatter3d` documentation and fix colorbar title ([#5418](https://github.com/holoviz/holoviews/pull/5418))

Bug fixes:

- Only trigger range-update once in callbacks ([#5558](https://github.com/holoviz/holoviews/pull/5558))
- Ensure dynamically created subplots can be updated ([#5555](https://github.com/holoviz/holoviews/pull/5555))
- Fix start of stack-level in deprecations ([#5569](https://github.com/holoviz/holoviews/pull/5569))
- When sorting colormap records, replace None with an empty string ([#5539](https://github.com/holoviz/holoviews/pull/5539))
- Fix annotator in Geoviews by adding deserialization of non-deserialized base64 data ([#5587](https://github.com/holoviz/holoviews/pull/5587))
- Fix `hv.Empty` not working in `AdjointLayout` plot ([#5584](https://github.com/holoviz/holoviews/pull/5584))
- Check for categorical data to histogram ([#5540](https://github.com/holoviz/holoviews/pull/5540))
- Fix `clim_percentile` ([#5495](https://github.com/holoviz/holoviews/pull/5495))

Compatibility:

- Compatibility with Shapely 2.0 ([#5561](https://github.com/holoviz/holoviews/pull/5561))
- Compatibility with Numpy 1.24 ([#5581](https://github.com/holoviz/holoviews/pull/5581))
- Compatibility with Ibis 4.0 ([#5588](https://github.com/holoviz/holoviews/pull/5588))

Documentation:

- Installation instructions update ([#5562](https://github.com/holoviz/holoviews/pull/5562))
- Use OSM for reference tile source in notebook documentation ([#5536](https://github.com/holoviz/holoviews/pull/5536))
- Enhance Tiles example notebook ([#5563](https://github.com/holoviz/holoviews/pull/5563))

Maintenance:

- Various fixes and general maintenance of the CI ([#5384](https://github.com/holoviz/holoviews/pull/5384), [#5573](https://github.com/holoviz/holoviews/pull/5573), [#5576](https://github.com/holoviz/holoviews/pull/5576), [#5582](https://github.com/holoviz/holoviews/pull/5582))
- Updated codebase to modern Python conventions ([#5509](https://github.com/holoviz/holoviews/pull/5509), [#5577](https://github.com/holoviz/holoviews/pull/5577))
- Renamed `master` branch to `main` ([#5579](https://github.com/holoviz/holoviews/pull/5579))
- Update binder link and dependency pinning ([#5583](https://github.com/holoviz/holoviews/pull/5583))
- Update copyright to only contain start year ([#5580](https://github.com/holoviz/holoviews/pull/5580))


Version 1.15.3
==============
**December 6, 2022**

This release contains a small number of important bug fixes and
adds support for Python 3.11. Many thanks to our maintainers
@Hoxbro, @maximlt and @jlstevens.

Bug Fixes:

  - Fix for empty opts warning and incorrect clearing semantics
    ([#5496](https://github.com/holoviz/holoviews/pull/5496))
  - Fix potential race condition in the Options system
    ([#5535](https://github.com/holoviz/holoviews/pull/5535))

Enhancements:

  - Add support to Python 3.11
    ([#5513](https://github.com/holoviz/holoviews/pull/5513))
  - Cleanup the top `__init__` module
    ([#5516](https://github.com/holoviz/holoviews/pull/5516))

Documentation:

  - Fixes to release notes and CHANGELOG
    ([#5506](https://github.com/holoviz/holoviews/pull/5506))

Version 1.15.2
==============
**November 3, 2022**

This release contains a small number of important bug fixes. Many thanks
to @stanwest for his contribution and thank you to our maintainers
@Hoxbro, @maximlt, @jlstevens, @jbednar, and @philippjfr.

Bug fixes:

- Fix support for jupyterlite
  ([#5502](https://github.com/holoviz/holoviews/pull/5502))
- Improve error message for `hv.opts` without a plotting backend
  ([#5494](https://github.com/holoviz/holoviews/pull/5494))
- Fix warnings exposed in CI logs
  ([#5470](https://github.com/holoviz/holoviews/pull/5470))
- Thanks to @maximlt for various CI fixes
  ([#5484](https://github.com/holoviz/holoviews/pull/5484),
  [#5498](https://github.com/holoviz/holoviews/pull/5498),
  [#5485](https://github.com/holoviz/holoviews/pull/5485))

Enhancement:

- Allow Dimension objects to accept a dictionary specification
  ([#5333](https://github.com/holoviz/holoviews/pull/5333))
- Refactor to remove iterrows for loop from `connect_edges_pd`
  ([#5473](https://github.com/holoviz/holoviews/pull/5473))

Deprecations:

Promoted `DeprecationWarning` to `FutureWarning` when using pandas
`DataFrame`s with non-string column names. This will not change any
functionality but will start warning users about functionality that will
be deprecated in future.

- Upgrade warning for invalid dataframe column names
  ([#5472](https://github.com/holoviz/holoviews/pull/5472))

Version 1.15.1
==============
**October 4, 2022**

This release contains a small number of important bug fixes. Many thanks
to all our new contributors @MarcSkovMadsen, @j-svensmark, @ceball,
@droumis, @ddrinka, @Jhsmit and @stanwest as well as a special thanks to
@Hoxbro for his many bug fixes. An additional thank you goes out to
@maximlt, @philippjfr, @jbednar and @jlstevens.

Enhancements:

- Sort output of `decimate` operation so that it can be used with connected Elements
  (Curve, Area, etc.) ([#5452](https://github.com/holoviz/holoviews/pull/5452))
- Ensure HoloViews is importable from a pyodide webworker
  ([#5410](https://github.com/holoviz/holoviews/pull/5410))
- Add support for stepwise Area plots
  ([#5390](https://github.com/holoviz/holoviews/pull/5390))
- Better error message for `hv.Cycle` when incompatible backend activated
  ([#5379](https://github.com/holoviz/holoviews/pull/5379))
- Improvements to VSCode notebook support
  ([#5398](https://github.com/holoviz/holoviews/pull/5398))
- Protect matplotlib tests from global styles
  ([#5311](https://github.com/holoviz/holoviews/pull/5311))
- Faster hashing for arrays and pandas objects
  ([#5455](https://github.com/holoviz/holoviews/pull/5455))
- Add pre-commit hooks to CI actions and fixes to pytest configuration
  ([#5385](https://github.com/holoviz/holoviews/pull/5385),
   [#5440](https://github.com/holoviz/holoviews/pull/5440))

Bug Fixes:

- Allow import of numpy 1.12
  ([#5367](https://github.com/holoviz/holoviews/pull/5367))
- Fixes handling of iterables in Overlays
  ([#5320](https://github.com/holoviz/holoviews/pull/5320))
- Always return a string when using `hv.Dimension.pprint_value`
  ([#5383](https://github.com/holoviz/holoviews/pull/5383))
- Support widgets in slices for `loc` and `iloc`
  ([#5352](https://github.com/holoviz/holoviews/pull/5352))
- Take account of labeled dimension in Bokeh plotting classes
  ([#5404](https://github.com/holoviz/holoviews/pull/5404))
- Fix handling of pandas `Period` ranges
  ([#5393](https://github.com/holoviz/holoviews/pull/5393))
- Fixed declaration of `Scatter` to `Selection1DExpr`
  ([#5413](https://github.com/holoviz/holoviews/pull/5413))
- Ensure rangesupdate event fires on all plots with linked axes
  ([#5465](https://github.com/holoviz/holoviews/pull/5465))
- Fixed fallback to shapely spatial select
  ([#5468](https://github.com/holoviz/holoviews/pull/5468))
- Many thanks to @Hoxbro for many miscellaneous plotting fixes,
  including fixes to plotting of `BoxWhisker`, `VectorField` elements
  ([#5397](https://github.com/holoviz/holoviews/pull/5397),
   [#5450](https://github.com/holoviz/holoviews/pull/5450),
   [#5400](https://github.com/holoviz/holoviews/pull/5400),
   [#5409](https://github.com/holoviz/holoviews/pull/5409),
   [#5460](https://github.com/holoviz/holoviews/pull/5460)))
- Fixes to documentation building GitHub Action
  ([#5320](https://github.com/holoviz/holoviews/pull/5456),
  ([#5320](https://github.com/holoviz/holoviews/pull/5467)))

Documentation:

- Introduced module documentation
  ([#5362](https://github.com/holoviz/holoviews/pull/5362))
- Remove Python 2 references from README
  ([#5365](https://github.com/holoviz/holoviews/pull/5365))
- Update call to panel add_periodic_callback in Bokeh gallery example
  ([#5436](https://github.com/holoviz/holoviews/pull/5436))
- Added reference to example in `RangeToolLink`
  ([#5435](https://github.com/holoviz/holoviews/pull/5435))

API:

In future, HoloViews will not allow non-string values for pandas
DataFrame column names. This deprecation cycle starts by
issuing a `DeprecationWarning` that should not be visible to users.

- Issue DeprecationWarning for invalid DataFrame column types
  ([#5457](https://github.com/holoviz/holoviews/pull/5457))

Version 1.15.0
==============
**July 6, 2022**

This is a major release with a large number of new features and bug
fixes, as well as updates to Python and Panel compatibility.

Many thanks to the numerous users who filed bug reports, tested
development versions, and contributed a number of new features and bug
fixes, including special thanks to @ablythed @ahuang11
@douglas-raillard-arm @FloLangenfeld @HoxBro @ianthomas23 @jenssss
@pepijndevos @peterroelants @stas-sl @Yura52 for their contributions. In
addition, thanks to the maintainers @jbednar, @maximlt, @jlstevens and
@philippjfr for contributing to this release.

Compatibility:

- Python 2 support has finally been dropped with 1.14.9 as the last
  release supporting Python 2
- HoloViews now requires panel >0.13.1
  ([#4329](https://github.com/holoviz/holoviews/pull/4329))
- Colormaps for the output of the datashade operation have changed to
  address
  [holoviz/datashader#357](https://github.com/holoviz/datashader/issues/357);
  see `rescale_discrete_levels` below. To revert to the old colorbar
  behavior, set `ColorbarPlot.rescale_discrete_levels = False` in the
  `bokeh` or `mpl` plotting modules as appropriate.
- Updated Sankey algorithm means that some users may need to update the
  `node_padding` parameter for plots generated with earlier releases.

Major features:

After a long period of hotfix releases for the 1.14.9 series, many new
features on the main branch have been released. Features relating to
datashader support, linked selection and improvements to the Bokeh
plotting backend are called out in their own sections.

- Support constructor interface from a spatialpandas GeometryArray
  ([#5281](https://github.com/holoviz/holoviews/pull/5281))
- Allow plotting anonymous pandas.Series
  ([#5015](https://github.com/holoviz/holoviews/pull/5015))
- Add support for rendering in pyodide/pyscript
  ([#5338](https://github.com/holoviz/holoviews/pull/5338),
  [#5321](https://github.com/holoviz/holoviews/pull/5321),
  [#5275](https://github.com/holoviz/holoviews/pull/5275))

Datashader features:

The following new features have been added to the datashader support in
HoloViews, mainly focused on Datashader's new support for antialiasing
lines as well as the new `rescale_discrete_levels` colormapping option.


- Add automatic categorical legend for datashaded plots
  ([#4806](https://github.com/holoviz/holoviews/pull/4806))
- Implement `line_width` support when rasterizing spatialpandas paths
  ([#5280](https://github.com/holoviz/holoviews/pull/5280))
- Expose `rescale_discrete_levels` in the Bokeh backend
  ([#5312](https://github.com/holoviz/holoviews/pull/5312))
- Set `rescale_discrete_levels=True` by default
  ([#5268](https://github.com/holoviz/holoviews/pull/5268))

New linked selection features:

- Implement `linked_selection.filter` method
  ([#4999](https://github.com/holoviz/holoviews/pull/4999))
- Allow passing custom `selection_expr` to linked selections filter
  ([#5012](https://github.com/holoviz/holoviews/pull/5012))
- Fix `AdjointLayout` in `link_selections`
  ([#5030](https://github.com/holoviz/holoviews/pull/5030))

New features for the Bokeh plotting backend:

- Add `legend_labels` option to allow overriding legend labels
  ([#5342](https://github.com/holoviz/holoviews/pull/5342))
- Updated sankey algorithm to `d3-sankey-v0.12.3`
 ([#4707](https://github.com/holoviz/holoviews/pull/4707))

Other enhancements:

- Optimize and clean up options system
 ([#4954](https://github.com/holoviz/holoviews/pull/4954))
- Optimize lasso selection by applying box-select first
 ([#5061](https://github.com/holoviz/holoviews/pull/5061))
 https://github.com/holoviz/holoviews/pull/5061
- Support ibis-framework version 3
 ([#5292](https://github.com/holoviz/holoviews/pull/5292))
- Add `OpenTopoMap` as a tile source
 ([#5052](https://github.com/holoviz/holoviews/pull/5052))
- Show all histograms of an `Overlay`
([#5031](https://github.com/holoviz/holoviews/pull/5031))

Bug fixes:

- Fix batch watching and linking of parameters in Params stream
 ([#4960](https://github.com/holoviz/holoviews/pull/4960),
  [#4956](https://github.com/holoviz/holoviews/pull/4956))
- Ensure `Plot.refresh` is dispatched immediately if possible
 ([#5348](https://github.com/holoviz/holoviews/pull/5348))
- Fix datashader empty overlay aggregation
 ([#5334](https://github.com/holoviz/holoviews/pull/5334))
- Fixed missing handling of nodata for count aggregator with column
 ([#4951](https://github.com/holoviz/holoviews/pull/4951))
- Handle `pd.NA` as missing data in dtype=object column
 ([#5323](https://github.com/holoviz/holoviews/pull/5323))
- Forward `DynamicMap.hist` dimension parameter to histogram creation
 ([#5037](https://github.com/holoviz/holoviews/pull/5037))
- Remove numpy pin from examples
 ([#5285](https://github.com/holoviz/holoviews/pull/5285))
- Fix vmin/vmax deprecation on matplotlib HeatMapPlot
 ([#5300](https://github.com/holoviz/holoviews/pull/5300))
- Don't skip each renderer's `load_nb call` when multiple extension
 calls are made in a single cell
 ([#5302](https://github.com/holoviz/holoviews/pull/5302))
- Set plotly range correctly for log axis
 ([#5272](https://github.com/holoviz/holoviews/pull/5272))
- Sanitize uses of `contextlib.contextmanager`
 ([#5018](https://github.com/holoviz/holoviews/pull/5018))
- Ensure `overlay_aggregate` is not applied for anti-aliased lines
 ([#5266](https://github.com/holoviz/holoviews/pull/5266))
- Switch to using bokeh `rangesupdate` event for `Range` streams
 ([#5265](https://github.com/holoviz/holoviews/pull/5265))
- Fixes for bokeh `Callbacks`
 ([#5040](https://github.com/holoviz/holoviews/pull/5040))
- Fix for attribute error in matplotlib `CompositePlot`
 ([#4969](https://github.com/holoviz/holoviews/pull/4969))
- Silenced inappropriate deprecation warnings and updated deprecation
  settings in options system
 ([#5345](https://github.com/holoviz/holoviews/pull/5345),
  [#5346](https://github.com/holoviz/holoviews/pull/5346))

Documentation:

The following improvements to the documentation have been made:

- Fix `hv.help` when pattern is set
 ([#5330](https://github.com/holoviz/holoviews/pull/5330))
- Added release dates to changelog and releases
 ([#5027](https://github.com/holoviz/holoviews/pull/5027),
  [#5035](https://github.com/holoviz/holoviews/pull/5035))
- Removed unneeded list from dynamic map example
 ([#4953](https://github.com/holoviz/holoviews/pull/4953))
- Added FAQ about sharing only a single axis
 ([#5278](https://github.com/holoviz/holoviews/pull/5278))
- Miscellaneous fixes to Heatmap reference notebook and Continuous
  Coordinates user guide
 ([#5262](https://github.com/holoviz/holoviews/pull/5262))
- Added example of multiple RGB images as glyphs
 ([#5172](https://github.com/holoviz/holoviews/pull/5172))
- Trim trailing whitespaces
 ([#5019](https://github.com/holoviz/holoviews/pull/5019))
- Update outdated IOAM references
 ([#4985](https://github.com/holoviz/holoviews/pull/4985))

Testing infrastructure:

Many thanks to @maximlt for his work maintaining and fixing the testing
infrastructure across too many PRs to list here.

- Switch to pytest
 ([#4949](https://github.com/holoviz/holoviews/pull/4949))
- Test suite clean up and fix for the pip build
 ([#5326](https://github.com/holoviz/holoviews/pull/5326))
- Test updates following release of datashader 0.14.1
 ([#5344](https://github.com/holoviz/holoviews/pull/5344))


Version 1.14.9
==============
**May 6, 2022**

This release contains a small number of important bug fixes as well as
support for the newly added antialiasing option for line rendering in
datashader. Many thanks to @andriyot, @Hoxbro, @pepijndevos, @stas-sl,
@TheoMathurin, @maximlt, @jlstevens, @jbednar, and @philippjfr.

Enhancements:

- Improvements to extension loading, improving visual appearance in
  JupyterLab when no logo is used and a check to avoid loading
  unnecessary JavaScript.
  ([#5216](https://github.com/holoviz/holoviews/pull/5216),
   [#5249](https://github.com/holoviz/holoviews/pull/5249))
- Add support for setting antialiased line_width on datashader line
  aggregation as well as pixel_ratio setting
  ([#5264](https://github.com/holoviz/holoviews/pull/5264),
  [#5288](https://github.com/holoviz/holoviews/pull/5288))
- Added options to customize hover line_(width|join|cap|dash) properties
  ([#5211](https://github.com/holoviz/holoviews/pull/5211))
- Restored Python 2 compatibility that lapsed due to lack of CI testing
  since 1.14.3. This is expected to be the last release with Python 2 support.
  ([#5298](https://github.com/holoviz/holoviews/pull/5298))

Bug fixes:

- Fix to respect series order in stacked area plot
  ([#5236](https://github.com/holoviz/holoviews/pull/5236))
- Support buffer streams of unspecified length (#5247)
  ([#5247](https://github.com/holoviz/holoviews/pull/5247))
- Fixed log axis lower bound when data minimum is <= 0
  ([#5246](https://github.com/holoviz/holoviews/pull/5246))
- Declared GitHub project URL in setup.py
  ([#5227](https://github.com/holoviz/holoviews/pull/5227))
- Fixed streaming Psutil example application
  ([#5243](https://github.com/holoviz/holoviews/pull/5243))
- Respecting Renderer's center property for HoloViews pane
  ([#5197](https://github.com/holoviz/holoviews/pull/5197))

Documentation:

- Updated Large data guide to reflect changes in Datashader and
  antialiasing support
  ([#5267](https://github.com/holoviz/holoviews/pull/5267),
  [#5290](https://github.com/holoviz/holoviews/pull/5290))


Version 1.14.8
==============
**February 15, 2022**

This release contains a small number of important bug fixes as well as
fixes required for Python 3.9 and 3.10 support. Many thanks to @Hoxbro,
@maximlt, @jlstevens, @jbednar, and @philippjfr.

Bug fixes:

- Fixed xarray validation for aliased coordinate
  ([#5169](https://github.com/holoviz/holoviews/pull/5169))
- Fixed xaxis/yaxis options with Matplotlib
  ([#5200](https://github.com/holoviz/holoviews/pull/5200))
- Fixed nested widgets by handling list or tuple values in
  `resolve_dependent_value` utility
  ([#5184](https://github.com/holoviz/holoviews/pull/5184))
- Fixed issue handling multiple widgets without names
  ([#5185](https://github.com/holoviz/holoviews/pull/5185))
- Fix overlay of two-level categorical plots and HLine
  ([#5203](https://github.com/holoviz/holoviews/pull/5203))
- Added support for Ibis > 2.0
  ([#5204](https://github.com/holoviz/holoviews/pull/5204))
- Allow lower dimensional views on arbitrary dimensioned elements
  ([#5208](https://github.com/holoviz/holoviews/pull/5208))
- Fix escaping of HTML on Div element
  ([#5209](https://github.com/holoviz/holoviews/pull/5209))
- Miscellaneous fixes to unit tests, including cudf test fixes as well
  as addition of Python 3.9 and 3.10 to the test matrix
  ([#5166](https://github.com/holoviz/holoviews/pull/5166),
  [#5199](https://github.com/holoviz/holoviews/pull/5199),
  [#5201](https://github.com/holoviz/holoviews/pull/5201),
  [#5206](https://github.com/holoviz/holoviews/pull/5206))

Version 1.14.7
==============
**December 16, 2021**

This release contains a small number of important bug fixes. Many thanks
to @douglas-raillard-arm, @jenssss, @w31t1, @Hoxbro, @martinfleis, @maximlt,
@jlstevens, @jbednar, and @philippjfr.

Bug fixes:

- Support xyzservices.TileProvider as hv.Tiles input
  ([#5062](https://github.com/holoviz/holoviews/pull/5062))
- Allow reversed layout/overlay binary operators for `+` and `*` to be used with custom objects
  ([#5073](https://github.com/holoviz/holoviews/pull/5073))
- Fix internal numpy.round usage
  ([#5095](https://github.com/holoviz/holoviews/pull/5095))
- Remove dependency on recent Panel release by importing bokeh version from util module
  ([#5103](https://github.com/holoviz/holoviews/pull/5103))
- Add missing bounds for the cache_size Parameter
  ([#5105](https://github.com/holoviz/holoviews/pull/5105))
- Add current_key property to DynamicMap
  ([#5106](https://github.com/holoviz/holoviews/pull/5106))
- Pin freetype on Windows to avoid matplotlib error
  ([#5109](https://github.com/holoviz/holoviews/pull/5109))
- Handle the empty string as a group name
  ([#5131](https://github.com/holoviz/holoviews/pull/5131))
- Do not merge partially overlapping Stream callbacks
  ([#5133](https://github.com/holoviz/holoviews/pull/5133))
- Fix Violin matplotlib rendering with non-finite values
  ([#5135](https://github.com/holoviz/holoviews/pull/5135))
- Fix matplotlib colorbar labeling for dim expressions
  ([#5137](https://github.com/holoviz/holoviews/pull/5137))
- Fix datetime clipping on RangeXY stream
  ([#5138](https://github.com/holoviz/holoviews/pull/5138))
- Ensure FreehandDraw renders when styles are set
  ([#5139](https://github.com/holoviz/holoviews/pull/5139))
- Validate dimensionality of xarray interface data
  ([#5140](https://github.com/holoviz/holoviews/pull/5140))
- Preserve cols when overlaying on layout
  ([#5141](https://github.com/holoviz/holoviews/pull/5141))
- Fix Bars legend error when overlaid with annotation
  ([#5142](https://github.com/holoviz/holoviews/pull/5142))
- Fix plotly Bar plots containing NaNs
  ([#5143](https://github.com/holoviz/holoviews/pull/5143))
- Do not raise deprecated .opts warning for empty groups
  ([#5144](https://github.com/holoviz/holoviews/pull/5144))
- Handle unsigned integer dtype in datashader aggregate operation
  ([#5149](https://github.com/holoviz/holoviews/pull/5149))
- Delay projection comparison to optimize geoviews
  ([#5152](https://github.com/holoviz/holoviews/pull/5152))
- Utility to convert datetime64 to int64 and test suite maintenance
  ([#5157](https://github.com/holoviz/holoviews/pull/5157))
- Fix for Contours consistent of empty and nonempty paths
  ([#5162](https://github.com/holoviz/holoviews/pull/5162))
- Fixed docs:
    * Fix `fig_bounds` description in Plotting_with_Matplotlib.ipynb
      ([#4983](https://github.com/holoviz/holoviews/pull/4983))
    * Fix broken link in Gridded user guide
      ([#5098](https://github.com/holoviz/holoviews/pull/5098))
- Improved docs:
    * Switch to the Pydata Sphinx theme
      ([#5163](https://github.com/holoviz/holoviews/pull/5163))

Version 1.14.6
==============
**September 16, 2021**

This is a hotfix release with a number of important bug fixes. Most
importantly, this version supports the recent bokeh 2.4.0
release. Many thanks to @geronimos, @peterroelants,
@douglas-raillard-arm, @philippjfr and @jlstevens for contributing the
fixes in this release.

Bug fixes:

- Compatibility for bokeh 2.4 and fixes to processing of falsey
  properties and visible style property
  ([#5059](https://github.com/holoviz/holoviews/pull/5059),
  [#5063](https://github.com/holoviz/holoviews/pull/5063))
- Stricter validation of data.interface before calling subclass
  ([#5050](https://github.com/holoviz/holoviews/pull/5050))
- Fix to prevent options being ignored in some cases
  ([#5016](https://github.com/holoviz/holoviews/pull/5016))
- Improvements to linked selections including support for linked
  selection lasso for cudf and improved warnings
  ([#5044](https://github.com/holoviz/holoviews/pull/5044),
   [#5051](https://github.com/holoviz/holoviews/pull/5051))
- Respect apply_ranges at range computation level
  ([#5081](https://github.com/holoviz/holoviews/pull/5081))
- Keep ordering of kdim when stacking Areas
  ([#4971](https://github.com/holoviz/holoviews/pull/4971))
- Apply hover postprocessor on updates
  ([#5039](https://github.com/holoviz/holoviews/pull/5039))

Version 1.14.5
==============
**July 16, 2021**

This is a hotfix release with a number of important bug fixes. Most
importantly, this version supports for the recent pandas 1.3.0
release. Many thanks to @kgullikson88, @philippjfr and @jlstevens for
contributing the fixes in this release.

Bug fixes:

- Support for pandas>=1.3
  ([#5013](https://github.com/holoviz/holoviews/pull/5013))
- Various bug fixes relating to dim transforms including the use of
  parameters in slices and the use of getattribute
  ([#4993](https://github.com/holoviz/holoviews/pull/4993),
  [#5001](https://github.com/holoviz/holoviews/pull/5001),
  [#5005](https://github.com/holoviz/holoviews/pull/5005))


Version 1.14.4
==============
**May 18, 2021**

This release primarily focuses on a number of bug fixes. Many thanks to
@Hoxbro, @nitrocalcite, @brl0, @hyamanieu, @rafiyr, @jbednar, @jlstevens
and @philippjfr for contributing.

Enhancements:

- Re-enable `SaveTool` for plots with `Tiles`
  ([#4922](https://github.com/holoviz/holoviews/pull/4922))
- Enable dask `TriMesh` rasterization using datashader
  ([#4935](https://github.com/holoviz/holoviews/pull/4935))
- Use dataframe index for `TriMesh` node indices
  ([#4936](https://github.com/holoviz/holoviews/pull/4936))

Bug fixes:

- Fix hover for stacked `Bars`
  ([#4892](https://github.com/holoviz/holoviews/pull/4892))
- Check before dereferencing Bokeh colormappers
  ([#4902](https://github.com/holoviz/holoviews/pull/4902))
- Fix multiple parameterized inputs to `dim`
  ([#4903](https://github.com/holoviz/holoviews/pull/4903))
- Fix floating point error when generating bokeh Palettes
  ([#4911](https://github.com/holoviz/holoviews/pull/4911))
- Fix bug using dimensions with label on `Bars`
  ([#4929](https://github.com/holoviz/holoviews/pull/4929))
- Do not reverse colormaps with '_r' suffix a second time
  ([#4931](https://github.com/holoviz/holoviews/pull/4931))
- Fix remapping of `Params` stream parameter names
  ([#4932](https://github.com/holoviz/holoviews/pull/4932))
- Ensure `Area.stack` keeps labels
  ([#4937](https://github.com/holoviz/holoviews/pull/4937))

Documentation:

- Updated Dashboards user guide to show `pn.bind` first
  ([#4907](https://github.com/holoviz/holoviews/pull/4907))
- Updated docs to correctly declare Scatter kdims
  ([#4914](https://github.com/holoviz/holoviews/pull/4914))

Compatibility:

Unfortunately a number of tile sources are no longer publicly
available. Attempting to use these tile sources will now issue warnings
unless `hv.config.raise_deprecated_tilesource_exception` is set to
`True` in which case exceptions will be raised instead.

- The `Wikipedia` tile source is no longer available as it is no longer
  being served outside the wikimedia domain. As one of the most
  frequently used tile sources, HoloViews now issues a warning and
  switches to the OpenStreetMap (OSM) tile source instead.
- The `CartoMidnight` and `CartoEco` tile sources are no longer publicly
  available. Attempting to use these tile sources will result in a
  deprecation warning.

Version 1.14.3
==============
**April 8, 2021**

This release contains a small number of bug fixes, enhancements and
compatibility for the latest release of matplotlib. Many thanks to
@stonebig, @Hoxbro, @jlstevens, @jbednar and @philippjfr.

Enhancements:

- Allow applying linked selections to chained `DynamicMap`
  ([#4870](https://github.com/holoviz/holoviews/pull/4870))
- Issuing improved error message when `__radd__` called with an
  integer ([#4868](https://github.com/holoviz/holoviews/pull/4868))
- Implement `MultiInterface.assign`
  ([#4880](https://github.com/holoviz/holoviews/pull/4880))
- Handle tuple unit on xarray attribute
  ([#4881](https://github.com/holoviz/holoviews/pull/4881))
- Support selection masks and expressions on gridded data
  ([#4882](https://github.com/holoviz/holoviews/pull/4882))

Bug fixes:

- Handle empty renderers when merging `HoverTool.renderers`
  ([#4856](https://github.com/holoviz/holoviews/pull/4856))

Compatibility:

- Support matplotlib versions >=3.4
  ([#4878](https://github.com/holoviz/holoviews/pull/4878))

Version 1.14.2
==============
**March 2, 2021**

This release adds support for Bokeh 2.3, introduces a number of minor
enhancements, miscellaneous documentation improvements and a good number
of bug fixes.

Many thanks to the many contributors to this release, whether directly
by submitting PRs or by reporting issues and making
suggestions. Specifically, we would like to thank @philippjfr for the
Bokeh 2.3 compatibility updates, @kcpevey, @timgates42, and @scottstanie
for documentation improvements as well as @Hoxbro and @LunarLanding for
various bug fixes. In addition, thanks to the maintainers @jbednar,
@jlstevens and @philippjfr for contributing to this release.

Enhancements:

- Bokeh 2.3 compatibility
  ([#4805](https://github.com/holoviz/holoviews/pull/4805),
  [#4809](https://github.com/holoviz/holoviews/pull/4809))
- Supporting dictionary streams parameter in DynamicMaps and operations
  ([#4787](https://github.com/holoviz/holoviews/pull/4787),
   [#4818](https://github.com/holoviz/holoviews/pull/4818),
   [#4822](https://github.com/holoviz/holoviews/pull/4822))
- Support spatialpandas DaskGeoDataFrame
  ([#4792](https://github.com/holoviz/holoviews/pull/4792))
- Disable zoom on axis for geographic plots
  ([#4812](https://github.com/holoviz/holoviews/pull/4812)
- Add support for non-aligned data in Area stack classmethod
  ([#4836](https://github.com/holoviz/holoviews/pull/4836))
- Handle arrays and datetime ticks
  ([#4831](https://github.com/holoviz/holoviews/pull/4831))
- Support single-value numpy array as input to HLine and VLine
  ([#4798](https://github.com/holoviz/holoviews/pull/4798))

Bug fixes:

- Ensure link_inputs parameter on operations is passed to apply
  ([#4795](https://github.com/holoviz/holoviews/pull/4795))
- Fix for muted option on overlaid Bokeh plots
  ([#4830](https://github.com/holoviz/holoviews/pull/4830))
- Check for nested dim dependencies
  ([#4785](https://github.com/holoviz/holoviews/pull/4785))
- Fixed np.nanmax call when computing ranges
  ([#4847](https://github.com/holoviz/holoviews/pull/4847))
- Fix for Dimension pickling
  ([#4843](https://github.com/holoviz/holoviews/pull/4843))
- Fixes for dask backed elements in plotting
  ([#4813](https://github.com/holoviz/holoviews/pull/4813))
- Handle isfinite for NumPy and Pandas masked arrays
  ([#4817](https://github.com/holoviz/holoviews/pull/4817))
- Fix plotting Graph on top of Tiles/Annotation
  ([#4828](https://github.com/holoviz/holoviews/pull/4828))
- Miscellaneous fixes for the Bokeh plotting extension
  ([#4814](https://github.com/holoviz/holoviews/pull/4814),
  [#4839](https://github.com/holoviz/holoviews/pull/4839))
- Miscellaneous fixes for index based linked selections
  ([#4776](https://github.com/holoviz/holoviews/pull/4776))

Documentation:

- Expanded on Tap Stream example in Reference Gallery
  [#4782](https://github.com/holoviz/holoviews/pull/4782)
- Miscellaneous typo and broken link fixes
  ([#4783](https://github.com/holoviz/holoviews/pull/4783),
  [#4827](https://github.com/holoviz/holoviews/pull/4827),
  [#4844](https://github.com/holoviz/holoviews/pull/4844),
  [#4811](https://github.com/holoviz/holoviews/pull/4811))

Version 1.14.1
==============
**December 28, 2020**

This release contains a small number of bug fixes addressing
regressions. Many thanks to the contributors to this release including
@csachs, @GilShoshan94 and the maintainers @jlstevens, @jbednar and
@philippjfr.

Bug fixes:

- Fix issues with linked selections on tables
  ([#4758](https://github.com/holoviz/holoviews/pull/4758))
- Fix Heatmap alpha dimension transform
  ([#4757](https://github.com/holoviz/holoviews/pull/4757))
- Do not drop tools in linked selections
  ([#4756](https://github.com/holoviz/holoviews/pull/4756))
- Fixed access to possibly non-existent key
  ([#4742](https://github.com/holoviz/holoviews/pull/4742))

Documentation:

- Warn about disabled interactive features on website
  ([#4762](https://github.com/holoviz/holoviews/pull/4762))

Version 1.14.0
==============
**December 1, 2020**

This release brings a number of major features including a new
IbisInterface, new Plotly Dash support and greatly improved Plotly
support, and greatly improved interaction and integration with
Datashader. Many thanks to the many contributors to this release,
whether directly by submitting PRs or by reporting issues and making
suggestions. Specifically, we would like to thank @philippjfr,
@jonmmease, and @tonyfast for their work on the IbisInterface and
@jonmmease for improving Plotly support, as well as @kcpevey, @Hoxbro,
@marckassay, @mcepl, and @ceball for various other enhancements,
improvements to documentation and testing infrastructure.  In
addition, thanks to the maintainers @jbednar, @jlstevens and
@philippjfr for contributing to this release. This version includes a
large number of new features, enhancements, and bug fixes.

It is important to note that version 1.14 will be the last HoloViews
release supporting Python 2.

Major features:

- New Plotly Dash support
  ([#4605](https://github.com/holoviz/holoviews/pull/4605))
- New Plotly support for Tiles element
  ([#4686](https://github.com/holoviz/holoviews/pull/4686))
- New IbisInterface
  ([#4517](https://github.com/holoviz/holoviews/pull/4517))
- Greatly improved Datashader `rasterize()`
  ([#4567](https://github.com/holoviz/holoviews/pull/4567)).
  Previously, many of the features of Datashader were available only
  through `datashade`, which rendered data all the way to RGB pixels
  and thus prevented many client-side Bokeh features like hover,
  colorbars, dynamic colormaps, etc. `rasterize` now supports all
  these Bokeh features along with nearly all the Datashader features
  previously only available through `datashade`, including (now
  client-side) histogram equalization with `cnorm='eq_hist'` and easy
  control of transparency via a new `Dimension.nodata` parameter. See
  the [Large Data User
  Guide](https://holoviews.org/user_guide/Large_Data.html) for more
  information.

Enhancements:

- Implemented datashader aggregation of Rectangles
  ([#4701](https://github.com/holoviz/holoviews/pull/4701))
- New support for robust color limits (`clim_percentile`)
  ([#4712](https://github.com/holoviz/holoviews/pull/4712))
- Support for dynamic overlays in link_selections
  ([#4683](https://github.com/holoviz/holoviews/pull/4683))
- Allow clashing Param stream contents
  ([#4677](https://github.com/holoviz/holoviews/pull/4677))
- Ensured pandas does not convert times to UTC
  ([#4711](https://github.com/holoviz/holoviews/pull/4711))
- Removed all use of cyordereddict
  ([#4620](https://github.com/holoviz/holoviews/pull/4620))
- Testing infrastructure moved to GH Actions
  ([#4592](https://github.com/holoviz/holoviews/pull/4592))

Bug fixes:

- Ensure RangeXY returns x/y ranges in correct order (#4665)
  ([#4665](https://github.com/holoviz/holoviews/pull/4665))
- Fix datashader instability with Plotly by disabling padding for RGB elements
  ([#4705](https://github.com/holoviz/holoviews/pull/4705))
- Various Dask and cuDF histogram fixes
  ([#4691](https://github.com/holoviz/holoviews/pull/4691))
- Fix handling of custom matplotlib and bokeh colormaps
  ([#4693](https://github.com/holoviz/holoviews/pull/4693))
- Fix cuDF values implementation
  ([#4687](https://github.com/holoviz/holoviews/pull/4687))
- Fixed range calculation on HexTiles
  ([#4689](https://github.com/holoviz/holoviews/pull/4689))
- Use PIL for RGB.load_image
  ([#4639](https://github.com/holoviz/holoviews/pull/4639))

Documentation:

- Clarified data types accepted by Points
  ([#4430](https://github.com/holoviz/holoviews/pull/4430))
- Updated Introduction notebook
  ([#4682](https://github.com/holoviz/holoviews/pull/4682))
- Fixed releases urls
  ([#4672](https://github.com/holoviz/holoviews/pull/4672))

Compatibility:

- Warning when there are multiple kdims on Chart elements
  ([#4710](https://github.com/holoviz/holoviews/pull/4710))
- Set histogram `normed` option to False by default
  ([#4258](https://github.com/holoviz/holoviews/pull/4258))
- The default colormap in holoviews is now 'kbc_r' instead of
  'fire'; see issue
  [#3500](https://github.com/holoviz/holoviews/issues/3500) for details.
  This change was made mainly because the highest value of the fire colormap
  is white, which meant data was often not visible against a white
  background. To restore the old behavior you can set
  `hv.config.default_cmap='fire'`, which you can do via the extension e.g.
  `hv.extension('bokeh', config=dict(default_cmap='fire'))`. There is
  also `hv.config.default_gridded_cmap` which you can set to 'fire' if
  you wish to use the old colormap for the `Raster`, `Image` and
  `QuadMesh` element types. The default `HeatMap` colormap has also been
  set to 'kbc_r' for consistency and can be set back to the old value of
  'RdYlBu_r' via `hv.config.default_heatmap_cmap`.

Version 1.13.5
==============
**October 23, 2020**

This version contains numerous bug fixes and a number of enhancements.
Many thanks for contribution by @bryevdv, @jbednar, @jlstevens,
@jonmmease, @kcpevey and @philippjfr.

Enhancements:

- Refactor of link selections streams
  ([#4572](https://github.com/holoviz/holoviews/pull/4572))
- Add ability to listen to dataset linked_selection
  ([#4547](https://github.com/holoviz/holoviews/pull/4547))
- Added `selected` parameter to Bokeh PathPlot
  ([#4641](https://github.com/holoviz/holoviews/pull/4641))

Bug fixes:

- Improvements to iteration over Series in CuDF data backend
  ([#4624](https://github.com/holoviz/holoviews/pull/4624))
- Added .values_host calls needed for iteration in CuDF backend
  ([#4646](https://github.com/holoviz/holoviews/pull/4646))
- Fixed bug resetting ranges
  ([#4654](https://github.com/holoviz/holoviews/pull/4654))
- Fix bug matching elements to subplots in `DynamicMap` (#4649)
  ([#4649](https://github.com/holoviz/holoviews/pull/4649))
- Ensure consistent split `Violin` color assignment
  ([#4650](https://github.com/holoviz/holoviews/pull/4650))
- Ensure `PolyDrawCallback` always has vdim data
  ([#4644](https://github.com/holoviz/holoviews/pull/4644))
- Set default align in bokeh correctly
  ([#4637](https://github.com/holoviz/holoviews/pull/4637))
- Fixed deserialization of polygon/multi_line CDS data in bokeh backend
  ([#4631](https://github.com/holoviz/holoviews/pull/4631))

Documentation:

- Improved `Bars` reference example, demonstrating the dataframe constructor
  ([#4656](https://github.com/holoviz/holoviews/pull/4656))
- Various documentation fixes
  ([#4628](https://github.com/holoviz/holoviews/pull/4628))

Version 1.13.4
==============
**September 8, 2020**

This version fixes a large number of bugs particularly relating to
linked selections. Additionally it introduces some enhancements laying
the groundwork for future functionality. Many thanks for contribution
by @ruoyu0088, @hamogu, @Dr-Irv, @jonmmease, @justinbois, @ahuang11,
and the core maintainer @philippjfr.

Bug fixes:

- Fix the `.info` property to return the info
  ([#4513](https://github.com/holoviz/holoviews/pull/4513))
- Set `toolbar=True` the default in `save()`
  ([#4518](https://github.com/holoviz/holoviews/pull/4518))
- Fix bug when the default value is 0
  ([#4537](https://github.com/holoviz/holoviews/pull/4537))
- Ensure operations do not recursively accumulate pipelines
  ([#4544](https://github.com/holoviz/holoviews/pull/4544))
- Fixed whiskers for `BoxWhisker` so that they never point inwards
  ([#4548](https://github.com/holoviz/holoviews/pull/4548))
- Fix issues with boomeranging events when aspect is set
  ([#4569](https://github.com/holoviz/holoviews/pull/4569))
- Fix aspect if width/height has been constrained
  ([#4579](https://github.com/holoviz/holoviews/pull/4579))
- Fixed categorical handling in Geom plot types
  ([#4575](https://github.com/holoviz/holoviews/pull/4575))
- Do not attempt linking axes on annotations
  ([#4584](https://github.com/holoviz/holoviews/pull/4584))
- Reset `RangeXY` when `framewise` is set
  ([#4585](https://github.com/holoviz/holoviews/pull/4585))
- Add automatic collate for `Overlay` of `AdjointLayout`s
  ([#4586](https://github.com/holoviz/holoviews/pull/4586))
- Fixed color-ranging after box select on side histogram
  ([#4587](https://github.com/holoviz/holoviews/pull/4587))
- Use HTTPS throughout on homepage
  ([#4588](https://github.com/holoviz/holoviews/pull/4588))

Compatibility:

- Compatibility with bokeh 2.2 for CDSCallback
  ([#4568](https://github.com/holoviz/holoviews/pull/4568))
- Handle `rcParam` deprecations in matplotlib 3.3
  ([#4583](https://github.com/holoviz/holoviews/pull/4583))

Enhancements:

- Allow toggling the `selection_mode` on `link_selections` from the
  context menu in the bokeh toolbar
  ([#4604](https://github.com/holoviz/holoviews/pull/4604))
- Optimize options machinery
  ([#4545](https://github.com/holoviz/holoviews/pull/4545))
- Add new `Derived` stream class
  ([#4532](https://github.com/holoviz/holoviews/pull/4532))
- Set Panel state to busy during callbacks
  ([#4546](https://github.com/holoviz/holoviews/pull/4546))
- Support positional stream args in `DynamicMap` callback
  ([#4534](https://github.com/holoviz/holoviews/pull/4534))
- `legend_opts` implemented
  ([#4558](https://github.com/holoviz/holoviews/pull/4558))
- Add `History` stream
  ([#4554](https://github.com/holoviz/holoviews/pull/4554))
- Updated spreading operation to support aggregate arrays
  ([#4562](https://github.com/holoviz/holoviews/pull/4562))
- Add ability to supply `dim` transforms for all dimensions
  ([#4578](https://github.com/holoviz/holoviews/pull/4578))
- Add 'vline' and 'hline' Hover mode
  ([#4527](https://github.com/holoviz/holoviews/pull/4527))
- Allow rendering to pgf in matplotlib
  ([#4577](https://github.com/holoviz/holoviews/pull/4577))

Version 1.13.3
==============
**June 23, 2020**

This version introduces a number of enhancements of existing
functionality, particularly for features introduced in 1.13.0,
e.g. cuDF support and linked selections. In addition it introduces a
number of important bug fixes. Many thanks for contribution by
@kebowen730, @maximlt, @pretros1999, @alexbraditsas, @lelatbones,
@flothesof, @ruoyu0088, @cool-PR and the core maintainers @jbednar and
@philippjfr.

Enhancements:

- Expose `center` as an output rendering option
  ([#4365](https://github.com/holoviz/holoviews/pull/4365))
- Configurable throttling schemes for linked streams on the server
  ([#4372](https://github.com/holoviz/holoviews/pull/4372))
- Add support for lasso tool in linked selections
  ([#4362](https://github.com/holoviz/holoviews/pull/4362))
- Add support for NdOverlay in linked selections
  ([#4481](https://github.com/holoviz/holoviews/pull/4481))
- Add support for unwatching on `Params` stream
  ([#4417](https://github.com/holoviz/holoviews/pull/4417))
- Optimizations for the cuDF interface
  ([#4436](https://github.com/holoviz/holoviews/pull/4436))
- Add support for `by` aggregator in datashader operations
  ([#4438](https://github.com/holoviz/holoviews/pull/4438))
- Add support for cupy and dask histogram and box-whisker calculations
  ([#4447](https://github.com/holoviz/holoviews/pull/4447))
- Allow rendering HoloViews output as an ipywidget
  ([#4404](https://github.com/holoviz/holoviews/pull/4404))
- Allow `DynamicMap` callback to accept key dimension values as
  variable kwargs
  ([#4462](https://github.com/holoviz/holoviews/pull/4462))
- Delete toolbar by default when rendering bokeh plot to PNG
  ([#4422](https://github.com/holoviz/holoviews/pull/4422))
- Ensure `Bounds` and `Lasso` events only trigger on mouseup
  ([#4478](https://github.com/holoviz/holoviews/pull/4478))

Bug fixes:

- Eliminate circular references to allow immediate garbage collection
  ([#4368](https://github.com/holoviz/holoviews/pull/4368),
  [#4377](https://github.com/holoviz/holoviews/pull/4377))
- Allow bytes as categories
  ([#4392](https://github.com/holoviz/holoviews/pull/4392))
- Fix handling of zero as log colormapper lower bound
  ([#4383](https://github.com/holoviz/holoviews/pull/4383))
- Do not compute data ranges if Dimension.values is supplied
  ([#4416](https://github.com/holoviz/holoviews/pull/4416))
- Fix RangeXY updates when zooming on only one axis
  ([#4413](https://github.com/holoviz/holoviews/pull/4413))
- Ensure that ranges do not bounce when data_aspect is set
  ([#4431](https://github.com/holoviz/holoviews/pull/4431))
- Fix bug specifying a rotation for Box element
  ([#4460](https://github.com/holoviz/holoviews/pull/4460))
- Fix handling of datetimes in bokeh RectanglesPlot
  ([#4461](https://github.com/holoviz/holoviews/pull/4461))
- Fix bug normalizing ranges across multiple plots when framewise=True
  ([#4450](https://github.com/holoviz/holoviews/pull/4450))
- Fix bug coloring adjoined histograms
  ([#4458](https://github.com/holoviz/holoviews/pull/4458))
- Fix issues with ranges bouncing when PlotSize stream is attached
  ([#4480](https://github.com/holoviz/holoviews/pull/4480))
- Fix bug with hv.extension(inline=False)
  ([#4491](https://github.com/holoviz/holoviews/pull/4491))
- Handle missing categories on split Violin plot
  ([#4482](https://github.com/holoviz/holoviews/pull/4482))

Version 1.13.2
==============
**April 2, 2020**

This is a minor patch release fixing a number of regressions
introduced as part of the 1.13.x releases. Many thanks to the
contributors including @eddienko, @poplarShift, @wuyuani135, @maximlt
and the maintainer @philippjfr.

Enhancements:

- Add PressUp and PanEnd streams
  ([#4334](https://github.com/holoviz/holoviews/pull/4334))

Bug fixes:

- Fix regression in single node Sankey computation
  ([#4337](https://github.com/holoviz/holoviews/pull/4337))
- Fix color and alpha option on bokeh Arrow plot
  ([#4338](https://github.com/holoviz/holoviews/pull/4338))
- Fix undefined JS variables in various bokeh links
  ([#4341](https://github.com/holoviz/holoviews/pull/4341))
- Fix matplotlib >=3.2.1 deprecation warnings
  ([#4335](https://github.com/holoviz/holoviews/pull/4335))
- Fix handling of document in server mode
  ([#4355](https://github.com/holoviz/holoviews/pull/4355))

Version 1.13.1
==============
**March 25, 2020**

This is a minor patch release to fix issues compatibility with the
about to be released Bokeh 2.0.1 release. Additionally this release
makes Pandas a hard dependency, which was already implicitly the case
in 1.13.0 but not declared. Lastly this release contains a small number
of enhancements and bug fixes.

Enhancements:

- Add option to set Plotly plots to responsive
  ([#4319](https://github.com/holoviz/holoviews/pull/4319))
- Unified datetime formatting in bokeh hover info
  ([#4318](https://github.com/holoviz/holoviews/pull/4318))
- Allow using dim expressions as accessors
  ([#4311](https://github.com/holoviz/holoviews/pull/4311))
- Add explicit `.df` and `.xr` namespaces to `dim` expressions to
  allow using dataframe and xarray APIs
  ([#4320](https://github.com/holoviz/holoviews/pull/4320))
- Allow defining clim which defines only upper or lower bound and not
  both ([#4314](https://github.com/holoviz/holoviews/pull/4314))
- Improved exceptions when selected plotting extension is not loaded
  ([#4325](https://github.com/holoviz/holoviews/pull/4325))

Bug fixes:

- Fix regression in Overlay.relabel that occurred in 1.12.3 resulting
  in relabeling of contained elements by default
  ([#4246](https://github.com/holoviz/holoviews/pull/4246))
- Fix bug when updating bokeh Arrow elements
  ([#4313](https://github.com/holoviz/holoviews/pull/4313))
- Fix bug where Layout/Overlay constructors would drop items
  ([#4313](https://github.com/holoviz/holoviews/pull/4323))

Compatibility:

- Fix compatibility with Bokeh 2.0.1
  ([#4308](https://github.com/holoviz/holoviews/pull/4308))

Documentation:

- Update API reference manual
  ([#4316](https://github.com/holoviz/holoviews/pull/4316))

Version 1.13.0
==============
**March 20, 2020**

This release is packed full of features and includes a general
refactoring of how HoloViews renders widgets now built on top of the
Panel library. Many thanks to the many contributors to this release
either directly by submitting PRs or by reporting issues and making
suggestions. Specifically we would like to thank @poplarShift,
@jonmease, @flothesof, @julioasotodv, @ltalirz, @DancingQuanta, @ahuang,
@kcpevey, @Jacob-Barkhak, @nluetts, @harmbuisman, @ceball, @mgsnuno,
@srp3003, @jsignell as well as the maintainers @jbednar, @jlstevens and
@philippjfr for contributing to this release.  This version includes the
addition of a large number of features, enhancements and bug fixes:

Major features:

- Add `link_selection` to make custom linked brushing simple
  ([#3951](https://github.com/holoviz/holoviews/pull/3951))
- `link_selection` builds on new support for much more powerful
  data-transform pipelines: new `Dataset.transform` method
  ([#237](https://github.com/holoviz/holoviews/pull/237),
  [#3932](https://github.com/holoviz/holoviews/pull/3932)), `dim`
  expressions in `Dataset.select`
  ([#3920](https://github.com/holoviz/holoviews/pull/3920)), arbitrary
  method calls on `dim` expressions
  ([#4080](https://github.com/holoviz/holoviews/pull/4080)), and
  `Dataset.pipeline` and `Dataset.dataset` properties to track
  provenance of data
- Add Annotators to allow easily drawing, editing, and annotating visual
  elements ([#1185](https://github.com/holoviz/holoviews/pull/1185))
- Completely replaced custom Javascript widgets with Panel-based widgets
  allowing for customizable layout
  ([#84](https://github.com/holoviz/holoviews/pull/84),
  [#805](https://github.com/holoviz/holoviews/pull/805))
- Add `HSpan`, `VSpan`, `Slope`, `Segments` and `Rectangles` elements
  ([#3510](https://github.com/holoviz/holoviews/pull/3510),
  [#3532](https://github.com/holoviz/holoviews/pull/3532),
  [#4000](https://github.com/holoviz/holoviews/pull/4000))
- Add support for cuDF GPU dataframes, cuPy backed xarrays, and GPU
  datashading ([#3982](https://github.com/holoviz/holoviews/pull/3982))

Other features

- Add spatialpandas support and redesigned geometry interfaces for
  consistent roundtripping
  ([#4120](https://github.com/holoviz/holoviews/pull/4120))
- Support GIF rendering with Bokeh and Plotly backends
  ([#2956](https://github.com/holoviz/holoviews/pull/2956),
  [#4017](https://github.com/holoviz/holoviews/pull/4017))
- Support for Plotly `Bars`, `Bounds`, `Box`, `Ellipse`, `HLine`,
  `Histogram`, `RGB`, `VLine` and `VSpan` plots
- Add `UniformNdMapping.collapse` to collapse nested datastructures
  ([#4250](https://github.com/holoviz/holoviews/pull/4250))
- Add `CurveEdit` and `SelectionXY` streams
  ([#4119](https://github.com/holoviz/holoviews/pull/4119),
  [#4167](https://github.com/holoviz/holoviews/pull/4167))
- Add `apply_when` helper to conditionally apply operations
  ([#4289](https://github.com/holoviz/holoviews/pull/4289))
- Display Javascript callback errors in the notebook
  ([#4119](https://github.com/holoviz/holoviews/pull/4119))
- Add support for linked streams in Plotly backend to enable rich
  interactivity
  ([#3880](https://github.com/holoviz/holoviews/pull/3880),
  [#3912](https://github.com/holoviz/holoviews/pull/3912))

Enhancements:

- Support for packed values dimensions, e.g. 3D `RGB`/`HSV` arrays
  ([#550](https://github.com/holoviz/holoviews/pull/550),
  [#3983](https://github.com/holoviz/holoviews/pull/3983))
- Allow selecting/slicing datetimes with strings
  ([#886](https://github.com/holoviz/holoviews/pull/886))
- Support for datashading `Area`, `Spikes`, `Segments` and `Polygons`
  ([#4120](https://github.com/holoviz/holoviews/pull/4120))
- `HeatMap` now supports mixed categorical/numeric axes
  ([#2128](https://github.com/holoviz/holoviews/pull/2128))
- Use `__signature__` to generate .opts tab completions
  ([#4193](https://github.com/holoviz/holoviews/pull/4193))
- Allow passing element-specific keywords through `datashade` and
  `rasterize` ([#4077](https://github.com/holoviz/holoviews/pull/4077))
  ([#3967](https://github.com/holoviz/holoviews/pull/3967))
- Add `per_element` flag to `.apply` accessor
  ([#4119](https://github.com/holoviz/holoviews/pull/4119))
- Add `selected` plot option to control selected glyphs in bokeh
  ([#4281](https://github.com/holoviz/holoviews/pull/4281))
- Improve default `Sankey` `node_padding` heuristic
  ([#4253](https://github.com/holoviz/holoviews/pull/4253))
- Add `hooks` plot option for Plotly backend
  ([#4157](https://github.com/holoviz/holoviews/pull/4157))
- Support for split `Violin` plots in bokeh
  ([#4112](https://github.com/holoviz/holoviews/pull/4112))

Bug fixes:

- Fixed radial `HeatMap` sizing issues
  ([#4162](https://github.com/holoviz/holoviews/pull/4162))
- Switched to Panel for rendering machinery fixing various export issues
  ([#3683](https://github.com/holoviz/holoviews/pull/3683))
- Handle updating of user supplied `HoverTool` in bokeh
  ([#4266](https://github.com/holoviz/holoviews/pull/4266))
- Fix issues with single value datashaded plots
  ([#3673](https://github.com/holoviz/holoviews/pull/3673))
- Fix legend layout issues
  ([#3786](https://github.com/holoviz/holoviews/pull/3786))
- Fix linked axes issues with mixed date, categorical and numeric axes
  in bokeh ([#3845](https://github.com/holoviz/holoviews/pull/3845))
- Fixed handling of repeated dimensions in `PandasInterface`
  ([#4139](https://github.com/holoviz/holoviews/pull/4139))
- Fixed various issues related to widgets
  ([#3868](https://github.com/holoviz/holoviews/pull/3868),
  [#2885](https://github.com/holoviz/holoviews/pull/2885),
  [#1677](https://github.com/holoviz/holoviews/pull/1677),
  [#3212](https://github.com/holoviz/holoviews/pull/3212),
  [#1059](https://github.com/holoviz/holoviews/pull/1059),
  [#3027](https://github.com/holoviz/holoviews/pull/3027),
  [#3777](https://github.com/holoviz/holoviews/pull/3777))


Library compatibility:

- Better support for Pandas 1.0
  ([#4254](https://github.com/holoviz/holoviews/pull/4254))
- Compatibility with Bokeh 2.0
  ([#4226](https://github.com/holoviz/holoviews/pull/4226))

Migration notes:

- Geometry `.iloc` now indexes by geometry instead of by
  datapoint. Convert to dataframe or dictionary before using `.iloc` to
  access individual datapoints
  ([#4104](https://github.com/holoviz/holoviews/pull/4104))
- Padding around plot elements is now enabled by default, to revert set
  `hv.config.node_padding = 0`
  ([#1090](https://github.com/holoviz/holoviews/pull/1090))
- Removed Bars `group_index` and `stack_index` options, which are now
  controlled using the `stacked` option
  ([#3985](https://github.com/holoviz/holoviews/pull/3985))
- `.table` is deprecated; use `.collapse` method instead and cast to
  `Table` ([#3985](https://github.com/holoviz/holoviews/pull/3985))
- `HoloMap.split_overlays` is deprecated and is now a private method
  ([#3985](https://github.com/holoviz/holoviews/pull/3985))
- `Histogram.edges` and `Histogram.values` properties are deprecated; use
  `dimension_values`
  ([#3985](https://github.com/holoviz/holoviews/pull/3985))
- `Element.collapse_data` is deprecated; use the container's `.collapse`
  method instead
  ([#3985](https://github.com/holoviz/holoviews/pull/3985))
- `hv.output` `filename` argument is deprecated; use `hv.save` instead
  ([#3985](https://github.com/holoviz/holoviews/pull/3985))


Version 1.12.7
==============
**November 22, 2019**

This a very minor hotfix release fixing an important bug related to
axiswise normalization between plots. Many thanks to @srp3003 and
@philippjfr for contributing to this release.

Enhancements:

- Add styles attribute to PointDraw stream for consistency with other
  drawing streams
  ([#3819](https://github.com/holoviz/holoviews/issues/3819))

Bug fixes:

- Fixed shared_axes/axiswise regression
  ([#4097](https://github.com/holoviz/holoviews/pull/4097))


Version 1.12.6
==============
**October 8, 2019**

This is a minor release containing a large number of bug fixes thanks
to the contributions from @joelostblom, @ahuang11, @chbrandt,
@randomstuff, @jbednar and @philippjfr. It also contains a number of
enhancements. This is the last planned release in the 1.12.x series.

Enhancements:

- Ensured that shared_axes option on layout plots is respected across backends
  ([#3410](https://github.com/pyviz/holoviews/issues/3410))
- Allow plotting partially irregular (curvilinear) mesh
  ([#3952](https://github.com/pyviz/holoviews/issues/3952))
- Add support for dependent functions in dynamic operations
  ([#3975](https://github.com/pyviz/holoviews/issues/3975),
   [#3980](https://github.com/pyviz/holoviews/issues/3980))
- Add support for fast QuadMesh rasterization with datashader >= 0.8
  ([#4020](https://github.com/pyviz/holoviews/issues/4020))
- Allow passing Panel widgets as operation parameter
  ([#4028](https://github.com/pyviz/holoviews/issues/4028))

Bug fixes:

- Fixed issue rounding datetimes in Curve step interpolation
  ([#3958](https://github.com/pyviz/holoviews/issues/3958))
- Fix resampling of categorical colorcet colormaps
  ([#3977](https://github.com/pyviz/holoviews/issues/3977))
- Ensure that changing the Stream source deletes the old source
  ([#3978](https://github.com/pyviz/holoviews/issues/3978))
- Ensure missing hover tool does not break plot
  ([#3981](https://github.com/pyviz/holoviews/issues/3981))
- Ensure .apply work correctly on HoloMaps
  ([#3989](https://github.com/pyviz/holoviews/issues/3989),
   [#4025](https://github.com/pyviz/holoviews/issues/4025))
- Ensure Grid axes are always aligned in bokeh
  ([#3916](https://github.com/pyviz/holoviews/issues/3916))
- Fix hover tool on Image and Raster plots with inverted axis
  ([#4010](https://github.com/pyviz/holoviews/issues/4010))
- Ensure that DynamicMaps are still linked to streams after groupby
  ([#4012](https://github.com/pyviz/holoviews/issues/4012))
- Using hv.renderer no longer switches backends
  ([#4013](https://github.com/pyviz/holoviews/issues/4013))
- Ensure that Points/Scatter categorizes data correctly when axes are inverted
  ([#4014](https://github.com/pyviz/holoviews/issues/4014))
- Fixed error creating legend for matplotlib Image artists
  ([#4031](https://github.com/pyviz/holoviews/issues/4031))
- Ensure that unqualified Options objects are supported
  ([#4032](https://github.com/pyviz/holoviews/issues/4032))
- Fix bounds check when constructing Image with ImageInterface
  ([#4035](https://github.com/pyviz/holoviews/issues/4035))
- Ensure elements cannot be constructed with wrong number of columns
  ([#4040](https://github.com/pyviz/holoviews/issues/4040))
- Ensure streaming data works on bokeh server
  ([#4041](https://github.com/pyviz/holoviews/issues/4041))

Compatibility:

- Ensure HoloViews is fully compatible with xarray 0.13.0
  ([#3973](https://github.com/pyviz/holoviews/issues/3973))
- Ensure that deprecated matplotlib 3.1 rcparams do not warn
  ([#4042](https://github.com/pyviz/holoviews/issues/4042))
- Ensure compatibility with new legend options in bokeh 1.4.0
  ([#4036](https://github.com/pyviz/holoviews/issues/4036))

Version 1.12.5
==============
**August 14, 2019**

This is a very minor bug fix release ensuring compatibility with recent
releases of dask.

Compatibility:

- Ensure that HoloViews can be imported when dask is installed but
  dask.dataframe is not.
  ([#3900](https://github.com/pyviz/holoviews/issues/3900))
- Fix for rendering Scatter3D with matplotlib 3.1
  ([#3898](https://github.com/pyviz/holoviews/issues/3898))

Version 1.12.4
==============
**August 4, 2019**

This is a minor release with a number of bug and compatibility fixes
as well as a number of enhancements.

Many thanks to recent @henriqueribeiro, @poplarShift, @hojo590,
@stuarteberg, @justinbois, @schumann-tim, @ZuluPro and @jonmmease for
their contributions and the many users filing issues.

Enhancements:

- Add numpy log to dim transforms
  ([#3731](https://github.com/pyviz/holoviews/issues/3731))
- Make Buffer stream following behavior togglable
  ([#3823](https://github.com/pyviz/holoviews/issues/3823))
- Added internal methods to access dask arrays and made histogram
  operation operate on dask arrays
  ([#3854](https://github.com/pyviz/holoviews/issues/3854))
- Optimized range finding if Dimension.range is set
  ([#3860](https://github.com/pyviz/holoviews/issues/3860))
- Add ability to use functions annotated with param.depends as
  DynamicMap callbacks
  ([#3744](https://github.com/pyviz/holoviews/issues/3744))

Bug fixes:

- Fixed handling datetimes on Spikes elements
  ([#3736](https://github.com/pyviz/holoviews/issues/3736))
- Fix graph plotting for unsigned integer node indices
  ([#3773](https://github.com/pyviz/holoviews/issues/3773))
- Fix sort=False on GridSpace and GridMatrix
  ([#3769](https://github.com/pyviz/holoviews/issues/3769))
- Fix extent scaling on VLine/HLine annotations
  ([#3761](https://github.com/pyviz/holoviews/issues/3761))
- Fix BoxWhisker to match convention
  ([#3755](https://github.com/pyviz/holoviews/issues/3755))
- Improved handling of custom array types
  ([#3792](https://github.com/pyviz/holoviews/issues/3792))
- Allow setting cmap on HexTiles in matplotlib
  ([#3803](https://github.com/pyviz/holoviews/issues/3803))
- Fixed handling of data_aspect in bokeh backend
  ([#3848](https://github.com/pyviz/holoviews/issues/3848),
  [#3872](https://github.com/pyviz/holoviews/issues/3872))
- Fixed legends on bokeh Path plots
  ([#3809](https://github.com/pyviz/holoviews/issues/3809))
- Ensure Bars respect xlim and ylim
  ([#3853](https://github.com/pyviz/holoviews/issues/3853))
- Allow setting Chord edge colors using explicit colormapping
  ([#3734](https://github.com/pyviz/holoviews/issues/3734))
- Fixed bug in decimate operation
  ([#3875](https://github.com/pyviz/holoviews/issues/3875))

Compatibility:

- Improve compatibility with deprecated matplotlib rcparams
  ([#3745](https://github.com/pyviz/holoviews/issues/3745),
  [#3804](https://github.com/pyviz/holoviews/issues/3804))

Backwards incompatible changes:

- Unfortunately due to a major mixup the data_aspect option added in
  1.12.0 was not correctly implemented and fixing it changed its
  behavior significantly (inverting it entirely in some cases).
- A mixup in the convention used to compute the whisker of a
  box-whisker plots was fixed resulting in different results going
  forward.

Version 1.12.3
==============
**May 20, 2019**

This is a minor release primarily focused on a number of important bug
fixes. Thanks to our users for reporting issues, and special thanks to
the internal developers @philippjfr and @jlstevens and external
developers including @poplarShift, @fedario and @odoublewen for their
contributions.

Bug fixes:

- Fixed regression causing unhashable data to cause errors in streams
  ([#3681](https://github.com/pyviz/holoviews/issues/3681)
- Ensure that hv.help handles non-HoloViews objects
  ([#3689](https://github.com/pyviz/holoviews/issues/3689))
- Ensure that DataLink handles data containing NaNs
  ([#3694](https://github.com/pyviz/holoviews/issues/3694))
- Ensure that bokeh backend handles Cycle of markers
  ([#3706](https://github.com/pyviz/holoviews/issues/3706))
- Fix for using opts method on DynamicMap
  ([#3691](https://github.com/pyviz/holoviews/issues/3691))
- Ensure that bokeh backend handles DynamicMaps with variable length
  NdOverlay ([#3696](https://github.com/pyviz/holoviews/issues/3696))
- Fix default width/height setting for HeatMap
  ([#3703](https://github.com/pyviz/holoviews/issues/3703))
- Ensure that dask imports handle modularity
  ([#3685](https://github.com/pyviz/holoviews/issues/3685))
- Fixed regression in xarray data interface
  ([#3724](https://github.com/pyviz/holoviews/issues/3724))
- Ensure that RGB hover displays the integer RGB value
  ([#3727](https://github.com/pyviz/holoviews/issues/3727))
- Ensure that param streams handle subobjects
  ([#3728](https://github.com/pyviz/holoviews/pull/3728))

Version 1.12.2
==============
**May 1, 2019**

This is a minor release with a number of important bug fixes and a
small number of enhancements. Many thanks to our users for reporting
these issues, and special thanks to our internal developers
@philippjfr, @jlstevens and @jonmease and external contributors
including @ahuang11 and @arabidopsis for their contributions to the
code and the documentation.

Enhancements:

- Add styles argument to draw tool streams to allow cycling colors
  and other styling when drawing glyphs
  ([#3612](https://github.com/pyviz/holoviews/pull/3612))
- Add ability to define alpha on (data)shade operation
  ([#3611](https://github.com/pyviz/holoviews/pull/3611))
- Ensure that categorical plots respect Dimension.values order
  ([#3675](https://github.com/pyviz/holoviews/pull/3675))

Compatibility:

- Compatibility with Plotly 3.8
  ([#3644](https://github.com/pyviz/holoviews/pull/3644))

Bug fixes:

- Ensure that bokeh server plot updates have the exclusive Document
  lock ([#3621](https://github.com/pyviz/holoviews/pull/3621))
- Ensure that Dimensioned streams are inherited on `__mul__`
  ([#3658](https://github.com/pyviz/holoviews/pull/3658))
- Ensure that bokeh hover tooltips are updated when dimensions change
  ([#3609](https://github.com/pyviz/holoviews/pull/3609))
- Fix DynamicMap.event method for empty streams
  ([#3564](https://github.com/pyviz/holoviews/pull/3564))
- Fixed handling of datetimes on Path plots
  ([#3464](https://github.com/pyviz/holoviews/pull/3464),
   [#3662](https://github.com/pyviz/holoviews/pull/3662))
- Ensure that resampling operations do not cause event loops
  ([#3614](https://github.com/pyviz/holoviews/issues/3614))

Backward compatibility:

- Added color cycles on Violin and BoxWhisker elements due to earlier
  regression ([#3592](https://github.com/pyviz/holoviews/pull/3592))

Version 1.12.1
==============
**April 10, 2019**

This is a minor release that pins to the newly released Bokeh 1.1 and
adds support for parameter instances as streams:

Enhancements:

- Add support for passing in parameter instances as streams
  ([#3616](https://github.com/pyviz/holoviews/pull/3616))

Version 1.12.0
==============
**April 2, 2019**

This release provides a number of exciting new features as well as a set
of important bug fixes. Many thanks to our users for reporting these
issues, and special thanks to @ahuang11, @jonmmease, @poplarShift,
@reckoner, @scottclowe and @syhooper for their contributions to the code
and the documentation.

Features:

- New plot options for controlling layouts including a responsive mode
  as well as improved control over aspect using the newly updated bokeh
  layout engine ([#3450](https://github.com/pyviz/holoviews/pull/3450),
  [#3575](https://github.com/pyviz/holoviews/pull/3575))
- Added a succinct and powerful way of creating DynamicMaps from
  functions and methods via the new `.apply` method
  ([#3554](https://github.com/pyviz/holoviews/pull/3554),
  [#3474](https://github.com/pyviz/holoviews/pull/3474))

Enhancements:

- Added a number of new plot options including a clabel param for
  colorbars ([#3517](https://github.com/pyviz/holoviews/pull/3517)),
  exposed Sankey font size
  ([#3535](https://github.com/pyviz/holoviews/pull/3535)) and added a
  radius for bokeh nodes
  ([#3556](https://github.com/pyviz/holoviews/pull/3556))
- Switched notebook output to use an HTML mime bundle instead of
  separate HTML and JS components
  ([#3574](https://github.com/pyviz/holoviews/pull/3574))
- Improved support for style mapping constant values via
  `dim.categorize`
  ([#3578](https://github.com/pyviz/holoviews/pull/3578))

Bug fixes:

- Fixes for colorscales and colorbars
  ([#3572](https://github.com/pyviz/holoviews/pull/3572),
  [#3590](https://github.com/pyviz/holoviews/pull/3590))
- Other miscellaneous fixes
([#3530](https://github.com/pyviz/holoviews/pull/3530),
[#3536](https://github.com/pyviz/holoviews/pull/3536),
[#3546](https://github.com/pyviz/holoviews/pull/3546),
[#3560](https://github.com/pyviz/holoviews/pull/3560),
[#3571](https://github.com/pyviz/holoviews/pull/3571),
[#3580](https://github.com/pyviz/holoviews/pull/3580),
[#3584](https://github.com/pyviz/holoviews/pull/3584),
[#3585](https://github.com/pyviz/holoviews/pull/3585),
[#3594](https://github.com/pyviz/holoviews/pull/3594))


Version 1.11.3
==============
**February 25, 2019**

This is the last micro-release in the 1.11 series providing a number
of important fixes. Many thanks to our users for reporting these
issues and @poplarShift and @henriqueribeiro for contributing a number
of crucial fixes.

Bug fixes:

- All unused Options objects are now garbage collected fixing the last
  memory leak ([#3438](https://github.com/pyviz/holoviews/pull/3438))
- Ensured updating of size on matplotlib charts does not error
  ([#3442](https://github.com/pyviz/holoviews/pull/3442))
- Fix casting of datetimes on dask dataframes
  ([#3460](https://github.com/pyviz/holoviews/pull/3460))
- Ensure that calling redim does not break streams and links
  ([#3478](https://github.com/pyviz/holoviews/pull/3478))
- Ensure that matplotlib polygon plots close the edge path
  ([#3477](https://github.com/pyviz/holoviews/pull/3477))
- Fixed bokeh ArrowPlot error handling colorbars
  ([#3476](https://github.com/pyviz/holoviews/pull/3476))
- Fixed bug in angle conversion on the VectorField if invert_axes
  ([#3488](https://github.com/pyviz/holoviews/pull/3488))
- Ensure that all non-Annotation elements support empty constructors
  ([#3511](https://github.com/pyviz/holoviews/pull/3511))
- Fixed bug handling out-of-bounds errors when using tap events on
  datetime axis
  ([#3519](https://github.com/pyviz/holoviews/pull/3519))

Enhancements:

- Apply Labels element offset using a bokeh transform allowing Labels
  element to share data with original data
  ([#3445](https://github.com/pyviz/holoviews/pull/3445))
- Allow using datetimes in xlim/ylim/zlim
  ([#3491](https://github.com/pyviz/holoviews/pull/3491))
- Optimized rendering of TriMesh wireframes
  ([#3495](https://github.com/pyviz/holoviews/pull/3495))
- Add support for datetime formatting when hovering on Image/Raster
  ([#3520](https://github.com/pyviz/holoviews/pull/3520))
- Added Tiles element from GeoViews
  ([#3515](https://github.com/pyviz/holoviews/pull/3515))

Version 1.11.2
==============
**January 28, 2019**

This is a minor bug fix release with a number of small but important
bug fixes. Special thanks to @darynwhite for his contributions.

Bug fixes:

- Compatibility with pandas 0.24.0 release
  ([#3433](https://github.com/pyviz/holoviews/pull/3433))
- Fixed timestamp selections on streams
  ([#3427](https://github.com/pyviz/holoviews/pull/3427))
- Fixed persisting options during clone on Overlay
  ([#3435](https://github.com/pyviz/holoviews/pull/3435))
- Ensure cftime datetimes are displayed as a slider
  ([#3413](https://github.com/pyviz/holoviews/pull/3413))

Enhancements:

- Allow defining hook on backend load
  ([#3429](https://github.com/pyviz/holoviews/pull/3429))
- Improvements for handling graph attributes in Graph.from_networkx
  ([#3432](https://github.com/pyviz/holoviews/pull/3432))


Version 1.11.1
==============
**January 17, 2019**

This is a minor bug fix release with a number of important bug fixes,
enhancements and updates to the documentation. Special thanks to
@ahuang11, @garibarba and @Safrone for their contributions.

Bug fixes:

- Fixed bug plotting adjoined histograms in matplotlib
  ([#3377](https://github.com/pyviz/holoviews/pull/3377))
- Fixed bug updating bokeh RGB alpha value
  ([#3371](https://github.com/pyviz/holoviews/pull/3371))
- Handled issue when colorbar limits were equal in bokeh
  ([#3382](https://github.com/pyviz/holoviews/pull/3382))
- Fixed bugs plotting empty Violin and BoxWhisker elements
  ([#3397](https://github.com/pyviz/holoviews/pull/3397),
  [#3405](https://github.com/pyviz/holoviews/pull/3405))
- Fixed handling of characters that have no uppercase on Layout and
  Overlay objects
  (([#3403](https://github.com/pyviz/holoviews/pull/3403))
- Fixed bug updating Polygon plots in bokeh
  ([#3409](https://github.com/pyviz/holoviews/pull/3409))

Enhancements:

- Provide control over gridlines ticker and mirrored axis ticker by
  default ([#3398](https://github.com/pyviz/holoviews/pull/3377))
- Enabled colorbars on CompositePlot classes such as Graphs, Chords
  etc. ([#3397](https://github.com/pyviz/holoviews/pull/3396))
- Ensure that xarray backend retains dimension metadata when casting
  element ([#3401](https://github.com/pyviz/holoviews/pull/3401))
- Consistently support clim options
  ([#3382](https://github.com/pyviz/holoviews/pull/3382))

Documentation:

- Completed updates from .options to .opts API in the documentation
  ([#3364]((https://github.com/pyviz/holoviews/pull/3364),
  [#3367]((https://github.com/pyviz/holoviews/pull/3367))


Version 1.11.0
==============
**December 24, 2018**

This is a major release containing a large number of features and API
improvements. Specifically this release was devoted to improving the
general usability and accessibility of the HoloViews API and
deprecating parts of the API in anticipation for the 2.0 release.
To enable deprecation warnings for these deprecations set:

    hv.config.future_deprecations = True

The largest updates to the API relate to the options system which is now
more consistent, has better validation and better supports notebook
users without requiring IPython magics. The new `dim` transform
generalizes the mapping from data dimensions to visual dimensions,
greatly increasing the expressive power of the options system. Please
consult the updated user guides for more information.

Special thanks for the contributions by Andrew Huang (@ahuang11),
Julia Signell (@jsignell), Jon Mease (@jonmmease), and Zachary Barry
(@zbarry).

Features:

- Generalized support for style mapping using `dim` transforms
  ([2152](https://github.com/pyviz/holoviews/pull/2152))
- Added alternative to opts magic with tab-completion
  ([#3173](https://github.com/pyviz/holoviews/pull/3173))
- Added support for Polygons with holes and improved contours
  operation ([#3092](https://github.com/pyviz/holoviews/pull/3092))
- Added support for Links to express complex interactivity in JS
  ([#2832](https://github.com/pyviz/holoviews/pull/2832))
- Plotly improvements including support for plotly 3.0
  ([#3194](https://github.com/pyviz/holoviews/pull/3194)), improved
  support for containers
  ([#3255](https://github.com/pyviz/holoviews/pull/3255)) and support
  for more elements
  ([#3256](https://github.com/pyviz/holoviews/pull/3256))
- Support for automatically padding plots using new `padding` option
  ([#2293](https://github.com/pyviz/holoviews/pull/2293))
- Added `xlim`/`ylim` plot options to simplify setting axis ranges
  ([#2293](https://github.com/pyviz/holoviews/pull/2293))
- Added `xlabel`/`ylabel` plot options to simplify overriding axis
  labels ([#2833](https://github.com/pyviz/holoviews/issues/2833))
- Added `xformatter`/`yformatter` plot options to easily override tick
  formatter ([#3042](https://github.com/pyviz/holoviews/pull/3042))
- Added `active_tools` options to allow defining tools to activate on
  bokeh plot initialization
  ([#3251](https://github.com/pyviz/holoviews/pull/3251))
- Added `FreehandDraw` stream to allow freehand drawing on bokeh plots
  ([#2937](https://github.com/pyviz/holoviews/pull/2937))
- Added support for `cftime` types for dates which are not supported
  by standard datetimes and calendars
  ([#2728](https://github.com/pyviz/holoviews/pull/2728))
- Added top-level `save` and `render` functions to simplify exporting
  plots ([#3134](https://github.com/pyviz/holoviews/pull/3134))
- Added support for updating Bokeh bokeh legends
  ([#3139](https://github.com/pyviz/holoviews/pull/3139))
- Added support for indicating directed graphs with arrows
  ([#2521](https://github.com/pyviz/holoviews/issues/2521))

Enhancements:

- Improved import times
  ([#3055](https://github.com/pyviz/holoviews/pull/3055))
- Adopted Google style docstring and documented most core methods and
  classes ([#3128](https://github.com/pyviz/holoviews/pull/3128)

Bug fixes:

- GIF rendering fixed under Windows
  ([#3151](https://github.com/pyviz/holoviews/issues/3151))
- Fixes for hover on Path elements in bokeh
  ([#2472](https://github.com/pyviz/holoviews/issues/2427),
  [#2872](https://github.com/pyviz/holoviews/issues/2872))
- Fixes for handling TriMesh value dimensions on rasterization
  ([#3050](https://github.com/pyviz/holoviews/pull/3050))

Deprecations:

- `finalize_hooks` renamed to `hooks`
  ([#3134](https://github.com/pyviz/holoviews/pull/3134))
- All `*_index` and related options are now deprecated including
  `color_index`, `size_index`, `scaling_method`, `scaling_factor`,
  `size_fn` ([#2152](https://github.com/pyviz/holoviews/pull/2152))
- Bars `group_index`, `category_index` and `stack_index` are deprecated in
  favor of stacked option
  ([#2828](https://github.com/pyviz/holoviews/issues/2828))
- iris interface was moved to GeoViews
  ([#3054](https://github.com/pyviz/holoviews/pull/3054))
- Top-level namespace was cleaned up
  ([#2224](https://github.com/pyviz/holoviews/pull/2224))
- `ElementOpration`, `Layout.display` and `mdims` argument to `.to`
  now fully removed
  ([#3128](https://github.com/pyviz/holoviews/pull/3128))
- `Element.mapping`, `ItemTable.values`, `Element.table`,
  `HoloMap.split_overlays`, `ViewableTree.from_values`,
  `ViewableTree.regroup` and `Element.collapse_data` methods now
  marked for deprecation
  ([#3128](https://github.com/pyviz/holoviews/pull/3128))


Version 1.10.8
==============
**October 29, 2018**


This a likely the last hotfix release in the 1.10.x series containing
fixes for compatibility with bokeh 1.0 and matplotlib 3.0. It also
contains a wide array of fixes contributed and reported by users:

Special thanks for the contributions by Andrew Huang (@ahuang11),
Julia Signell (@jsignell), and Zachary Barry (@zbarry).

Enhancements:

- Add support for labels, choord, hextiles and area in `.to` interface
  ([#2924](https://github.com/pyviz/holoviews/pull/2923))
- Allow defining default bokeh themes as strings on Renderer
  ([#2972](https://github.com/pyviz/holoviews/pull/2972))
- Allow specifying fontsize for categorical axis ticks in bokeh
  ([#3047](https://github.com/pyviz/holoviews/pull/3047))
- Allow hiding toolbar without disabling tools
  ([#3074](https://github.com/pyviz/holoviews/pull/3074))
- Allow specifying explicit colormapping on non-categorical data
  ([#3071](https://github.com/pyviz/holoviews/pull/3071))
- Support for displaying xarray without explicit coordinates
  ([#2968](https://github.com/pyviz/holoviews/pull/2968))

Fixes:

- Ensured that objects are garbage collected when using
  linked streams ([#2111](https://github.com/pyviz/holoviews/issues/2111))
- Allow dictionary data to reference values which are not dimensions
  ([#2855](https://github.com/pyviz/holoviews/pull/2855),
  [#2859](https://github.com/pyviz/holoviews/pull/2859))
- Fixes for zero and non-finite ranges in datashader operation
  ([#2860](https://github.com/pyviz/holoviews/pull/2860),
  [#2863](https://github.com/pyviz/holoviews/pull/2863),
  [#2869](https://github.com/pyviz/holoviews/pull/2869))
- Fixes for CDSStream and drawing tools on bokeh server
  ([#2915](https://github.com/pyviz/holoviews/pull/2915))
- Fixed issues with nans, datetimes and streaming on Area and Spread
  elements ([#2951](https://github.com/pyviz/holoviews/pull/2951),
  [c55b044](https://github.com/pyviz/holoviews/commit/c55b044))
- General fixes for datetime handling
  ([#3005](https://github.com/pyviz/holoviews/pull/3005),
  [#3045](https://github.com/pyviz/holoviews/pull/3045),
  [#3075](https://github.com/pyviz/holoviews/pull/3074))
- Fixed handling of curvilinear and datetime coordinates on QuadMesh
  ([#3017](https://github.com/pyviz/holoviews/pull/3017),
  [#3081](https://github.com/pyviz/holoviews/pull/3081))
- Fixed issue when inverting a shared axis in bokeh
  ([#3083](https://github.com/pyviz/holoviews/pull/3083))
- Fixed formatting of values in HoloMap widgets
  ([#2954](https://github.com/pyviz/holoviews/pull/2954))
- Fixed setting fontsize for z-axis label
  ([#2967](https://github.com/pyviz/holoviews/pull/2967))

Compatibility:

- Suppress warnings about rcParams in matplotlib 3.0
  ([#3013](https://github.com/pyviz/holoviews/pull/3013),
  [#3058](https://github.com/pyviz/holoviews/pull/3058),
  [#3104](https://github.com/pyviz/holoviews/pull/3104))
- Fixed incompatibility with Python <=3.5
  ([#3073](https://github.com/pyviz/holoviews/pull/3073))
- Fixed incompatibility with bokeh >=1.0
  ([#3051](https://github.com/pyviz/holoviews/pull/3051))

Documentation:

- Completely overhauled the FAQ
  ([#2928](https://github.com/pyviz/holoviews/pull/2928),
  [#2941](https://github.com/pyviz/holoviews/pull/2941),
  [#2959](https://github.com/pyviz/holoviews/pull/2959),
  [#3025](https://github.com/pyviz/holoviews/pull/3025))

Version 1.10.7
==============
**July 8, 2018**

This a very minor hotfix release mostly containing fixes for datashader
aggregation of empty datasets:

Fixes:

- Fix datashader aggregation of empty and zero-range data
  ([#2860](https://github.com/pyviz/holoviews/pull/2860),
  [#2863](https://github.com/pyviz/holoviews/pull/2863))
- Disable validation for additional, non-referenced keys in the
  DictInterface ([#2860](https://github.com/pyviz/holoviews/pull/2860))
- Fixed frame lookup for non-overlapping dimensions
  ([#2861](https://github.com/pyviz/holoviews/pull/2861))
- Fixed ticks on log Colorbar if low value <= 0
  ([#2865](https://github.com/pyviz/holoviews/pull/2865))


Version 1.10.6
==============
**June 29, 2018**

This another minor bug fix release in the 1.10 series and likely the
last one before the upcoming 1.11 release. In addition to some important
fixes relating to datashading and the handling of dask data, this
release includes a number of enhancements and fixes.

Enhancements:

- Added the ability to specify color intervals using the color_levels
  plot options ([#2797](https://github.com/pyviz/holoviews/pull/2797))
- Allow defining port and multiple websocket origins on BokehRenderer.app
  ([#2801](https://github.com/pyviz/holoviews/pull/2801))
- Support for datetimes in Curve step interpolation
  ([#2757](https://github.com/pyviz/holoviews/pull/2757))
- Add ability to mute legend by default
  ([#2831](https://github.com/pyviz/holoviews/pull/2831))
- Implemented ability to collapse and concatenate gridded data
  ([#2762](https://github.com/pyviz/holoviews/pull/2762))
- Add support for cumulative histogram and explicit bins
  ([#2812](https://github.com/pyviz/holoviews/pull/2812))

Fixes:

- Dataset discovers multi-indexes on dask dataframes
  ([#2789](https://github.com/pyviz/holoviews/pull/2789))
- Fixes for datashading NdOverlays with datetime axis and data with
  zero range ([#2829](https://github.com/pyviz/holoviews/pull/2829),
  [#2842](https://github.com/pyviz/holoviews/pull/2842))


Version 1.10.5
==============
**June 5, 2018**

This is a minor bug fix release containing a mixture of small
enhancements, a number of important fixes and improved compatibility
with pandas 0.23.

Enhancements:

- Graph.from_networkx now extracts node and edge attributes from
  networkx graphs
  ([#2714](https://github.com/pyviz/holoviews/pull/2714))
- Added throttling support to scrubber widget
  ([#2748](https://github.com/pyviz/holoviews/pull/2748))
- histogram operation now works on datetimes
  ([#2719](https://github.com/pyviz/holoviews/pull/2719))
- Legends on NdOverlay containing overlays now supported
  ([#2755](https://github.com/pyviz/holoviews/pull/2755))
- Dataframe indexes may now be referenced in ``.to`` conversion
  ([#2739](https://github.com/pyviz/holoviews/pull/2739))
- Reindexing a gridded Dataset without arguments now behaves
  consistently with NdMapping types and drops scalar dimensions making
  it simpler to drop dimensions after selecting
  ([#2746](https://github.com/pyviz/holoviews/pull/2746))

Fixes:

- Various fixes for QuadMesh support including support for contours,
  nan coordinates and inverted coordinates
  ([#2691](https://github.com/pyviz/holoviews/pull/2691),
  [#2702](https://github.com/pyviz/holoviews/pull/2702),
  [#2771](https://github.com/pyviz/holoviews/pull/2771))
- Fixed bugs laying out complex layouts in bokeh
  ([#2740](https://github.com/pyviz/holoviews/pull/2740))
- Fix for adding value dimensions to an xarray dataset
  ([#2761](https://github.com/pyviz/holoviews/pull/2761))

Compatibility:

- Addressed various deprecation warnings generated by pandas 0.23
  ([#2699](https://github.com/pyviz/holoviews/pull/2699),
  [#2725](https://github.com/pyviz/holoviews/pull/2725),
  [#2767](https://github.com/pyviz/holoviews/pull/2767))


Version 1.10.4
==============
**May 14, 2018**

This is a minor bug fix release including a number of crucial fixes
for issues reported by our users.

Enhancement:

- Allow setting alpha on Image/RGB/HSV and Raster types in bokeh
  ([#2680](https://github.com/pyviz/holoviews/pull/2680))

Fixes:

- Fixed bug running display multiple times in one cell
  ([#2677](https://github.com/pyviz/holoviews/pull/2677))
- Avoid sending hover data unless explicitly requested
  ([#2681](https://github.com/pyviz/holoviews/pull/2681))
- Fixed bug slicing xarray with tuples
  ([#2674](https://github.com/pyviz/holoviews/pull/2674))

Version 1.10.3
==============
**May 8, 2018**

This is a minor bug fix release including a number of crucial fixes
for issues reported by our users.

Enhancement:

- The dimensions of elements may now be changed allowing updates to
  axis labels and table column headers
  ([#2666](https://github.com/pyviz/holoviews/pull/2666))

Fixes:

- Fix for ``labelled`` plot option
  ([#2643](https://github.com/pyviz/holoviews/pull/2643))
- Optimized initialization of dynamic plots specifying a large
  parameter space
  ([#2646](https://github.com/pyviz/holoviews/pull/2646))
- Fixed unicode and reversed axis slicing issues in XArrayInterface
  ([#2658](https://github.com/pyviz/holoviews/issues/2658),
   [#2653](https://github.com/pyviz/holoviews/pull/2653))
- Fixed widget sorting issues when applying dynamic groupby
  ([#2641](https://github.com/pyviz/holoviews/issues/2641))

API:

- The PlotReset reset parameter was renamed to resetting to avoid
  clash with a method
  ([#2665](https://github.com/pyviz/holoviews/pull/2665))
- PolyDraw tool data parameter now always indexed with 'xs' and 'ys'
  keys for consistency
  ([#2650](https://github.com/pyviz/holoviews/issues/2650))

Version 1.10.2
==============
**April 30, 2018**

This is a minor bug fix release with a number of small fixes for
features and regressions introduced in 1.10:

Enhancement:

- Exposed Image hover functionality for upcoming bokeh 0.12.16 release
  ([#2625](https://github.com/pyviz/holoviews/pull/2625))

Fixes:

- Minor fixes for newly introduced elements and plots including Chord
  ([#2581](https://github.com/pyviz/holoviews/issues/2581)) and
  RadialHeatMap
  ([#2610](https://github.com/pyviz/holoviews/issues/2610)
- Fixes for .options method including resolving style and plot option
  clashes ([#2411](https://github.com/pyviz/holoviews/issues/2411)) and
  calling it without arguments
  ([#2630](https://github.com/pyviz/holoviews/pull/2630))
- Fixes for IPython display function
  ([#2587](https://github.com/pyviz/holoviews/issues/2587)) and
  display_formats
  ([#2592](https://github.com/pyviz/holoviews/issues/2592))

Deprecations:

- BoxWhisker and Bars ``width`` bokeh style options and Arrow
  matplotlib ``fontsize`` option are deprecated
  ([#2411](https://github.com/pyviz/holoviews/issues/2411))


Version 1.10.1
==============
**April 20, 2018**

This is a minor bug fix release with a number of fixes for regressions
and minor bugs introduced in the 1.10.0 release:

Fixes:

- Fixed static HTML export of notebooks
  ([#2574](https://github.com/pyviz/holoviews/pull/2574))
- Ensured Chord element allows recurrent edges
  ([#2583](https://github.com/pyviz/holoviews/pull/2583))
- Restored behavior for inferring key dimensions order from XArray
  Dataset ([#2579](https://github.com/pyviz/holoviews/pull/2579))
- Fixed Selection1D stream on bokeh server after changes in bokeh
  0.12.15 ([#2586](https://github.com/pyviz/holoviews/pull/2586))


Version 1.10.0
==============
**April 17, 2018**

This is a major release with a large number of new features and bug
fixes, as well as a small number of API changes. Many thanks to the
numerous users who filed bug reports, tested development versions, and
contributed a number of new features and bug fixes, including special
thanks to @mansenfranzen, @ea42gh, @drs251 and @jakirkham.


JupyterLab support:

- Full compatibility with JupyterLab when installing the
  jupyterlab_holoviews extension
  ([#687](https://github.com/pyviz/holoviews/issues/687))

New components:

- Added [``Sankey`` element](http://holoviews.org/reference/elements/bokeh/Sankey.html)
  to plot directed flow graphs
  ([#1123](https://github.com/pyviz/holoviews/issues/1123))
- Added [``TriMesh`` element](http://holoviews.org/reference/elements/bokeh/TriMesh.html)
  and datashading operation to plot small and large irregular meshes
  ([#2143](https://github.com/pyviz/holoviews/issues/2143))
- Added a [``Chord`` element](http://holoviews.org/reference/elements/bokeh/Chord.html)
  to draw flow graphs between different
  nodes ([#2137](https://github.com/pyviz/holoviews/issues/2137),
  [#2143](https://github.com/pyviz/holoviews/pull/2143))
- Added [``HexTiles`` element](http://holoviews.org/reference/elements/bokeh/HexTiles.html)
  to plot data binned into a hexagonal grid
  ([#1141](https://github.com/pyviz/holoviews/issues/1141))
- Added [``Labels`` element](http://holoviews.org/reference/elements/bokeh/Labels.html)
  to plot a large number of text labels at once (as data rather than as annotations)
  ([#1837](https://github.com/pyviz/holoviews/issues/1837))
- Added [``Div`` element](http://holoviews.org/reference/elements/bokeh/Div.html)
  to add arbitrary HTML elements to a Bokeh layout
  ([#2221](https://github.com/pyviz/holoviews/issues/2221))
- Added
  [``PointDraw``](http://holoviews.org/reference/streams/bokeh/PointDraw.html),
  [``PolyDraw``](http://holoviews.org/reference/streams/bokeh/PolyDraw.html),
  [``BoxEdit``](http://holoviews.org/reference/streams/bokeh/BoxEdit.html), and
  [``PolyEdit``](http://holoviews.org/reference/streams/bokeh/PolyEdit.html)
  streams to allow drawing, editing, and annotating glyphs on a Bokeh
  plot, and syncing the resulting data to Python
  ([#2268](https://github.com/pyviz/holoviews/issues/2459))

Features:

- Added [radial ``HeatMap``](http://holoviews.org/reference/elements/bokeh/RadialHeatMap.html)
  option to allow plotting heatmaps with a cyclic x-axis
  ([#2139](https://github.com/pyviz/holoviews/pull/2139))
- All elements now support declaring bin edges as well as centers
  allowing ``Histogram`` and ``QuadMesh`` to become first class
  ``Dataset`` types
  ([#547](https://github.com/pyviz/holoviews/issues/547))
- When using widgets, their initial or default value can now be
  set via the `Dimension.default` parameter
  ([#704](https://github.com/pyviz/holoviews/issues/704))
- n-dimensional Dask arrays are now supported directly via the gridded
  dictionary data interface
  ([#2305](https://github.com/pyviz/holoviews/pull/2305))
- Added new [Styling Plots](http://holoviews.org/user_guide/Styling_Plots.html)
  and [Colormaps](http://holoviews.org/user_guide/Colormaps.html)
  user guides, including new functionality for working with colormaps.

Enhancements:

- Improvements to exceptions
  ([#1127](https://github.com/pyviz/holoviews/issues/1127))
- Toolbar position and merging (via a new ``merge_toolbar``
  option) can now be controlled for Layout and Grid plots
  ([#1977](https://github.com/pyviz/holoviews/issues/1977))
- Bokeh themes can now be applied at the renderer level
  ([#1861](https://github.com/pyviz/holoviews/issues/1861))
- Dataframe and Series index can now be referenced by name when
  constructing an element
  ([#2000](https://github.com/pyviz/holoviews/issues/2000))
- Option-setting methods such as ``.opts``, ``.options`` and
  ``hv.opts`` now allow specifying the backend instead of defaulting
  to the current backend
  ([#1801](https://github.com/pyviz/holoviews/issues/1801))
- Handled API changes in streamz 0.3.0 in Buffer stream
  ([#2409](https://github.com/pyviz/holoviews/issues/2409))
- Supported GIF output on windows using new Matplotlib pillow
  animation support
  ([#385](https://github.com/pyviz/holoviews/issues/385))
- Provided simplified interface to ``rasterize`` most element types
  using datashader
  ([#2465](https://github.com/pyviz/holoviews/pull/2465))
- ``Bivariate`` element now support ``levels`` as a plot option
  ([#2099](https://github.com/pyviz/holoviews/issues/2099))
- ``NdLayout`` and ``GridSpace`` now consistently support ``*``
  overlay operation
  ([#2075](https://github.com/pyviz/holoviews/issues/2075))
- The Bokeh backend no longer has a hard dependency on Matplotlib
  ([#829](https://github.com/pyviz/holoviews/issues/829))
- ``DynamicMap`` may now return (``Nd``)``Overlay`` with varying
  number of elements
  ([#1388](https://github.com/pyviz/holoviews/issues/1388))
- In the notebook, deleting or re-executing a cell will now delete
  the plot and clean up any attached streams
  ([#2141](https://github.com/pyviz/holoviews/issues/2141))
- Added ``color_levels`` plot option to set discrete number of levels
  during colormapping
  ([#2483](https://github.com/pyviz/holoviews/pull/2483))
- Expanded the [Large Data](http://holoviews.org/user_guide/Large_Data.html)
  user guide to show examples of all Element and Container types
  supported for datashading and give performance guidelines.

Fixes:

- ``Layout`` and ``Overlay`` objects no longer create lower-case nodes
  on attribute access
  ([#2331](https://github.com/pyviz/holoviews/pull/2331))
- ``Dimension.step`` now correctly respects both integer and float
  steps ([#1707](https://github.com/pyviz/holoviews/issues/1707))
- Fixed timezone issues when using linked streams on datetime axes
  ([#2459](https://github.com/pyviz/holoviews/issues/2459))


Changes affecting backwards compatibility:

- Image elements now expect and validate regular sampling
  ([#1869](https://github.com/pyviz/holoviews/issues/1869)); for
  genuinely irregularly sampled data QuadMesh should be used.
- Tabular elements will no longer default to use ``ArrayInterface``,
  instead preferring pandas and dictionary data formats
  ([#1236](https://github.com/pyviz/holoviews/issues/1236))
- ``Cycle``/``Palette`` values are no longer zipped together; instead
  they now cycle independently
  ([#2333](https://github.com/pyviz/holoviews/pull/2333))
- The default color ``Cycle`` was expanded to provide more unique colors
  ([#2483](https://github.com/pyviz/holoviews/pull/2483))
- Categorical colormapping was made consistent across backends,
  changing the behavior of categorical Matplotlib colormaps
  ([#2483](https://github.com/pyviz/holoviews/pull/2483))
- Disabled auto-indexable property of the Dataset baseclass, i.e. if a
  single column is supplied no integer index column is added
  automatically ([#2522](https://github.com/pyviz/holoviews/pull/2522))


Version 1.9.5
=============
**March 2, 2018**

This release includes a very small number of minor bugfixes and a new
feature to simplify setting options in python:

Enhancements:

-  Added .options method for simplified options setting.
   ([\#2306](https://github.com/pyviz/holoviews/pull/2306))

Fixes:

-  Allow plotting bytes datausing the Bokeh backend in python3
   ([\#2357](https://github.com/pyviz/holoviews/pull/2357))
-  Allow .range to work on data with heterogeneous types in Python 3
   ([\#2345](https://github.com/pyviz/holoviews/pull/2345))
-  Fixed bug streaming data containing datetimes using bokeh>=0.12.14
   ([\#2383](https://github.com/pyviz/holoviews/pull/2383))


Version 1.9.4
=============
**February 16, 2018**

This release contains a small number of important bug fixes:

-    Compatibility with recent versions of Dask and pandas
     ([\#2329](https://github.com/pyviz/holoviews/pull/2329))
-    Fixed bug referencing columns containing non-alphanumeric characters
     in Bokeh Tables ([\#2336](https://github.com/pyviz/holoviews/pull/2336))
-    Fixed issue in regrid operation
     ([2337](https://github.com/pyviz/holoviews/pull/2337))
-    Fixed issue when using datetimes with datashader when processing
     ranges ([\#2344](https://github.com/pyviz/holoviews/pull/2344))


Version 1.9.3
=============
**February 11, 2018**

This release contains a number of important bug fixes and minor
enhancements.

Particular thanks to @jbampton, @ea42gh, @laleph, and @drs251 for a
number of fixes and improvements to the documentation.

Enhancements:

-    Optimized rendering of stream based OverlayPlots
     ([\#2253](https://github.com/pyviz/holoviews/pull/2253))
-    Added ``merge_toolbars`` and ``toolbar`` options to control
     toolbars on ``Layout`` and Grid plots
     ([\#2289](https://github.com/pyviz/holoviews/pull/2289))
-    Optimized rendering of ``VectorField``
     ([\#2314](https://github.com/pyviz/holoviews/pull/2289))
-    Improvements to documentation
     ([\#2198](https://github.com/pyviz/holoviews/pull/2198),
     [\#2220](https://github.com/pyviz/holoviews/pull/2220),
     [\#2233](https://github.com/pyviz/holoviews/pull/2233),
     [\#2235](https://github.com/pyviz/holoviews/pull/2235),
     [\#2316](https://github.com/pyviz/holoviews/pull/2316))
-    Improved Bokeh ``Table`` formatting
     ([\#2267](https://github.com/pyviz/holoviews/pull/2267))
-    Added support for handling datetime.date types
     ([\#2267](https://github.com/pyviz/holoviews/pull/2267))
-    Add support for pre- and post-process hooks on operations
     ([\#2246](https://github.com/pyviz/holoviews/pull/2246),
	  [\#2334](https://github.com/pyviz/holoviews/pull/2334))

Fixes:

-    Fix for Bokeh server widgets
     ([\#2218](https://github.com/pyviz/holoviews/pull/2218))
-    Fix using event based streams on Bokeh server
     ([\#2239](https://github.com/pyviz/holoviews/pull/2239),
     [\#2256](https://github.com/pyviz/holoviews/pull/2256))
-    Switched to drawing ``Distribution``, ``Area`` and ``Spread``
     using patch glyphs in Bokeh fixing legends
     ([\#2225](https://github.com/pyviz/holoviews/pull/2225))
-    Fixed categorical coloring of ``Polygons``/``Path`` elements in
     Matplotlib ([\#2259](https://github.com/pyviz/holoviews/pull/2259))
-    Fixed bug computing categorical datashader aggregates
     ([\#2295](https://github.com/pyviz/holoviews/pull/2295))
-    Allow using ``Empty`` object in ``AdjointLayout``
     ([\#2275](https://github.com/pyviz/holoviews/pull/2275))


API Changes:

-    Renamed ``Trisurface`` to ``TriSurface`` for future consistency
     ([\#2219](https://github.com/pyviz/holoviews/pull/2219))


Version 1.9.2
=============
**December 11, 2017**

This release is a minor bug fix release patching various issues
which were found in the 1.9.1 release.

Enhancements:

-   Improved the Graph element, optimizing the constructor
    and adding support for defining a `edge_color_index`
    ([\#2145](https://github.com/pyviz/holoviews/pull/2145))
-   Added support for adding jitter to Bokeh Scatter and Points plots
    ([e56208](https://github.com/pyviz/holoviews/commit/e56208e1eb6e1e4af67b6a3ffbb5a925bfc37e14))

Fixes:

-   Ensure dimensions, group and label are inherited when casting
    Image to QuadMesh
    ([\#2144](https://github.com/pyviz/holoviews/pull/2144))
-   Handle compatibility for Bokeh version >= 0.12.11
    ([\#2159](https://github.com/pyviz/holoviews/pull/2159))
-   Fixed broken Bokeh ArrowPlot
    ([\#2172](https://github.com/pyviz/holoviews/pull/2172))
-   Fixed Pointer based streams on datetime axes
    ([\#2179](https://github.com/pyviz/holoviews/pull/2179))
-   Allow constructing and plotting of empty Distribution and
    Bivariate elements
    ([\#2190](https://github.com/pyviz/holoviews/pull/2190))
-   Added support for hover info on Bokeh BoxWhisker plots
    ([\#2187](https://github.com/pyviz/holoviews/pull/2187))
-   Fixed bug attaching streams to (Nd)Overlay types
    ([\#2194](https://github.com/pyviz/holoviews/pull/2194))


Version 1.9.1
=============
**November 13, 2017**

This release is a minor bug fix release patching various issues
which were found in the 1.9.0 release.

Enhancements:

-   Exposed min_alpha parameter on datashader shade and datashade
    operations ([\#2109](https://github.com/pyviz/holoviews/pull/2109))

Fixes:

-   Fixed broken Bokeh server linked stream throttling
    ([\#2112](https://github.com/pyviz/holoviews/pull/2112))
-   Fixed bug in Bokeh callbacks preventing linked streams using
    Bokeh's on_event callbacks from working
    ([\#2112](https://github.com/pyviz/holoviews/pull/2112))
-   Fixed insufficient validation issue for Image and bugs when
    applying regrid operation to xarray based Images
    ([\#2117](https://github.com/pyviz/holoviews/pull/2117))
-   Fixed handling of dimensions and empty elements in univariate_kde
    and bivariate_kde operations
    ([\#2103](https://github.com/pyviz/holoviews/pull/2103))

Version 1.9.0
=============
**November 3, 2017**

This release includes a large number of long awaited features,
improvements and bug fixes, including streaming and graph support,
binary transfer of Bokeh data, fast Image/RGB regridding, first-class
statistics elements and a complete overhaul of the geometry elements.

Particular thanks to all users and contributors who have reported
issues and submitted pull requests.

Features:

-   The kdim and vdim keyword arguments are now positional making the
    declaration of elements less verbose (e.g. Scatter(data, 'x',
    'y')) ([\#1946](https://github.com/pyviz/holoviews/pull/1946))
-   Added Graph, Nodes, and EdgePaths elements adding support for
    plotting network graphs
    ([\#1829](https://github.com/pyviz/holoviews/pull/1829))
-   Added datashader based regrid operation for fast Image and RGB
    regridding ([\#1773](https://github.com/pyviz/holoviews/pull/1773))
-   Added support for binary transport when plotting with Bokeh,
    providing huge speedups for dynamic plots
    ([\#1894](https://github.com/pyviz/holoviews/pull/1894),
    [\#1896](https://github.com/pyviz/holoviews/pull/1896))
-   Added Pipe and Buffer streams for streaming data support
    ([\#2011](https://github.com/pyviz/holoviews/pull/2011))
-   Add support for datetime axes on Image, RGB and when applying
    datashading and regridding operations
    ([\#2023](https://github.com/pyviz/holoviews/pull/2023))
-   Added Distribution and Bivariate as first class elements which can
    be plotted with Matplotlib and Bokeh without depending on seaborn
    ([\#1985](https://github.com/pyviz/holoviews/pull/1985))
-   Completely overhauled support for plotting geometries with Path,
    Contours and Polygons elements including support for coloring
    individual segments and paths by value
    ([\#1991](https://github.com/pyviz/holoviews/pull/1991))

Enhancements:

-   Add support for adjoining all elements on Matplotlib plots
    ([\#1033](https://github.com/pyviz/holoviews/pull/1033))
-   Improved exception handling for data interfaces
    ([\#2041](https://github.com/pyviz/holoviews/pull/2041))
-   Add groupby argument to histogram operation
    ([\#1725](https://github.com/pyviz/holoviews/pull/1725))
-   Add support for reverse sort on Dataset elements
    ([\#1843](https://github.com/pyviz/holoviews/pull/1843))
-   Added support for invert_x/yaxis on all elements
    ([\#1872](https://github.com/pyviz/holoviews/pull/1872),
    [\#1919](https://github.com/pyviz/holoviews/pull/1919))

Fixes:

-   Fixed a bug in Matplotlib causing the first frame in gif and mp4
    getting stuck
    ([\#1922](https://github.com/pyviz/holoviews/pull/1922))
-   Fixed various issues with support for new nested categorical axes
    in Bokeh ([\#1933](https://github.com/pyviz/holoviews/pull/1933))
-   A large range of other bug fixes too long to list here.

Changes affecting backwards compatibility:

-   The contours operation no longer overlays the contours on top of
    the supplied Image by default and returns a single
    Contours/Polygons rather than an NdOverlay of them
    ([\#1991](https://github.com/pyviz/holoviews/pull/1991))
-   The values of the Distribution element should now be defined as a
    key dimension
    ([\#1985](https://github.com/pyviz/holoviews/pull/1985))
-   The seaborn interface was removed in its entirety being replaced
    by first class support for statistics elements such as
    Distribution and Bivariate
    ([\#1985](https://github.com/pyviz/holoviews/pull/1985))
-   Since kdims and vdims can now be passed as positional arguments
    the bounds argument on Image is no longer positional
    ([\#1946](https://github.com/pyviz/holoviews/pull/1946)).
-   The datashade and shade cmap was reverted back to blue due to issues
    with the fire cmap against a white background.
    ([\#2078](https://github.com/pyviz/holoviews/pull/2078))
-   Dropped all support for Bokeh versions older than 0.12.10
-   histogram operation now returns Histogram elements with less
    generic value dimension and customizable label
    ([\#1836](https://github.com/pyviz/holoviews/pull/1836))


Version 1.8.4
=============
**September 13, 2017**

This bugfix release includes a number of critical fixes for compatibility
with Bokeh 0.12.9 along with various other bug fixes. Many thanks to our
users for various detailed bug reports, feedback and contributions.

Fixes:

-   Fixes to register BoundsXY stream.
    ([\#1826](https://github.com/pyviz/holoviews/pull/1826))
-   Fix for Bounds streams on Bokeh server.
    ([\#1883](https://github.com/pyviz/holoviews/pull/1883))
-   Compatibility with Matplotlib 2.1
    ([\#1842](https://github.com/pyviz/holoviews/pull/1842))
-   Fixed bug in scrubber widget and support for scrubbing discrete
    DynamicMaps ([\#1832](https://github.com/pyviz/holoviews/pull/1832))
-   Various fixes for compatibility with Bokeh 0.12.9
    ([\#1849](https://github.com/pyviz/holoviews/pull/1849),
    [\#1866](https://github.com/pyviz/holoviews/pull/1886))
-   Fixes for setting QuadMesh ranges.
    ([\#1876](https://github.com/pyviz/holoviews/pull/1876))
-   Fixes for inverting Image/RGB/Raster axes in Bokeh.
    ([\#1872](https://github.com/pyviz/holoviews/pull/1872))


Version 1.8.3
=============
**August 21, 2017**

This bugfix release fixes a number of minor issues identified since the
last release:

Features:

-   Add support for setting the Bokeh sizing_mode as a plot option
    ([\#1813](https://github.com/pyviz/holoviews/pull/1813))

Fixes:

-   Handle StopIteration on DynamicMap correctly.
    ([\#1792](https://github.com/pyviz/holoviews/pull/1792))
-   Fix bug with linked streams on empty source element
    ([\#1725](https://github.com/pyviz/holoviews/pull/1806))
-   Compatibility with latest datashader 0.6.0 release
    ([\#1773](https://github.com/pyviz/holoviews/pull/1773))
-   Fixed missing HTML closing tag in extension
    ([\#1797](https://github.com/pyviz/holoviews/issues/1797),
     [\#1809](https://github.com/pyviz/holoviews/pull/1809))
-   Various fixes and improvements for documentation
    ([\#1664](https://github.com/pyviz/holoviews/pull/1664),
    [\#1796](https://github.com/pyviz/holoviews/pull/1796))


Version 1.8.2
=============
**August 4, 2017**

This bugfix release addresses a number of minor issues identified since
the 1.8.1 release:

Feature:

-   Added support for groupby to histogram operation.
    ([\#1725](https://github.com/pyviz/holoviews/pull/1725))

Fixes:

-   Fixed problem with HTML export due to new extension logos.
    ([\#1778](https://github.com/pyviz/holoviews/pull/1778))
-   Replaced deprecated ``__call__`` usage with opts method throughout codebase.
    ([\#1759](https://github.com/pyviz/holoviews/pull/1759),
    [\#1763](https://github.com/pyviz/holoviews/pull/1763),
    [\#1779](https://github.com/pyviz/holoviews/pull/1779))
-   Fixed pip installation.
    ([\#1782](https://github.com/pyviz/holoviews/pull/1782))
-   Fixed miscellaneous bugs
   ([\#1724](https://github.com/pyviz/holoviews/pull/1724),
   [\#1739](https://github.com/pyviz/holoviews/pull/1739),
   [\#1711](https://github.com/pyviz/holoviews/pull/1711))

Version 1.8.1
=============
**July 7, 2017**

This bugfix release addresses a number of minor issues identified since
the 1.8 release:

Feature:

-   All enabled plotting extension logos now shown
    ([\#1694](https://github.com/pyviz/holoviews/pull/1694))

Fixes:

-   Updated search ordering when looking for holoviews.rc
    ([\#1700](https://github.com/pyviz/holoviews/pull/1700))
-   Fixed lower bound inclusivity bug when no upper bound supplied
    ([\#1686](https://github.com/pyviz/holoviews/pull/1686))
-   Raise SkipRendering error when plotting nested layouts
    ([\#1687](https://github.com/pyviz/holoviews/pull/1687))
-   Added safety margin for grid axis constraint issue
    ([\#1695](https://github.com/pyviz/holoviews/pull/1685))
-   Fixed bug when using +framewise
    ([\#1685](https://github.com/pyviz/holoviews/pull/1685))
-   Fixed handling of Spacer models in sparse grid
    ([\#1682](https://github.com/pyviz/holoviews/pull/))
-   Renamed Bounds to BoundsXY for consistency
    ([\#1672](https://github.com/pyviz/holoviews/pull/1672))
-   Fixed Bokeh log axes with axis lower bound &lt;=0
    ([\#1691](https://github.com/pyviz/holoviews/pull/1691))
-   Set default datashader cmap to fire
    ([\#1697](https://github.com/pyviz/holoviews/pull/1697))
-   Set SpikesPlot color index to None by default
    ([\#1671](https://github.com/pyviz/holoviews/pull/1671))
-   Documentation fixes
    ([\#1662](https://github.com/pyviz/holoviews/pull/1662),
    [\#1665](https://github.com/pyviz/holoviews/pull/1665),
    [\#1690](https://github.com/pyviz/holoviews/pull/1690),
    [\#1692](https://github.com/pyviz/holoviews/pull/1692),
    [\#1658](https://github.com/pyviz/holoviews/pull/1658))

Version 1.8.0
=============
**June 29, 2017**

This release includes a complete and long awaited overhaul of the
HoloViews documentation and website, with a new gallery, getting-started
section, and logo. In the process, we have also improved and made small
fixes to all of the major new functionality that appeared in 1.7.0 but
was not properly documented until now. We want to thank all our old and
new contributors for providing feedback, bug reports, and pull requests.

Major features:

-   Completely overhauled the documentation and website
    ([\#1384](https://github.com/pyviz/holoviews/pull/1384),
    [\#1473](https://github.com/pyviz/holoviews/pull/1473),
    [\#1476](https://github.com/pyviz/holoviews/pull/1476),
    [\#1473](https://github.com/pyviz/holoviews/pull/1473),
    [\#1537](https://github.com/pyviz/holoviews/pull/1537),
    [\#1585](https://github.com/pyviz/holoviews/pull/1585),
    [\#1628](https://github.com/pyviz/holoviews/pull/1628),
    [\#1636](https://github.com/pyviz/holoviews/pull/1636))
-   Replaced dependency on bkcharts with new Bokeh bar plot
    ([\#1416](https://github.com/pyviz/holoviews/pull/1416)) and Bokeh
    BoxWhisker plot
    ([\#1604](https://github.com/pyviz/holoviews/pull/1604))
-   Added support for drawing the `Arrow` annotation in Bokeh
    ([\#1608](https://github.com/pyviz/holoviews/pull/1608))
-   Added periodic method DynamicMap to schedule recurring events
    ([\#1429](https://github.com/pyviz/holoviews/pull/1429))
-   Cleaned up the API for deploying to Bokeh server
    ([\#1444](https://github.com/pyviz/holoviews/pull/1444),
    [\#1469](https://github.com/pyviz/holoviews/pull/1469),
    [\#1486](https://github.com/pyviz/holoviews/pull/1486))
-   Validation of invalid backend specific options
    ([\#1465](https://github.com/pyviz/holoviews/pull/1465))
-   Added utilities and entry points to convert notebooks to scripts
    including magics
    ([\#1491](https://github.com/pyviz/holoviews/pull/1491))
-   Added support for rendering to png in Bokeh backend
    ([\#1493](https://github.com/pyviz/holoviews/pull/1493))
-   Made Matplotlib and Bokeh styling more consistent and dropped custom
    Matplotlib rc file
    ([\#1518](https://github.com/pyviz/holoviews/pull/1518))
-   Added `iloc` and `ndloc` method to allow integer based indexing on
    tabular and gridded datasets
    ([\#1435](https://github.com/pyviz/holoviews/pull/1435))
-   Added option to restore case sensitive completion order by setting
    `hv.extension.case_sensitive_completion=True` in python or via
    holoviews.rc file
    ([\#1613](https://github.com/pyviz/holoviews/pull/1613))

Other new features and improvements:

-   Optimized datashading of `NdOverlay`
    ([\#1430](https://github.com/pyviz/holoviews/pull/1430))
-   Expose last `DynamicMap` args and kwargs on Callable
    ([\#1453](https://github.com/pyviz/holoviews/pull/1453))
-   Allow colormapping `Contours` Element
    ([\#1499](https://github.com/pyviz/holoviews/pull/1499))
-   Add support for fixed ticks with labels in Bokeh backend
    ([\#1503](https://github.com/pyviz/holoviews/pull/1503))
-   Added a `clim` parameter to datashade controlling the color range
    ([\#1508](https://github.com/pyviz/holoviews/pull/1508))
-   Add support for wrapping xarray DataArrays containing Dask arrays
    ([\#1512](https://github.com/pyviz/holoviews/pull/1512))
-   Added support for aggregating to target `Image` dimensions in
    datashader `aggregate` operation
    ([\#1513](https://github.com/pyviz/holoviews/pull/1513))
-   Added top-level hv.extension and `hv.renderer` utilities
    ([\#1517](https://github.com/pyviz/holoviews/pull/1517))
-   Added support for `Splines` defining multiple cubic splines in Bokeh
    ([\#1529](https://github.com/pyviz/holoviews/pull/1529))
-   Add support for redim.label to quickly define dimension labels
    ([\#1541](https://github.com/pyviz/holoviews/pull/1541))
-   Add `BoundsX` and `BoundsY` streams
    ([\#1554](https://github.com/pyviz/holoviews/pull/1554))
-   Added support for adjoining empty plots
    ([\#1561](https://github.com/pyviz/holoviews/pull/1561))
-   Handle zero-values correctly when using `logz` colormapping option
    in Matplotlib
    ([\#1576](https://github.com/pyviz/holoviews/pull/1576))
-   Define a number of `Cycle` and `Palette` defaults across backends
    ([\#1605](https://github.com/pyviz/holoviews/pull/1605))
-   Many other small improvements and fixes
    ([\#1399](https://github.com/pyviz/holoviews/pull/1399),
    [\#1400](https://github.com/pyviz/holoviews/pull/1400),
    [\#1405](https://github.com/pyviz/holoviews/pull/1405),
    [\#1412](https://github.com/pyviz/holoviews/pull/1412),
    [\#1413](https://github.com/pyviz/holoviews/pull/1413),
    [\#1418](https://github.com/pyviz/holoviews/pull/1418),
    [\#1439](https://github.com/pyviz/holoviews/pull/1439),
    [\#1442](https://github.com/pyviz/holoviews/pull/1442),
    [\#1443](https://github.com/pyviz/holoviews/pull/1443),
    [\#1467](https://github.com/pyviz/holoviews/pull/1467),
    [\#1485](https://github.com/pyviz/holoviews/pull/1485),
    [\#1505](https://github.com/pyviz/holoviews/pull/1505),
    [\#1493](https://github.com/pyviz/holoviews/pull/1493),
    [\#1509](https://github.com/pyviz/holoviews/pull/1509),
    [\#1524](https://github.com/pyviz/holoviews/pull/1524),
    [\#1543](https://github.com/pyviz/holoviews/pull/1543),
    [\#1547](https://github.com/pyviz/holoviews/pull/1547),
    [\#1560](https://github.com/pyviz/holoviews/pull/1560),
    [\#1603](https://github.com/pyviz/holoviews/pull/1603))

Changes affecting backwards compatibility:

-   Renamed `ElementOperation` to `Operation`
    ([\#1421](https://github.com/pyviz/holoviews/pull/1421))
-   Removed `stack_area` operation in favor of `Area.stack` classmethod
    ([\#1515](https://github.com/pyviz/holoviews/pull/1515))
-   Removed all mpld3 support
    ([\#1516](https://github.com/pyviz/holoviews/pull/1516))
-   Added `opts` method on all types, replacing the now-deprecated
    `__call__` syntax to set options
    ([\#1589](https://github.com/pyviz/holoviews/pull/1589))
-   Styling changes for both Matplotlib and Bokeh, which can be reverted
    for a notebook with the `config` option of `hv.extension`. For
    instance, `hv.extension('bokeh', config=dict(style_17=True))`
    ([\#1518](https://github.com/pyviz/holoviews/pull/1518))

Version 1.7.0
=============
**April 25, 2017**

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
    [\#832](https://github.com/pyviz/holoviews/pull/832),
    [\#838](https://github.com/pyviz/holoviews/pull/838),
    [\#842](https://github.com/pyviz/holoviews/pull/842),
    [\#844](https://github.com/pyviz/holoviews/pull/844),
    [\#845](https://github.com/pyviz/holoviews/pull/845),
    [\#846](https://github.com/pyviz/holoviews/pull/846),
    [\#858](https://github.com/pyviz/holoviews/pull/858),
    [\#860](https://github.com/pyviz/holoviews/pull/860),
    [\#889](https://github.com/pyviz/holoviews/pull/889),
    [\#904](https://github.com/pyviz/holoviews/pull/904),
    [\#913](https://github.com/pyviz/holoviews/pull/913),
    [\#933](https://github.com/pyviz/holoviews/pull/933),
    [\#962](https://github.com/pyviz/holoviews/pull/962),
    [\#964](https://github.com/pyviz/holoviews/pull/964),
    [\#1094](https://github.com/pyviz/holoviews/pull/1094),
    [\#1256](https://github.com/pyviz/holoviews/pull/1256),
    [\#1274](https://github.com/pyviz/holoviews/pull/1274),
    [\#1297](https://github.com/pyviz/holoviews/pull/1297),
    [\#1301](https://github.com/pyviz/holoviews/pull/1301),
    [\#1303](https://github.com/pyviz/holoviews/pull/1303)).
-   Dynamic Callable API (PR
    [\#951](https://github.com/pyviz/holoviews/pull/951),
    [\#1103](https://github.com/pyviz/holoviews/pull/1103),
    [\#1029](https://github.com/pyviz/holoviews/pull/1029),
    [\#968](https://github.com/pyviz/holoviews/pull/968),
    [\#935](https://github.com/pyviz/holoviews/pull/935),
    [\#1063](https://github.com/pyviz/holoviews/pull/1063),
    [\#1260](https://github.com/pyviz/holoviews/pull/1260)).
-   Simpler and more powerful DynamicMap (PR
    [\#1238](https://github.com/pyviz/holoviews/pull/1238),
    [\#1240](https://github.com/pyviz/holoviews/pull/1240),
    [\#1243](https://github.com/pyviz/holoviews/pull/1243),
    [\#1257](https://github.com/pyviz/holoviews/pull/1257),
    [\#1267](https://github.com/pyviz/holoviews/pull/1267),
    [\#1302](https://github.com/pyviz/holoviews/pull/1302),
    [\#1304](https://github.com/pyviz/holoviews/pull/1304),
    [\#1305](https://github.com/pyviz/holoviews/pull/1305)).
-   Fully general support for Bokeh events (PR
    [\#892](https://github.com/pyviz/holoviews/pull/892),
    [\#1148](https://github.com/pyviz/holoviews/pull/1148),
    [\#1235](https://github.com/pyviz/holoviews/pull/1235)).
-   Datashader operations (PR
    [\#894](https://github.com/pyviz/holoviews/pull/894),
    [\#907](https://github.com/pyviz/holoviews/pull/907),
    [\#963](https://github.com/pyviz/holoviews/pull/963),
    [\#1125](https://github.com/pyviz/holoviews/pull/1125),
    [\#1281](https://github.com/pyviz/holoviews/pull/1281),
    [\#1306](https://github.com/pyviz/holoviews/pull/1306)).
-   Support for Bokeh apps and Bokeh Server (PR
    [\#959](https://github.com/pyviz/holoviews/pull/959),
    [\#1283](https://github.com/pyviz/holoviews/pull/1283)).
-   Working with renderers interactively outside the notebook (PR
    [\#1214](https://github.com/pyviz/holoviews/pull/1214)).
-   Support for Matplotlib 2.0 (PR
    [\#867](https://github.com/pyviz/holoviews/pull/867),
    [\#868](https://github.com/pyviz/holoviews/pull/868),
    [\#1131](https://github.com/pyviz/holoviews/pull/1131),
    [\#1264](https://github.com/pyviz/holoviews/pull/1264),
    [\#1266](https://github.com/pyviz/holoviews/pull/1266)).
-   Support for Bokeh 0.12.2, 0.12.3, 0.12.4, and 0.12.5 (PR
    [\#899](https://github.com/pyviz/holoviews/pull/899),
    [\#900](https://github.com/pyviz/holoviews/pull/900),
    [\#1007](https://github.com/pyviz/holoviews/pull/1007),
    [\#1036](https://github.com/pyviz/holoviews/pull/1036),
    [\#1116](https://github.com/pyviz/holoviews/pull/1116)).
-   Many new features for the Bokeh backend: widgets editable (PR
    [\#1247](https://github.com/pyviz/holoviews/pull/1247)), selection
    colors and interactive legends (PR
    [\#1220](https://github.com/pyviz/holoviews/pull/1220)), GridSpace
    axes (PR [\#1150](https://github.com/pyviz/holoviews/pull/1150)),
    categorical axes and colormapping (PR
    [\#1089](https://github.com/pyviz/holoviews/pull/1089),
    [\#1137](https://github.com/pyviz/holoviews/pull/1137)), computing
    plot size (PR
    [\#1140](https://github.com/pyviz/holoviews/pull/1140)), GridSpaces
    inside Layouts (PR
    [\#1104](https://github.com/pyviz/holoviews/pull/1104)), Layout/Grid
    titles (PR [\#1017](https://github.com/pyviz/holoviews/pull/1017)),
    histogram with live colormapping (PR
    [\#928](https://github.com/pyviz/holoviews/pull/928)), colorbars (PR
    [\#861](https://github.com/pyviz/holoviews/pull/861)),
    finalize\_hooks (PR
    [\#1040](https://github.com/pyviz/holoviews/pull/1040)), labelled and
    show\_frame options (PR
    [\#863](https://github.com/pyviz/holoviews/pull/863),
    [\#1013](https://github.com/pyviz/holoviews/pull/1013)), styling
    hover glyphs (PR
    [\#1286](https://github.com/pyviz/holoviews/pull/1286)), hiding
    legends on BarPlot (PR
    [\#837](https://github.com/pyviz/holoviews/pull/837)), VectorField
    plot (PR [\#1196](https://github.com/pyviz/holoviews/pull/1196)),
    Histograms now have same color cycle as mpl
    ([\#1008](https://github.com/pyviz/holoviews/pull/1008)).
-   Implemented convenience redim methods to easily set dimension
    ranges, values etc. (PR
    [\#1302](https://github.com/pyviz/holoviews/pull/1302))
-   Made methods on and operations applied to DynamicMap lazy
    ([\#422](https://github.com/pyviz/holoviews/pull/422),
    [\#588](https://github.com/pyviz/holoviews/pull/588),
    [\#1188](https://github.com/pyviz/holoviews/pull/1188),
    [\#1240](https://github.com/pyviz/holoviews/pull/1240),
    [\#1227](https://github.com/pyviz/holoviews/pull/1227))
-   Improved documentation (PR
    [\#936](https://github.com/pyviz/holoviews/pull/936),
    [\#1070](https://github.com/pyviz/holoviews/pull/1070),
    [\#1242](https://github.com/pyviz/holoviews/pull/1242),
    [\#1273](https://github.com/pyviz/holoviews/pull/1273),
    [\#1280](https://github.com/pyviz/holoviews/pull/1280)).
-   Improved error handling (PR
    [\#906](https://github.com/pyviz/holoviews/pull/906),
    [\#932](https://github.com/pyviz/holoviews/pull/932),
    [\#939](https://github.com/pyviz/holoviews/pull/939),
    [\#949](https://github.com/pyviz/holoviews/pull/949),
    [\#1011](https://github.com/pyviz/holoviews/pull/1011),
    [\#1290](https://github.com/pyviz/holoviews/pull/1290),
    [\#1262](https://github.com/pyviz/holoviews/pull/1262),
    [\#1295](https://github.com/pyviz/holoviews/pull/1295)), including
    re-enabling option system keyword validation (PR
    [\#1277](https://github.com/pyviz/holoviews/pull/1277)).
-   Improved testing (PR
    [\#834](https://github.com/pyviz/holoviews/pull/834),
    [\#871](https://github.com/pyviz/holoviews/pull/871),
    [\#881](https://github.com/pyviz/holoviews/pull/881),
    [\#941](https://github.com/pyviz/holoviews/pull/941),
    [\#1117](https://github.com/pyviz/holoviews/pull/1117),
    [\#1153](https://github.com/pyviz/holoviews/pull/1153),
    [\#1171](https://github.com/pyviz/holoviews/pull/1171),
    [\#1207](https://github.com/pyviz/holoviews/pull/1207),
    [\#1246](https://github.com/pyviz/holoviews/pull/1246),
    [\#1259](https://github.com/pyviz/holoviews/pull/1259),
    [\#1287](https://github.com/pyviz/holoviews/pull/1287)).

Other new features and improvements:

-   Operations for timeseries (PR
    [\#1172](https://github.com/pyviz/holoviews/pull/1172)),
    downsample\_columns (PR
    [\#903](https://github.com/pyviz/holoviews/pull/903)),
    interpolate\_curve (PR
    [\#1097](https://github.com/pyviz/holoviews/pull/1097)), and stacked
    area (PR [\#1193](https://github.com/pyviz/holoviews/pull/1193)).
-   Dataset types can be declared as empty by passing an empty list (PR
    [\#1355](https://github.com/pyviz/holoviews/pull/1355))
-   Plot or style options for Curve interpolation (PR
    [\#1097](https://github.com/pyviz/holoviews/pull/1097)), transposing
    layouts (PR [\#1100](https://github.com/pyviz/holoviews/pull/1100)),
    multiple paths (PR
    [\#997](https://github.com/pyviz/holoviews/pull/997)), and norm for
    ColorbarPlot (PR
    [\#957](https://github.com/pyviz/holoviews/pull/957)).
-   Improved options inheritance for more intuitive behavior (PR
    [\#1275](https://github.com/pyviz/holoviews/pull/1275)).
-   Image interface providing similar functionality for Image and
    non-Image types (making GridImage obsolete) (PR
    [\#994](https://github.com/pyviz/holoviews/pull/994)).
-   Dask data interface (PR
    [\#974](https://github.com/pyviz/holoviews/pull/974),
    [\#991](https://github.com/pyviz/holoviews/pull/991)).
-   xarray aggregate/reduce (PR
    [\#1192](https://github.com/pyviz/holoviews/pull/1192)).
-   Indicate color clipping and control clipping colors (PR
    [\#686](https://github.com/pyviz/holoviews/pull/686)).
-   Better datetime handling (PR
    [\#1098](https://github.com/pyviz/holoviews/pull/1098)).
-   Gridmatrix diagonal types (PR
    [\#1194](https://github.com/pyviz/holoviews/pull/1194),
    [\#1027](https://github.com/pyviz/holoviews/pull/1027)).
-   log option for histogram operation (PR
    [\#929](https://github.com/pyviz/holoviews/pull/929)).
-   Perceptually uniform fire colormap (PR
    [\#943](https://github.com/pyviz/holoviews/pull/943)).
-   Support for adjoining overlays (PR
    [\#1213](https://github.com/pyviz/holoviews/pull/1213)).
-   coloring weighted average in SideHistogram (PR
    [\#1087](https://github.com/pyviz/holoviews/pull/1087)).
-   HeatMap allows displaying multiple values on hover (PR
    [\#849](https://github.com/pyviz/holoviews/pull/849)).
-   Allow casting Image to QuadMesh (PR
    [\#1282](https://github.com/pyviz/holoviews/pull/1282)).
-   Unused columns are now preserved in gridded groupby (PR
    [\#1154](https://github.com/pyviz/holoviews/pull/1154)).
-   Optimizations and fixes for constructing Layout/Overlay types (PR
    [\#952](https://github.com/pyviz/holoviews/pull/952)).
-   DynamicMap fixes (PR
    [\#848](https://github.com/pyviz/holoviews/pull/848),
    [\#883](https://github.com/pyviz/holoviews/pull/883),
    [\#911](https://github.com/pyviz/holoviews/pull/911),
    [\#922](https://github.com/pyviz/holoviews/pull/922),
    [\#923](https://github.com/pyviz/holoviews/pull/923),
    [\#927](https://github.com/pyviz/holoviews/pull/927),
    [\#944](https://github.com/pyviz/holoviews/pull/944),
    [\#1170](https://github.com/pyviz/holoviews/pull/1170),
    [\#1227](https://github.com/pyviz/holoviews/pull/1227),
    [\#1270](https://github.com/pyviz/holoviews/pull/1270)).
-   Bokeh-backend fixes including handling of empty frames
    ([\#835](https://github.com/pyviz/holoviews/pull/835)), faster
    updates ([\#905](https://github.com/pyviz/holoviews/pull/905)), hover
    tool fixes ([\#1004](https://github.com/pyviz/holoviews/pull/1004),
    [\#1178](https://github.com/pyviz/holoviews/pull/1178),
    [\#1092](https://github.com/pyviz/holoviews/pull/1092),
    [\#1250](https://github.com/pyviz/holoviews/pull/1250)) and many more
    (PR [\#537](https://github.com/pyviz/holoviews/pull/537),
    [\#851](https://github.com/pyviz/holoviews/pull/851),
    [\#852](https://github.com/pyviz/holoviews/pull/852),
    [\#854](https://github.com/pyviz/holoviews/pull/854),
    [\#880](https://github.com/pyviz/holoviews/pull/880),
    [\#896](https://github.com/pyviz/holoviews/pull/896),
    [\#898](https://github.com/pyviz/holoviews/pull/898),
    [\#921](https://github.com/pyviz/holoviews/pull/921),
    [\#934](https://github.com/pyviz/holoviews/pull/934),
    [\#1004](https://github.com/pyviz/holoviews/pull/1004),
    [\#1010](https://github.com/pyviz/holoviews/pull/1010),
    [\#1014](https://github.com/pyviz/holoviews/pull/1014),
    [\#1030](https://github.com/pyviz/holoviews/pull/1030),
    [\#1069](https://github.com/pyviz/holoviews/pull/1069),
    [\#1072](https://github.com/pyviz/holoviews/pull/1072),
    [\#1085](https://github.com/pyviz/holoviews/pull/1085),
    [\#1157](https://github.com/pyviz/holoviews/pull/1157),
    [\#1086](https://github.com/pyviz/holoviews/pull/1086),
    [\#1169](https://github.com/pyviz/holoviews/pull/1169),
    [\#1195](https://github.com/pyviz/holoviews/pull/1195),
    [\#1263](https://github.com/pyviz/holoviews/pull/1263)).
-   Matplotlib-backend fixes and improvements (PR
    [\#864](https://github.com/pyviz/holoviews/pull/864),
    [\#873](https://github.com/pyviz/holoviews/pull/873),
    [\#954](https://github.com/pyviz/holoviews/pull/954),
    [\#1037](https://github.com/pyviz/holoviews/pull/1037),
    [\#1068](https://github.com/pyviz/holoviews/pull/1068),
    [\#1128](https://github.com/pyviz/holoviews/pull/1128),
    [\#1132](https://github.com/pyviz/holoviews/pull/1132),
    [\#1143](https://github.com/pyviz/holoviews/pull/1143),
    [\#1163](https://github.com/pyviz/holoviews/pull/1163),
    [\#1209](https://github.com/pyviz/holoviews/pull/1209),
    [\#1211](https://github.com/pyviz/holoviews/pull/1211),
    [\#1225](https://github.com/pyviz/holoviews/pull/1225),
    [\#1269](https://github.com/pyviz/holoviews/pull/1269),
    [\#1300](https://github.com/pyviz/holoviews/pull/1300)).
-   Many other small improvements and fixes (PR
    [\#830](https://github.com/pyviz/holoviews/pull/830),
    [\#840](https://github.com/pyviz/holoviews/pull/840),
    [\#841](https://github.com/pyviz/holoviews/pull/841),
    [\#850](https://github.com/pyviz/holoviews/pull/850),
    [\#855](https://github.com/pyviz/holoviews/pull/855),
    [\#856](https://github.com/pyviz/holoviews/pull/856),
    [\#859](https://github.com/pyviz/holoviews/pull/859),
    [\#865](https://github.com/pyviz/holoviews/pull/865),
    [\#893](https://github.com/pyviz/holoviews/pull/893),
    [\#897](https://github.com/pyviz/holoviews/pull/897),
    [\#902](https://github.com/pyviz/holoviews/pull/902),
    [\#912](https://github.com/pyviz/holoviews/pull/912),
    [\#916](https://github.com/pyviz/holoviews/pull/916),
    [\#925](https://github.com/pyviz/holoviews/pull/925),
    [\#938](https://github.com/pyviz/holoviews/pull/938),
    [\#940](https://github.com/pyviz/holoviews/pull/940),
    [\#948](https://github.com/pyviz/holoviews/pull/948),
    [\#950](https://github.com/pyviz/holoviews/pull/950),
    [\#955](https://github.com/pyviz/holoviews/pull/955),
    [\#956](https://github.com/pyviz/holoviews/pull/956),
    [\#967](https://github.com/pyviz/holoviews/pull/967),
    [\#970](https://github.com/pyviz/holoviews/pull/970),
    [\#972](https://github.com/pyviz/holoviews/pull/972),
    [\#973](https://github.com/pyviz/holoviews/pull/973),
    [\#981](https://github.com/pyviz/holoviews/pull/981),
    [\#992](https://github.com/pyviz/holoviews/pull/992),
    [\#998](https://github.com/pyviz/holoviews/pull/998),
    [\#1009](https://github.com/pyviz/holoviews/pull/1009),
    [\#1012](https://github.com/pyviz/holoviews/pull/1012),
    [\#1016](https://github.com/pyviz/holoviews/pull/1016),
    [\#1023](https://github.com/pyviz/holoviews/pull/1023),
    [\#1034](https://github.com/pyviz/holoviews/pull/1034),
    [\#1043](https://github.com/pyviz/holoviews/pull/1043),
    [\#1045](https://github.com/pyviz/holoviews/pull/1045),
    [\#1046](https://github.com/pyviz/holoviews/pull/1046),
    [\#1048](https://github.com/pyviz/holoviews/pull/1048),
    [\#1050](https://github.com/pyviz/holoviews/pull/1050),
    [\#1051](https://github.com/pyviz/holoviews/pull/1051),
    [\#1054](https://github.com/pyviz/holoviews/pull/1054),
    [\#1060](https://github.com/pyviz/holoviews/pull/1060),
    [\#1062](https://github.com/pyviz/holoviews/pull/1062),
    [\#1074](https://github.com/pyviz/holoviews/pull/1074),
    [\#1082](https://github.com/pyviz/holoviews/pull/1082),
    [\#1084](https://github.com/pyviz/holoviews/pull/1084),
    [\#1088](https://github.com/pyviz/holoviews/pull/1088),
    [\#1093](https://github.com/pyviz/holoviews/pull/1093),
    [\#1099](https://github.com/pyviz/holoviews/pull/1099),
    [\#1115](https://github.com/pyviz/holoviews/pull/1115),
    [\#1119](https://github.com/pyviz/holoviews/pull/1119),
    [\#1121](https://github.com/pyviz/holoviews/pull/1121),
    [\#1130](https://github.com/pyviz/holoviews/pull/1130),
    [\#1133](https://github.com/pyviz/holoviews/pull/1133),
    [\#1151](https://github.com/pyviz/holoviews/pull/1151),
    [\#1152](https://github.com/pyviz/holoviews/pull/1152),
    [\#1155](https://github.com/pyviz/holoviews/pull/1155),
    [\#1156](https://github.com/pyviz/holoviews/pull/1156),
    [\#1158](https://github.com/pyviz/holoviews/pull/1158),
    [\#1162](https://github.com/pyviz/holoviews/pull/1162),
    [\#1164](https://github.com/pyviz/holoviews/pull/1164),
    [\#1174](https://github.com/pyviz/holoviews/pull/1174),
    [\#1175](https://github.com/pyviz/holoviews/pull/1175),
    [\#1180](https://github.com/pyviz/holoviews/pull/1180),
    [\#1187](https://github.com/pyviz/holoviews/pull/1187),
    [\#1197](https://github.com/pyviz/holoviews/pull/1197),
    [\#1202](https://github.com/pyviz/holoviews/pull/1202),
    [\#1205](https://github.com/pyviz/holoviews/pull/1205),
    [\#1206](https://github.com/pyviz/holoviews/pull/1206),
    [\#1210](https://github.com/pyviz/holoviews/pull/1210),
    [\#1217](https://github.com/pyviz/holoviews/pull/1217),
    [\#1219](https://github.com/pyviz/holoviews/pull/1219),
    [\#1228](https://github.com/pyviz/holoviews/pull/1228),
    [\#1232](https://github.com/pyviz/holoviews/pull/1232),
    [\#1241](https://github.com/pyviz/holoviews/pull/1241),
    [\#1244](https://github.com/pyviz/holoviews/pull/1244),
    [\#1245](https://github.com/pyviz/holoviews/pull/1245),
    [\#1249](https://github.com/pyviz/holoviews/pull/1249),
    [\#1254](https://github.com/pyviz/holoviews/pull/1254),
    [\#1255](https://github.com/pyviz/holoviews/pull/1255),
    [\#1271](https://github.com/pyviz/holoviews/pull/1271),
    [\#1276](https://github.com/pyviz/holoviews/pull/1276),
    [\#1278](https://github.com/pyviz/holoviews/pull/1278),
    [\#1285](https://github.com/pyviz/holoviews/pull/1285),
    [\#1288](https://github.com/pyviz/holoviews/pull/1288),
    [\#1289](https://github.com/pyviz/holoviews/pull/1289)).

Changes affecting backwards compatibility:

-   Automatic coloring and sizing on Points now disabled (PR
    [\#748](https://github.com/pyviz/holoviews/pull/748)).
-   Deprecated max\_branches output magic option (PR
    [\#1293](https://github.com/pyviz/holoviews/pull/1293)).
-   Deprecated GridImage (PR
    [\#1292](https://github.com/pyviz/holoviews/pull/1292),
    [\#1223](https://github.com/pyviz/holoviews/pull/1223)).
-   Deprecated NdElement (PR
    [\#1191](https://github.com/pyviz/holoviews/pull/1191)).
-   Deprecated DFrame conversion methods (PR
    [\#1065](https://github.com/pyviz/holoviews/pull/1065)).
-   Banner text removed from notebook\_extension() (PR
    [\#1231](https://github.com/pyviz/holoviews/pull/1231),
    [\#1291](https://github.com/pyviz/holoviews/pull/1291)).
-   Bokeh's Matplotlib compatibility module removed (PR
    [\#1239](https://github.com/pyviz/holoviews/pull/1239)).
-   ls as Matplotlib linestyle alias dropped (PR
    [\#1203](https://github.com/pyviz/holoviews/pull/1203)).
-   mdims argument of conversion interface renamed to groupby (PR
    [\#1066](https://github.com/pyviz/holoviews/pull/1066)).
-   Replaced global alias state with Dimension.label
    ([\#1083](https://github.com/pyviz/holoviews/pull/1083)).
-   DynamicMap only update ranges when set to framewise
-   Deprecated DynamicMap sampled, bounded, open and generator modes
    ([\#969](https://github.com/pyviz/holoviews/pull/969),
    [\#1305](https://github.com/pyviz/holoviews/pull/1305))
-   Layout.display method is now deprecated
    ([\#1026](https://github.com/pyviz/holoviews/pull/1026))
-   Layout fix for Matplotlib figures with non-square aspects introduced
    in 1.6.2 (PR [\#826](https://github.com/pyviz/holoviews/pull/826)),
    now enabled by default.

Version 1.6.2
=============
**August 23, 2016**

Bug fix release with various fixes for gridded data backends and
optimizations for Bokeh.

-   Optimized Bokeh event messaging, reducing the average json payload
    by 30-50% (PR [\#807](https://github.com/pyviz/holoviews/pull/807)).
-   Fixes for correctly handling NdOverlay types returned by DynamicMaps
    (PR [\#814](https://github.com/pyviz/holoviews/pull/814)).
-   Added support for datetime64 handling in Matplotlib and support for
    datetime formatters on Dimension.type\_formatters (PR
    [\#816](https://github.com/pyviz/holoviews/pull/816)).
-   Fixed handling of constant dimensions when slicing xarray datasets
    (PR [\#817](https://github.com/pyviz/holoviews/pull/817)).
-   Fixed support for passing custom dimensions to iris Datasets (PR
    [\#818](https://github.com/pyviz/holoviews/pull/818)).
-   Fixed support for add\_dimension on xarray interface (PR
    [\#820](https://github.com/pyviz/holoviews/pull/820)).
-   Improved extents computation on Matplotlib SpreadPlot (PR
    [\#821](https://github.com/pyviz/holoviews/pull/821)).
-   Bokeh backend avoids sending data for static frames and empty events
    (PR [\#822](https://github.com/pyviz/holoviews/pull/822)).
-   Added major layout fix for figures with non-square aspects, reducing
    the amount of unnecessary whitespace (PR
    [\#826](https://github.com/pyviz/holoviews/pull/826)). Disabled by
    default until 1.7 release but can be enabled with:

``` {.sourceCode .python}
from holoviews.plotting.mpl import LayoutPlot
LayoutPlot.v17_layout_format = True
LayoutPlot.vspace = 0.3
```

Version 1.6.1
=============
**July 27, 2016**

Bug fix release following the 1.6 major release with major bug fixes for
the grid data interfaces and improvements to the options system.

-   Ensured that style options incompatible with active backend are
    ignored (PR [\#802](https://github.com/pyviz/holoviews/pull/802)).
-   Added support for placing legends outside the plot area in Bokeh (PR
    [\#801](https://github.com/pyviz/holoviews/pull/801)).
-   Fix to ensure Bokeh backend does not depend on pandas (PR
    [\#792](https://github.com/pyviz/holoviews/pull/792)).
-   Fixed option system to ensure correct inheritance when redefining
    options (PR [\#796](https://github.com/pyviz/holoviews/pull/796)).
-   Major refactor and fixes for the grid based data backends (iris,
    xarray and arrays with coordinates) ensuring the data is oriented
    and transposed correctly (PR
    [\#794](https://github.com/pyviz/holoviews/pull/794)).

Version 1.6
===========
**July 14, 2016**

A major release with an optional new data interface based on xarray,
support for batching Bokeh plots for huge increases in performance,
support for Bokeh 0.12 and various other fixes and improvements.

Features and improvements:

-   Made VectorFieldPlot more general with support for independent
    coloring and scaling (PR
    [\#701](https://github.com/pyviz/holoviews/pull/701)).
-   Iris interface now allows tuple and dict formats in the constructor
    (PR [\#709](https://github.com/pyviz/holoviews/pull/709).
-   Added support for dynamic groupby on all data interfaces (PR
    [\#711](https://github.com/pyviz/holoviews/pull/711)).
-   Added an xarray data interface (PR
    [\#713](https://github.com/pyviz/holoviews/pull/713)).
-   Added the redim method to all Dimensioned objects making it easy to
    quickly change dimension names and attributes on nested objects
    [\#715](https://github.com/pyviz/holoviews/pull/715)).
-   Added support for batching plots (PR
    [\#715](https://github.com/pyviz/holoviews/pull/717)).
-   Support for Bokeh 0.12 release (PR
    [\#725](https://github.com/pyviz/holoviews/pull/725)).
-   Added support for logz option on Bokeh Raster plots (PR
    [\#729](https://github.com/pyviz/holoviews/pull/729)).
-   Bokeh plots now support custom tick formatters specified via
    Dimension value\_format (PR
    [\#728](https://github.com/pyviz/holoviews/pull/728)).

Version 1.5
===========
**May 12, 2016**

A major release with a large number of new features including new data
interfaces for grid based data, major improvements for DynamicMaps and a
large number of bug fixes.

Features and improvements:

-   Added a grid based data interface to explore n-dimensional gridded
    data easily (PR
    [\#562](https://github.com/pyviz/holoviews/pull/542)).
-   Added data interface based on [iris
    Cubes](http://scitools.org.uk/iris/docs/v1.9.2/index.html) (PR
    [\#624](https://github.com/pyviz/holoviews/pull/624)).
-   Added support for dynamic operations and overlaying of DynamicMaps
    (PR [\#588](https://github.com/pyviz/holoviews/pull/588)).
-   Added support for applying groupby operations to DynamicMaps (PR
    [\#667](https://github.com/pyviz/holoviews/pull/667)).
-   Added dimension value formatting in widgets (PR
    [\#562](https://github.com/pyviz/holoviews/issues/562)).
-   Added support for indexing and slicing with a function (PR
    [\#619](https://github.com/pyviz/holoviews/pull/619)).
-   Improved throttling behavior on widgets (PR
    [\#596](https://github.com/pyviz/holoviews/pull/596)).
-   Major refactor of Matplotlib plotting classes to simplify
    implementing new Element plots (PR
    [\#438](https://github.com/pyviz/holoviews/pull/438)).
-   Added Renderer.last\_plot attribute to allow easily debugging or
    modifying the last displayed plot (PR
    [\#538](https://github.com/pyviz/holoviews/pull/538)).
-   Added Bokeh QuadMeshPlot (PR
    [\#661](https://github.com/pyviz/holoviews/pull/661)).

Bug fixes:

-   Fixed overlaying of 3D Element types (PR
    [\#504](https://github.com/pyviz/holoviews/pull/504)).
-   Fix for Bokeh hovertools with dimensions with special characters (PR
    [\#524](https://github.com/pyviz/holoviews/pull/524)).
-   Fixed bugs in seaborn Distribution Element (PR
    [\#630](https://github.com/pyviz/holoviews/pull/630)).
-   Fix for inverted Raster.reduce method (PR
    [\#672](https://github.com/pyviz/holoviews/pull/672)).
-   Fixed Store.add\_style\_opts method (PR
    [\#587](https://github.com/pyviz/holoviews/pull/587)).
-   Fixed bug preventing simultaneous logx and logy plot options (PR
    [\#554](https://github.com/pyviz/holoviews/pull/554)).

Backwards compatibility:

-   Renamed `Columns` type to `Dataset` (PR
    [\#620](https://github.com/pyviz/holoviews/issues/620)).

Version 1.4.3
=============
**February 11, 2016**

A minor bugfix release to patch a number of small but important issues.

Fixes and improvements:

-   Added a [DynamicMap
    Tutorial](http://holoviews.org/Tutorials/Dynamic_Map.html) to
    explain how to explore very large or continuous parameter spaces in
    HoloViews ([PR
    \#470](https://github.com/pyviz/holoviews/issues/470)).
-   Various fixes and improvements for DynamicMaps including slicing
    ([PR \#488](https://github.com/pyviz/holoviews/issues/488)) and
    validation ([PR
    \#483](https://github.com/pyviz/holoviews/issues/478)) and
    serialization ([PR
    \#483](https://github.com/pyviz/holoviews/issues/478))
-   Widgets containing Matplotlib plots now display the first frame from
    cache providing at least the initial frame when exporting
    DynamicMaps ([PR
    \#486](https://github.com/pyviz/holoviews/issues/483))
-   Fixed plotting Bokeh plots using widgets in live mode, after changes
    introduced in latest Bokeh version (commit
    [1b87c91e9](https://github.com/pyviz/holoviews/commit/1b87c91e9e7cf35b267344ccd4a2fa91dd052890)).
-   Fixed issue in coloring Point/Scatter objects by values ([Issue
    \#467](https://github.com/pyviz/holoviews/issues/467)).

Backwards compatibility:

-   The behavior of the `scaling_factor` on Point and Scatter plots has
    changed now simply multiplying `area` or `width` (as defined by the
    `scaling_method`). To disable scaling points by a dimension set
    `size_index=None`.
-   Removed hooks to display 3D Elements using the `BokehMPLRawWrapper`
    in Bokeh ([PR \#477](https://github.com/pyviz/holoviews/pull/477))
-   Renamed the DynamicMap mode `closed` to `bounded` ([PR
    \#477](https://github.com/pyviz/holoviews/pull/485))

Version 1.4.2
=============
**February 7, 2016**

Over the past month since the 1.4.1 release, we have improved our
infrastructure for building documentation, updated the main website and
made several additional usability improvements.

Documentation changes:

-   Major overhaul of website and notebook building making it much
    easier to test user contributions ([Issue
    \#180](https://github.com/pyviz/holoviews/issues/180), [PR
    \#429](https://github.com/pyviz/holoviews/pull/429))
-   Major rewrite of the documentation ([PR
    \#401](https://github.com/pyviz/holoviews/pull/401), [PR
    \#411](https://github.com/pyviz/holoviews/pull/411))
-   Added Columnar Data Tutorial and removed most of Pandas Conversions
    as it is now supported by the core.

Fixes and improvements:

-   Major improvement for grid based layouts with varying aspects ([PR
    \#457](https://github.com/pyviz/holoviews/pull/457))
-   Fix for interleaving %matplotline inline and holoviews plots ([Issue
    \#179](https://github.com/pyviz/holoviews/issues/179))
-   Matplotlib legend z-orders and updating fixed ([Issue
    \#304](https://github.com/pyviz/holoviews/issues/304), [Issue
    \#305](https://github.com/pyviz/holoviews/issues/305))
-   `color_index` and `size_index` plot options support specifying
    dimension by name ([Issue
    \#391](https://github.com/pyviz/holoviews/issues/391))
-   Added `Area` Element type for drawing area under or between Curves.
    ([PR \#427](https://github.com/pyviz/holoviews/pull/427))
-   Fixed issues where slicing would remove styles applied to
    an Element. ([Issue
    \#423](https://github.com/pyviz/holoviews/issues/423), [PR
    \#439](https://github.com/pyviz/holoviews/pull/439))
-   Updated the `title_format` plot option to support a `{dimensions}`
    formatter ([PR \#436](https://github.com/pyviz/holoviews/pull/436))
-   Improvements to Renderer API to allow JS and CSS requirements for
    exporting standalone widgets ([PR
    \#426](https://github.com/pyviz/holoviews/pull/426))
-   Compatibility with the latest Bokeh 0.11 release ([PR
    \#393](https://github.com/pyviz/holoviews/pull/393))

Version 1.4.1
=============
**December 22, 2015**

Over the past two weeks since the 1.4 release, we have implemented
several important bug fixes and have made several usability
improvements.

New features:

-   Improved help system. It is now possible to recursively list all the
    applicable documentation for a composite object. In addition, the
    documentation may now be filtered using a regular
    expression pattern. ([PR
    \#370](https://github.com/pyviz/holoviews/pull/370))
-   HoloViews now supports multiple active display hooks making it
    easier to use nbconvert. For instance, PNG data will be embedded in
    the notebook if the argument display\_formats=\['html','png'\] is
    supplied to the notebook\_extension. ([PR
    \#355](https://github.com/pyviz/holoviews/pull/355))
-   Improvements to the display of DynamicMaps as well as many new
    improvements to the Bokeh backend including better VLines/HLines and
    support for the Bars element. ([PR
    \#367](https://github.com/pyviz/holoviews/pull/367) , [PR
    \#362](https://github.com/pyviz/holoviews/pull/362), [PR
    \#339](https://github.com/pyviz/holoviews/pull/339)).
-   New Spikes and BoxWhisker elements suitable for representing
    distributions as a sequence of lines or as a box-and-whisker plot.
    ([PR \#346](https://github.com/pyviz/holoviews/pull/346), [PR
    \#339](https://github.com/pyviz/holoviews/pull/339))
-   Improvements to the notebook\_extension. For instance,
    executing hv.notebook\_extension('bokeh') will now load BokehJS and
    automatically activate the Bokeh backend (if available).
-   Significant performance improvements when using the groupby
    operation on HoloMaps and when working with highly
    nested datastructures. ([PR
    \#349](https://github.com/pyviz/holoviews/pull/349), [PR
    \#359](https://github.com/pyviz/holoviews/pull/359))

Notable bug fixes:

-   DynamicMaps are now properly integrated into the style system and
    can be customized in the same way as HoloMaps. ([PR
    \#368](https://github.com/pyviz/holoviews/pull/368))
-   Widgets now work correctly when unicode is used in the dimension
    labels and values ([PR
    \#376](https://github.com/pyviz/holoviews/pull/376)).

Version 1.4.0
=============
**December 4, 2015**

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
    \#284](https://github.com/pyviz/holoviews/pull/284))
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
    \#250](https://github.com/pyviz/holoviews/pull/250))
-   DynamicMap: The new DynamicMap class allows HoloMap data to be
    generated on-the-fly while running a Jupyter IPython
    notebook kernel. Allows visualization of unbounded data streams and
    smooth exploration of large continuous parameter spaces. ([PR
    \#278](https://github.com/pyviz/holoviews/pull/278))

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
    BokehJS tools and the IPython kernel, e.g. allowing downsampling
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
**July 6, 2015**

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
**July 1, 2015**

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
**June 27, 2015**

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
-   Defaulting to Matplotlib ticking behavior (62e1e58).
-   Allowed specifying fixed figure sizes to Matplotlib via `fig_inches`
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
    positions (and labels), and a Matplotlib tick locator.
-   Changed backend options in %output magic, `nbagg` and `d3` are now
    modes of the Matplotlib backend and can be selected with
    `backend='matplotlib:nbagg'` and
    `backend='matplotlib:mpld3'` respectively. The 'd3' and 'nbagg'
    options remain supported but will be deprecated in future.
-   Customizations should no longer be applied directly to
    `Store.options`; the `Store.options(backend='matplotlib')` object
    should be customized instead. There is no longer a need to call the
    deprecated `Store.register_plots` method.

Version 1.2.0
=============
**May 27, 2015**

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
**April 15, 2015**

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
**March 26, 2015**

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
**March 16, 2015**

First public release available on GitHub and PyPI.
