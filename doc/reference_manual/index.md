# API Reference Manual

To learn how to use HoloViews, we recommend that people start with the
[Getting Started guide], and then use the online help features from within
IPython Notebook to learn more. Each option should have corresponding
tab-completion and help provided, and this is usually enough to
determine how to use the HoloViews components.

Even so, for a truly comprehensive reference to every object in
HoloViews, we provide this API documentation. The API reference manual
is generated directly from documentation and declarations in the
source code, and is thus often much more verbose than necessary,
because many little-used yet often-duplicated methods are listed for
each class. Still, the reference for a given component does provide a
comprehensive listing of all attributes and methods, inherited or
otherwise, which is can be difficult to obtain from the source code
directly.

______________________________________________________________________

## Module Structure

### HoloViews subpackages

[annotators]

: Helper functions and classes to annotate visual elements.

[core]

: Base classes implementing the core data structures of HoloViews.

[core.data]

: Data Interface classes allowing HoloViews to work with different types of data.

[element]

: Elements that form the basis of more complex visualizations.

[ipython]

: Interface to IPython notebook, including magics and display hooks.

[operation]

: Operations applied to transform existing Elements or other data structures.

[plotting]

: Base plotting classes and utilities.

[plotting.bokeh]

: Bokeh plotting classes and utilities.

[plotting.mpl]

: Matplotlib plotting classes and utilities.

[plotting.plotly]

: Plotly plotting classes and utilities.

[selection]

: Helper functions to apply linked brushing and selections

[streams]

: Stream classes to provide interactivity for DynamicMap

[util]

: High-level utilities

```{toctree}
:hidden: true
:maxdepth: 2

annotators <holoviews.annotators>
core <holoviews.core>
core.data <holoviews.core.data>
element <holoviews.element>
ipython <holoviews.ipython>
operation <holoviews.operation>
plotting <holoviews.plotting>
plotting.bokeh <holoviews.plotting.bokeh>
plotting.matplotlib <holoviews.plotting.mpl>
plotting.plotly <holoviews.plotting.plotly>
selection <holoviews.selection>
streams <holoviews.streams>
util <holoviews.util>
```

[annotators]: holoviews.annotators.html
[core]: holoviews.core.html
[core.data]: holoviews.core.data.html
[element]: holoviews.element.html
[getting started guide]: ../getting_started/index.html
[ipython]: holoviews.ipython.html
[operation]: holoviews.operation.html
[plotting]: holoviews.plotting.html
[plotting.bokeh]: holoviews.plotting.bokeh.html
[plotting.mpl]: holoviews.plotting.mpl.html
[plotting.plotly]: holoviews.plotting.plotly.html
[selection]: holoviews.selection.html
[streams]: holoviews.streams.html
[user manual]: ../User_Manual/index.html
[util]: holoviews.util.html
