# User Guide

The User Guide is the primary resource documenting key concepts that
will help you use HoloViews in your work. For newcomers, a gentle
introduction to HoloViews can be found in our [Getting Started] guide and an overview of some interesting HoloViews examples can be
found in our [Gallery]. If you are looking for a specific component
(or wish to view the available range of primitives), see our
[Reference Gallery].

## Core guides

These user guides provide detailed explanation of some of the core
concepts in HoloViews:

[Annotating your Data]

: Core concepts when annotating your data with semantic metadata.

[Composing Elements]

: Composing your data into layouts and overlays with the `+` and `*` operators.

[Applying Customizations]

: Using the options system to declare customizations.

[Style Mapping]

: Mapping your data to the visual attributes of your plot.

[Dimensioned Containers]

: Multi-dimensional containers for animating and faceting your data flexibly.

[Building Composite Objects]

: How to build and work with nested composite objects.

[Live Data]

: Lazily generate data on the fly and generate engaging interactive visualizations.

[Tabular Datasets]

: Explore tabular data with [NumPy](https://www.numpy.org/), [pandas](https://pandas.pydata.org/) and [dask](https://www.dask.org/).

[Gridded Datasets]

: Explore gridded data (n-dimensional arrays) with [NumPy](https://www.numpy.org/) and [XArray](https://xarray.dev/).

[Geometry Data]

: Represent and visualize path and polygon geometries with support for multi-geometries and value dimensions.

[Indexing and Selecting Data]

: Select and index subsets of your data with HoloViews.

[Transforming Elements]

: Apply operations to your data that can be used to build data analysis pipelines

[Responding to Events]

: Allow your visualizations to respond to Python events using the 'streams' system.

[Custom Interactivity]

: Use [Bokeh](https://bokeh.org) and 'linked streams' to directly interact with your visualizations.

[Data Processing Pipelines]

: Chain operations to build sophisticated, interactive and lazy data analysis pipelines.

[Working with large data]

: Leverage Datashader to interactively explore millions or billions of datapoints.

[Interactive Hover for Big Data]

: Use the `selector` with Datashader to enable fast, interactive hover tooltips that
reveal individual data points without sacrificing aggregation.

[Working with Streaming Data]

: Demonstrates how to leverage streaming with HoloViews.

[Creating interactive dashboards]

: Use external widget libraries to build custom, interactive dashboards.

## Supplementary guides

These guides provide detail about specific additional features in HoloViews:

[Configuring HoloViews]

: Information about configuration options.

[Customizing Plots]

: How to customize plots including their titles, axis labels, ranges, ticks and more.

[Colormaps]

: Overview of colormaps available, including when and how to use each type.

[Plotting with Bokeh]

: Styling options and unique [Bokeh](https://bokeh.org) features such as plot tools and using bokeh models directly.

[Deploying Bokeh Apps]

: Using [bokeh server](https://docs.bokeh.org/en/latest/docs/user_guide/server.html) using scripts and notebooks.

[Linking Bokeh plots]

: Using Links to define custom interactions on a plot without a Python server

[Plotting with matplotlib]

: Styling options and unique Matplotlib features such as GIF/MP4 support.

[Working with renderers and plots]

: Using the `Renderer` and `Plot` classes for access to the plotting machinery.

[Using linked brushing to cross-filter complex datasets]

: Explains how to use the `link_selections` helper to cross-filter multiple elements.

[Using Annotators to edit and label data]

: Explains how to use the `annotate` helper to edit and annotate elements with the help of drawing tools and editable tables.

[Exporting and Archiving]

: Archive both your data and visualization in scripts and notebooks.

[Continuous Coordinates]

: How continuous coordinates are handled, specifically focusing on `Image` and `Raster` types.

[Interactive Hover for Big Data]

: Explains how to use interactive hover tools with large datasets.

[Notebook Magics]

: IPython magics supported in Jupyter Notebooks.

```{toctree}
:hidden: true
:maxdepth: 2
:titlesonly: true

 Annotating your Data <Annotating_Data>
 Composing Elements <Composing_Elements>
 Applying Customizations <Applying_Customizations>
 Style Mapping <Style_Mapping>
 Dimensioned Containers <Dimensioned_Containers>
 Building Composite Objects <Building_Composite_Objects>
 Live Data <Live_Data>
 Tabular Datasets <Tabular_Datasets>
 Gridded Datasets <Gridded_Datasets>
 Geometry Data <Geometry_Data>
 Indexing and Selecting Data <Indexing_and_Selecting_Data>
 Transforming Elements <Transforming_Elements>
 Responding to Events <Responding_to_Events>
 Custom Interactivity <Custom_Interactivity>
 Data Processing Pipelines <Data_Pipelines>
 Creating interactive network graphs <Network_Graphs>
 Working with large data <Large_Data>
 Interactive Hover for Big Data <Interactive_Hover_for_Big_Data>
 Working with streaming data <Streaming_Data>
 Creating interactive dashboards <Dashboards>
 Configuring HoloViews <Configuring>
 Customizing Plots <Customizing_Plots>
 Colormaps <Colormaps>
 Plotting with Bokeh <Plotting_with_Bokeh>
 Deploying Bokeh Apps <Deploying_Bokeh_Apps>
 Linking Bokeh plots <Linking_Plots>
 Plotting with matplotlib <Plotting_with_Matplotlib>
 Working with Plot and Renderers <Plots_and_Renderers>
 Linked Brushing <Linked_Brushing>
 Annotators <Annotators>
 Exporting and Archiving <Exporting_and_Archiving>
 Continuous Coordinates <Continuous_Coordinates>
 Notebook Magics <Notebook_Magics>
```

[annotating your data]: Annotating_Data.html
[applying customizations]: Applying_Customizations.html
[building composite objects]: Building_Composite_Objects.html
[colormaps]: Colormaps.html
[composing elements]: Composing_Elements.html
[configuring holoviews]: Configuring.html
[continuous coordinates]: Continuous_Coordinates.html
[creating interactive dashboards]: Dashboards.html
[custom interactivity]: Custom_Interactivity.html
[customizing plots]: Customizing_Plots.html
[data processing pipelines]: Data_Pipelines.html
[deploying bokeh apps]: Deploying_Bokeh_Apps.html
[dimensioned containers]: Dimensioned_Containers.html
[exporting and archiving]: Exporting_and_Archiving.html
[gallery]: ../gallery/index.html
[geometry data]: Geometry_Data.html
[getting started]: ../getting_started/index.html
[gridded datasets]: Gridded_Datasets.html
[indexing and selecting data]: Indexing_and_Selecting_Data.html
[interactive hover for big data]: Interactive_Hover_for_Big_Data.html
[linking bokeh plots]: Linking_Plots.html
[live data]: Live_Data.html
[notebook magics]: Notebook_Magics.html
[plotting with bokeh]: Plotting_with_Bokeh.html
[plotting with matplotlib]: Plotting_with_Matplotlib.html
[reference gallery]: ../reference/index.html
[responding to events]: Responding_to_Events.html
[style mapping]: Style_Mapping.html
[tabular datasets]: Tabular_Datasets.html
[transforming elements]: Transforming_Elements.html
[using annotators to edit and label data]: Annotators.html
[using linked brushing to cross-filter complex datasets]: Linked_Brushing.html
[working with large data]: Large_Data.html
[working with renderers and plots]: Plots_and_Renderers.html
[working with streaming data]: Streaming_Data.html
