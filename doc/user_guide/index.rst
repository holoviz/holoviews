User Guide
==========

The User Guide is the primary resource documenting key concepts that
will help you use HoloViews in your work. For newcomers, a gentle
introduction to HoloViews can be found in our `Getting Started`_ guide and an overview of some interesting HoloViews examples can be
found in our `Gallery`_. If you are looking for a specific component
(or wish to view the available range of primitives), see our
`Reference Gallery`_.

Core guides
-----------

These user guides provide detailed explanation of some of the core
concepts in HoloViews:

`Annotating your Data`_
 Core concepts when annotating your data with semantic metadata.

`Composing Elements`_
 Composing your data into layouts and overlays with the ``+`` and ``*`` operators.

`Applying Customizations`_
 Using the options system to declare customizations.

`Style Mapping`_
 Mapping your data to the visual attributes of your plot.

`Dimensioned Containers`_
 Multi-dimensional containers for animating and faceting your data flexibly.

`Building Composite Objects`_
 How to build and work with nested composite objects.

`Live Data`_
 Lazily generate data on the fly and generate engaging interactive visualizations.

`Tabular Datasets`_
 Explore tabular data with `NumPy <https://www.numpy.org/>`_, `pandas <https://pandas.pydata.org/>`_ and `dask <https://www.dask.org/>`_.

`Gridded Datasets`_
 Explore gridded data (n-dimensional arrays) with `NumPy <https://www.numpy.org/>`_ and `XArray <https://xarray.dev/>`_.

`Geometry Data`_
 Represent and visualize path and polygon geometries with support for multi-geometries and value dimensions.

`Indexing and Selecting Data`_
 Select and index subsets of your data with HoloViews.

`Transforming Elements`_
 Apply operations to your data that can be used to build data analysis pipelines

`Responding to Events`_
 Allow your visualizations to respond to Python events using the 'streams' system.

`Custom Interactivity`_
 Use `Bokeh <https://bokeh.org>`_ and 'linked streams' to directly interact with your visualizations.

`Data Processing Pipelines`_
 Chain operations to build sophisticated, interactive and lazy data analysis pipelines.

`Working with large data`_
 Leverage Datashader to interactively explore millions or billions of datapoints.

`Interactive Hover for Big Data`_
 Use the ``selector`` with Datashader to enable fast, interactive hover tooltips that
 reveal individual data points without sacrificing aggregation.

`Working with Streaming Data`_
 Demonstrates how to leverage the streamz library with HoloViews to work with streaming datasets.

`Creating interactive dashboards`_
 Use external widget libraries to build custom, interactive dashboards.


Supplementary guides
--------------------

These guides provide detail about specific additional features in HoloViews:

`Configuring HoloViews`_
 Information about configuration options.

`Customizing Plots`_
 How to customize plots including their titles, axis labels, ranges, ticks and more.

`Colormaps`_
 Overview of colormaps available, including when and how to use each type.

`Plotting with Bokeh`_
 Styling options and unique `Bokeh <https://bokeh.org>`_ features such as plot tools and using bokeh models directly.

`Deploying Bokeh Apps`_
 Using `bokeh server <https://docs.bokeh.org/en/latest/docs/user_guide/server.html>`_ using scripts and notebooks.

`Linking Bokeh plots`_
 Using Links to define custom interactions on a plot without a Python server

`Plotting with matplotlib`_
 Styling options and unique Matplotlib features such as GIF/MP4 support.

`Working with renderers and plots`_
 Using the ``Renderer`` and ``Plot`` classes for access to the plotting machinery.

`Using linked brushing to cross-filter complex datasets`_
 Explains how to use the ``link_selections`` helper to cross-filter multiple elements.

`Using Annotators to edit and label data`_
 Explains how to use the ``annotate`` helper to edit and annotate elements with the help of drawing tools and editable tables.

`Exporting and Archiving`_
 Archive both your data and visualization in scripts and notebooks.

`Continuous Coordinates`_
 How continuous coordinates are handled, specifically focusing on ``Image`` and ``Raster`` types.

`Interactive Hover for Big Data`_
 Explains how to use interactive hover tools with large datasets.

`Notebook Magics`_
 IPython magics supported in Jupyter Notebooks.

.. toctree::
   :titlesonly:
   :hidden:
   :maxdepth: 2

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

.. _Getting Started: ../getting_started/index.html
.. _Gallery: ../gallery/index.html
.. _Reference Gallery: ../reference/index.html
.. _Annotating your Data: Annotating_Data.html
.. _Composing Elements: Composing_Elements.html
.. _Applying Customizations: Applying_Customizations.html
.. _Style Mapping: Style_Mapping.html
.. _Dimensioned Containers: Dimensioned_Containers.html
.. _Building Composite Objects: Building_Composite_Objects.html
.. _Live Data: Live_Data.html
.. _Tabular Datasets: Tabular_Datasets.html
.. _Gridded Datasets: Gridded_Datasets.html
.. _Geometry Data: Geometry_Data.html
.. _Indexing and Selecting Data: Indexing_and_Selecting_Data.html
.. _Transforming Elements: Transforming_Elements.html
.. _Responding to Events: Responding_to_Events.html
.. _Custom Interactivity: Custom_Interactivity.html
.. _Data Processing Pipelines: Data_Pipelines.html
.. _Working with large data: Large_Data.html
.. _Interactive Hover for Big Data: Interactive_Hover_for_Big_Data.html
.. _Working with Streaming Data: Streaming_Data.html
.. _Creating interactive dashboards: Dashboards.html
.. _Configuring HoloViews: Configuring.html
.. _Customizing Plots: Customizing_Plots.html
.. _Colormaps: Colormaps.html
.. _Plotting with Bokeh: Plotting_with_Bokeh.html
.. _Deploying Bokeh Apps: Deploying_Bokeh_Apps.html
.. _Linking Bokeh plots: Linking_Plots.html
.. _Plotting with matplotlib: Plotting_with_Matplotlib.html
.. _Working with renderers and plots: Plots_and_Renderers.html
.. _Using linked brushing to cross-filter complex datasets: Linked_Brushing.html
.. _Using Annotators to edit and label data: Annotators.html
.. _Exporting and Archiving: Exporting_and_Archiving.html
.. _Continuous Coordinates: Continuous_Coordinates.html
.. _Interactive Hover for Big Data: Interactive_Hover_for_Big_Data.html
.. _Notebook Magics: Notebook_Magics.html
