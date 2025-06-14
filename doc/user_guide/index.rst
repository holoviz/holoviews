User Guide
==========

The User Guide is the primary resource documenting key concepts that
will help you use HoloViews in your work. For newcomers, a gentle
introduction to HoloViews can be found in our `Getting Started <../getting_started/index.html>`_
guide and an overview of some interesting HoloViews examples can be
found in our `Gallery <../gallery/index.html>`_. If you are looking for a specific component
(or wish to view the available range of primitives), see our
`Reference Gallery <../reference/index.html>`_.

Core guides
-----------

These user guides provide detailed explanation of some of the core
concepts in HoloViews:

`Annotating your Data <Annotating_Data.html>`_
 Core concepts when annotating your data with semantic metadata.

`Composing Elements <Composing_Elements.html>`_
 Composing your data into layouts and overlays with the ``+`` and ``*`` operators.

`Applying Customizations <Applying_Customizations.html>`_
 Using the options system to declare customizations.

`Style Mapping <Style_Mapping.html>`_
 Mapping your data to the visual attributes of your plot.

`Dimensioned Containers <Dimensioned_Containers.html>`_
 Multi-dimensional containers for animating and faceting your data flexibly.

`Building Composite Objects <Building_Composite_Objects.html>`_
 How to build and work with nested composite objects.

`Live Data <Live_Data.html>`_
 Lazily generate data on the fly and generate engaging interactive visualizations.

`Tabular Datasets <Tabular_Datasets.html>`_
 Explore tabular data with `NumPy <http://www.numpy.org/>`_, `pandas <http://pandas.pydata.org/>`_ and `dask <http://dask.pydata.org/>`_.

`Gridded Datasets <Gridded_Datasets.html>`_
 Explore gridded data (n-dimensional arrays) with `NumPy <http://www.numpy.org/>`_ and `XArray <http://xarray.pydata.org/>`_.

`Geometry Data <Geometry_Data.html>`_
 Working with and representing geometry data such as lines, multi-lines, polygons, multi-polygons and contours.

`Indexing and Selecting Data <Indexing_and_Selecting_Data.html>`_
 Select and index subsets of your data with HoloViews.

`Transforming Elements <Transforming_Elements.html>`_
 Apply operations to your data that can be used to build data analysis pipelines

`Responding to Events <Responding_to_Events.html>`_
 Allow your visualizations to respond to Python events using the 'streams' system.

`Custom Interactivity <Custom_Interactivity.html>`_
 Use `Bokeh <https://bokeh.pydata.org>`_ and 'linked streams' to directly interact with your visualizations.

`Data Processing Pipelines <Data_Pipelines.html>`_
 Chain operations to build sophisticated, interactive and lazy data analysis pipelines.

`Creating interactive network graphs <Network_Graphs.html>`_
 Using the Graph element to display small and large networks interactively.

`Working with large data <Large_Data.html>`_
 Leverage Datashader to interactively explore millions or billions of datapoints.

`Interactive Hover for Big Data <Interactive_Hover_for_Big_Data.html>`_
 Use the ``selector`` with Datashader to enable fast, interactive hover tooltips that
 reveal individual data points without sacrificing aggregation.

`Working with Streaming Data <Streaming_Data.html>`_
 Demonstrates how to leverage HoloViews to work with streaming datasets.

`Creating interactive dashboards <Dashboards.html>`_
 Use external widget libraries to build custom, interactive dashboards.


Supplementary guides
--------------------

These guides provide detail about specific additional features in HoloViews:

`Configuring HoloViews <Configuring.html>`_
 Information about configuration options.

`Customizing Plots <Customizing_Plots.html>`_
 How to customize plots including their titles, axis labels, ranges, ticks and more.

`Colormaps <Colormaps.html>`_
 Overview of colormaps available, including when and how to use each type.

`Plotting with Bokeh <Plotting_with_Bokeh.html>`_
 Styling options and unique `Bokeh <bokeh.pydata.org>`_ features such as plot tools and using bokeh models directly.

`Deploying Bokeh Apps <Deploying_Bokeh_Apps.html>`_
 Using `bokeh server <http://bokeh.pydata.org/en/latest/docs/user_guide/server.html>`_ using scripts and notebooks.

`Linking Bokeh plots <Linking_Plots.html>`_
 Using Links to define custom interactions on a plot without a Python server

`Plotting with matplotlib <Plotting_with_Matplotlib.html>`_
 Styling options and unique Matplotlib features such as GIF/MP4 support.

`Plotting with plotly <Plotting_with_Plotly.html>`_
 Styling options and unique plotly features, focusing on 3D plotting.

`Working with renderers and plots <Plots_and_Renderers.html>`_
 Using the ``Renderer`` and ``Plot`` classes for access to the plotting machinery.

`Using linked brushing to cross-filter complex datasets <Linked_Brushing.html>`_
 Explains how to use the ``link_selections`` helper to cross-filter multiple elements.

`Using Annotators to edit and label data <Annotators.html>`_
 Explains how to use the ``annotate`` helper to edit and annotate elements with the help of drawing tools and editable tables.

`Exporting and Archiving <Exporting_and_Archiving.html>`_
 Archive both your data and visualization in scripts and notebooks.

`Continuous Coordinates <Continuous_Coordinates.html>`_
 How continuous coordinates are handled, specifically focusing on ``Image`` and ``Raster`` types.

`Notebook Magics <Notebook_Magics.html>`_
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
    Customizing Plots <Customizing_Plots>
    Colormaps <Colormaps>
    Plotting with Bokeh <Plotting_with_Bokeh>
    Deploying Bokeh Apps <Deploying_Bokeh_Apps>
    Linking Bokeh plots <Linking_Plots>
    Plotting with matplotlib <Plotting_with_Matplotlib>
    Plotting with plotly <Plotting_with_Plotly>
    Working with Plot and Renderers <Plots_and_Renderers>
    Linked Brushing <Linked_Brushing>
    Annotators <Annotators>
    Exporting and Archiving <Exporting_and_Archiving>
    Continuous Coordinates <Continuous_Coordinates>
    Notebook Magics <Notebook_Magics>
