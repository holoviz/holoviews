[![PyPI](https://img.shields.io/pypi/v/holoviews.svg)](https://pypi.python.org/pypi/holoviews)
[![Conda](https://anaconda.org/ioam/holoviews/badges/installer/conda.svg)](https://anaconda.org/ioam/holoviews)
[![Downloads](https://anaconda.org/ioam/holoviews/badges/downloads.svg)](https://anaconda.org/ioam/holoviews)
[![BuildStatus](https://travis-ci.org/ioam/holoviews.svg?branch=master)](https://travis-ci.org/ioam/holoviews)
[![holoviewsDocs](http://buildbot.holoviews.org:8010/png?builder=website)](http://buildbot.holoviews.org:8010/waterfall)
[![Coveralls](https://img.shields.io/coveralls/ioam/holoviews.svg)](https://coveralls.io/r/ioam/holoviews)
[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/ioam/holoviews?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![MyBinder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/ioam/holoviews-contrib)

# HoloViews <img src="http://assets.holoviews.org/logo/holoviews_color_icon_500x500.png" alt="HoloViews logo" height="40px" align="right" />

**Stop plotting your data - annotate your data and let it visualize
itself.**

<div>
<div >
  <a href="http://holoviews.org/gallery/demos/bokeh/iris_splom_example.html">
    <img src="http://build.holoviews.org/_images/iris_splom_example.png" width='20%'> </img> </a>
  <a href="http://holoviews.org/getting_started/Gridded_Datasets.html">
    <img src="http://assets.holoviews.org/collage/cells.png" width='22%'> </img>  </a>
  <a href="http://holoviews.org/gallery/demos/bokeh/scatter_economic.html">
    <img src="http://build.holoviews.org/_images/scatter_economic.png" width='43%'> </img> </a>
</div>

<div >
  <a href="http://holoviews.org/gallery/demos/bokeh/square_limit.html">
    <img src="http://build.holoviews.org/_images/square_limit.png" width='20%'> </a>
  <a href="http://holoviews.org/gallery/demos/bokeh/bars_economic.html">
    <img src="http://build.holoviews.org/_images/bars_economic.png" width='20%'> </a>
  <a href="http://holoviews.org/gallery/demos/bokeh/texas_choropleth_example.html">
    <img src="http://build.holoviews.org/_images/texas_choropleth_example.png" width='20%'> </a>
  <a href="http://holoviews.org/gallery/demos/bokeh/verhulst_mandelbrot.html">
    <img src="http://build.holoviews.org/_images/verhulst_mandelbrot.png" width='20%'> </a>
</div>
<div >
    <a href="http://holoviews.org/gallery/demos/bokeh/dropdown_economic.html">
      <img src="http://assets.holoviews.org/collage/dropdown.gif" width='31%'> </a>
    <a href="http://holoviews.org/gallery/demos/bokeh/dragon_curve.html">
      <img src="http://assets.holoviews.org/collage/dragon_fractal.gif" width='26%'> </a>
    <a href="http://holoviews.org/gallery/apps/bokeh/nytaxi_hover.html">
      <img src="http://assets.holoviews.org/collage/ny_datashader.gif" width='31%'> </a>
</div>
</div>

HoloViews requires [Param](http://ioam.github.com/param/) and
[Numpy](http://www.numpy.org/) and is designed to work together with
[Matplotlib](http://matplotlib.org/) or
[Bokeh](http://bokeh.pydata.org), making use of the [Jupyter/IPython
Notebook](http://jupyter.org).

You can get the latest version of HoloViews from the ioam conda channel:

> conda install -c ioam holoviews

Or clone holoviews directly from GitHub with:

    git clone git://github.com/ioam/holoviews.git

Please visit [our website](http://holoviews.org) for official releases,
installation instructions, documentation, and many detailed [example
notebooks and tutorials](http://holoviews.org/Tutorials). Additional
user contributed notebooks may be found in the
[holoviews-contrib](https://github.com/ioam/holoviews-contrib)
repository including examples that may be run live on
[mybinder.org](http://mybinder.org/repo/ioam/holoviews-contrib).

For general discussion, we have a [gitter
channel](https://gitter.im/ioam/holoviews). In addition we have [a
user-contributed wiki](https://github.com/ioam/holoviews-contrib/wiki)
describing current work-in-progress and experimental features. If you
find any bugs or have any feature suggestions please file a GitHub Issue
or submit a pull request.

Usage
-----

Once you've installed HoloViews, you can get started by launching
Jupyter Notebook:

    jupyter notebook

Now you can download the [tutorial
notebooks](http://assets.holoviews.org/notebooks-1.7.zip). unzip them
somewhere Jupyter Notebook can find them, and then open the
Homepage.ipynb tutorial or any of the others in the Notebook. Enjoy
exploring your data!

Note: When running HoloViews in Jupyter Notebook 5.0 a data rate limit
was introduced which severely limits the output that HoloViews can
display. This limit will be removed again in the upcoming 5.1 release,
in the meantime you can raise the limit manually by overriding the
default `iopub_data_rate_limit`:

    jupyter notebook --NotebookApp.iopub_data_rate_limit=100000000

Alternatively you can set a higher default in the user configuration
file in `~/.jupyter/jupyter_notebook_config.py`, by adding:

    c.NotebookApp.iopub_data_rate_limit=100000000

If the configuration file does not exist generate one first using:

    jupyter notebook --generate-config

Features
--------

**Overview**

-   Lets you build data structures that both contain and visualize
    your data.
-   Includes a rich [library of composable
    elements](http://www.holoviews.org/Tutorials/Elements.html) that can
    be overlaid, nested and positioned with ease.
-   Supports [rapid data
    exploration](http://www.holoviews.org/Tutorials/Exploring_Data.html)
    that naturally develops into a [fully reproducible
    workflow](http://www.holoviews.org/Tutorials/Exporting.html).
-   Create interactive visualizations that can be controlled via widgets
    or via custom events in Python using the 'streams' system. When
    using the bokeh backend, you can use streams to directly interact
    with your plots.
-   Rich semantics for [indexing and slicing of data in arbitrarily
    high-dimensional
    spaces](http://www.holoviews.org/Tutorials/Sampling_Data.html).
-   Plotting output using the
    [Matplotlib](http://www.holoviews.org/Tutorials/Elements.html),
    [Bokeh](http://www.holoviews.org/Tutorials/Bokeh_Elements.html), and
    [plotly](http://plot.ly/) backends.
-   A variety of data interfaces to work with tabular and N-dimensional
    array data using [NumPy](http://www.numpy.org/),
    [pandas](http://pandas.pydata.org/),
    [dask](http://dask.pydata.org/en/latest/),
    [iris](http://scitools.org.uk/iris/) and
    [xarray](http://xarray.pydata.org/en/stable/).
-   Every parameter of every object includes
    easy-to-access documentation.
-   All features [available in vanilla Python 2 or
    3](http://www.holoviews.org/Tutorials/Options.html), with
    minimal dependencies.

**Support for maintainable, reproducible research**

-   Supports a truly reproducible workflow by minimizing the code needed
    for analysis and visualization.
-   Already used in a variety of research projects, from conception to
    final publication.
-   All HoloViews objects can be pickled and unpickled.
-   Provides comparison utilities for testing, so you know when your
    results have changed and why.
-   Core data structures only depend on the numpy and param libraries.
-   Provides [export and archival
    facilities](http://www.holoviews.org/Tutorials/Exporting.html) for
    keeping track of your work throughout the lifetime of a project.

**Analysis and data access features**

-   Allows you to annotate your data with dimensions, units, labels and
    data ranges.
-   Easily [slice and
    access](http://www.holoviews.org/Tutorials/Sampling_Data.html)
    regions of your data, no matter how high the dimensionality.
-   Apply any suitable function to collapse your data or
    reduce dimensionality.
-   Helpful textual representation to inform you how every level of your
    data may be accessed.
-   Includes small library of common operations for any scientific or
    engineering data.
-   Highly extensible: add new operations to easily apply the data
    transformations you need.

**Visualization features**

-   Useful default settings make it easy to inspect data, with
    minimal code.
-   Powerful normalization system to make understanding your data across
    plots easy.
-   Build [complex animations or interactive visualizations in
    seconds](http://www.holoviews.org/Tutorials/Exploring_Data.html)
    instead of hours or days.
-   Refine the visualization of your data interactively
    and incrementally.
-   Separation of concerns: all visualization settings are kept separate
    from your data objects.
-   Support for fully interactive plots using the [Bokeh
    backend](http://www.holoviews.org/Tutorials/Bokeh_Backend.html).

**Jupyter Notebook support**

-   Support for all recent releases of IPython and Jupyter Notebooks.
-   Automatic tab-completion everywhere.
-   Exportable sliders and scrubber widgets.
-   Custom interactivity using streams and notebook comms to dynamically
    updating plots.
-   Automatic display of animated formats in the notebook or for export,
    including gif, webm, and mp4.
-   Useful IPython magics for configuring global display options and for
    customizing objects.
-   [Automatic archival and export of
    notebooks](http://www.holoviews.org/Tutorials/Exporting.html),
    including extracting figures as SVG, generating a static HTML copy
    of your results for reference, and storing your optional metadata
    like version control information.
