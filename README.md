[![PyPI](https://img.shields.io/pypi/v/holoviews.svg)](https://pypi.python.org/pypi/holoviews)
[![Conda](https://anaconda.org/ioam/holoviews/badges/installer/conda.svg)](https://anaconda.org/ioam/holoviews)
[![Downloads](https://anaconda.org/ioam/holoviews/badges/downloads.svg)](https://anaconda.org/ioam/holoviews)
[![BuildStatus](https://travis-ci.org/ioam/holoviews.svg?branch=master)](https://travis-ci.org/ioam/holoviews)
[![holoviewsDocs](http://buildbot.holoviews.org:8010/png?builder=website)](http://buildbot.holoviews.org:8010/waterfall)
[![Coveralls](https://img.shields.io/coveralls/ioam/holoviews.svg)](https://coveralls.io/r/ioam/holoviews)
[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/ioam/holoviews?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

# <img src="http://assets.holoviews.org/logo/holoviews_color_icon_500x500.png" alt="HoloViews logo" height="40px" align="left" /> HoloViews

**Stop plotting your data - annotate your data and let it visualize
itself.**

HoloViews is an
[open-source](https://github.com/ioam/holoviews/blob/master/LICENSE.txt)
Python library designed to make data analysis and visualization seamless
and simple. With HoloViews, you can usually express what you want to do
in very few lines of code, letting you focus on what you are trying to
explore and convey, not on the process of plotting. 

Check out the [HoloViews web site](http://holoviews.org) for extensive examples and documentation.

<div>
<div >
  <a href="http://holoviews.org/gallery/demos/bokeh/iris_splom_example.html">
    <img src="http://holoviews.org/_images/iris_splom_example.png" width='20%'> </img> </a>
  <a href="http://holoviews.org/getting_started/Gridded_Datasets.html">
    <img src="http://assets.holoviews.org/collage/cells.png" width='22%'> </img>  </a>
  <a href="http://holoviews.org/gallery/demos/bokeh/scatter_economic.html">
    <img src="http://holoviews.org/_images/scatter_economic.png" width='43%'> </img> </a>
</div>

<div >
  <a href="http://holoviews.org/gallery/demos/bokeh/square_limit.html">
    <img src="http://holoviews.org/_images/square_limit.png" width='20%'> </a>
  <a href="http://holoviews.org/gallery/demos/bokeh/bars_economic.html">
    <img src="http://holoviews.org/_images/bars_economic.png" width='20%'> </a>
  <a href="http://holoviews.org/gallery/demos/bokeh/texas_choropleth_example.html">
    <img src="http://holoviews.org/_images/texas_choropleth_example.png" width='20%'> </a>
  <a href="http://holoviews.org/gallery/demos/bokeh/verhulst_mandelbrot.html">
    <img src="http://holoviews.org/_images/verhulst_mandelbrot.png" width='20%'> </a>
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


Installation
============

HoloViews works with 
[Python 2.7 and Python 3](https://travis-ci.org/ioam/holoviews)
on Linux, Windows, or Mac, and provides optional extensions for working with the 
[Jupyter/IPython Notebook](http://jupyter.org).

The recommended way to install HoloViews is using the
[conda](http://conda.pydata.org/docs/) command provided by
[Anaconda](http://docs.continuum.io/anaconda/install) or
[Miniconda](http://conda.pydata.org/miniconda.html):

    conda install -c ioam holoviews bokeh

This command will install the typical packages most useful with
HoloViews, though HoloViews itself depends only on
[Numpy](http://numpy.org) and [Param](http://ioam.github.com/param).
Additional installation and configuration options are described in the
[user guide](http://holoviews.org/user_guide/Installing_and_Configuring.html).

You can also clone holoviews directly from GitHub and install it with:

    git clone git://github.com/ioam/holoviews.git
    cd holoviews
    pip install -e .

Usage
-----

Once you've installed HoloViews, you can get a copy of all the
examples shown on the website:

    holoviews --install-examples
    cd holoviews-examples

And then you can launch Jupyter Notebook to explore them:

    jupyter notebook --NotebookApp.iopub_data_rate_limit=100000000

(Increasing the rate limit in this way is [required for the current 5.0
Jupyter version](http://holoviews.org/user_guide/Installing_and_Configuring.html), but
should not be needed in later Jupyter releases.)

For general discussion, we have a [gitter channel](https://gitter.im/ioam/holoviews).
If you find any bugs or have any feature suggestions please file a GitHub 
[issue](https://github.com/ioam/holoviews/issues)
or submit a [pull request](https://help.github.com/articles/about-pull-requests).
