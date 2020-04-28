[![PyPI](https://img.shields.io/pypi/v/holoviews.svg)](https://pypi.python.org/pypi/holoviews)
[![Conda](https://anaconda.org/pyviz/holoviews/badges/installer/conda.svg)](https://anaconda.org/pyviz/holoviews)
[![Downloads](https://s3.amazonaws.com/pubbadges/holoviews_current.svg)](https://anaconda.org/pyviz/holoviews)
[![BuildStatus](https://travis-ci.org/holoviz/holoviews.svg?branch=master)](https://travis-ci.org/holoviz/holoviews)
[![Coveralls](https://img.shields.io/coveralls/pyviz/holoviews.svg)](https://coveralls.io/r/pyviz/holoviews)
[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/pyviz/pyviz?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Binder](https://img.shields.io/badge/Launch%20JupyterLab-v1.13.2-579ACA.svg?logo=data:image/png;base64,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)](https://mybinder.org/v2/gh/holoviz/holoviews/v1.13.2?urlpath=lab/tree/examples)

# <img src="https://assets.holoviews.org/logo/holoviews_color_icon_500x500.png" alt="HoloViews logo" height="40px" align="left" /> HoloViews

**Stop plotting your data - annotate your data and let it visualize
itself.**

HoloViews is an
[open-source](https://github.com/pyviz/holoviews/blob/master/LICENSE.txt)
Python library designed to make data analysis and visualization seamless
and simple. With HoloViews, you can usually express what you want to do
in very few lines of code, letting you focus on what you are trying to
explore and convey, not on the process of plotting. 

Check out the [HoloViews web site](http://holoviews.org) for extensive examples and documentation.

<div>
<div >
  <a href="http://holoviews.org/gallery/demos/bokeh/iris_splom_example.html">
    <img src="http://holoviews.org/_images/iris_splom_example_large.png" width='20%'> </img> </a>
  <a href="http://holoviews.org/getting_started/Gridded_Datasets.html">
    <img src="https://assets.holoviews.org/collage/cells.png" width='22%'> </img>  </a>
  <a href="http://holoviews.org/gallery/demos/bokeh/scatter_economic.html">
    <img src="http://holoviews.org/_images/scatter_economic_large.png" width='43%'> </img> </a>
</div>

<div >
  <a href="http://holoviews.org/gallery/demos/bokeh/square_limit.html">
    <img src="http://holoviews.org/_images/square_limit_large.png" width='20%'> </a>
  <a href="http://holoviews.org/gallery/demos/bokeh/bars_economic.html">
    <img src="http://holoviews.org/_images/bars_economic.png" width='20%'> </a>
  <a href="http://holoviews.org/gallery/demos/bokeh/texas_choropleth_example.html">
    <img src="http://holoviews.org/_images/texas_choropleth_example_large.png" width='20%'> </a>
  <a href="http://holoviews.org/gallery/demos/bokeh/verhulst_mandelbrot.html">
    <img src="http://holoviews.org/_images/verhulst_mandelbrot.png" width='20%'> </a>
</div>
<div >
    <a href="http://holoviews.org/gallery/demos/bokeh/dropdown_economic.html">
      <img src="https://assets.holoviews.org/collage/dropdown.gif" width='31%'> </a>
    <a href="http://holoviews.org/gallery/demos/bokeh/dragon_curve.html">
      <img src="https://assets.holoviews.org/collage/dragon_fractal.gif" width='26%'> </a>
    <a href="http://holoviews.org/gallery/apps/bokeh/nytaxi_hover.html">
      <img src="https://assets.holoviews.org/collage/ny_datashader.gif" width='31%'> </a>
</div>
</div>


Installation
============

HoloViews works with 
[Python 2.7 and Python 3](https://travis-ci.org/holoviz/holoviews)
on Linux, Windows, or Mac, and provides optional extensions for working with the 
[Jupyter/IPython Notebook](http://jupyter.org).

The recommended way to install HoloViews is using the
[conda](http://conda.pydata.org/docs/) command provided by
[Anaconda](http://docs.continuum.io/anaconda/install) or
[Miniconda](http://conda.pydata.org/miniconda.html):

    conda install -c pyviz holoviews bokeh

This command will install the typical packages most useful with
HoloViews, though HoloViews itself depends only on
[Numpy](http://numpy.org) and [Param](https://param.holoviz.org).
Additional installation and configuration options are described in the
[user guide](http://holoviews.org/user_guide/Installing_and_Configuring.html).

You can also clone holoviews directly from GitHub and install it with:

    git clone git://github.com/holoviz/holoviews.git
    cd holoviews
    pip install -e .

Usage
-----

Once you've installed HoloViews, you can get a copy of all the
examples shown on the website:

    holoviews --install-examples
    cd holoviews-examples

And then you can launch Jupyter Notebook to explore them:

    jupyter notebook

To work with JupyterLab you will also need the PyViz JupyterLab
extension:

    conda install -c conda-forge jupyterlab
    jupyter labextension install @pyviz/jupyterlab_pyviz

Once you have installed JupyterLab and the extension launch it with::

    jupyter-lab

For more details on setup and configuration see [our website](http://holoviews.org/user_guide/Installing_and_Configuring.html).

For general discussion, we have a [gitter channel](https://gitter.im/pyviz/pyviz).
If you find any bugs or have any feature suggestions please file a GitHub 
[issue](https://github.com/pyviz/holoviews/issues)
or submit a [pull request](https://help.github.com/articles/about-pull-requests).
