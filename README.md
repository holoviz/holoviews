# <img src="https://assets.holoviews.org/logo/holoviews_color_icon_500x500.png" alt="HoloViews logo" height="40px" align="left" /> HoloViews

**Stop plotting your data - annotate your data and let it visualize
itself.**

|    |    |
| --- | --- |
| Build Status | [![Build Status](https://github.com/holoviz/holoviews/workflows/tests/badge.svg?query=branch:main)](https://github.com/holoviz/holoviews/actions/workflows/test.yaml?query=branch%3Amain) |
| Coverage | [![codecov](https://codecov.io/gh/holoviz/holoviews/branch/main/graph/badge.svg)](https://codecov.io/gh/holoviz/holoviews) |
| Latest dev release | [![Github tag](https://img.shields.io/github/tag/holoviz/holoviews.svg?label=tag&colorB=11ccbb)](https://github.com/holoviz/holoviews/tags) [![dev-site](https://img.shields.io/website-up-down-green-red/http/dev.holoviews.org.svg?label=dev%20website)](http://dev.holoviews.org) |
| Latest release | [![Github release](https://img.shields.io/github/release/holoviz/holoviews.svg?label=tag&colorB=11ccbb)](https://github.com/holoviz/holoviews/releases) [![PyPI version](https://img.shields.io/pypi/v/holoviews.svg?colorB=cc77dd)](https://pypi.python.org/pypi/holoviews) [![holoviews version](https://img.shields.io/conda/v/pyviz/holoviews.svg?colorB=4488ff&style=flat)](https://anaconda.org/pyviz/holoviews) [![conda-forge version](https://img.shields.io/conda/v/conda-forge/holoviews.svg?label=conda%7Cconda-forge&colorB=4488ff)](https://anaconda.org/conda-forge/holoviews) [![defaults version](https://img.shields.io/conda/v/anaconda/holoviews.svg?label=conda%7Cdefaults&style=flat&colorB=4488ff)](https://anaconda.org/anaconda/holoviews) |
| Python | [![Python support](https://img.shields.io/pypi/pyversions/holoviews.svg)](https://pypi.org/project/holoviews/) |
| Docs | [![DocBuildStatus](https://github.com/holoviz/holoviews/workflows/docs/badge.svg?query=branch%3Amain)](https://github.com/holoviz/holoviews/actions?query=workflow%3Adocs+branch%3Amain) [![site](https://img.shields.io/website-up-down-green-red/https/holoviews.org.svg)](https://holoviews.org) |
| Binder | [![Binder](https://img.shields.io/badge/Launch%20JupyterLab-v1.15.4-579ACA.svg?logo=data:image/png;base64,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)](https://mybinder.org/v2/gh/holoviz/holoviews/v1.15.4?urlpath=lab/tree/examples) |
| Support | [![Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fdiscourse.holoviz.org)](https://discourse.holoviz.org/) |


HoloViews is an
[open-source](https://github.com/holoviz/holoviews/blob/main/LICENSE.txt)
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
[Python 3](https://github.com/holoviz/holoviews/actions?query=workflow%3Apytest)
on Linux, Windows, or Mac, and works seamlessly with
[Jupyter Notebook and JupyterLab](https://jupyter.org).

The recommended way to install HoloViews is using the
[conda](https://docs.conda.io/projects/conda/en/latest/index.html) command provided by
[Anaconda](https://docs.anaconda.com/anaconda/install/) or
[Miniconda](https://docs.conda.io/en/latest/miniconda.html):

    conda install -c pyviz holoviews bokeh

This command will install the typical packages most useful with
HoloViews, though HoloViews itself depends only on
[Numpy](https://numpy.org) [Pandas](https://pandas.pydata.org) and [Param](https://param.holoviz.org).
Additional installation and configuration options are described in the
[user guide](https://holoviews.org/user_guide/Installing_and_Configuring.html).

You can also clone holoviews directly from GitHub and install it with:

    git clone git://github.com/holoviz/holoviews.git
    cd holoviews
    pip install -e .

Usage
-----

Once you've installed HoloViews, you can get a copy of all the examples shown on this website:

    holoviews --install-examples
    cd holoviews-examples

Now you can launch Jupyter Notebook or JupyterLab to explore them:

    jupyter notebook

    jupyter lab

If you are working with a JupyterLab version <2.0 you will also need the PyViz JupyterLab
extension:

    jupyter labextension install @pyviz/jupyterlab_pyviz

For more details on setup and configuration see [our website](https://holoviews.org/user_guide/Installing_and_Configuring.html).

For general discussion, we have a [gitter channel](https://gitter.im/pyviz/pyviz).
If you find any bugs or have any feature suggestions please file a GitHub
[issue](https://github.com/holoviz/holoviews/issues)
or submit a [pull request](https://help.github.com/articles/about-pull-requests).
