# Installing HoloViews

The quickest and easiest way to get the latest version of all the
recommended packages for working with HoloViews on Linux, Windows, or
Mac systems is via the
[conda](https://docs.conda.io/projects/conda/en/latest/) command
provided by the [Anaconda](https://docs.anaconda.com/anaconda/install/)
or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
scientific Python distributions:

    conda install holoviews

This installation includes the default [Matplotlib](https://matplotlib.org)
plotting library backend, the more interactive [Bokeh](https://bokeh.pydata.org) plotting library backend.

A similar set of packages can be installed using `pip`, if that command
is available on your system:

    pip install "holoviews[recommended]"

`pip` also supports other installation options, including a minimal
install of only the packages necessary to generate and manipulate
HoloViews objects without visualization:

    pip install holoviews

This minimal install will install only the required packages, for HoloViews to run.
This makes it very easy to integrate HoloViews into your workflow or as part of another project.

Now that you are set up you can get a copy of all the examples shown on
this website:

    holoviews --install-examples
    cd holoviews-examples

Once you've installed HoloViews examples, you can get started by launching
Jupyter Notebook

    jupyter notebook

Or JupyterLab

    jupyter lab

Both can be installed with pip or conda:

    pip install jupyterlab
    conda install jupyterlab

For helping develop HoloViews see the [developer guide](developer_guide/index).
