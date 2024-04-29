# Installing HoloViews

The quickest and easiest way to get the latest version of all the
recommended packages for working with HoloViews on Linux, Windows, or
Mac systems is via the
[conda](https://docs.conda.io/projects/conda/en/latest/) command
provided by the [Anaconda](https://docs.anaconda.com/anaconda/install/)
or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
scientific Python distributions:

    conda install -c pyviz holoviews

This recommended installation includes the default
[Matplotlib](http://matplotlib.org) plotting library backend, the more
interactive [Bokeh](http://bokeh.pydata.org) plotting library backend,
and the [Jupyter Notebook](http://jupyter.org).

A similar set of packages can be installed using `pip`, if that command
is available on your system:

    pip install "holoviews[recommended]"

`pip` also supports other installation options, including a minimal
install of only the packages necessary to generate and manipulate
HoloViews objects without visualization:

    pip install holoviews

This minimal install includes only three required libraries
[Param](https://param.holoviz.org/), [Numpy](https://numpy.org) and,
[pandas](https://pandas.pydata.org/), which makes it very easy to
integrate HoloViews into your workflow or as part of another project.

Alternatively, you can ask `pip` to install a larger set of packages
that provide additional functionality in HoloViews:

    pip install "holoviews[examples]"

This option installs all the required and recommended packages, in
addition to all all libraries required for running all the examples.

Lastly, to get _everything_ including the test dependencies, you can
use:

    pip install "holoviews[all]"

Between releases, development snapshots are made available on conda and
can be installed using:

    conda install -c pyviz/label/dev holoviews

To get the very latest development version using `pip`, you can use:

    pip install git+https://github.com/holoviz/holoviews.git

The alternative approach using git archive (e.g
`pip install https://github.com/holoviz/holoviews/archive/main.zip`) is
_not_ recommended as you will have incomplete version strings.

Anyone interested in following development can get the very latest
version by cloning the git repository:

    git clone https://github.com/holoviz/holoviews.git

To make this code available for import you then need to run:

    python setup.py develop

And you can then update holoviews at any time to the latest version by
running:

    git pull

Once you\'ve installed HoloViews, you can get started by launching
Jupyter Notebook:

    jupyter notebook

To work with JupyterLab\>2.0 you won\'t need to install anything else,
however for older versions you should also install the PyViz extension:

    jupyter labextension install @pyviz/jupyterlab_pyviz

Once you have installed JupyterLab and the extension launch it with:

    jupyter lab

Now that you are set up you can get a copy of all the examples shown on
this website:

    holoviews --install-examples
    cd holoviews-examples
