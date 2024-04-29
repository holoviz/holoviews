# Installing and Configuring Holoviews

HoloViews can be installed on any platform where [NumPy](http://numpy.org) and Python 3 are available.

That said, HoloViews is designed to work closely with many other libraries, which can make installation and configuration more complicated. This user guide page describes some of these less-common or not-required options that may be helpful for some users.

## Other installation options

The main [installation instructions](http://holoviews.org/#installation) should be sufficient for most users, but you may also want the [Matplotlib](http://matplotlib.org) and [Plotly](https://plot.ly/python/) backends, which are required for some of the examples:

    conda install matplotlib plotly

HoloViews can also be installed using one of these `pip` commands:

    pip install holoviews
    pip install 'holoviews[recommended]'
    pip install 'holoviews[extras]'
    pip install 'holoviews[all]'

The first option installs just the bare library and the [NumPy](http://numpy.org) and [Param](https://github.com/holoviz/param) libraries, which is all you need on your system to generate and work with HoloViews objects without visualizing them. The other options install additional libraries that are often useful, with the `recommended` option being similar to the `conda` install command above.

Between releases, development snapshots are made available as conda packages:

    conda install -c pyviz/label/dev holoviews

To get the very latest development version you can clone our git
repository and put it on the Python path:

    git clone https://github.com/holoviz/holoviews.git
    cd holoviews
    pip install -e .

## JupyterLab configuration

To work with JupyterLab you will also need the HoloViews JupyterLab
extension:

```
conda install -c conda-forge jupyterlab
jupyter labextension install @pyviz/jupyterlab_pyviz
```

Once you have installed JupyterLab and the extension launch it with:

```
jupyter-lab
```

## `hv.config` settings

The default HoloViews installation will use the latest defaults and options available, which is appropriate for new users. If you want to work with code written for older HoloViews versions, you can use the top-level `hv.config` object to control various backwards-compatibility options:

- `future_deprecations`: Enables warnings about future deprecations (introduced in 1.11).
- `warn_options_call`: Warn when using the to-be-deprecated `__call__` syntax for specifying options, instead of the recommended `.opts` method.

It is recommended you set `warn_options_call` to `True` in your holoviews.rc file (see section below).

It is possible to set the configuration using `hv.config` directly:

```python
import holoviews as hv
hv.config(future_deprecations=True)
```

However, because in some cases this configuration needs to be declared before the plotting extensions are imported, the recommended way of setting configuration options is:

```python
hv.extension('bokeh', config=dict(future_deprecations=True))
```

In addition to backwards-compatibility options, `hv.config` holds some global options:

- `image_rtol`: The tolerance used to enforce regular sampling for regular, gridded data. Used to validate `Image` data.

This option allows you to set the `rtol` parameter of [`Image`](../reference/elements/bokeh/Image.ipynb) elements globally.

## Improved tab-completion

Both `Layout` and `Overlay` are designed around convenient tab-completion, with the expectation of upper-case names being listed first. In recent versions of Jupyter/IPython there has been a regression whereby the tab-completion is no longer case-sensitive. This can be fixed with:

```python
import holoviews as hv
hv.extension(case_sensitive_completion=True)
```

## The holoviews.rc file

HoloViews searches for the first rc file it finds in the following places (in order):

1. `holoviews.rc` in the parent directory of the top-level `__init__.py` file (useful for developers working out of the HoloViews git repo)
2. `~/.holoviews.rc`
3. `~/.config/holoviews/holoviews.rc`

The rc file location can be overridden via the `HOLOVIEWSRC` environment variable.

The rc file is a Python script, executed as HoloViews is imported. An example rc file to include various options discussed above might look like this:

```
import holoviews as hv
hv.config(warn_options_call=True)
hv.extension.case_sensitive_completion=True
```
