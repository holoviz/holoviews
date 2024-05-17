# Configuring Holoviews

## `hv.config` settings

The default HoloViews installation will use the latest defaults and options available, which is appropriate for new users.
If you want to work with code written for older HoloViews versions, you can use the top-level `hv.config` object to control various backwards-compatibility options:

- `future_deprecations`: Enables warnings about future deprecations.
- `warn_options_call`: Warn when using the to-be-deprecated `__call__` syntax for specifying options, instead of the recommended `.opts` method.

It is recommended you set `warn_options_call` to `True` in your holoviews.rc file (see section below).

It is possible to set the configuration using `hv.config` directly:

```python
import holoviews as hv

hv.config(future_deprecations=True)
```

However, because in some cases this configuration needs to be declared before the plotting extensions are imported, the recommended way of setting configuration options is:

```python
hv.extension("bokeh", config=dict(future_deprecations=True))
```

In addition to backwards-compatibility options, `hv.config` holds some global options:

- `image_rtol`: The tolerance used to enforce regular sampling for regular, gridded data. Used to validate `Image` data.

This option allows you to set the `rtol` parameter of [`Image`](../reference/elements/bokeh/Image.ipynb) elements globally.

## Improved tab-completion

Both `Layout` and `Overlay` are designed around convenient tab-completion, with the expectation of upper-case names being listed first.

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

```python
import holoviews as hv

hv.config(warn_options_call=True)
hv.extension.case_sensitive_completion = True
```
