# Configuring Holoviews

## `hv.config` settings

The default HoloViews installation will use the latest defaults and options available, which is appropriate for new users.

It is possible to set the configuration using `hv.config` directly:

```python
import holoviews as hv

hv.config(no_padding=True)
```

However, because in some cases this configuration needs to be declared before the plotting extensions are imported, the recommended way of setting configuration options is:

```python
hv.extension("bokeh", config=dict(no_padding=True))
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

1. `HOLOVIEWSRC` environment variable if it is a valid path
1. `holoviews.rc` in the parent directory of the top-level `__init__.py` file (useful for developers working out of the HoloViews git repo)
1. `~/.holoviews.rc`
1. `~/.config/holoviews/holoviews.rc`

The rc file is a Python script and can be loaded with:

```python
import holoviews as hv

hv.extensions(load_rc=True)
```

An example rc file to include various options discussed above might look like this:

```python
import holoviews as hv

hv.config(warn_options_call=True)
hv.extension.case_sensitive_completion = True
```
