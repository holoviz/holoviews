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

## The HoloViews RC file

If the `HOLOVIEWSRC` environment variable is a valid path, HoloViews will load the configuration from that file.
This allows users to set their preferred options globally without needing to modify their scripts each time.
An example of an RC file to include the various options discussed above might look like this:

```python
import holoviews as hv

hv.config(no_padding=True)
hv.extension.case_sensitive_completion = True
```
