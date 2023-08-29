try:
    from .datashader import shade
except ImportError:
    shade = None


class flatten_stack(shade):
    """
    Thin wrapper around datashader's shade operation to flatten
    ImageStacks into RGB elements.

    Used for the MPL and Plotly backends because these backends
    do not natively support ImageStacks, unlike Bokeh.
    """

    def _process(self, element, key=None):
        if shade is None:
            raise ImportError('Flattening ImageStacks requires datashader.')
        return super()._process(element, key=key)
