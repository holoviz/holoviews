import asyncio
import sys

from panel.io.pyodide import write

from ..core.dimension import LabelledData
from ..core.options import Store
from ..util import extension as _extension

def write_html(obj):
    if hasattr(sys.stdout, '_out'):
        out = sys.stdout._out # type: ignore
    else:
        raise ValueError("Could not determine target node to write to.")
    asyncio.create_task(write(out, obj))
    return {'text/plain': ''}, {}

def write_image(element, fmt):
    """
    Used to render elements to an image format (svg or png) if requested
    in the display formats.
    """
    if fmt not in Store.display_formats:b
        return None

    backend = Store.current_backend
    if type(element) not in Store.registry[backend]:
        return None
    renderer = Store.renderers[backend]
    plot = renderer.get_plot(element)

    # Current renderer does not support the image format
    if fmt not in renderer.param.objects('existing')['fig'].objects:
        return None

    data, info = renderer(plot, fmt=fmt)
    return {info['mime_type']: data}, {}

def write_png(obj):
    return write_image(element, 'png')

def write_svg(obj):
    return write_image(element, 'svg')

class pyodide_extension(_extension):

    _loaded = False

    def __call__(self, *args, **params):
        super().__call__(*args, **params)
        if not self._loaded:
            Store.output_settings.initialize(list(Store.renderers.keys()))
            Store.set_display_hook('html+js', LabelledData, write_output)
            Store.set_display_hook('png', LabelledData, write_png)
            Store.set_display_hook('svg', LabelledData, write_svg)
            pyodide_extension._loaded = True
