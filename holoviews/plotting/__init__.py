"""
HoloViews plotting sub-system the defines the interface to be used by
any third-party plotting/rendering package.

This file defines the HTML tags used to wrap renderered output for
display in the IPython Notebook (optional).
"""

from .. import DEFAULT_RENDERER
from ..core.options import Cycle
from .plot import Plot
from .renderer import Renderer, MIME_TYPES

# Tags used when visual output is to be embedded in HTML
IMAGE_TAG = "<img src='{src}' style='max-width:100%; margin: auto; display: block; {css}'/>"
VIDEO_TAG = """
<video controls style='max-width:100%; margin: auto; display: block; {css}'>
<source src='{src}' type='{mime_type}'>
Your browser does not support the video tag.
</video>"""
PDF_TAG = "<iframe src='{src}' style='width:100%; margin: auto; display: block; {css}'></iframe>"


HTML_TAGS = {
    'base64': 'data:{mime_type};base64,{b64}', # Use to embed data
    'svg':  IMAGE_TAG,
    'png':  IMAGE_TAG,
    'gif':  IMAGE_TAG,
    'webm': VIDEO_TAG,
    'mp4':  VIDEO_TAG,
    'pdf':  PDF_TAG
}

DEFAULT_RENDER_CLASS=None

def public(obj):
    global DEFAULT_RENDER_CLASS
    if not isinstance(obj, type): return False
    is_plot_or_cycle = any([issubclass(obj, bc) for bc in [Plot, Cycle]])
    is_renderer = any([issubclass(obj, bc) for bc in [Renderer]])
    if is_renderer and (obj is not Renderer):
        DEFAULT_RENDER_CLASS = obj
    return (is_plot_or_cycle or is_renderer)

# Load the default renderer
if DEFAULT_RENDERER=='matplotlib':
    from .mpl import *

_public = list(set([_k for _k, _v in locals().items() if public(_v)]))
__all__ = _public
