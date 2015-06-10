"""
Public API for all plotting renderers supported by HoloViews,
regardless of plotting package or backend.
"""

import param
from . import MIME_TYPES
from ..core.io import Exporter
from ..core.options import Store

from param.parameterized import bothmethod

class PlotRenderer(Exporter):
    """
    The job of a PlotRenderer is to turn the plotting state held
    within Plot classes into concrete output in the form of the PNG,
    SVG, MP4 or WebM formats (among others).
    """

    fig = param.ObjectSelector(default='svg',
                               objects=['png', 'svg', 'pdf', None], doc="""
        Output render format for static figures. If None, no figure
        rendering will occur. """)

    holomap = param.ObjectSelector(default='gif',
                                   objects=['webm','mp4', 'gif', None], doc="""
        Output render multi-frame (typically animated) format. If
        None, no multi-frame rendering will occur.""")

    size=param.Integer(100, doc="""
        The rendered size as a percentage size""")

    fps=param.Integer(20, doc="""
        Rendered fps (frames per second) for animated formats.""")

    dpi=param.Integer(None, allow_None=True, doc="""
        The render resolution in dpi (dots per inch)""")

    info_fn = param.Callable(None, allow_None=True, constant=True,  doc="""
        PlotRenderers do not support the saving of object info metadata""")

    key_fn = param.Callable(None, allow_None=True, constant=True,  doc="""
        PlotRenderers do not support the saving of object key metadata""")

    # Error messages generated when testing potentially supported formats
    HOLOMAP_FORMAT_ERROR_MESSAGES = {}

    def __init__(self, **params):
        super(MPLPlotRenderer, self).__init__(**params)


    def __call__(self, obj, fmt=None):
        """
        Render the supplied HoloViews component using the appropriate
        backend. The output is not a file format but a suitable,
        in-memory byte stream together with any suitable metadata.
        """
        # Example of the return format where the first value is the rendered data.
        return None, {'file-ext':fmt, 'mime_type':MIME_TYPES[fmt]}

    @classmethod
    def get_plot_size(cls, obj, percent_size):
        """
        Given an object and a percentage size (as supplied by the
        %output magic) return the appropriate sizing plot
        option. Default plot sizes at the plotting class level should
        be taken into account.
        """
        raise NotImplementedError


    @bothmethod
    def save(self_or_cls, obj, basename, fmt=None, key={}, info={}, options=None, **kwargs):
        """
        Given an object, a basename for the output file, a file format
        and some options, save the element in a suitable format to disk.
        """
        raise NotImplementedError
