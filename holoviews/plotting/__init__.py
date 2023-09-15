"""
HoloViews plotting sub-system the defines the interface to be used by
any third-party plotting/rendering package.

This file defines the HTML tags used to wrap rendered output for
display in the IPython Notebook (optional).
"""
from ..core.options import Cycle, Compositor
from ..element import Area, Image, ImageStack, QuadMesh, Polygons, Raster, RGB
from ..element.sankey import _layout_sankey, Sankey
from .plot import Plot
from .renderer import Renderer, HTML_TAGS # noqa (API import)
from .util import apply_nodata, flatten_stack, list_cmaps # noqa (API import)
from ..operation.stats import univariate_kde, bivariate_kde

Compositor.register(Compositor("Image", apply_nodata, None,
                               'data', transfer_options=True,
                               transfer_parameters=True,
                               output_type=Image,
                               backends=['bokeh', 'matplotlib', 'plotly']))
Compositor.register(Compositor("Raster", apply_nodata, None,
                               'data', transfer_options=True,
                               transfer_parameters=True,
                               output_type=Raster,
                               backends=['bokeh', 'matplotlib', 'plotly']))
Compositor.register(Compositor("QuadMesh", apply_nodata, None,
                               'data', transfer_options=True,
                               transfer_parameters=True,
                               output_type=QuadMesh,
                               backends=['bokeh', 'matplotlib', 'plotly']))
Compositor.register(Compositor("Distribution", univariate_kde, None,
                               'data', transfer_options=True,
                               transfer_parameters=True,
                               output_type=Area,
                               backends=['bokeh', 'matplotlib', 'plotly']))
Compositor.register(Compositor("Bivariate", bivariate_kde, None,
                               'data', transfer_options=True,
                               transfer_parameters=True,
                               output_type=Polygons,
                               backends=['bokeh', 'matplotlib']))
Compositor.register(Compositor("Sankey", _layout_sankey, None,
                               'data', transfer_options=True,
                               transfer_parameters=True,
                               output_type=Sankey))

Compositor.register(Compositor("ImageStack", apply_nodata, None,
                               'data', transfer_options=True,
                               transfer_parameters=True,
                               output_type=ImageStack,
                               backends=['bokeh']))

Compositor.register(Compositor("ImageStack", flatten_stack, None,
                               'data', transfer_options=True,
                               transfer_parameters=True,
                               output_type=RGB,
                               backends=['matplotlib', 'plotly']))


DEFAULT_CYCLE = ['#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b', '#17becf',
                 '#9467bd', '#d62728', '#1f77b4', '#e377c2', '#8c564b', '#bcbd22']

Cycle.default_cycles['default_colors'] = DEFAULT_CYCLE

def public(obj):
    if not isinstance(obj, type): return False
    is_plot_or_cycle = any([issubclass(obj, bc) for bc in [Plot, Cycle]])
    is_renderer = any([issubclass(obj, bc) for bc in [Renderer]])
    return (is_plot_or_cycle or is_renderer)

_public = list({_k for _k, _v in locals().items() if public(_v)})
__all__ = _public
