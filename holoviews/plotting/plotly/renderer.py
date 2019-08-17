from __future__ import absolute_import, division, unicode_literals

import base64

import param
import panel as pn

with param.logging_level('CRITICAL'):
    import plotly.graph_objs as go

from panel.pane import Viewable
from param.parameterized import bothmethod

from ..renderer import Renderer, MIME_TYPES, HTML_TAGS
from ...core.options import Store
from ...core import HoloMap
from .callbacks import callbacks
from .util import clean_internal_figure_properties



def _PlotlyHoloviewsPane(fig_dict):
    """
    Custom Plotly pane constructor for use by the HoloViews Pane.
    """

    # Remove internal HoloViews properties
    clean_internal_figure_properties(fig_dict)
    
    plotly_pane = pn.pane.Plotly(fig_dict, viewport_update_policy='mouseup')
    
    # Register callbacks on pane
    for callback_cls in callbacks.values():
        plotly_pane.param.watch(
            lambda event, cls=callback_cls: cls.update_streams_from_property_update(event.new, event.obj.object),
            callback_cls.callback_property,
        )
    return plotly_pane


class PlotlyRenderer(Renderer):

    backend = param.String(default='plotly', doc="The backend name.")

    fig = param.ObjectSelector(default='auto', objects=['html', 'png', 'svg', 'auto'], doc="""
        Output render format for static figures. If None, no figure
        rendering will occur. """)

    mode_formats = {'fig': ['html', 'png', 'svg'],
                    'holomap': ['widgets', 'scrubber', 'auto']}

    widgets = ['scrubber', 'widgets']

    _loaded = False

    _render_with_panel = True

    @bothmethod
    def get_plot_state(self_or_cls, obj, doc=None, renderer=None, **kwargs):
        """
        Given a HoloViews Viewable return a corresponding figure dictionary.
        Allows cleaning the dictionary of any internal properties that were added
        """
        fig_dict = super(PlotlyRenderer, self_or_cls).get_plot_state(obj, renderer, **kwargs)

        # Remove internal properties (e.g. '_id', '_dim')
        clean_internal_figure_properties(fig_dict)

        # Run through Figure constructor to normalize keys
        # (e.g. to expand magic underscore notation)
        fig_dict = go.Figure(fig_dict).to_dict()

        # Remove template
        fig_dict.get('layout', {}).pop('template', None)
        return fig_dict


    def _figure_data(self, plot, fmt, as_script=False, **kwargs):
        # Wrapping plot.state in go.Figure here performs validation
        # and applies any default theme.
        figure = go.Figure(plot.state)

        if fmt in ('png', 'svg'):
            import plotly.io as pio
            data = pio.to_image(figure, fmt)

            if fmt == 'svg':
                data = data.decode('utf-8')

            if as_script:
                b64 = base64.b64encode(data).decode("utf-8")
                (mime_type, tag) = MIME_TYPES[fmt], HTML_TAGS[fmt]
                src = HTML_TAGS['base64'].format(mime_type=mime_type, b64=b64)
                div = tag.format(src=src, mime_type=mime_type, css='')
                return div
            else:
                return data
        else:
            raise ValueError("Unsupported format: {fmt}".format(fmt=fmt))


    @classmethod
    def plot_options(cls, obj, percent_size):
        factor = percent_size / 100.0
        obj = obj.last if isinstance(obj, HoloMap) else obj
        plot = Store.registry[cls.backend].get(type(obj), None)
        options = plot.lookup_options(obj, 'plot').options
        width = options.get('width', plot.width) * factor
        height = options.get('height', plot.height) * factor
        return dict(options, **{'width':int(width), 'height': int(height)})


    @classmethod
    def load_nb(cls, inline=True):
        """
        Loads the plotly notebook resources.
        """
        import panel.models.plotly # noqa
        cls._loaded = True


def _activate_plotly_backend(renderer):
    if renderer == "plotly":
        pn.pane.HoloViews._panes["plotly"] = _PlotlyHoloviewsPane

Store._backend_switch_hooks.append(_activate_plotly_backend)
