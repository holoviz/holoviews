from __future__ import absolute_import, division, unicode_literals

import base64
import json

import param
with param.logging_level('CRITICAL'):
    from plotly.offline.offline import utils, get_plotlyjs, init_notebook_mode
    import plotly.graph_objs as go

from ..renderer import Renderer, MIME_TYPES, HTML_TAGS
from ...core.options import Store
from ...core import HoloMap
from .widgets import PlotlyScrubberWidget, PlotlySelectionWidget


plotly_msg_handler = """
/* Backend specific body of the msg_handler, updates displayed frame */
var plot = $('#{plot_id}')[0];
var data = JSON.parse(msg);
$.each(data.data, function(i, obj) {{
  $.each(Object.keys(obj), function(j, key) {{
    plot.data[i][key] = obj[key];
  }});
}});
var plotly = window._Plotly || window.Plotly;
plotly.relayout(plot, data.layout);
plotly.redraw(plot);
"""


class PlotlyRenderer(Renderer):

    backend = param.String(default='plotly', doc="The backend name.")

    fig = param.ObjectSelector(default='auto', objects=['html', 'json', 'png', 'svg', 'auto'], doc="""
        Output render format for static figures. If None, no figure
        rendering will occur. """)

    mode_formats = {'fig': {'default': ['html', 'png', 'svg', 'json']},
                    'holomap': {'default': ['widgets', 'scrubber', 'auto']}}

    widgets = {'scrubber': PlotlyScrubberWidget,
               'widgets': PlotlySelectionWidget}

    backend_dependencies = {'js': (get_plotlyjs(),)}

    comm_msg_handler = plotly_msg_handler

    _loaded = False

    def __call__(self, obj, fmt='html', divuuid=None):
        plot, fmt =  self._validate(obj, fmt)
        mime_types = {'file-ext':fmt, 'mime_type': MIME_TYPES[fmt]}

        if isinstance(plot, tuple(self.widgets.values())):
            return plot(), mime_types
        elif fmt in ('html', 'png', 'svg'):
            return self._figure_data(plot, fmt, divuuid=divuuid), mime_types
        elif fmt == 'json':
            return self.diff(plot), mime_types


    def diff(self, plot, serialize=True):
        """
        Returns a json diff required to update an existing plot with
        the latest plot data.
        """
        diff = plot.state
        if serialize:
            return json.dumps(diff, cls=utils.PlotlyJSONEncoder)
        else:
            return diff


    def _figure_data(self, plot, fmt=None, divuuid=None, comm=True, as_script=False, width=800, height=600):
        # Wrapping plot.state in go.Figure here performs validation
        # and applies any default theme.
        figure = go.Figure(plot.state)

        if fmt in ('png', 'svg'):
            import plotly.io as pio
            data = pio.to_image(figure, fmt)
            if as_script:
                b64 = base64.b64encode(data).decode("utf-8")
                (mime_type, tag) = MIME_TYPES[fmt], HTML_TAGS[fmt]
                src = HTML_TAGS['base64'].format(mime_type=mime_type, b64=b64)
                div = tag.format(src=src, mime_type=mime_type, css='')
                js = ''
                return div, js
            return data

        if divuuid is None:
            divuuid = plot.id

        jdata = json.dumps(figure.data, cls=utils.PlotlyJSONEncoder)
        jlayout = json.dumps(figure.layout, cls=utils.PlotlyJSONEncoder)

        config = {}
        config['showLink'] = False
        jconfig = json.dumps(config)

        if as_script:
            header = 'window.PLOTLYENV=window.PLOTLYENV || {};'
        else:
            header = ('<script type="text/javascript">'
                      'window.PLOTLYENV=window.PLOTLYENV || {};'
                      '</script>')

        script = '\n'.join([
            'var plotly = window._Plotly || window.Plotly;'
            'plotly.plot("{id}", {data}, {layout}, {config}).then(function() {{',
            '    var elem = document.getElementById("{id}.loading"); elem.parentNode.removeChild(elem);',
            '}})']).format(id=divuuid,
                           data=jdata,
                           layout=jlayout,
                           config=jconfig)

        html = ('<div id="{id}.loading" style="color: rgb(50,50,50);">'
                'Drawing...</div>'
                '<div id="{id}" style="height: {height}; width: {width};" '
                'class="plotly-graph-div">'
                '</div>'.format(id=divuuid, height=height, width=width))
        if as_script:
            return html, header + script

        content = (
            '{html}'
            '<script type="text/javascript">'
            '  {script}'
            '</script>'
        ).format(html=html, script=script)
        return '\n'.join([header, content])


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
        from IPython.display import publish_display_data
        cls._loaded = True
        init_notebook_mode(connected=not inline)
        publish_display_data(data={MIME_TYPES['jlab-hv-load']:
                                   get_plotlyjs()})
