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

// Restore axis range extents set by the user
for (var key in data.layout) {{
    if(key.slice(1, 5) === 'axis') {{
        if (plot.layout[key] && plot.layout[key].range) {{
            data.layout[key].range = plot.layout[key].range;
        }}
    }}
}}

plotly.relayout(plot, data.layout);
plotly.redraw(plot);
"""


def build_new_plot_js(plot_id,
                      data,
                      layout,
                      config,
                      comm_ids,
                      timeout,
                      debounce,
                      callback_handlers):

    # Based on JS_CALLBACK from pyviz_comms
    return """
var plotdiv = document.getElementById('{plot_id}');
var plotly = window._Plotly || window.Plotly;
plotly.newPlot(plotdiv, {data}, {layout}, {config}).then(function() {{
    var elem = document.getElementById("{plot_id}.loading");
    elem.parentNode.removeChild(elem);

    function unique_events(events) {{
        // Processes the event queue ignoring duplicate events
        // of the same type
        var unique = [];
        var unique_events = [];
        for (var i=0; i<events.length; i++) {{
            var _tmpevent = events[i];
            event = _tmpevent[0];
            data = _tmpevent[1];
            if (unique_events.indexOf(event)===-1) {{
                unique.unshift(data);
                unique_events.push(event);
            }}
        }}
        return unique;
    }}

    function process_events(comm_status) {{
        // Iterates over event queue and sends events via Comm
        var events = unique_events(comm_status.event_buffer);
        for (var i=0; i<events.length; i++) {{
            var data = events[i];
            var comm = window.PyViz.comms[data["comm_id"]];
            comm.send(data);
        }}
        comm_status.event_buffer = [];
    }}

    function on_msg(msg) {{
        // Receives acknowledgement from Python, processing event
        // and unblocking Comm if event queue empty
        msg = JSON.parse(msg.content.data);
        var comm_id = msg["comm_id"]
        var comm_status = window.PyViz.comm_status[comm_id];
        if (comm_status.event_buffer.length) {{
            process_events(comm_status);
            comm_status.blocked = true;
            comm_status.time = Date.now()+{debounce};
        }} else {{
            comm_status.blocked = false;
        }}
        comm_status.event_buffer = [];
        if ((msg.msg_type == "Ready") && msg.content) {{
            console.log("Python callback returned following output:", msg.content);
        }} else if (msg.msg_type == "Error") {{
            console.log("Python failed with the following traceback:", msg['traceback'])
        }}
    }}

    function send_msg(data, comm_id, event_name) {{
        // Initialize event queue and timeouts for Comm
        var comm_status = window.PyViz.comm_status[comm_id];
        if (comm_status === undefined) {{
            comm_status = {{event_buffer: [], blocked: false, time: Date.now()}}
            window.PyViz.comm_status[comm_id] = comm_status
        }}

        // Add current event to queue and process queue if not blocked
        data['comm_id'] = comm_id;
        timeout = comm_status.time + {timeout};
        if ((comm_status.blocked && (Date.now() < timeout))) {{
            comm_status.event_buffer.unshift([event_name, data]);
        }} else {{
            comm_status.event_buffer.unshift([event_name, data]);
            setTimeout(function() {{ process_events(comm_status); }}, {debounce});
            comm_status.blocked = true;
            comm_status.time = Date.now()+{debounce};
        }}

    }}

    // Initialize Comms
    var comm_ids = {comm_ids};
    for (var i=0; i<comm_ids.length; i++) {{
        var comm_id = comm_ids[i];
        if (window.PyViz.comm_manager == undefined) {{ return }}
        window.PyViz.comm_manager.get_client_comm("{plot_id}", comm_id, on_msg);
    }}

    {callback_handlers}
}})
""".format(plot_id=plot_id,
           data=data,
           layout=layout,
           config=config,
           timeout=timeout,
           debounce=debounce,
           callback_handlers=callback_handlers,
           comm_ids=comm_ids)


def _to_figure_uid(fig_dict):
    """
    Convert plotly figure dict to a graph_objs.Figure object while
    preserving the trace UIDs that we use to associate traces with comms
    on the JavaScript side
    """
    fig = go.Figure(fig_dict)
    for trace, state_trace in zip(fig.data, fig_dict.get('data', [])):
        trace.uid = state_trace.get('uid', trace.uid)
    return fig


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
            return json.dumps(_to_figure_uid(diff), cls=utils.PlotlyJSONEncoder)
        else:
            return diff


    def _figure_data(self, plot, fmt=None, divuuid=None, comm=True, as_script=False, width=800, height=600):
        # Wrapping plot.state in go.Figure here performs validation
        # and applies any default theme.
        figure = _to_figure_uid(plot.state)

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

        # Add handlers for each callback type
        callbacks = [cb for cbs in plot.traverse(
            lambda x: getattr(x, 'callbacks', [])) for cb in cbs]

        comm_ids = [cb.comm.id for cb in callbacks]
        jcomm_ids = json.dumps(comm_ids)

        callback_handlers = ''
        callback_classes = set([type(cb) for cb in callbacks])
        for callback_class in callback_classes:
            callbacks_of_class = [cb for cb in callbacks
                                  if isinstance(cb, callback_class)]
            if callbacks_of_class:
                callback_handlers += callback_class.build_callback_js(callbacks_of_class)

        timeout = 20000
        debounce = 20
        script = build_new_plot_js(
            plot_id=divuuid,
            data=jdata,
            layout=jlayout,
            config=jconfig,
            comm_ids=jcomm_ids,
            timeout=timeout,
            debounce=debounce,
            callback_handlers=callback_handlers)

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
