import json
from holoviews.streams import Stream, Selection1D, RangeXY


class MessageCallback(object):
    """
    A MessageCallback is an abstract baseclass used to supply Streams
    with events originating from plotly plot interactions. The baseclass
    defines how messages are handled and the basic specification required
    to define a Callback.
    """

    def __init__(self, plot, streams, source, **params):
        self.plot = plot
        self.streams = streams
        if plot.renderer.mode != 'server':
            self.comm = plot.renderer.comm_manager.get_client_comm(
                on_msg=self.on_msg)
        else:
            self.comm = None
        self.source = source

    def cleanup(self):
        self.plot = None
        self.source = None
        self.streams = []
        if self.comm:
            try:
                self.comm.close()
            except:
                pass

    def on_msg(self, msg):
        streams = []
        for stream in self.streams:
            processed_msg = self._process_msg(msg)
            if not processed_msg:
                continue

            stream.update(**processed_msg)
            streams.append(stream)
        try:
            Stream.trigger(streams)
        except Exception as e:
            raise e

    def _process_msg(self, msg):
        """
        Subclassable method to preprocess JSON message in callback
        before passing to stream.
        """
        return msg


class Selection1DCallback(MessageCallback):
    @staticmethod
    def build_callback_js(callbacks):
        trace_to_comm = {cb.plot.trace_uid: cb.comm.id for cb in callbacks}
        return """
    plotdiv.on('plotly_selected', function(eventData) {{
        var trace_to_comm = {trace_to_comm};
        
        if (eventData === undefined) {{
            return;
        }}

        // Initialize selection array for each trace
        var inds_by_trace = {{}};
        for (var trace_uid in trace_to_comm) {{
            inds_by_trace[trace_uid] = []
        }}

        // Append selection indexes per trace
        eventData.points.forEach(function(pt) {{
            var trace_uid = plotdiv.data[pt.curveNumber].uid;
            var inds = inds_by_trace[trace_uid];
            if (inds !== undefined) {{
                inds.push(pt.pointNumber);
            }}
        }});

        // Send selection1d messages
        for (var trace_uid in trace_to_comm) {{
            var comm_id = trace_to_comm[trace_uid];
            var data = {{index: inds_by_trace[trace_uid]}}
            send_msg(data, comm_id, 'selection1d')
        }}
    }});
""".format(trace_to_comm=json.dumps(trace_to_comm))


class RangeXYCallback(MessageCallback):
    @staticmethod
    def build_callback_js(callbacks):
        trace_to_comm = {cb.plot.trace_uid: cb.comm.id for cb in callbacks}
        return """
    plotdiv.on('plotly_relayout', function(eventData) {{
        var trace_to_comm = {trace_to_comm};
        
        if (eventData === undefined) {{
            return;
        }}
        
        var traces = plotdiv._fullData;
        for (var trace_uid in trace_to_comm) {{
            for (var trace_ind=0; trace_ind<traces.length; trace_ind++) {{
                if (traces[trace_ind].uid === trace_uid) {{
                    var xaxis = traces[trace_ind].xaxis
                    var yaxis = traces[trace_ind].yaxis
                    
                    if (xaxis === undefined || yaxis === undefined) {{
                        continue
                    }}
                    
                    var xaxis_subplot = "xaxis" + xaxis.slice(1);
                    var yaxis_subplot = "yaxis" + xaxis.slice(1);

                    if (eventData[xaxis_subplot + '.range[0]'] !== undefined ||
                        eventData[xaxis_subplot + '.autorange'] !== undefined ||
                        eventData[yaxis_subplot + '.range[0]'] !== undefined ||
                        eventData[yaxis_subplot + '.autorange'] !== undefined) {{
                        
                        var data = {{
                            x_range: plotdiv._fullLayout[xaxis_subplot]['range'],
                            y_range: plotdiv._fullLayout[yaxis_subplot]['range']
                        }}
                        var comm_id = trace_to_comm[trace_uid];
                        send_msg(data, comm_id, 'rangexy');
                    }}
                }}
            }}
        }}
        
    }});
""".format(trace_to_comm=json.dumps(trace_to_comm))

    def _process_msg(self, msg):
        """
        Subclassable method to preprocess JSON message in callback
        before passing to stream.
        """
        # Convert x_range and y_range to tuples
        msg['x_range'] = tuple(msg['x_range'])
        msg['y_range'] = tuple(msg['y_range'])
        return msg


callbacks = Stream._callbacks['plotly']
callbacks[Selection1D] = Selection1DCallback
callbacks[RangeXY] = RangeXYCallback
