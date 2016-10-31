from __future__ import unicode_literals

import json
from functools import partial

import param
from bokeh.io import _CommsHandle
from bokeh.util.notebook import get_comms
from bokeh.models.widgets import Select, Slider, AutocompleteInput, TextInput
from bokeh.layouts import layout, gridplot, widgetbox, row, column

from ...core import Store, NdMapping, OrderedDict
from ...core.util import drop_streams, unique_array, isnumeric, wrap_tuple_streams
from ..widgets import NdWidget, SelectionWidget, ScrubberWidget
from .util import serialize_json



class BokehServerWidgets(param.Parameterized):
    """
    """

    position = param.ObjectSelector(default='right',
        objects=['right', 'left', 'above', 'below'])

    sizing_mode = param.ObjectSelector(default='fixed',
        objects=['fixed', 'stretch_both', 'scale_width',
                 'scale_height', 'scale_both'])

    def __init__(self, plot, renderer=None, **params):
        super(BokehServerWidgets, self).__init__(**params)
        self.plot = plot
        streams = []
        for stream in plot.streams:
            if any(k in plot.dimensions for k in stream.contents):
                streams.append(stream)
        self.dimensions, self.keys = drop_streams(streams,
                                                  plot.dimensions,
                                                  plot.keys)
        if renderer is None:
            backend = Store.current_backend
            self.renderer = Store.renderers[backend]
        else:
            self.renderer = renderer
        # Create mock NdMapping to hold the common dimensions and keys
        self.mock_obj = NdMapping([(k, None) for k in self.keys],
                                  kdims=self.dimensions)
        self.widgets, self.lookups = self.get_widgets()
        self.reverse_lookups = {d: {v: k for k, v in item.items()}
                                for d, item in self.lookups.items()}
        self.subplots = {}
        if self.plot.renderer.mode == 'default':
            self.attach_callbacks()
        self.state = self.init_layout()


    def get_widgets(self):
        # Generate widget data
        widgets = OrderedDict()
        lookups = {}
        for idx, dim in enumerate(self.mock_obj.kdims):
            label, lookup = None, None
            if self.plot.dynamic:
                if dim.values:
                    if all(isnumeric(v) for v in dim.values):
                        values = dim.values
                        labels = [unicode(dim.pprint_value(v)) for v in dim.values]
                        label = AutocompleteInput(value=labels[0], completions=labels,
                                                  title=dim.pprint_label)
                        widget = Slider(value=0, end=len(dim.values)-1, title=None, step=1)
                        lookup = zip(values, labels)
                    else:
                        values = [(v, dim.pprint_value(v)) for v in dim.values]
                        widget = Select(title=dim.pprint_label, value=dim_vals[0][0],
                                        options=values)
                else:
                    start = dim.soft_range[0] if dim.soft_range[0] else dim.range[0]
                    end = dim.soft_range[1] if dim.soft_range[1] else dim.range[1]
                    int_type = isinstance(dim.type, type) and issubclass(dim.type, int)
                    if isinstance(dim_range, int) or int_type:
                        step = 1
                    else:
                        step = 10**(round(math.log10(dim_range))-3)
                    label = TextInput(value=str(start), title=dim.pprint_label)
                    widget = Slider(value=start, start=start,
                                    end=end, step=step, title=None)
            else:
                values = (dim.values if dim.values else
                            list(unique_array(self.mock_obj.dimension_values(dim.name))))
                labels = [str(dim.pprint_value(v)) for v in values]
                if isinstance(values[0], np.datetime64) or isnumeric(values[0]):
                    label = AutocompleteInput(value=labels[0], completions=labels,
                                              title=dim.pprint_label)
                    widget = Slider(value=0, end=len(dim.values)-1, title=None)
                else:
                    widget = Select(title=dim.pprint_label, value=values[0],
                                    options=list(zip(values, labels)))
                lookup = zip(values, labels)
            if label:
                label.on_change('value', partial(self.update, dim.pprint_label, 'label'))
            widget.on_change('value', partial(self.update, dim.pprint_label, 'widget'))
            widgets[dim.pprint_label] = (label, widget)
            if lookup:
                lookups[dim.pprint_label] = OrderedDict(lookup)
        return widgets, lookups


    def init_layout(self):
        widgets = [widget for d in self.widgets.values()
                   for widget in d if widget]
        wbox = widgetbox(widgets, width=200)
        if self.position in ['right', 'below']:
            plots = [self.plot.state, wbox]
        else:
            plots = [wbox, self.plot.state]
        layout_fn = row if self.position in ['left', 'right'] else column
        layout = layout_fn(plots, sizing_mode=self.sizing_mode)
        return layout


    def attach_callbacks(self):
        """
        Attach callbacks to interact with Comms.
        """
        pass


    def update(self, dim, widget_type, attr, old, new):
        """
        Handle update events on bokeh server.
        """
        label, widget = self.widgets[dim]
        if widget_type == 'label':
            if isinstance(label, AutocompleteInput):
                value = self.reverse_lookups[dim][new]
                widget.value = value
            else:
                widget.value = new
        else:
            if label:
                text = self.lookups[dim][new]
                label.value = text
        key = []
        for dim, (label, widget) in self.widgets.items():
            if label:
                if isinstance(label, AutocompleteInput):
                    val = self.lookups[dim].keys()[widget.value]
                else:
                    val = new
            else:
                val = widget.value
            key.append(val)
        key = wrap_tuple_streams(tuple(key), self.plot.dimensions,
                                 self.plot.streams)
        self.plot.update(key)



class BokehWidget(NdWidget):

    css = param.String(default='bokehwidgets.css', doc="""
        Defines the local CSS file to be loaded for this widget.""")

    extensionjs = param.String(default='bokehwidgets.js', doc="""
        Optional javascript extension file for a particular backend.""")

    def _get_data(self):
        # Get initial frame to draw immediately
        init_frame = self._plot_figure(0, fig_format='html')
        data = super(BokehWidget, self)._get_data()
        return dict(data, init_frame=init_frame)

    def encode_frames(self, frames):
        if self.export_json:
            self.save_json(frames)
            frames = {}
        else:
            frames = json.dumps(frames).replace('</', r'<\/')
        return frames

    def _plot_figure(self, idx, fig_format='json'):
        """
        Returns the figure in html format on the
        first call and
        """
        self.plot.update(idx)
        if self.embed or fig_format == 'html':
            if fig_format == 'html':
                msg = self.renderer.html(self.plot, fig_format)
            else:
                json_patch = self.renderer.diff(self.plot, serialize=False)
                msg = dict(patch=json_patch, root=self.plot.state._id)
                msg = serialize_json(msg)
            return msg

class BokehSelectionWidget(BokehWidget, SelectionWidget):
    pass

class BokehScrubberWidget(BokehWidget, ScrubberWidget):
    pass
