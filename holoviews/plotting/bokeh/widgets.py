from __future__ import unicode_literals

import math
import json
from functools import partial

import param
import numpy as np
from bokeh.models.widgets import Select, Slider, AutocompleteInput, TextInput, Div
from bokeh.layouts import widgetbox, row, column

from ...core import Store, NdMapping, OrderedDict
from ...core.util import (drop_streams, unique_array, isnumeric,
                          wrap_tuple_streams, unicode)
from ..renderer import MIME_TYPES
from ..widgets import NdWidget, SelectionWidget, ScrubberWidget
from .util import serialize_json



class BokehServerWidgets(param.Parameterized):
    """
    BokehServerWidgets create bokeh widgets corresponding to all the
    key dimensions found on a BokehPlot instance. It currently supports
    to types of widgets sliders (which may be discrete or continuous)
    and dropdown widgets letting you select non-numeric values.
    """

    display_options = param.Dict(default={}, doc="""
        Additional options controlling display options of the widgets.""")

    editable = param.Boolean(default=False, doc="""
        Whether the slider text fields should be editable. Disabled
        by default for a more compact widget layout.""")

    position = param.ObjectSelector(default='right',
        objects=['right', 'left', 'above', 'below'])

    sizing_mode = param.ObjectSelector(default='fixed',
        objects=['fixed', 'stretch_both', 'scale_width',
                 'scale_height', 'scale_both'])

    width = param.Integer(default=250, doc="""
        Width of the widget box in pixels""")

    basejs = param.String(default=None, precedence=-1, doc="""
        Defines the local CSS file to be loaded for this widget.""")

    extensionjs = param.String(default=None, precedence=-1, doc="""
        Optional javascript extension file for a particular backend.""")

    css = param.String(default=None, precedence=-1, doc="""
        Defines the local CSS file to be loaded for this widget.""")

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
                                  kdims=list(self.dimensions))
        self.widgets, self.lookups = self.get_widgets()
        self.subplots = {}
        if self.plot.renderer.mode == 'default':
            self.attach_callbacks()
        self.state = self.init_layout()
        self._queue = []
        self._active = False

        if hasattr(self.plot.document, 'on_session_destroyed'):
            self.plot.document.on_session_destroyed(self.plot._session_destroy)


    @classmethod
    def create_widget(self, dim, holomap=None, editable=False):
        """"
        Given a Dimension creates bokeh widgets to select along that
        dimension. For numeric data a slider widget is created which
        may be either discrete, if a holomap is supplied or the
        Dimension.values are set, or a continuous widget for
        DynamicMaps. If the slider is discrete the returned mapping
        defines a mapping between values and labels making it possible
        sync the two slider and label widgets. For non-numeric data
        a simple dropdown selection widget is generated.
        """
        label, mapping = None, None
        if holomap is None:
            if dim.values:
                if dim.default is None:
                    default = dim.values[0]
                elif dim.default not in dim.values:
                    raise ValueError("%s dimension default %r is not in dimension values: %s"
                                     % (dim, dim.default, dim.values))
                else:
                    default = dim.default
                value = dim.values.index(default)

                if all(isnumeric(v) for v in dim.values):
                    values = sorted(dim.values)
                    labels = [unicode(dim.pprint_value(v)) for v in values]
                    if editable:
                        label = AutocompleteInput(value=labels[value], completions=labels,
                                                  title=dim.pprint_label)
                    else:
                        label = Div(text='<b>%s</b>' % dim.pprint_value_string(labels[value]))
                    widget = Slider(value=value, start=0, end=len(dim.values)-1, title=None, step=1)
                    mapping = list(enumerate(zip(values, labels)))
                else:
                    values = [(v, dim.pprint_value(v)) for v in dim.values]
                    widget = Select(title=dim.pprint_label, value=values[value][0],
                                    options=values)
            else:
                start = dim.soft_range[0] if dim.soft_range[0] else dim.range[0]
                end = dim.soft_range[1] if dim.soft_range[1] else dim.range[1]
                dim_range = end - start
                int_type = isinstance(dim.type, type) and issubclass(dim.type, int)
                if dim.step is not None:
                    step = dim.step
                elif isinstance(dim_range, int) or int_type:
                    step = 1
                else:
                    step = 10**((round(math.log10(dim_range))-3))

                if dim.default is None:
                    default = start
                elif (dim.default < start or dim.default > end):
                    raise ValueError("%s dimension default %r is not in the provided range: %s"
                                     % (dim, dim.default, (start, end)))
                else:
                    default = dim.default

                if editable:
                    label = TextInput(value=str(default), title=dim.pprint_label)
                else:
                    label = Div(text='<b>%s</b>' % dim.pprint_value_string(default))
                widget = Slider(value=default, start=start,
                                end=end, step=step, title=None)
        else:
            values = (dim.values if dim.values else
                      list(unique_array(holomap.dimension_values(dim.name))))
            if dim.default is None:
                default = values[0]
            elif dim.default not in values:
                raise ValueError("%s dimension default %r is not in dimension values: %s"
                                 % (dim, dim.default, values))
            else:
                default = dim.default
            if isinstance(values[0], np.datetime64) or isnumeric(values[0]):
                values = sorted(values)
                labels = [dim.pprint_value(v) for v in values]
                value = values.index(default)
                if editable:
                    label = AutocompleteInput(value=labels[value], completions=labels,
                                              title=dim.pprint_label)
                else:
                    label = Div(text='<b>%s</b>' % (dim.pprint_value_string(labels[value])))
                widget = Slider(value=value, start=0, end=len(values)-1, title=None, step=1)
            else:
                labels = [dim.pprint_value(v) for v in values]
                widget = Select(title=dim.pprint_label, value=default,
                                options=list(zip(values, labels)))
            mapping = list(enumerate(zip(values, labels)))
        return widget, label, mapping


    def get_widgets(self):
        """
        Creates a set of widgets representing the dimensions on the
        plot object used to instantiate the widgets class.
        """
        widgets = OrderedDict()
        mappings = {}
        for dim in self.mock_obj.kdims:
            holomap = None if self.plot.dynamic else self.mock_obj
            widget, label, mapping = self.create_widget(dim, holomap, self.editable)
            if label is not None and not isinstance(label, Div):
                label.on_change('value', partial(self.on_change, dim, 'label'))
            widget.on_change('value', partial(self.on_change, dim, 'widget'))
            widgets[dim.pprint_label] = (label, widget)
            if mapping:
                mappings[dim.pprint_label] = OrderedDict(mapping)
        return widgets, mappings


    def init_layout(self):
        widgets = [widget for d in self.widgets.values()
                   for widget in d if widget]
        wbox = widgetbox(widgets, width=self.width)
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


    def on_change(self, dim, widget_type, attr, old, new):
        self._queue.append((dim, widget_type, attr, old, new))
        if not self._active:
            self.plot.document.add_timeout_callback(self.update, 50)
            self._active = True


    def update(self):
        """
        Handle update events on bokeh server.
        """
        if not self._queue:
            return

        dim, widget_type, attr, old, new = self._queue[-1]
        self._queue = []
        dim_label = dim.pprint_label

        label, widget = self.widgets[dim_label]
        if widget_type == 'label':
            if isinstance(label, AutocompleteInput):
                value = [new]
                widget.value = value
            else:
                widget.value = float(new)
        elif label:
            lookups = self.lookups.get(dim_label)
            if not self.editable:
                if lookups:
                    new = lookups[widget.value][1]
                label.text = '<b>%s</b>' % dim.pprint_value_string(new)
            elif isinstance(label, AutocompleteInput):
                text = lookups[new][1]
                label.value = text
            else:
                label.value = dim.pprint_value(new)

        key = []
        for dim, (label, widget) in self.widgets.items():
            lookups = self.lookups.get(dim)
            if label and lookups:
                val = lookups[widget.value][0]
            else:
                val = widget.value
            key.append(val)
        key = wrap_tuple_streams(tuple(key), self.plot.dimensions,
                                 self.plot.streams)
        self.plot.update(key)
        self._active = False



class BokehWidget(NdWidget):

    css = param.String(default='bokehwidgets.css', doc="""
        Defines the local CSS file to be loaded for this widget.""")

    extensionjs = param.String(default='bokehwidgets.js', doc="""
        Optional javascript extension file for a particular backend.""")

    def _get_data(self):
        # Get initial frame to draw immediately
        msg, metadata = self.renderer.components(self.plot, comm=False)
        data = super(BokehWidget, self)._get_data()
        return dict(data, init_html=msg['text/html'],
                    init_js=msg[MIME_TYPES['js']],
                    plot_id=self.plot.state._id)

    def encode_frames(self, frames):
        if self.export_json:
            self.save_json(frames)
            frames = {}
        else:
            frames = json.dumps(frames).replace('</', r'<\/')
        return frames

    def get_frames(self):
        nframes = len(self.plot)
        if self.embed:
            self.plot.update(nframes-1)
            frames = OrderedDict([(idx, self._plot_figure(idx))
                                  for idx in range(nframes)])
        else:
            frames = {}
        return self.encode_frames(frames)

    def _plot_figure(self, idx, fig_format='json'):
        """
        Returns the figure in html format on the
        first call and
        """
        self.plot.update(idx)
        if self.embed:
            patch = self.renderer.diff(self.plot, binary=False)
            msg = serialize_json(dict(content=patch.content,
                                      root=self.plot.state._id))
            return msg


class BokehSelectionWidget(BokehWidget, SelectionWidget):
    pass


class BokehScrubberWidget(BokehWidget, ScrubberWidget):
    pass
