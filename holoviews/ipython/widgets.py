import os, sys, math, time, uuid, json, warnings
from unittest import SkipTest

import numpy as np

try:
    import IPython
    from IPython.core.display import clear_output
except:
    clear_output = None
    raise SkipTest("IPython extension requires IPython >= 0.12")
from IPython.display import display
try:
    # Silence the annoying FutureWarning in IPython 3.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from IPython.html import widgets
        from IPython.html.widgets import FloatSliderWidget
except:
    widgets = None
    FloatSliderWidget = object

# IPython 0.13 does not have version_info
ipython2 = hasattr(IPython, 'version_info') and (IPython.version_info[0] == 2)

import param

from ..core import OrderedDict, NdMapping
from ..core.util import ProgressIndicator
from ..plotting import Plot
from .magics import OutputMagic


class ProgressBar(ProgressIndicator):
    """
    A simple text progress bar suitable for both the IPython notebook
    and the IPython interactive prompt.
    """

    display = param.ObjectSelector(default='stdout',
                  objects=['stdout', 'disabled', 'broadcast'], doc="""
       Parameter to control display of the progress bar. By default,
       progress is shown on stdout but this may be disabled e.g. for
       jobs that log standard output to file.

       If the output mode is set to 'broadcast', a socket is opened on
       a stated port to broadcast the completion percentage. The
       RemoteProgress class may then be used to view the progress from
       a different process.""")

    width = param.Integer(default=70, doc="""
        The width of the progress bar as the number of chararacters""")

    fill_char = param.String(default='#', doc="""
        The character used to fill the progress bar.""")

    blank_char = param.String(default=' ', doc="""
        The character for the blank portion of the progress bar.""")

    elapsed_time = param.Boolean(default=True, doc="""
        If enabled, the progress bar will disappear and display the
        total elapsed time once 100% completion is reached.""")

    cache = {}

    def __init__(self, **params):
        self.start_time = None
        super(ProgressBar,self).__init__(**params)

    def __call__(self, percentage):
        " Update the progress bar within the specified percent_range"
        if self.start_time is None: self.start_time = time.time()
        span = (self.percent_range[1]-self.percent_range[0])
        percentage = self.percent_range[0] + ((percentage/100.0) * span)

        if self.display == 'disabled': return
        elif self.display == 'stdout':
            if percentage==100 and self.elapsed_time:
                elapsed = time.time() -  self.start_time
                if clear_output and not ipython2: clear_output()
                if clear_output and ipython2: clear_output(wait=True)
                sys.stdout.write('\r' + '100%% %s %02d:%02d:%02d'
                                 % (self.label.lower(), elapsed//3600,
                                    elapsed//60, elapsed%60))
                return
            else:
                self._stdout_display(percentage)
            return

        if 'socket' not in self.cache:
            self.cache['socket'] = self._get_socket()

        if self.cache['socket'] is not None:
            self.cache['socket'].send('%s|%s' % (percentage, self.label))


    def _stdout_display(self, percentage):
        if clear_output and not ipython2: clear_output()
        if clear_output and ipython2: clear_output(wait=True)
        percent_per_char = 100.0 / self.width
        char_count = int(math.floor(percentage/percent_per_char)
                         if percentage<100.0 else self.width)
        blank_count = self.width - char_count
        sys.stdout.write('\r' + "%s[%s%s] %0.1f%%"
                         % (self.label+':\n' if self.label else '',
                            self.fill_char * char_count,
                            ' '*len(self.fill_char) * blank_count,
                            percentage))
        sys.stdout.flush()
        time.sleep(0.0001)

    def _get_socket(self, min_port=8080, max_port=8100, max_tries=20):
        import zmq
        context = zmq.Context()
        sock = context.socket(zmq.PUB)
        try:
            port = sock.bind_to_random_port('tcp://*',
                                            min_port=min_port,
                                            max_port=max_port,
                                            max_tries=max_tries)
            self.message("Progress broadcast bound to port %d" % port)
            return sock
        except:
            self.message("No suitable port found for progress broadcast.")
            return None


class RemoteProgress(ProgressBar):
    """
    Connect to a progress bar in a separate process with output_mode
    set to 'broadcast' in order to display the results (to stdout).
    """

    hostname=param.String(default='localhost', doc="""
      Hostname where progress is being broadcast.""")

    port = param.Integer(default=8080,
                         doc="""Target port on hostname.""")

    def __init__(self, port, **params):
        super(RemoteProgress, self).__init__(port=port, **params)

    def __call__(self):
        import zmq
        context = zmq.Context()
        sock = context.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, '')
        sock.connect('tcp://' + self.hostname +':'+str(self.port))
        # Get progress via socket
        percent = None
        while True:
            try:
                message= sock.recv()
                [percent_str, label] = message.split('|')
                percent = float(percent_str)
                self.label = label
                super(RemoteProgress, self).__call__(percent)
            except KeyboardInterrupt:
                if percent is not None:
                    self.message("Exited at %.3f%% completion" % percent)
                break
            except:
                self.message("Could not process socket message: %r"
                             % message)


class RunProgress(ProgressBar):
    """
    RunProgress breaks up the execution of a slow running command so
    that the level of completion can be displayed during execution.

    This class is designed to run commands that take a single numeric
    argument that acts additively. Namely, it is expected that a slow
    running command 'run_hook(X+Y)' can be arbitrarily broken up into
    multiple, faster executing calls 'run_hook(X)' and 'run_hook(Y)'
    without affecting the overall result.

    For instance, this is suitable for simulations where the numeric
    argument is the simulated time - typically, advancing 10 simulated
    seconds takes about twice as long as advancing by 5 seconds.
    """

    interval = param.Number(default=100, doc="""
        The run interval used to break up updates to the progress bar.""")

    run_hook = param.Callable(default=param.Dynamic.time_fn.advance, doc="""
        By default updates time in param which is very fast and does
        not need a progress bar. Should be set to some slower running
        callable where display of progress level is desired.""")


    def __init__(self, **params):
        super(RunProgress,self).__init__(**params)

    def __call__(self, value):
        """
        Execute the run_hook to a total of value, breaking up progress
        updates by the value specified by interval.
        """
        completed = 0
        while (value - completed) >= self.interval:
            self.run_hook(self.interval)
            completed += self.interval
            super(RunProgress, self).__call__(100 * (completed / float(value)))
        remaining = value - completed
        if remaining != 0:
            self.run_hook(remaining)
            super(RunProgress, self).__call__(100)



class FixedValueSliderWidget(FloatSliderWidget):
    """
    Subclass of FloatSliderWidget that jumps discretely
    between a set of supplied values.
    """

    def __init__(self, values=[], *args, **kwargs):
        value = round(values[0], 5)
        vmin = min(values)
        vmax = max(values)
        step = min(abs(np.diff(values))) if len(values) > 1 else 0
        self.values = np.array(values)
        widgets.DOMWidget.__init__(self, step=step, min=vmin, max=vmax,
                                   value=value, *args, **kwargs)
        self.on_trait_change(self._snap_value, ['value'])
        self.time = time.time()


    def _snap_value(self, name, old_val, new_val):
        """
        Snap value to the closest specified value.
        """
        if self.time+0.05 > time.time():
            return
        diffs = np.abs(self.values - new_val)
        idx = np.argmin(diffs)
        val = self.values[idx]
        if val != self.value:
            self.value = round(val, 5)
            self.time = time.time()


def isnumeric(val):
    try:
        float(val)
        return True
    except:
        return False


def get_plot_size():
    factor = OutputMagic.options['size'] / 100.0
    return (Plot.figure_inches[0] * factor,
            Plot.figure_inches[1] * factor)


class NdWidget(param.Parameterized):
    """
    NdWidget is an abstract base class implementing a method to
    find the dimensions and keys of any ViewableElement, GridSpace or UniformNdMapping type.
    In the process it creates a mock_obj to hold the dimensions
    and keys.
    """

    def __init__(self, plot, **params):
        super(NdWidget, self).__init__(**params)
        self.plot = plot
        self.dimensions = plot.dimensions
        self.keys = plot.keys
        # Create mock NdMapping to hold the common dimensions and keys
        self.mock_obj = NdMapping([(k, None) for k in self.keys],
                                  key_dimensions=self.dimensions)


    def _plot_figure(self, idx):
        from .display_hooks import display_figure
        fig = self.plot[idx]
        if OutputMagic.options['backend'] == 'd3':
            import mpld3
            mpld3.plugins.connect(fig, mpld3.plugins.MousePosition(fontsize=14))
            return mpld3.fig_to_dict(fig)
        return display_figure(fig)



class IPySelectionWidget(NdWidget):
    """
    Interactive widget to select and view ViewableElement objects contained
    in an NdMapping. ViewSelector creates Slider and Dropdown widgets
    for each dimension contained within the supplied object and
    an image widget for the plotted ViewableElement. All widgets are dynamically
    updated to match the current selection.
    """

    cached = param.Boolean(default=True, doc="""
        Whether to cache the ViewableElement plots when initializing the object.""")

    css = param.Dict(default={'margin-left': 'auto',
                              'margin-right': 'auto'}, doc="""
                              CSS to apply to the widgets.""")

    def __init__(self, plot, **params):
        super(IPySelectionWidget, self).__init__(plot, **params)

        if widgets is None:
            raise ImportError('ViewSelector requires IPython >= 2.0.')

        self._initialize_widgets()
        self.refresh = True

        if self.cached:
            self.frames = OrderedDict((k, self._plot_figure(idx))
                                      for idx, k in enumerate(self.keys))


    def _initialize_widgets(self):
        """
        Initialize widgets and dimension values.
        """

        self.pwidgets = {}
        self.dim_val = {}
        for didx, dim in enumerate(self.mock_obj.key_dimensions):
            all_vals = [k[didx] for k in self.keys]

            # Initialize dimension value
            vals = self._get_dim_vals(list(self.keys[0]), didx)
            self.dim_val[dim.name] = vals[0]

            # Initialize widget
            if isnumeric(vals[0]):
                widget_type = FixedValueSliderWidget
            else:
                widget_type = widgets.DropdownWidget
                all_vals = dict((str(v), v) for v in all_vals)
            self.pwidgets[dim.name] = widget_type(values=sorted(set(all_vals)))


    def __call__(self):
        # Initalize image widget
        if (OutputMagic.options['backend'] == 'mpld3'
            or OutputMagic.options['fig'] =='svg'):
            self.image_widget = widgets.HTMLWidget()
        else:
            self.image_widget = widgets.ImageWidget()

        if self.cached:
            self.image_widget.value = list(self.frames.values())[0]
        else:
            self.image_widget.value = self._plot_figure(0)
        self.image_widget.set_css(self.css)

        # Initialize interactive widgets
        interactive_widget = widgets.interactive(self.update_widgets,
                                                 **self.pwidgets)
        interactive_widget.set_css(self.css)

        # Display widgets
        display(interactive_widget)
        display(self.image_widget)
        return '' # Suppresses outputting ViewableElement repr when called through hook


    def _get_dim_vals(self, indices, idx):
        """
        Get the dimension values along the supplied dimension,
        computed from the supplied indices into the mock_obj.
        """
        indices[idx] = slice(None)
        vals = [k[idx] if isinstance(k, tuple) else k
                for k in self.mock_obj[tuple(indices)].keys()]
        return vals


    def update_widgets(self, **kwargs):
        """
        Callback method to process the new keys, find the closest matching
        ViewableElement and update all the widgets.
        """

        # Do nothing if dimension values are unchanged
        if all(v == self.dim_val[k] for k, v in kwargs.items()):
            return

        # Place changed dimensions first
        changed_fn = lambda x: x[1] == self.dim_val[x[0]]
        dimvals = sorted(kwargs.items(), key=changed_fn)

        # Find the closest matching key along each dimension and update
        # the matching widget accordingly.
        checked = [slice(None) for i in range(self.mock_obj.ndims)]
        for dim, val in dimvals:
            if not isnumeric(val): val = str(val)
            dim_idx = self.mock_obj.get_dimension_index(dim)
            widget = self.pwidgets[dim]
            vals = self._get_dim_vals(checked, dim_idx)
            if val not in vals:
                if isnumeric(val):
                    val = vals[np.argmin(np.abs(np.array(vals) - val))]
                else:
                    val = str(vals[0])
            checked[dim_idx] = val
            self.dim_val[dim] = val
            widget.value = round(val, 5) if isnumeric(val) else val

        # Update frame
        checked = tuple(checked)
        if self.cached:
            self.image_widget.value = self.frames[checked]
        else:
            self.image_widget.value = self._plot_figure(self.keys.index(checked))



class ScrubberWidget(NdWidget):
    """
    ScrubberWidget generates a basic animation widget with a slider
    and various play/rewind/stepping options. It has been adapted
    from Jake Vanderplas' JSAnimation library, which was released
    under BSD license.

    Optionally the individual plots can be exported to json, which can
    be dynamically loaded by serving the data the data for each frame
    on a simple server.
    """

    template = param.String('jsscrubber.jinja', doc="""
        The jinja2 template used to generate the html output.""")

    #######################
    # JSON export options #
    #######################

    export_json = param.Boolean(default=False, doc="""Whether to export
         plots as json files, which can be dynamically loaded through
         a callback from the slider.""")

    json_path = param.String(default='./json_figures', doc="""
         If export_json is True the json files will be written to this
         directory.""")

    server_url = param.String(default='', doc="""If export_json is
         True the slider widget will expect to be served the plot data
         from this URL. Data should be served from:
         server_url/fig_{id}/{frame}.""")

    ##############################
    # Javascript include options #
    ##############################

    mpld3_url = 'https://mpld3.github.io/js/mpld3.v0.3git.js'
    d3_url = 'https://cdnjs.cloudflare.com/ajax/libs/d3/3.4.13/d3.js'

    def __init__(self, plot, **params):
        super(ScrubberWidget, self).__init__(plot, **params)
        self.frames = OrderedDict((idx, self._plot_figure(idx))
                                  for idx in range(len(self.plot)))


    def get_frames(self, id):
        use_mpld3 = OutputMagic.options['backend'] == 'd3'
        frames = {idx: frame if use_mpld3 or self.export_json else
                  str(frame) for idx, frame in enumerate(self.frames.values())}
        encoder = {}
        if use_mpld3:
            import mpld3
            encoder = dict(cls=mpld3._display.NumpyEncoder)

        if self.export_json:
            if not os.path.isdir(self.json_path):
                os.mkdir(self.json_path)
            with open(self.json_path+'/fig_%s.json' % id, 'wb') as f:
                json.dump(frames, f, **encoder)
            frames = {}
        elif use_mpld3:
            frames = json.dumps(frames, **encoder)
        return frames


    def render_html(self, data):
        # Set up jinja2 templating
        import jinja2
        path, _ = os.path.split(os.path.abspath(__file__))
        templateLoader = jinja2.FileSystemLoader(searchpath=path)
        templateEnv = jinja2.Environment(loader=templateLoader)
        template = templateEnv.get_template(self.template)

        return template.render(**data)


    def __call__(self):
        id = uuid.uuid4().hex
        frames = self.get_frames(id)

        data = {'id': id, 'Nframes': len(self.plot),
                'interval': int(1000. / OutputMagic.options['fps']),
                'frames': frames,
                'load_json': str(self.export_json).lower(),
                'server': self.server_url,
                'mpld3_url': self.mpld3_url,
                'd3_url': self.d3_url[:-3],
                'mpld3': str(OutputMagic.options['backend'] == 'd3').lower()}

        return self.render_html(data)



class SelectionWidget(ScrubberWidget):
    """
    Javascript based widget to select and view ViewableElement objects contained
    in an NdMapping. For each dimension in the NdMapping a slider or
    dropdown selection widget is created and can be used to select the
    html output associated with the selected ViewableElement type. Supports
    selection of any HoloViews static output type including png, svg
    and mpld3 output. Unlike the IPSelectionWidget, this widget type
    is exportable to a static html.

    Just like the ScrubberWidget the data can be optionally saved
    to json and dynamically loaded from a server.
    """

    template = param.String('jsslider.jinja', doc="""
        The jinja2 template used to generate the html output.""")

    ##############################
    # Javascript include options #
    ##############################

    jqueryui_url = 'https://code.jquery.com/ui/1.10.4/jquery-ui.min.js'

    def __init__(self, plot, **params):
        NdWidget.__init__(self, plot, **params)
        self.frames = OrderedDict((k, self._plot_figure(idx))
                                  for idx, k in enumerate(self.keys))


    def get_widgets(self):
        # Generate widget data
        widgets = []
        dimensions = []
        init_dim_vals = []
        for idx, dim in enumerate(self.mock_obj.key_dimensions):
            dim_vals = dim.values if dim.values else sorted(set(self.mock_obj.dimension_values(dim.name)))
            dim_vals = [v for v in dim_vals if v is not None]
            if isnumeric(dim_vals[0]):
                dim_vals = [round(v, 10) for v in dim_vals]
                widget_type = 'slider'
            else:
                widget_type = 'dropdown'
            init_dim_vals.append(dim_vals[0])
            dim_str = dim.name.replace(' ', '_')
            visibility = 'visibility: visible' if len(dim_vals) > 1 else 'visibility: hidden; height: 0;'
            widgets.append(dict(dim=dim_str, dim_idx=idx, vals=repr(dim_vals),
                                type=widget_type, visibility=visibility))
            dimensions.append(dim_str)
        return widgets, dimensions, init_dim_vals


    def get_key_data(self):
        # Generate key data
        key_data = OrderedDict()
        for i, k in enumerate(self.mock_obj.data.keys()):
            key = [("%.1f" % v if v % 1 == 0 else "%.10f" % v)
                   if isnumeric(v) else v for v in k]
            key_data[str(tuple(key))] = i
        return json.dumps(key_data)


    def __call__(self):
        id = uuid.uuid4().hex
        widgets, dimensions, init_dim_vals = self.get_widgets()
        key_data = self.get_key_data()
        frames = self.get_frames(id)

        data = {'id': id, 'Nframes': len(self.mock_obj),
                'Nwidget': self.mock_obj.ndims,
                'frames': frames, 'dimensions': dimensions,
                'key_data': key_data, 'widgets': widgets,
                'init_dim_vals': init_dim_vals,
                'load_json': str(self.export_json).lower(),
                'server': self.server_url,
                'mpld3_url': self.mpld3_url,
                'jqueryui_url': self.jqueryui_url[:-3],
                'd3_url': self.d3_url[:-3],
                'delay': int(1000./OutputMagic.options['fps']),
                'notFound': "<h2 style='vertical-align: middle'>No frame at selected dimension value.<h2>",
                'mpld3': str(OutputMagic.options['backend'] == 'd3').lower()}

        return self.render_html(data)



def progress(iterator, enum=False, length=None):
    """
    A helper utility to display a progress bar when iterating over a
    collection of a fixed length or a generator (with a declared
    length).

    If enum=True, then equivalent to enumerate with a progress bar.
    """
    progress = ProgressBar()
    length = len(iterator) if length is None else length
    gen = enumerate(iterator)
    while True:
        i, val = next(gen)
        progress((i+1.0)/length * 100)
        if enum:
            yield i, val
        else:
            yield val
