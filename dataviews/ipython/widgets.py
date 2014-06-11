import sys, math, time

import numpy as np
from collections import OrderedDict

from nose.plugins.skip import SkipTest

try:
    import IPython
    from IPython.core.display import clear_output
except:
    clear_output = None
    raise SkipTest("IPython extension requires IPython >= 0.12")
from IPython.display import display
from IPython.core.pylabtools import print_figure
try:
    from IPython.html import widgets
    from IPython.html.widgets import FloatSliderWidget
except:
    widgets = None
    FloatSliderWidget = object

try:
    import mpld3
except:
    mpld3 = None
ipython2 = (IPython.version_info[0] == 2)

import param

from .. import GridLayout, NdMapping, Stack
from ..options import options
from ..views import Layout, Overlay, View
from ..plots import GridLayoutPlot, Plot
from .magics import ViewMagic


class ProgressBar(param.Parameterized):
    """
    A simple text progress bar suitable for both the IPython notebook
    and the IPython interactive prompt.
    """

    label = param.String(default='Progress', allow_None=True, doc="""
        The label of the current progress bar.""")

    width = param.Integer(default=70, doc="""
        The width of the progress bar as the number of chararacters""")

    fill_char = param.String(default='#', doc="""
        The character used to fill the progress bar.""")

    blank_char = param.String(default=' ', doc="""
        The character for the blank portion of the progress bar.""")

    percent_range = param.NumericTuple(default=(0.0,100.0), doc="""
        The total percentage spanned by the progress bar when called
        with a value between 0% and 100%. This allows an overall
        completion in percent to be broken down into smaller sub-tasks
        that individually complete to 100 percent.""")

    def __init__(self, **kwargs):
        super(ProgressBar,self).__init__(**kwargs)

    def __call__(self, percentage):
        " Update the progress bar within the specified percent_range"
        span = (self.percent_range[1]-self.percent_range[0])
        percentage = self.percent_range[0] + ((percentage/100.0) * span)
        if clear_output and not ipython2: clear_output()
        if clear_output and ipython2: clear_output(wait=True)
        percent_per_char = 100.0 / self.width
        char_count = int(math.floor(percentage/percent_per_char)
                         if percentage<100.0 else self.width)
        blank_count = self.width - char_count
        sys.stdout.write('\r' + "%s[%s%s] %0.1f%%" % (self.label+':\n' if self.label else '',
                                                      self.fill_char * char_count,
                                                      ' '*len(self.fill_char)*blank_count,
                                                      percentage))
        sys.stdout.flush()
        time.sleep(0.0001)



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


    def __init__(self, **kwargs):
        super(RunProgress,self).__init__(**kwargs)

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
    factor = ViewMagic.PERCENTAGE_SIZE / 100.0
    return (Plot.size[0] * factor,
            Plot.size[1] * factor)


class ViewSelector(param.Parameterized):
    """
    Interactive widget to select and view View objects contained
    in an NdMapping. ViewSelector creates Slider and Dropdown widgets
    for each dimension contained within the supplied object and
    an image widget for the plotted View. All widgets are dynamically
    updated to match the current selection.
    """

    cached = param.Boolean(default=True, doc="""
        Whether to cache the View plots when initializing the object.""")

    css = param.Dict(default={'margin-left': 'auto',
                              'margin-right': 'auto'}, doc="""
                              CSS to apply to the widgets.""")

    def __init__(self, view, **params):
        super(ViewSelector, self).__init__(**params)

        if widgets is None:
            raise ImportError('ViewSelector requires IPython >= 2.0.')

        self._process_view(view)
        self._initialize_widgets()
        self.refresh = True

        if self.cached:
            self.frames = OrderedDict((k, self._plot_figure(idx))
                                      for idx, k in enumerate(self._keys))

    def _plot_figure(self, idx):
        if ViewMagic.FIGURE_FORMAT == 'mpld3' and mpld3:
            fig = self.plot[idx]
            from mpld3 import plugins
            plugins.connect(fig, plugins.MousePosition(fontsize=14))
            return mpld3.fig_to_html(fig)
        else:
            return print_figure(self.plot[idx], ViewMagic.FIGURE_FORMAT)


    def _process_view(self, view):
        """
        Determine the dimensions and keys to be turned into widgets and
        initialize the plots.
        """
        if isinstance(view, (GridLayout, Layout)):
            view = GridLayout([view]) if isinstance(view, Layout) else view
            shape = view.shape
            grid_size = (shape[1]*get_plot_size()[1],
                         shape[0]*get_plot_size()[0])
            self.plot = GridLayoutPlot(view, **dict(size=grid_size))

            keys_list = []
            for v in view:
                if isinstance(v, Layout): v = v.main
                if isinstance(v, Overlay): v = v[0]
                if isinstance(v, View):
                    v = v.stack_type([((0,), v)], dimensions=['Frame'])
                keys_list.append(list(v._data.keys()))

            # Check if all elements in the Grid have common dimensions
            if all(x == keys_list[0] for x in keys_list):
                self._keys = keys_list[0]
                element = view[0, 0]
                if isinstance(element, Layout):
                    self.dimensions = element.main.dimensions
                else:
                    self.dimensions = element.dimensions
            else:
                self._keys = [(k,) for k in range(len(view))]
                self.dimensions = ['Frame']
        elif isinstance(view, (View, Stack)):
            if isinstance(view, View):
                view = view.stack_type([((0,), view)], dimensions=['Frame'])
            opts = dict(options.plotting(view).opts, size=get_plot_size())
            self.plot = Plot.defaults[view.type](view, **opts)
            self._keys = view._data.keys()
            self.dimensions = view.dimensions

        # Create mock NdMapping to hold the common dimensions and keys
        self.mock_obj = NdMapping([(k, 0) for k in self._keys],
                                  dimensions=self.dimensions)


    def _initialize_widgets(self):
        """
        Initialize widgets and dimension values.
        """

        self.pwidgets = {}
        self.dim_val = {}
        for didx, dim in enumerate(self.mock_obj.dimension_labels):
            all_vals = [k[didx] for k in self._keys]

            # Initialize dimension value
            vals = self._get_dim_vals(list(self._keys[0]), didx)
            self.dim_val[dim] = vals[0]

            # Initialize widget
            if isnumeric(vals[0]):
                widget_type = FixedValueSliderWidget
            else:
                widget_type = widgets.DropdownWidget
                all_vals = dict((str(v), v) for v in all_vals)
            self.pwidgets[dim] = widget_type(values=all_vals)


    def __call__(self):
        # Initalize image widget
        if (ViewMagic.FIGURE_FORMAT == 'mpld3' and mpld3) or ViewMagic.FIGURE_FORMAT == 'svg':
            self.image_widget = widgets.HTMLWidget()
        else:
            self.image_widget = widgets.ImageWidget()

        if self.cached:
            self.image_widget.value = list(self.frames.values())[0]
        else:
            self.image_widget.value = self._plot_figure(0)
        self.image_widget.set_css(self.css)

        # Initialize interactive widgets
        interactive_widget = widgets.interactive(self.update_widgets, **self.pwidgets)
        interactive_widget.set_css(self.css)

        # Display widgets
        display(interactive_widget)
        display(self.image_widget)
        return '' # Suppresses outputting View repr when called through hook


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
        View and update all the widgets.
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
            dim_idx = self.mock_obj.dim_index(dim)
            widget = self.pwidgets[dim]
            vals = self._get_dim_vals(checked, dim_idx)
            if val not in vals:
                if isnumeric(val):
                    val = vals[np.argmin(np.abs(np.array(vals) - val))]
                else:
                    val = str(vals[0])
            if isnumeric(val):
                if len(vals) == 0: widget.step = 0
                widget.min = min(vals)
                widget.max = max(vals)
            checked[dim_idx] = val
            self.dim_val[dim] = val
            widget.value = round(val, 5) if isnumeric(val) else val

        # Update frame
        checked = tuple(checked)
        if self.cached:
            self.image_widget.value = self.frames[checked]
        else:
            self.image_widget.value = self._plot_figure(self._keys.index(checked))
