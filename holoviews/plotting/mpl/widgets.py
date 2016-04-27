import uuid, json, warnings
import param

from ..widgets import NdWidget, SelectionWidget, ScrubberWidget

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from matplotlib.backends.backend_nbagg import CommSocket
except ImportError:
    CommSocket = object

class WidgetCommSocket(CommSocket):
    """
    CustomCommSocket provides communication between the IPython
    kernel and a matplotlib canvas element in the notebook.
    A CustomCommSocket is required to delay communication
    between the kernel and the canvas element until the widget
    has been rendered in the notebook.
    """

    def __init__(self, manager):
        self.supports_binary = None
        self.manager = manager
        self.uuid = str(uuid.uuid4())
        self.html = "<div id=%r></div>" % self.uuid

    def start(self):
        try:
            # Jupyter/IPython 4.0
            from ipykernel.comm import Comm
        except:
            # IPython <=3.0
            from IPython.kernel.comm import Comm

        try:
            self.comm = Comm('matplotlib', data={'id': self.uuid})
        except AttributeError:
            raise RuntimeError('Unable to create an IPython notebook Comm '
                               'instance. Are you in the IPython notebook?')
        self.comm.on_msg(self.on_message)
        self.comm.on_close(lambda close_message: self.manager.clearup_closed())


class MPLWidget(NdWidget):

    CDN = param.Dict(default=dict(NdWidget.CDN, mpld3='https://mpld3.github.io/js/mpld3.v0.3git.js',
                                  d3='https://cdnjs.cloudflare.com/ajax/libs/d3/3.4.13/d3.js'))

    extensionjs = param.String(default='mplwidgets.js', doc="""
        Optional javascript extension file for a particular backend.""")

    template = param.String(default='mplwidgets.jinja')

    def __init__(self, plot, renderer=None, **params):
        super(MPLWidget, self).__init__(plot, renderer, **params)
        if self.renderer.mode == 'nbagg':
            self.cached = False
            self.initialize_connection(plot)


    def _plot_figure(self, idx):
        with self.renderer.state():
            self.plot.update(idx)
            if self.renderer.mode == 'mpld3':
                figure_format = 'json'
            elif self.renderer.fig == 'auto':
                figure_format = self.renderer.params('fig').objects[0]
            else:
                figure_format = self.renderer.fig
            return self.renderer.html(self.plot, figure_format)


    def update(self, key):
        if self.plot.dynamic == 'bounded' and not isinstance(key, int):
            key = tuple(dim.values[k] if dim.values else k
                        for dim, k in zip(self.mock_obj.kdims, tuple(key)))

        if self.renderer.mode == 'nbagg':
            if not self.manager._shown:
                self.comm.start()
                self.manager.add_web_socket(self.comm)
                self.manager._shown = True
            fig = self.plot[key]
            fig.canvas.draw_idle()
            return ''
        frame = self._plot_figure(key)
        if self.renderer.mode == 'mpld3':
            frame = self.encode_frames({0: frame})
        return frame


    def get_frames(self):
        if self.renderer.mode == 'nbagg':
            self.manager.display_js()
            frames = {0: self.comm.html}
        elif self.embed:
            return super(MPLWidget, self).get_frames()
        else:
            frames = {0: self._plot_figure(0)}
        return self.encode_frames(frames)


    def encode_frames(self, frames):
        if self.export_json:
            self.save_json(frames)
            return {}
        elif not isinstance(frames, dict):
            pass
        elif self.renderer.mode == 'mpld3':
            import mpld3
            encoder = dict(cls=mpld3._display.NumpyEncoder)
            frames = {idx: frame for idx, frame in frames.items()}
            frames = json.dumps(frames, **encoder)
        else:
            frames = {idx: frame for idx, frame in frames.items()}
        return super(MPLWidget, self).encode_frames(frames)


    def initialize_connection(self, plot):
        plot.update(0)
        self.manager = self.renderer.get_figure_manager(plot.state)
        self.comm = WidgetCommSocket(self.manager)


class MPLSelectionWidget(MPLWidget, SelectionWidget):
    pass

class MPLScrubberWidget(MPLWidget, ScrubberWidget):
    pass
