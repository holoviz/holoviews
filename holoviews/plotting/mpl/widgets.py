import uuid, json
import param

from ..widgets import NdWidget, SelectionWidget, ScrubberWidget

try:
    from matplotlib.backends.backend_nbagg import CommSocket
except:
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

    template = param.String(default='widget.jinja')

    def __init__(self, plot, renderer=None, **params):
        super(MPLWidget, self).__init__(plot, renderer, **params)
        if self.renderer.mode == 'nbagg':
            self.cached = False
            self.initialize_connection(plot)


    def _plot_figure(self, idx):
        with self.renderer.state():
            self.plot.update(idx)
            css = self.display_options.get('css', {})
            if self.renderer.mode == 'mpld3':
                figure_format = 'json'
            else:
                figure_format = self.display_options.get('figure_format',
                                                         self.renderer.fig)
            return self.renderer.html(self.plot, figure_format, css=css)


    def update(self, n):
        if self.renderer.mode == 'nbagg':
            if not self.manager._shown:
                self.comm.start()
                self.manager.add_web_socket(self.comm)
                self.manager._shown = True
            fig = self.plot[n]
            fig.canvas.draw_idle()
            return ''
        frame = self._plot_figure(n)
        if self.renderer.mode == 'mpld3':
            frame = self.encode_frames({0: frame})
        return frame


    def get_frames(self):
        if self.renderer.mode == 'nbagg':
            self.manager.display_js()
            frames = {0: self.comm.html}
        elif self.embed:
            frames = self.frames
        else:
            frames = {0: self._plot_figure(0)}
            if self.renderer.mode == 'mpld3':
                self.frames[0] = frames
            else:
                self.frames.update(frames)
        return self.encode_frames(frames)


    def encode_frames(self, frames):
        if self.renderer.mode == 'mpld3':
            import mpld3
            encoder = dict(cls=mpld3._display.NumpyEncoder)
            frames = {idx: frame for idx, frame in frames.items()}
            frames = json.dumps(frames, **encoder)
        else:
            frames = {idx: frame for idx, frame in frames.items()}
        return super(MPLWidget, self).encode_frames(frames)


    def initialize_connection(self, plot):
        plot.update(0)
        self.manager = self.renderer.get_figure_manager(plot)
        self.comm = WidgetCommSocket(self.manager)


class SelectionWidget(MPLWidget, SelectionWidget):
    pass

class ScrubberWidget(MPLWidget, ScrubberWidget):
    pass
