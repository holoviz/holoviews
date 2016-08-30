import uuid, json, warnings
import param

from ..widgets import NdWidget, SelectionWidget, ScrubberWidget


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


    def _plot_figure(self, idx):
        with self.renderer.state():
            self.plot.update(idx)
            if self.renderer.mode == 'mpld3':
                figure_format = 'json'
            elif self.renderer.fig == 'auto':
                figure_format = self.renderer.params('fig').objects[0]
            else:
                figure_format = self.renderer.fig
            return self.renderer.html(self.plot, figure_format, comm=False)


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
        self.plot.push()
        return "Complete"

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
            frames = dict(frames)
            return json.dumps(frames, **encoder)
        else:
            frames = dict(frames)
            return json.dumps(frames)



class MPLSelectionWidget(MPLWidget, SelectionWidget):
    pass

class MPLScrubberWidget(MPLWidget, ScrubberWidget):
    pass
