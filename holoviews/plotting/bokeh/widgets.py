import json
import param

from ..widgets import NdWidget, SelectionWidget, ScrubberWidget

class BokehWidget(NdWidget):

    template = param.String(default='./bokehwidget.jinja')

    def __init__(self, plot, renderer=None, **params):
        super(BokehWidget, self).__init__(plot, renderer, **params)
        self.initialized = False


    def get_frames(self):
        frames = {0: self._plot_figure(0)}
        return self.encode_frames(frames)

    def encode_frames(self, frames):
        frames = json.dumps(frames).replace('</', r'<\/')
        return frames

    def _plot_figure(self, idx):
        redraw = not self.initialized
        self.plot.update(idx)
        figure_format = 'html' if redraw else 'json'
        self.initialized = True
        return self.renderer.html(self.plot, figure_format)



class SelectionWidget(BokehWidget, SelectionWidget):
    pass

class ScrubberWidget(BokehWidget, ScrubberWidget):
    pass
