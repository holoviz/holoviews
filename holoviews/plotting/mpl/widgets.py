import json
import param

from ..widgets import NdWidget, SelectionWidget, ScrubberWidget


class MPLWidget(NdWidget):

    extensionjs = param.String(default='mplwidgets.js', doc="""
        Optional javascript extension file for a particular backend.""")


    def get_frames(self):
        if self.embed:
            return super(MPLWidget, self).get_frames()
        else:
            frames = {0: self._plot_figure(self.init_key)}
        return self.encode_frames(frames)


    def _plot_figure(self, idx):
        with self.renderer.state():
            self.plot.update(idx)
            if self.renderer.fig == 'auto':
                figure_format = self.renderer.params('fig').objects[0]
            else:
                figure_format = self.renderer.fig
            return self.renderer._figure_data(self.plot, figure_format, as_script=True)[0]


    def encode_frames(self, frames):
        if self.export_json:
            self.save_json(frames)
            return {}
        elif not isinstance(frames, dict):
            pass
        else:
            frames = dict(frames)
            return json.dumps(frames)



class MPLSelectionWidget(MPLWidget, SelectionWidget):
    pass

class MPLScrubberWidget(MPLWidget, ScrubberWidget):
    pass
