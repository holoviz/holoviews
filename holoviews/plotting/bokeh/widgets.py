import os, json
import param

from ..widgets import NdWidget, SelectionWidget, ScrubberWidget

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
        return self.renderer.html(self.plot, fig_format)


class BokehSelectionWidget(BokehWidget, SelectionWidget):
    pass

class BokehScrubberWidget(BokehWidget, ScrubberWidget):
    pass
