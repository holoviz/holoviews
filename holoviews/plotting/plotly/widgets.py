import json
import param

from ..widgets import NdWidget, SelectionWidget, ScrubberWidget

class PlotlyWidget(NdWidget):

    extensionjs = param.String(default='plotlywidgets.js', doc="""
        Optional javascript extension file for a particular backend.""")

    def _get_data(self):
        # Get initial frame to draw immediately
        init_frame = self._plot_figure(0, fig_format='html')
        data = super(PlotlyWidget, self)._get_data()
        return dict(data, init_frame=init_frame)

    def encode_frames(self, frames):
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
                msg = self.renderer.figure_data(self.plot,
                                                divuuid=self.id, comm=False)
            else:
                msg = self.renderer.diff(self.plot)
            return msg


class PlotlySelectionWidget(PlotlyWidget, SelectionWidget):
    pass

class PlotlyScrubberWidget(PlotlyWidget, ScrubberWidget):
    pass
