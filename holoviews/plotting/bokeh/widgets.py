import json

import param
import bokeh
from bokeh.io import Document
from bokeh.io import _CommsHandle
from bokeh.util.notebook import get_comms

from ..widgets import NdWidget, SelectionWidget, ScrubberWidget
from .util import compute_static_patch

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
        if self.embed or fig_format == 'html':
            html = self.renderer.html(self.plot, fig_format)
            return html
        else:
            doc = self.plot.document
            if hasattr(doc, 'last_comms_handle'):
                handle = doc.last_comms_handle
            else:
                handle = _CommsHandle(get_comms(doc.last_comms_target),
                                      doc, doc.to_json())
                doc.last_comms_handle = handle

            plotobjects = [h for handles in self.plot.traverse(lambda x: x.current_handles)
                           for h in handles]
            msg = compute_static_patch(doc, plotobjects)
            handle.comms.send(json.dumps(msg))
            return 'Complete'


class BokehSelectionWidget(BokehWidget, SelectionWidget):
    pass

class BokehScrubberWidget(BokehWidget, ScrubberWidget):
    pass
