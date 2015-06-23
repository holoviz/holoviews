import json

from ..widgets import NdWidget, SelectionWidget, ScrubberWidget

class BokehWidget(NdWidget):

    def encode_frames(self, frames):
        frames = json.dumps(frames).replace('</', r'<\/')
        return frames


class SelectionWidget(BokehWidget, SelectionWidget):
    pass

class ScrubberWidget(BokehWidget, ScrubberWidget):
    pass
