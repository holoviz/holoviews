import param

from ..plot import DimensionedPlot
from .renderer import PlotlyRenderer

class PlotlyPlot(DimensionedPlot):
    
    width = param.Integer(default=400)
    height = param.Integer(default=400)

    renderer = PlotlyRenderer

    @property
    def state(self):
        """
        The plotting state that gets updated via the update method and
        used by the renderer to generate output.
        """
        return self.handles['fig']
