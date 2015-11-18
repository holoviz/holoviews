import param

from ..plot import DimensionedPlot
from .renderer import PlotlyRenderer

class PlotlyPlot(DimensionedPlot):
    
    width = param.Integer(default=400)
    height = param.Integer(default=400)

    renderer = PlotlyRenderer
