import param

from .chart import AreaPlot
from .path import PolygonPlot


class DistributionPlot(AreaPlot):
    """
    DistributionPlot visualizes a distribution of values as a KDE.
    """

    bandwidth = param.Number(default=None, doc="""
        The bandwidth of the kernel for the density estimate.""")

    cut = param.Number(default=3, doc="""
        Draw the estimate to cut * bw from the extreme data points.""")

    filled = param.Boolean(default=True, doc="""
        Whether the bivariate contours should be filled.""")


class BivariatePlot(PolygonPlot):
    """
    Bivariate plot visualizes two-dimensional kernel density
    estimates. Additionally, by enabling the joint option, the
    marginals distributions can be plotted alongside each axis (does
    not animate or compose).
    """

    bandwidth = param.Number(default=None, doc="""
        The bandwidth of the kernel for the density estimate.""")

    cut = param.Number(default=3, doc="""
        Draw the estimate to cut * bw from the extreme data points.""")

    filled = param.Boolean(default=False, doc="""
        Whether the bivariate contours should be filled.""")
