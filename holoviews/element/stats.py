import param

from ..core.dimension import Dimension
from .chart import Chart, Scatter


class Bivariate(Chart):
    """
    Bivariate Views are containers for two dimensional data,
    which is to be visualized as a kernel density estimate. The
    data should be supplied as an Nx2 array, containing the x-
    and y-data.
    """

    kdims = param.List(default=[Dimension('x'), Dimension('y')])

    vdims = param.List(default=[], bounds=(0,1))

    group = param.String(default="Bivariate", constant=True)



class Distribution(Chart):
    """
    Distribution Views provide a container for data to be
    visualized as a one-dimensional distribution. The data should
    be supplied as a simple one-dimensional array or
    list. Internally it uses Seaborn to make all the conversions.
    """

    kdims = param.List(default=[])

    group = param.String(default='Distribution', constant=True)

    vdims = param.List(default=[Dimension('Value')])

    _auto_indexable_1d = False


class Regression(Scatter):
    """
    Regression is identical to a Scatter plot but is visualized
    using the Seaborn regplot interface. This allows it to
    implement linear regressions, confidence intervals and a lot
    more.
    """

    group = param.String(default='Regression', constant=True)


