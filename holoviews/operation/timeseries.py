import numpy as np
import param

from ..core import ElementOperation, Element
from ..core.util import pd
from ..element import Scatter

DF_INTERFACES = []
try:
    from ..core.data import PandasInterface
    DF_INTERFACES.append(PandasInterface)
except:
    pass

try:
    from ..core.data import DaskInterface
    DF_INTERFACES.append(DaskInterface)
except:
    pass


def get_df_data(element):
    """
    Return element data as dataframe, avoiding casting if already
    in dataframe format
    """
    if element.interface in DF_INTERFACES:
        return element.data
    else:
        return element.dframe()


class rolling(ElementOperation):
    """
    Applies a function over a rolling window.
    """

    center = param.Boolean(default=True, doc="""
        Whether to set the x-coordinate at the center or right edge
        of the window.""")

    function = param.Callable(default=np.mean, doc="""
        The function to apply over the rolling window.""")

    rolling_window = param.Integer(default=10, doc="""
        The window size over which to apply the function.""")

    def _apply(self, element, key=None):
        xdim = element.kdims[0].name
        df = get_df_data(element)
        df = df.set_index(xdim).rolling(window=self.p.rolling_window, center=self.p.center)
        return element.clone(df.apply(self.p.function).reset_index())

    def _process(self, element, key=None):
        return element.map(self._apply, Element)


class resample(ElementOperation):
    """
    Resamples a timeseries of dates with a frequency and function
    """

    edge = param.ObjectSelector(default='right', doc="""
        The bin edge to label the bin with.""")

    function = param.Callable(default=np.mean, doc="""
        The function to apply over the rolling window.""")

    rule = param.String(default='D', doc="""
        A string representing the time interval over which to apply the resampling""")

    def _apply(self, element, key=None):
        xdim = element.kdims[0].name
        df = get_df_data(element)
        df = df.set_index(xdim).resample(rule=self.p.rule, label=self.p.edge)
        return element.clone(df.apply(self.p.function).reset_index())

    def _process(self, element, key=None):
        return element.map(self._apply, Element)


class rolling_outlier_std(ElementOperation):
    """
    Detect outliers for using the standard devitation within a rolling window.

    Outliers are the array elements outside `sigma` standard deviations from
    the smoothed trend line, as calculated from the trend line residuals.
    """

    rolling_window = param.Integer(default=10, doc="""
        The window size of which within the rolling std is computed.""")

    sigma = param.Number(default=2.0, doc="""
        Minimum sigma before a value is considered an outlier.""")

    def _apply(self, element, key=None):
        sigma, window = self.p.sigma, self.p.rolling_window
        ys = element.dimension_values(1)

        # Calculate the variation in the distribution of the residual
        avg = pd.Series(ys).rolling(window, center=True).mean()
        residual = ys - avg
        std = pd.Series(residual).rolling(window, center=True).std()

        # Get indices of outliers
        outliers = (np.abs(residual) > std * sigma).values
        return element[outliers].clone(new_type=Scatter, group='Outliers')

    def _process(self, element, key=None):
        return element.map(self._apply, Element)
