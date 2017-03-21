import param
import numpy as np
import pandas as pd

from ..core import ElementOperation, Element
from ..core.data import PandasInterface
from ..element import Scatter


class rolling(ElementOperation):
    """
    Applies a function over a rolling window.
    """

    center = param.Boolean(default=True, doc="""
        Whether to set the x-coordinate at the center or right edge
        of the window.""")

    function = param.Callable(default=np.mean, doc="""
        The function to apply over the rolling window.""")

    min_periods = param.Integer(default=None, doc="""
       Minimum number of observations in window required to have a
       value (otherwise result is NA).""")

    rolling_window = param.Integer(default=10, doc="""
        The window size over which to apply the function.""")

    window_type = param.ObjectSelector(default=None,
        objects=['boxcar', 'triang', 'blackman', 'hamming', 'bartlett',
                 'parzen', 'bohman', 'blackmanharris', 'nuttall',
                 'barthann', 'kaiser', 'gaussian', 'general_gaussian',
                 'slepian'], doc="The type of the window to apply")

    def _process_layer(self, element, key=None):
        xdim = element.kdims[0].name
        df = PandasInterface.as_dframe(element)
        roll_kwargs = {'window': self.p.rolling_window,
                       'center': self.p.center,
                       'win_type': self.p.window_type,
                       'min_periods': self.p.min_periods}
        df = df.set_index(xdim).rolling(**roll_kwargs)
        if roll_kwargs['window'] is None:
            rolled = df.apply(self.p.function)
        else:
            if self.p.function is np.mean:
                rolled = df.mean()
            elif self.p.function is np.sum:
                rolled = df.sum()
            else:
                raise ValueError("Rolling window function only supports "
                                 "mean and sum when custom window_type is supplied")
        return element.clone(rolled.reset_index())

    def _process(self, element, key=None):
        return element.map(self._process_layer, Element)


class resample(ElementOperation):
    """
    Resamples a timeseries of dates with a frequency and function
    """

    closed = param.ObjectSelector(default=None, objects=['left', 'right'],
        doc="Which side of bin interval is closed")

    function = param.Callable(default=np.mean, doc="""
        The function to apply over the rolling window.""")

    label = param.ObjectSelector(default='right', doc="""
        The bin edge to label the bin with.""")

    rule = param.String(default='D', doc="""
        A string representing the time interval over which to apply the resampling""")

    def _process_layer(self, element, key=None):
        df = PandasInterface.as_dframe(element)
        xdim = element.kdims[0].name
        resample_kwargs = {'rule': self.p.rule, 'label': self.p.label,
                           'closed': self.p.closed}
        df = df.set_index(xdim).resample(**resample_kwargs)
        return element.clone(df.apply(self.p.function).reset_index())

    def _process(self, element, key=None):
        return element.map(self._process_layer, Element)


class rolling_outlier_std(ElementOperation):
    """
    Detect outliers using the standard deviation within a rolling window.

    Outliers are the array elements outside `sigma` standard deviations from
    the smoothed trend line, as calculated from the trend line residuals.
    """

    rolling_window = param.Integer(default=10, doc="""
        The window size of which within the rolling std is computed.""")

    sigma = param.Number(default=2.0, doc="""
        Minimum sigma before a value is considered an outlier.""")

    def _process_layer(self, element, key=None):
        sigma, window = self.p.sigma, self.p.rolling_window
        ys = element.dimension_values(1)

        # Calculate the variation in the distribution of the residual
        avg = pd.Series(ys).rolling(window, center=True).mean()
        residual = ys - avg
        std = pd.Series(residual).rolling(window, center=True).std()

        # Get indices of outliers
        outliers = (np.abs(residual) > std * sigma).values
        return element[outliers].clone(new_type=Scatter)

    def _process(self, element, key=None):
        return element.map(self._process_layer, Element)
