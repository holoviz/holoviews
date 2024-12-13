import types
from datetime import datetime as dt


import numpy as np
import pandas as pd
from pandas.core.arrays.masked import BaseMaskedArray
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.generic import ABCExtensionArray, ABCIndex, ABCSeries


# Types
generator_types = (zip, range, types.GeneratorType)
pandas_datetime_types = (pd.Timestamp, DatetimeTZDtype, pd.Period)
pandas_timedelta_types = (pd.Timedelta,)
datetime_types = (np.datetime64, dt.datetime, dt.date, dt.time, *pandas_datetime_types)
timedelta_types = (np.timedelta64, dt.timedelta, *pandas_timedelta_types)
arraylike_types = (np.ndarray, ABCSeries, ABCIndex, ABCExtensionArray)
masked_types = (BaseMaskedArray,)

try:
    import cftime
    cftime_types = (cftime.datetime,)
    datetime_types += cftime_types
except ImportError:
    cftime_types = ()
_STANDARD_CALENDARS = {'standard', 'gregorian', 'proleptic_gregorian'}
