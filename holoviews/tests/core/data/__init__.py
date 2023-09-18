import warnings
from contextlib import contextmanager


@contextmanager
def ignore_xarray_nanosecond_warning():
    # xarray emit a warning when datetime/timedelta is not nanosecond,
    # and then convert it to nanosecond precision. The full warning is:
    #   Converting non-nanosecond precision {case} values to nanosecond precision.
    #   This behavior can eventually be relaxed in xarray, as it is an artifact from
    #   pandas which is now beginning to support non-nanosecond precision values.
    #   This warning is caused by passing non-nanosecond np.datetime64 or
    #   np.timedelta64 values to the DataArray or Variable constructor; it can be
    #   silenced by converting the values to nanosecond precision ahead of time.
    #
    # Note the Pandas version is 2.0
    msg = r"Converting non-nanosecond precision \w+ values to nanosecond precision"
    try:
        cw = warnings.catch_warnings()
        cw.__enter__()
        warnings.filterwarnings("ignore", message=msg, category=UserWarning)
        yield
    finally:
        cw.__exit__()
