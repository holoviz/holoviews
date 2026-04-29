from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import TYPE_CHECKING

import param
import pytest

from holoviews.core.util.dependencies import _is_installed
from holoviews.util.warnings import deprecated

cwd = os.path.abspath(os.path.split(__file__)[0])
sys.path.insert(0, os.path.join(cwd, ".."))


LEVELS = {"CRITICAL": 50, "ERROR": 40, "WARNING": 30, "INFO": 20, "DEBUG": 10, "VERBOSE": 0}


class MockLoggingHandler(logging.Handler):
    """
    Mock logging handler to check for expected logs used by
    LoggingComparisonTestCase.

    Messages are available from an instance's ``messages`` dict, in
    order, indexed by a lowercase log level string (e.g., 'debug',
    'info', etc.)."""

    def __init__(self, *args, **kwargs):
        self.messages = {
            "DEBUG": [],
            "INFO": [],
            "WARNING": [],
            "ERROR": [],
            "CRITICAL": [],
            "VERBOSE": [],
        }
        self.param_methods = {
            "WARNING": "param.param.warning()",
            "INFO": "param.param.message()",
            "VERBOSE": "param.param.verbose()",
            "DEBUG": "param.param.debug()",
        }
        super().__init__(*args, **kwargs)

    def emit(self, record):
        "Store a message to the instance's messages dictionary"
        self.acquire()
        try:
            self.messages[record.levelname].append(record.getMessage())
        finally:
            self.release()

    def reset(self):
        self.acquire()
        self.messages = {
            "DEBUG": [],
            "INFO": [],
            "WARNING": [],
            "ERROR": [],
            "CRITICAL": [],
            "VERBOSE": [],
        }
        self.release()

    def tail(self, level, n=1):
        "Returns the last n lines captured at the given level"
        return [str(el) for el in self.messages[level][-n:]]

    def assertEndsWith(self, level, substring):
        deprecated("1.25.0", "assertEndsWith", "assert_endswith")
        self.assert_endswith(level, substring)

    def assert_endswith(self, level, substring):
        """
        Assert that the last line captured at the given level ends with
        a particular substring.
        """
        msg = "\n\n{method}: {last_line}\ndoes not end with:\n{substring}"
        last_line = self.tail(level, n=1)
        if len(last_line) == 0:
            raise AssertionError(f"Missing {self.param_methods[level]} output: {substring!r}")
        if not last_line[0].endswith(substring):
            raise AssertionError(
                msg.format(
                    method=self.param_methods[level],
                    last_line=repr(last_line[0]),
                    substring=repr(substring),
                )
            )
        else:
            self.messages[level].pop(-1)

    def assertContains(self, level, substring):
        deprecated("1.25.0", "assertEndsWith", "assert_endswith")
        self.assert_contains(level, substring)

    def assert_contains(self, level, substring):
        """
        Assert that the last line captured at the given level contains a
        particular substring.
        """
        msg = "\n\n{method}: {last_line}\ndoes not contain:\n{substring}"
        last_line = self.tail(level, n=1)
        if len(last_line) == 0:
            raise AssertionError(f"Missing {self.param_methods[level]} output: {substring!r}")
        if substring not in last_line[0]:
            raise AssertionError(
                msg.format(
                    method=self.param_methods[level],
                    last_line=repr(last_line[0]),
                    substring=repr(substring),
                )
            )
        else:
            self.messages[level].pop(-1)


class LoggingComparisonTestCase:
    """
    ComparisonTestCase with support for capturing param logging output.

    Subclasses must call super setUp to make the
    tests independent. Testing can then be done via the
    self.log_handler.tail and self.log_handler.assertEndsWith methods.
    """

    def __init_subclass__(self, *args, **kwargs):
        deprecated(
            "1.25.0",
            "Inheriting from 'holoviews.tests.utils.LoggingComparisonTestCase'",
            "holoviews.tests.utils.LoggingComparison",
            repr_old=False,
        )
        super().__init_subclass__(*args, **kwargs)

    def setup_method(self):
        log = param.parameterized.get_logger()
        self.handlers = log.handlers
        log.handlers = []
        self.log_handler = MockLoggingHandler(level="DEBUG")
        log.addHandler(self.log_handler)

    def teardown_method(self):
        log = param.parameterized.get_logger()
        log.handlers = self.handlers
        messages = self.log_handler.messages
        self.log_handler.reset()
        for level, msgs in messages.items():
            for msg in msgs:
                log.log(LEVELS[level], msg)


class LoggingComparison:
    """
    Comparison with support for capturing param logging output.

    Testing can then be done via the
    self.log_handler.tail and self.log_handler.assertEndsWith methods.
    """

    @pytest.fixture(autouse=True)
    def _setup_logger(self):
        log = param.parameterized.get_logger()
        self.handlers = log.handlers
        log.handlers = []
        self.log_handler = MockLoggingHandler(level="DEBUG")
        log.addHandler(self.log_handler)
        try:
            yield
        finally:
            log = param.parameterized.get_logger()
            log.handlers = self.handlers
            messages = self.log_handler.messages
            self.log_handler.reset()
            for level, msgs in messages.items():
                for msg in msgs:
                    log.log(LEVELS[level], msg)


def optional_dependencies(*names: tuple[str]):
    """Check if a dependency is installed and return the module and a fixture that skips test."""
    if all(_is_installed(n) for n in names):
        return importlib.import_module(names[0])


if TYPE_CHECKING:
    import dask
    import dask.array as da
    import dask.dataframe as dd
    import datashader as ds
    import ibis
    import matplotlib as mpl
    import networkx as nx
    import notebook
    import plotly
    import polars as pl
    import scipy
    import shapely
    import spatialpandas as spd
    import tsdownsample
    import xarray as xr
else:
    dask = optional_dependencies("dask")
    da = optional_dependencies("dask.array")
    dd = optional_dependencies("dask.dataframe", "pyarrow")
    ds = optional_dependencies("datashader")
    ibis = optional_dependencies("ibis")
    mpl = optional_dependencies("matplotlib")
    nx = optional_dependencies("networkx")
    notebook = optional_dependencies("notebook")
    plotly = optional_dependencies("plotly")
    pl = optional_dependencies("polars")
    scipy = optional_dependencies("scipy")
    shapely = optional_dependencies("shapely")
    spd = optional_dependencies("spatialpandas")
    tsdownsample = optional_dependencies("tsdownsample")
    xr = optional_dependencies("xarray")


dask_skip = pytest.mark.skipif(dask is None, reason="dask is not installed")
da_skip = pytest.mark.skipif(da is None, reason="dask.array is not installed")
dd_skip = pytest.mark.skipif(dd is None, reason="dask.dataframe is not installed")
ds_skip = pytest.mark.skipif(ds is None, reason="datashader is not installed")
ibis_skip = pytest.mark.skipif(ibis is None, reason="ibis is not installed")
mpl_skip = pytest.mark.skipif(mpl is None, reason="matplotlib is not installed")
nx_skip = pytest.mark.skipif(nx is None, reason="networkx is not installed")
notebook_skip = pytest.mark.skipif(notebook is None, reason="notebook is not installed")
plotly_skip = pytest.mark.skipif(plotly is None, reason="plotly is not installed")
pl_skip = pytest.mark.skipif(pl is None, reason="polars is not installed")
scipy_skip = pytest.mark.skipif(scipy is None, reason="scipy is not installed")
shapely_skip = pytest.mark.skipif(shapely is None, reason="shapely is not installed")
spd_skip = pytest.mark.skipif(spd is None, reason="spatialpandas is not installed")
tsdownsample_skip = pytest.mark.skipif(
    tsdownsample is None, reason="tsdownsample is not installed"
)
xr_skip = pytest.mark.skipif(xr is None, reason="xarray is not installed")
