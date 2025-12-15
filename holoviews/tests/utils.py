from __future__ import annotations

import importlib
import logging
import os
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Literal, overload

import param
import pytest

from holoviews.core.util.dependencies import _is_installed
from holoviews.util.warnings import deprecated

if TYPE_CHECKING:
    import dask
    import dask.array as da
    import dask.dataframe as dd
    import datashader
    import ibis
    import matplotlib as mpl
    import networkx as nx
    import notebook
    import plotly
    import pyparsing
    import scipy
    import shapely
    import spatialpandas
    import tsdownsample
    import xarray
    from _pytest.mark.structures import MarkDecorator

    MaybeModuleType = ModuleType | None

cwd = os.path.abspath(os.path.split(__file__)[0])
sys.path.insert(0, os.path.join(cwd, '..'))


LEVELS = {'CRITICAL': 50, 'ERROR': 40, 'WARNING': 30, 'INFO': 20,
          'DEBUG': 10, 'VERBOSE': 0}


class MockLoggingHandler(logging.Handler):
    """
    Mock logging handler to check for expected logs used by
    LoggingComparisonTestCase.

    Messages are available from an instance's ``messages`` dict, in
    order, indexed by a lowercase log level string (e.g., 'debug',
    'info', etc.)."""

    def __init__(self, *args, **kwargs):
        self.messages = {'DEBUG': [], 'INFO': [], 'WARNING': [],
                         'ERROR': [], 'CRITICAL': [], 'VERBOSE':[]}
        self.param_methods = {
            'WARNING':'param.param.warning()',
            'INFO':'param.param.message()',
            'VERBOSE':'param.param.verbose()',
            'DEBUG':'param.param.debug()'}
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
        self.messages = {'DEBUG': [], 'INFO': [], 'WARNING': [],
                         'ERROR': [], 'CRITICAL': [], 'VERBOSE':[]}
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
        msg='\n\n{method}: {last_line}\ndoes not end with:\n{substring}'
        last_line = self.tail(level, n=1)
        if len(last_line) == 0:
            raise AssertionError(f'Missing {self.param_methods[level]} output: {substring!r}')
        if not last_line[0].endswith(substring):
            raise AssertionError(msg.format(method=self.param_methods[level],
                                            last_line=repr(last_line[0]),
                                            substring=repr(substring)))
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
        msg='\n\n{method}: {last_line}\ndoes not contain:\n{substring}'
        last_line = self.tail(level, n=1)
        if len(last_line) == 0:
            raise AssertionError(f'Missing {self.param_methods[level]} output: {substring!r}')
        if substring not in last_line[0]:
            raise AssertionError(msg.format(method=self.param_methods[level],
                                            last_line=repr(last_line[0]),
                                            substring=repr(substring)))
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
        self.log_handler = MockLoggingHandler(level='DEBUG')
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
        self.log_handler = MockLoggingHandler(level='DEBUG')
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


@overload
def optional_dependencies(name: Literal["scipy"], /) -> tuple[scipy, MarkDecorator]: ...

@overload
def optional_dependencies(name: Literal["ibis"], /) -> tuple[ibis, MarkDecorator]: ...

@overload
def optional_dependencies(name: Literal["dask"], /) -> tuple[dask, MarkDecorator]: ...

@overload
def optional_dependencies(name: Literal["dask.array"], /) -> tuple[da, MarkDecorator]: ...

@overload
def optional_dependencies(name: Literal["dask.dataframe"], /) -> tuple[dd, MarkDecorator]: ...

@overload
def optional_dependencies(name: Literal["datashader"], /) -> tuple[datashader, MarkDecorator]: ...

@overload
def optional_dependencies(name: Literal["matplotlib"], /) -> tuple[mpl, MarkDecorator]: ...

@overload
def optional_dependencies(name: Literal["networkx"], /) -> tuple[nx, MarkDecorator]: ...

@overload
def optional_dependencies(name: Literal["notebook"], /) -> tuple[notebook, MarkDecorator]: ...

@overload
def optional_dependencies(name: Literal["plotly"], /) -> tuple[plotly, MarkDecorator]: ...

@overload
def optional_dependencies(name: Literal["pyparsing"], /) -> tuple[pyparsing, MarkDecorator]: ...

@overload
def optional_dependencies(name: Literal["shapely"], /) -> tuple[shapely, MarkDecorator]: ...

@overload
def optional_dependencies(name: Literal["spatialpandas"], /) -> tuple[spatialpandas, MarkDecorator]: ...

@overload
def optional_dependencies(name: Literal["tsdownsample"], /) -> tuple[tsdownsample, MarkDecorator]: ...

@overload
def optional_dependencies(name: Literal["xarray"], /) -> tuple[xarray, MarkDecorator]: ...


def optional_dependencies(name: str, /) -> tuple[MaybeModuleType, MarkDecorator]:
    """Check if a dependency is installed and return the module and a fixture that skips test.
    """
    if _is_installed(name):
        module = importlib.import_module(name)
    else:
        module = None

    fixture = pytest.mark.skipif(module is None, reason=f"{name} is not installed")
    return module, fixture
