import os, sys
import param
import logging

from holoviews.element.comparison import ComparisonTestCase

cwd = os.path.abspath(os.path.split(__file__)[0])
sys.path.insert(0, os.path.join(cwd, '..'))


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
        self.param_methods = {'WARNING':'param.warning()', 'INFO':'param.message()',
                              'VERBOSE':'param.verbose()', 'DEBUG':'param.debug()'}
        super(MockLoggingHandler, self).__init__(*args, **kwargs)

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
        """
        Assert that the last line captured at the given level ends with
        a particular substring.
        """
        msg='\n\n{method}: {last_line}\ndoes not end with:\n{substring}'
        last_line = self.tail(level, n=1)
        if len(last_line) == 0:
            raise AssertionError('Missing {method} output: {substring}'.format(
                method=self.param_methods[level], substring=repr(substring)))
        if not last_line[0].endswith(substring):
            raise AssertionError(msg.format(method=self.param_methods[level],
                                            last_line=repr(last_line[0]),
                                            substring=repr(substring)))

    def assertContains(self, level, substring):
        """
        Assert that the last line captured at the given level contains a
        particular substring.
        """
        msg='\n\n{method}: {last_line}\ndoes not contain:\n{substring}'
        last_line = self.tail(level, n=1)
        if len(last_line) == 0:
            raise AssertionError('Missing {method} output: {substring}'.format(
                method=self.param_methods[level], substring=repr(substring)))
        if substring not in last_line[0]:
            raise AssertionError(msg.format(method=self.param_methods[level],
                                            last_line=repr(last_line[0]),
                                            substring=repr(substring)))


class LoggingComparisonTestCase(ComparisonTestCase):
    """
    ComparisonTestCase with support for capturing param logging output.

    Subclasses must call super setUp to make the
    tests independent. Testing can then be done via the
    self.log_handler.tail and self.log_handler.assertEndsWith methods.
    """

    @classmethod
    def setUpClass(cls):
        super(LoggingComparisonTestCase, cls).setUpClass()
        log = param.parameterized.get_logger()
        cls.log_handler = MockLoggingHandler(level='DEBUG')
        log.addHandler(cls.log_handler)

    def setUp(self):
        super(LoggingComparisonTestCase, self).setUp()
        self.log_handler.reset()
