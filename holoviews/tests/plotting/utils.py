from io import StringIO
import logging

import param


class ParamLogStream:
    """
    Context manager that replaces the param logger and captures
    log messages in a StringIO stream.
    """

    def __enter__(self):
        self.stream = StringIO()
        self._handler = logging.StreamHandler(self.stream)
        self._logger = logging.getLogger('testlogger')
        for handler in self._logger.handlers:
            self._logger.removeHandler(handler)
        self._logger.addHandler(self._handler)
        self._param_logger = param.parameterized.logger
        param.parameterized.logger = self._logger
        return self

    def __exit__(self, *args):
        param.parameterized.logger = self._param_logger
        self._handler.close()
        self.stream.seek(0)
