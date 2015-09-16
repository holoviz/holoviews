import re

from ...core.util import basestring
from matplotlib import ticker

def wrap_formatter(formatter):
    """
    Wraps formatting function or string in
    appropriate matplotlib formatter type.
    """
    if isinstance(formatter, ticker.Formatter):
        return formatter
    elif callable(formatter):
        return ticker.FuncFormatter(formatter)
    elif isinstance(formatter, basestring):
        if re.findall(r"\{(\w+)\}", formatter):
            return ticker.StrMethodFormatter(formatter)
        else:
            return ticker.FormatStrFormatter(formatter)
