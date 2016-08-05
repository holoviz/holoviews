import re
import inspect
import warnings

import numpy as np
from matplotlib import ticker

from ...core.util import basestring


def wrap_formatter(formatter):
    """
    Wraps formatting function or string in
    appropriate matplotlib formatter type.
    """
    if isinstance(formatter, ticker.Formatter):
        return formatter
    elif callable(formatter):
        args = [arg for arg in inspect.getargspec(formatter).args
                if arg != 'self']
        wrapped = formatter
        if len(args) == 1:
            def wrapped(val, pos=None):
                return formatter(val)
        return ticker.FuncFormatter(wrapped)
    elif isinstance(formatter, basestring):
        if re.findall(r"\{(\w+)\}", formatter):
            return ticker.StrMethodFormatter(formatter)
        else:
            return ticker.FormatStrFormatter(formatter)

def unpack_adjoints(ratios):
    new_ratios = {}
    offset = 0
    for k, (num, ratios) in sorted(ratios.items()):
        unpacked = [[] for _ in range(num)]
        for r in ratios:
            nr = len(r)
            for i in range(num):
                unpacked[i].append(r[i] if i < nr else np.nan)
        for i, r in enumerate(unpacked):
            new_ratios[k+i+offset] = r
        offset += num-1
    return new_ratios

def normalize_ratios(ratios):
    normalized = {}
    for i, v in enumerate(zip(*ratios.values())):
        arr = np.array(v)
        normalized[i] = arr/float(np.nanmax(arr))
    return normalized

def compute_ratios(ratios, normalized=True):
    unpacked = unpack_adjoints(ratios)
    if normalized:
        unpacked = normalize_ratios(unpacked)
    sorted_ratios = sorted(unpacked.items())
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        return np.nanmax(np.vstack([v for _, v in sorted_ratios]), axis=0)
