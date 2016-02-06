import re

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
        return ticker.FuncFormatter(formatter)
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
    return np.nanmax(np.vstack([v for _, v in sorted_ratios]), axis=0)
