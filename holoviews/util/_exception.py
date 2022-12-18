import os
import inspect
from functools import lru_cache
from warnings import warn


@lru_cache(maxsize=None)
def deprecation_warning(msg, warning=FutureWarning):
    "To only run the warning once"

    # Finding the first stacklevel outside holoviews and param
    # Inspired by: pandas.util._exceptions.find_stack_level
    import holoviews as hv
    import param

    pkg_dir = os.path.dirname(hv.__file__)
    test_dir = os.path.join(pkg_dir, "tests")
    param_dir = os.path.dirname(param.__file__)

    frame = inspect.currentframe()
    stacklevel = 0
    while frame:
        fname = inspect.getfile(frame)
        if (fname.startswith(pkg_dir) or fname.startswith(param_dir)) and not fname.startswith(test_dir):
            frame = frame.f_back
            stacklevel += 1
        else:
            break

    warn(msg, warning, stacklevel=stacklevel)
