import sys, os

# Add param submodule to sys.path
cwd = os.path.abspath(os.path.split(__file__)[0])
sys.path.insert(0, os.path.join(cwd, '..', 'param'))

import param

__version__ = param.Version(release=(0,7), fpath=__file__,
                            commit="$Format:%h$", reponame='dataviews')

from .views import * # pyflakes:ignore (API import)
from .dataviews import * # pyflakes:ignore (API import)
from .sheetviews import * # pyflakes:ignore (API import)
from .ndmapping import * # pyflakes:ignore (API import)


def public(obj):
    if not isinstance(obj, type): return False
    baseclasses = [NdMapping, View, Dimension, Overlay]
    return any([issubclass(obj, bc) for bc in baseclasses])

_public = list(set([_k for _k, _v in locals().items() if public(_v)]))
__all__ = _public + ["boundingregion", "ipython", "plotting", "sheetcoords" ]
