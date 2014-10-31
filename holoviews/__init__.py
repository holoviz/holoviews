import os, sys

cwd = os.path.abspath(os.path.split(__file__)[0])
sys.path.insert(0, os.path.join(cwd, '..', 'param'))

import param
__version__ = param.Version(release=(0,7), fpath=__file__,
                            commit="$Format:%h$", reponame='holoviews')

from .core import options                     # pyflakes:ignore (API import)
from .core.dimension import Dimension         # pyflakes:ignore (API import)
from .core.boundingregion import BoundingBox  # pyflakes:ignore (API import)
from .core.layer import *                     # pyflakes:ignore (API import)
from .core.layout import *                    # pyflakes:ignore (API import)
from .core.sheetcoords import *               # pyflakes:ignore (API import)

from .ipython import *              # pyflakes:ignore (API import)
from .view import *                 # pyflakes:ignore (API import)
