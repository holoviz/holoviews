import os, sys

cwd = os.path.abspath(os.path.split(__file__)[0])
sys.path.insert(0, os.path.join(cwd, '..', 'param'))

import param
__version__ = param.Version(release=(0,8,1), fpath=__file__,
                            commit="$Format:%h$", reponame='holoviews')

from .core.dimension import Dimension                   # pyflakes:ignore (API import)
from .core.boundingregion import BoundingBox            # pyflakes:ignore (API import)
from .core.options import Options, Store, StoreOptions  # pyflakes:ignore (API import)
from .core.layout import *                              # pyflakes:ignore (API import)
from .core.element import *                             # pyflakes:ignore (API import)
from .core.overlay import *                             # pyflakes:ignore (API import)
from .core.sheetcoords import *                         # pyflakes:ignore (API import)
from .core.tree import *                                # pyflakes:ignore (API import)
from .core.io import FileArchive

from .interface import *                                             # pyflakes:ignore (API import)
from .operation import ElementOperation, MapOperation, TreeOperation # pyflakes:ignore (API import)
from .element import *                                               # pyflakes:ignore (API import)

try:
    from .ipython.archive import notebook_archive as archive
except:
    archive = FileArchive()
