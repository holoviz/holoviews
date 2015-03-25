from __future__ import print_function, absolute_import
import os, sys, re, pydoc

import numpy as np # pyflakes:ignore (API import)

_cwd = os.path.abspath(os.path.split(__file__)[0])
sys.path.insert(0, os.path.join(_cwd, '..', 'param'))

import param

__version__ = param.Version(release=(1,0,1), fpath=__file__,
                            commit="$Format:%h$", reponame='holoviews')

from .core import archive
from .core.dimension import OrderedDict, Dimension      # pyflakes:ignore (API import)
from .core.boundingregion import BoundingBox            # pyflakes:ignore (API import)
from .core.options import Options, Store, StoreOptions  # pyflakes:ignore (API import)
from .core.layout import *                              # pyflakes:ignore (API import)
from .core.element import *                             # pyflakes:ignore (API import)
from .core.overlay import *                             # pyflakes:ignore (API import)
from .core.tree import *                                # pyflakes:ignore (API import)

from .interface import *                                             # pyflakes:ignore (API import)
from .operation import ElementOperation, MapOperation, TreeOperation # pyflakes:ignore (API import)
from .element import *                                               # pyflakes:ignore (API import)



def help(obj, visualization=False, ansi=True):
    """
    Extended version of the built-in help that supports parameterized
    functions and objects. If ansi is set to False, all ANSI color
    codes are stripped out.
    """
    ansi_escape = re.compile(r'\x1b[^m]*m')
    parameterized_object = isinstance(obj, param.Parameterized)
    parameterized_class = (isinstance(obj,type)
                           and  issubclass(obj,param.Parameterized))

    if parameterized_object or parameterized_class:
        if Store.registry.get(obj if parameterized_class else type(obj), False):
            if visualization is False:
                print("\nTo view the visualization options applicable to this object or class, use:\n\n"
                      "   holoviews.help(obj, visualization=True)\n")
            else:
                Store.info(obj, ansi=ansi)
                return
        info = param.ipython.ParamPager()(obj)
        if ansi is False:
            info = ansi_escape.sub('', info)
        print(info)
    else:
        pydoc.help(obj)
