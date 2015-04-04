from unittest import SkipTest
import matplotlib.pyplot as plt

import holoviews
from ..element.comparison import ComparisonTestCase
from ..interface.collector import Collector
from .archive import notebook_archive
from .magics import load_magics
from .display_hooks import animate, set_display_hooks, OutputMagic
from .parser import Parser
from .widgets import RunProgress

from param import ipython as param_ext

try:    from matplotlib import animation
except: animation = None

Collector.interval_hook = RunProgress
holoviews.archive = notebook_archive

def supported_formats(optional_formats):
    "Optional formats that are actually supported"
    supported = []
    for fmt in optional_formats:
        try:
            anim = animation.FuncAnimation(plt.figure(),
                                           lambda x: x, frames=[0,1])
            animate(anim, 72, *OutputMagic.ANIMATION_OPTS[fmt])
            supported.append(fmt)
        except: pass
    return supported



class IPTestCase(ComparisonTestCase):
    """
    This class extends ComparisonTestCase to handle IPython specific
    objects and support the execution of cells and magic.
    """

    def setUp(self):
        super(IPTestCase, self).setUp()
        try:
            import IPython
            from IPython.display import HTML, SVG
            self.ip = IPython.InteractiveShell()
            if self.ip is None:
                raise TypeError()
        except Exception:
                raise SkipTest("IPython could not be started")

        self.addTypeEqualityFunc(HTML, self.skip_comparison)
        self.addTypeEqualityFunc(SVG,  self.skip_comparison)

    def skip_comparison(self, obj1, obj2, msg): pass

    def get_object(self, name):
        obj = self.ip._object_find(name).obj
        if obj is None:
            raise self.failureException("Could not find object %s" % name)
        return obj


    def cell(self, line):
        "Run an IPython cell"
        self.ip.run_cell(line, silent=True)

    def cell_magic(self, *args, **kwargs):
        "Run an IPython cell magic"
        self.ip.run_cell_magic(*args, **kwargs)


    def line_magic(self, *args, **kwargs):
        "Run an IPython line magic"
        self.ip.run_line_magic(*args, **kwargs)



# Populating the namespace for keyword evaluation
from ..core.options import Cycle, Palette # pyflakes:ignore (namespace import)
import numpy as np                        # pyflakes:ignore (namespace import)

Parser.namespace = {'np':np, 'Cycle':Cycle, 'Palette': Palette}

_loaded = False
def load_ipython_extension(ip):

    global _loaded
    if not _loaded:
        _loaded = True

        param_ext.load_ipython_extension(ip, verbose=False)

        load_magics(ip)
        valid_formats = supported_formats(OutputMagic.optional_formats)
        OutputMagic.register_supported_formats(valid_formats)
        set_display_hooks(ip)

def unload_ipython_extension(ip):
    global _loaded
    _loaded = False
