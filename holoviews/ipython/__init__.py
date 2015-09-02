import os
import base64
from unittest import SkipTest

import jinja2
from IPython.display import display, HTML

import holoviews
from ..core.util import find_file
from ..element.comparison import ComparisonTestCase
from ..interface.collector import Collector
from .. import plotting
from ..plotting.renderer import Renderer
from ..plotting.widgets import NdWidget
from .archive import notebook_archive
from .magics import load_magics
from .display_hooks import display      # pyflakes:ignore (API import)
from .display_hooks import set_display_hooks, OutputMagic
from .parser import Parser
from .widgets import RunProgress

from param import ipython as param_ext

Collector.interval_hook = RunProgress
holoviews.archive = notebook_archive


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


def load_notebook():
    """
    Displays javascript and CSS to initialize HoloViews widgets
    """

    # Get all the widgets and find the set of required js widget files
    widgets = [wdgt for cls in Renderer.__subclasses__()
               for wdgt in cls.widgets.values()]
    css = list({wdgt.css for wdgt in widgets})
    basejs = list({wdgt.basejs for wdgt in widgets})
    extensionjs = list({wdgt.extensionjs for wdgt in widgets})

    # Join all the js widget code into one string
    path = os.path.dirname(os.path.abspath(plotting.__file__))
    widgetjs = '\n'.join(open(find_file(path, f), 'r').read()
                         for f in basejs + extensionjs
                         if f is not None )
    widgetcss = '\n'.join(open(find_file(path, f), 'r').read()
                          for f in css if f is not None)

    logo_path = os.path.join(path, '../../doc/_static/holoviews_logo.png')
    logobase64 = base64.b64encode(open(logo_path, "rb").read())

    # Evaluate load_notebook.html template with widgetjs code
    templateLoader = jinja2.FileSystemLoader(os.path.dirname(os.path.abspath(__file__)))
    jinjaEnv = jinja2.Environment(loader=templateLoader)
    template = jinjaEnv.get_template('load_notebook.html')
    display(HTML(template.render({'widgetjs': widgetjs,
                                  'widgetcss': widgetcss,
                                  'logob64': logobase64})))


# Populating the namespace for keyword evaluation
from ..core.options import Cycle, Palette, Store # pyflakes:ignore (namespace import)
import numpy as np                               # pyflakes:ignore (namespace import)

Parser.namespace = {'np':np, 'Cycle':Cycle, 'Palette': Palette}

_loaded = False
def load_ipython_extension(ip):

    global _loaded
    if not _loaded:
        _loaded = True
        param_ext.load_ipython_extension(ip, verbose=False)
        load_magics(ip)
        OutputMagic.register_supported_formats(OutputMagic.optional_formats)
        set_display_hooks(ip)
        load_notebook()

def unload_ipython_extension(ip):
    global _loaded
    _loaded = False
