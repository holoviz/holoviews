import os
import base64
from unittest import SkipTest

import param
import jinja2
from IPython.display import HTML

import holoviews
from ..core.options import Store
from ..element.comparison import ComparisonTestCase
from ..interface.collector import Collector
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


def load_hvjs(logo=False, JS=True, message='HoloViewsJS successfully loaded.'):
    """
    Displays javascript and CSS to initialize HoloViews widgets.
    """
    # Evaluate load_notebook.html template with widgetjs code
    if JS:
        widgetjs, widgetcss = Renderer.embed_assets()
    else:
        widgetjs, widgetcss = '', ''
    templateLoader = jinja2.FileSystemLoader(os.path.dirname(os.path.abspath(__file__)))
    jinjaEnv = jinja2.Environment(loader=templateLoader)
    template = jinjaEnv.get_template('load_notebook.html')
    display(HTML(template.render({'widgetjs': widgetjs,
                                  'widgetcss': widgetcss,
                                  'logo': logo,
                                  'message':message})))


# Populating the namespace for keyword evaluation
from ..core.options import Cycle, Palette # pyflakes:ignore (namespace import)
import numpy as np                        # pyflakes:ignore (namespace import)

Parser.namespace = {'np':np, 'Cycle':Cycle, 'Palette': Palette}

class notebook_extension(param.ParameterizedFunction):
    """
    Parameterized function to initialize notebook resources
    and register magics.
    """

    css = param.String(default='', doc="Optional CSS rule set to apply to the notebook.")

    logo = param.Boolean(default=True, doc="Toggles display of HoloViews logo")

    width = param.Number(default=None, bounds=(0, 100), doc="""
        Width of the notebook as a percentage of the browser screen window width.""")

    display_formats = param.List(default=['html'], doc="""
        A list of formats that are rendered to the notebook where
        multiple formats may be selected at once (although only one
        format will be displayed).

        Although the 'html' format is supported across backends, other
        formats supported by the current backend (e.g 'png' and 'svg'
        using the matplotlib backend) may be used. This may be useful to
        export figures to other formats such as PDF with nbconvert. """)

    ip = param.Parameter(default=None, doc="IPython kernel instance")

    _loaded = False

    def __call__(self, **params):
        resources = self._get_resources(params)
        ip = params.pop('ip', None)
        p = param.ParamOverrides(self, params)
        Store.display_formats = p.display_formats

        if 'html' not in p.display_formats and len(p.display_formats) > 1:
            msg = ('Output magic unable to control displayed format '
                   'as IPython notebook uses fixed precedence '
                   'between %r' % p.display_formats)
            display(HTML('<b>Warning</b>: %s' % msg))

        if notebook_extension._loaded == False:
            ip = get_ipython() if ip is None else ip
            param_ext.load_ipython_extension(ip, verbose=False)
            load_magics(ip)
            OutputMagic.initialize()
            set_display_hooks(ip)
            notebook_extension._loaded = True

        css = ''
        if p.width is not None:
            css += '<style>div.container { width: %s%% }</style>' % p.width
        if p.css:
            css += '<style>%s</style>' % p.css
        if css:
            display(HTML(css))

        resources = list(resources)
        if len(resources) == 0: return

        # Create a message for the logo (if shown)
        js_names = {'holoviews':'HoloViewsJS'} # resource : displayed name
        loaded = ', '.join(js_names[r] if r in js_names else r.capitalize()+'JS'
                           for r in resources)

        load_hvjs(logo=p.logo,
                  JS=('holoviews' in resources),
                  message = '%s successfully loaded in this cell.' % loaded)
        for r in [r for r in resources if r != 'holoviews']:
            Store.renderers[r].load_nb()


    def _get_resources(self, params):
        """
        Finds the list of resources from the keyword parameters and pops
        them out of the params dictionary.
        """
        resources = []
        disabled = []
        for resource in ['holoviews'] + list(Store.renderers.keys()):
            if resource in params:
                setting = params.pop(resource)
                if setting is True and resource != 'matplotlib':
                    resources.append(resource)
                if setting is False:
                    disabled.append(resource)

        if ('holoviews' not in disabled) and ('holoviews' not in resources):
            resources = ['holoviews'] + resources
        return resources


    @param.parameterized.bothmethod
    def tab_completion_docstring(self_or_cls):
        """
        Generates a docstring that can be used to enable tab-completion
        of resources.
        """
        elements = ['%s=Boolean' %k for k in list(Store.renderers.keys())]
        for name, p in self_or_cls.params().items():
            param_type = p.__class__.__name__
            elements.append("%s=%s" % (name, param_type))

        return "params(%s)" % ', '.join(['holoviews=Boolean'] + elements)


notebook_extension.__doc__ = notebook_extension.tab_completion_docstring()

def load_ipython_extension(ip):
    notebook_extension(ip=ip)

def unload_ipython_extension(ip):
    notebook_extension._loaded = False
