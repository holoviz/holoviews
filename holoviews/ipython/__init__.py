import os
from unittest import SkipTest

import param
import holoviews

from IPython import version_info
from IPython.core.completer import IPCompleter
from IPython.display import HTML, publish_display_data
from param import ipython as param_ext

from ..core.dimension import LabelledData
from ..core.tree import AttrTree
from ..core.options import Store
from ..element.comparison import ComparisonTestCase
from ..util import extension
from ..plotting.renderer import Renderer
from .magics import load_magics
from .display_hooks import display  # noqa (API import)
from .display_hooks import pprint_display, png_display, svg_display


AttrTree._disabled_prefixes = ['_repr_','_ipython_canary_method_should_not_exist']

def show_traceback():
    """
    Display the full traceback after an abbreviated traceback has occurred.
    """
    from .display_hooks import FULL_TRACEBACK
    print(FULL_TRACEBACK)


class IPTestCase(ComparisonTestCase):
    """
    This class extends ComparisonTestCase to handle IPython specific
    objects and support the execution of cells and magic.
    """

    def setUp(self):
        super().setUp()
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


class notebook_extension(extension):
    """
    Notebook specific extension to hv.extension that offers options for
    controlling the notebook environment.
    """

    css = param.String(default='', doc="Optional CSS rule set to apply to the notebook.")

    logo = param.Boolean(default=True, doc="Toggles display of HoloViews logo")

    inline = param.Boolean(default=True, doc="""
        Whether to inline JS and CSS resources.
        If disabled, resources are loaded from CDN if one is available.""")

    width = param.Number(default=None, bounds=(0, 100), doc="""
        Width of the notebook as a percentage of the browser screen window width.""")

    display_formats = param.List(default=['html'], doc="""
        A list of formats that are rendered to the notebook where
        multiple formats may be selected at once (although only one
        format will be displayed).

        Although the 'html' format is supported across backends, other
        formats supported by the current backend (e.g. 'png' and 'svg'
        using the matplotlib backend) may be used. This may be useful to
        export figures to other formats such as PDF with nbconvert.""")

    allow_jedi_completion = param.Boolean(default=False, doc="""
       Whether to allow jedi tab-completion to be enabled in IPython.
       Disabled by default because many HoloViews features rely on
       tab-completion machinery not supported when using jedi.""")

    case_sensitive_completion = param.Boolean(default=False, doc="""
       Whether to monkey patch IPython to use the correct tab-completion
       behavior. """)

    _loaded = False

    def __call__(self, *args, **params):
        comms = params.pop('comms', None)
        super().__call__(*args, **params)
        # Abort if IPython not found
        try:
            ip = params.pop('ip', None) or get_ipython() # noqa (get_ipython)
        except:
            return

        # Notebook archive relies on display hooks being set to work.
        try:
            if version_info[0] >= 4:
                import nbformat # noqa (ensures availability)
            else:
                from IPython import nbformat # noqa (ensures availability)
            try:
                from .archive import notebook_archive
                holoviews.archive = notebook_archive
            except AttributeError as e:
                if str(e) != "module 'tornado.web' has no attribute 'asynchronous'":
                    raise

        except ImportError:
            pass

        # Not quite right, should be set when switching backends
        if 'matplotlib' in Store.renderers and not notebook_extension._loaded:
            svg_exporter = Store.renderers['matplotlib'].instance(holomap=None,fig='svg')
            holoviews.archive.exporters = [svg_exporter] + holoviews.archive.exporters

        p = param.ParamOverrides(self, {k:v for k,v in params.items() if k!='config'})
        if p.case_sensitive_completion:
            from IPython.core import completer
            completer.completions_sorting_key = self.completions_sorting_key
        if not p.allow_jedi_completion and hasattr(IPCompleter, 'use_jedi'):
            ip.run_line_magic('config', 'IPCompleter.use_jedi = False')

        resources = self._get_resources(args, params)

        Store.display_formats = p.display_formats
        if 'html' not in p.display_formats and len(p.display_formats) > 1:
            msg = ('Output magic unable to control displayed format '
                   'as IPython notebook uses fixed precedence '
                   'between %r' % p.display_formats)
            display(HTML('<b>Warning</b>: %s' % msg))

        loaded = notebook_extension._loaded
        if loaded == False:
            param_ext.load_ipython_extension(ip, verbose=False)
            load_magics(ip)
            Store.output_settings.initialize(list(Store.renderers.keys()))
            Store.set_display_hook('html+js', LabelledData, pprint_display)
            Store.set_display_hook('png', LabelledData, png_display)
            Store.set_display_hook('svg', LabelledData, svg_display)
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

        from panel import config
        if hasattr(config, 'comms') and comms:
            config.comms = comms

        same_cell_execution = getattr(self, '_repeat_execution_in_cell', False)
        for r in [r for r in resources if r != 'holoviews']:
            Store.renderers[r].load_nb(inline=p.inline)
        if not same_cell_execution:
            Renderer.load_nb(inline=p.inline)

        if hasattr(ip, 'kernel') and not loaded:
            Renderer.comm_manager.get_client_comm(notebook_extension._process_comm_msg,
                                                  "hv-extension-comm")

        # Create a message for the logo (if shown)
        if not same_cell_execution and p.logo:
            self.load_logo(logo=p.logo,
                           bokeh_logo=  p.logo and ('bokeh' in resources),
                           mpl_logo=    p.logo and (('matplotlib' in resources)
                                                    or resources==['holoviews']),
                           plotly_logo= p.logo and ('plotly' in resources))

    @classmethod
    def completions_sorting_key(cls, word):
        "Fixed version of IPyton.completer.completions_sorting_key"
        prio1, prio2 = 0, 0
        if word.startswith('__'):  prio1 = 2
        elif word.startswith('_'): prio1 = 1
        if word.endswith('='):     prio1 = -1
        if word.startswith('%%'):
            if not "%" in word[2:]:
                word = word[2:];   prio2 = 2
        elif word.startswith('%'):
            if not "%" in word[1:]:
                word = word[1:];   prio2 = 1
        return prio1, word, prio2


    def _get_resources(self, args, params):
        """
        Finds the list of resources from the keyword parameters and pops
        them out of the params dictionary.
        """
        resources = []
        disabled = []
        for resource in ['holoviews'] + list(Store.renderers.keys()):
            if resource in args:
                resources.append(resource)

            if resource in params:
                setting = params.pop(resource)
                if setting is True and resource != 'matplotlib':
                    if resource not in resources:
                        resources.append(resource)
                if setting is False:
                    disabled.append(resource)

        unmatched_args = set(args) - set(resources)
        if unmatched_args:
            display(HTML('<b>Warning:</b> Unrecognized resources %s'
                         % ', '.join(unmatched_args)))

        resources = [r for r in resources if r not in disabled]
        if ('holoviews' not in disabled) and ('holoviews' not in resources):
            resources = ['holoviews'] + resources
        return resources

    @classmethod
    def load_logo(cls, logo=False, bokeh_logo=False, mpl_logo=False, plotly_logo=False):
        """
        Allow to display Holoviews' logo and the plotting extensions' logo.
        """
        import jinja2

        templateLoader = jinja2.FileSystemLoader(os.path.dirname(os.path.abspath(__file__)))
        jinjaEnv = jinja2.Environment(loader=templateLoader)
        template = jinjaEnv.get_template('load_notebook.html')
        html = template.render({'logo':        logo,
                                'bokeh_logo':  bokeh_logo,
                                'mpl_logo':    mpl_logo,
                                'plotly_logo': plotly_logo})
        publish_display_data(data={'text/html': html})


notebook_extension.add_delete_action(Renderer._delete_plot)


def load_ipython_extension(ip):
    notebook_extension(ip=ip)

def unload_ipython_extension(ip):
    notebook_extension._loaded = False
