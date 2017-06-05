import os
from unittest import SkipTest

import param
from IPython import version_info
import holoviews
from param import ipython as param_ext
from IPython.display import HTML

from ..core.tree import AttrTree
from ..core.options import Store
from ..element.comparison import ComparisonTestCase
from ..interface.collector import Collector
from ..util import renderer
from ..plotting.renderer import Renderer
from .magics import load_magics
from .display_hooks import display  # noqa (API import)
from .display_hooks import set_display_hooks
from .widgets import RunProgress


Collector.interval_hook = RunProgress
AttrTree._disabled_prefixes = ['_repr_','_ipython_canary_method_should_not_exist']

def show_traceback():
    """
    Display the full traceback after an abbreviated traceback has occured.
    """
    from .display_hooks import FULL_TRACEBACK
    print(FULL_TRACEBACK)


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


class notebook_extension(renderer):
    """
    Notebook specific extension to hv.renderer that offers options for
    controlling the notebook environment.
    """

    css = param.String(default='', doc="Optional CSS rule set to apply to the notebook.")

    logo = param.Boolean(default=True, doc="Toggles display of HoloViews logo")

    inline = param.Boolean(default=True, doc="""Whether to inline JS and CSS resources,
        if disabled resources are loaded from CDN if one is available.""")

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

    _loaded = False

    def __call__(self, *args, **params):
        super(notebook_extension, self).__call__(*args, **params)
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
            from .archive import notebook_archive
            holoviews.archive = notebook_archive
        except ImportError:
            pass

        # Not quite right, should be set when switching backends
        if 'matplotlib' in Store.renderers and not notebook_extension._loaded:
            svg_exporter = Store.renderers['matplotlib'].instance(holomap=None,fig='svg')
            holoviews.archive.exporters = [svg_exporter] + holoviews.archive.exporters

        p = param.ParamOverrides(self, params)
        resources = self._get_resources(args, params)

        Store.display_formats = p.display_formats
        if 'html' not in p.display_formats and len(p.display_formats) > 1:
            msg = ('Output magic unable to control displayed format '
                   'as IPython notebook uses fixed precedence '
                   'between %r' % p.display_formats)
            display(HTML('<b>Warning</b>: %s' % msg))

        if notebook_extension._loaded == False:
            param_ext.load_ipython_extension(ip, verbose=False)
            load_magics(ip)
            Store.output_settings.initialize(list(Store.renderers.keys()))
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

        Renderer.load_nb()
        for r in [r for r in resources if r != 'holoviews']:
            Store.renderers[r].load_nb(inline=p.inline)

        # Create a message for the logo (if shown)
        self.load_hvjs(logo=p.logo, JS=('holoviews' in resources), message='')



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
    def load_hvjs(cls, logo=False, JS=True, message='HoloViewsJS successfully loaded.'):
        """
        Displays javascript and CSS to initialize HoloViews widgets.
        """
        import jinja2
        # Evaluate load_notebook.html template with widgetjs code
        if JS:
            widgetjs, widgetcss = Renderer.html_assets(extras=False, backends=[])
        else:
            widgetjs, widgetcss = '', ''
        templateLoader = jinja2.FileSystemLoader(os.path.dirname(os.path.abspath(__file__)))
        jinjaEnv = jinja2.Environment(loader=templateLoader)
        template = jinjaEnv.get_template('load_notebook.html')
        display(HTML(template.render({'widgetjs': widgetjs,
                                      'widgetcss': widgetcss,
                                      'logo': logo,
                                      'message':message})))

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
