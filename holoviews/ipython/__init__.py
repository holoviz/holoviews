import os

import param
from bokeh.settings import settings as bk_settings
from IPython.core.completer import IPCompleter
from IPython.display import HTML, publish_display_data
from param import ipython as param_ext

import holoviews as hv

from ..core.dimension import LabelledData
from ..core.options import Store
from ..core.tree import AttrTree
from ..util import extension
from .display_hooks import display, png_display, pprint_display, svg_display
from .magics import load_magics

AttrTree._disabled_prefixes = ['_repr_','_ipython_canary_method_should_not_exist']

def show_traceback():
    """Display the full traceback after an abbreviated traceback has occurred.

    """
    from .display_hooks import FULL_TRACEBACK
    print(FULL_TRACEBACK)


def __getattr__(attr):
    if attr == "IPTestCase":
        from ..element.comparison import IPTestCase
        from ..util.warnings import deprecated
        deprecated("1.23.0", old="holoviews.ipython.IPTestCase", new="holoviews.element.comparison.IPTestCase")
        return IPTestCase
    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")

HOLOVIEWS_B64_LOGO = (
    "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"
)
class notebook_extension(extension):
    """Notebook specific extension to hv.extension that offers options for
    controlling the notebook environment.

    """

    css = param.String(default='', doc="Optional CSS rule set to apply to the notebook.")

    logo = param.ClassSelector(default=True, class_=(bool, dict), doc="""
        Controls logo display. Dictionary option must include the keys
        `logo_link`, `logo_src`, and `logo_title`.""")

    inline = param.Boolean(default=False, doc="""
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

    allow_jedi_completion = param.Boolean(default=True, doc="""
       Whether to allow jedi tab-completion to be enabled in IPython.""")

    case_sensitive_completion = param.Boolean(default=False, doc="""
       Whether to monkey patch IPython to use the correct tab-completion
       behavior. """)

    enable_mathjax = param.Boolean(default=False, doc="""
        Whether to load bokeh-mathjax bundle in the notebook.""")

    _loaded = False

    def __call__(self, *args, **params):
        comms = params.pop('comms', None)
        super().__call__(*args, **params)
        # Abort if IPython not found
        try:
            ip = params.pop('ip', None) or get_ipython() # noqa (get_ipython)
        except Exception:
            return

        # Notebook archive relies on display hooks being set to work.
        try:
            import nbformat  # noqa: F401

            try:
                from .archive import notebook_archive
                hv.archive = notebook_archive
            except AttributeError as e:
                if str(e) != "module 'tornado.web' has no attribute 'asynchronous'":
                    raise

        except ImportError:
            pass

        # Not quite right, should be set when switching backends
        if 'matplotlib' in Store.renderers and not notebook_extension._loaded:
            svg_exporter = Store.renderers['matplotlib'].instance(holomap=None,fig='svg')
            hv.archive.exporters = [svg_exporter, *hv.archive.exporters]

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
                   f'between {p.display_formats!r}')
            display(HTML(f'<b>Warning</b>: {msg}'))

        loaded = notebook_extension._loaded
        if loaded == False:
            param_ext.load_ipython_extension(ip, verbose=False)
            load_magics(ip)
            Store.output_settings.initialize(list(Store.renderers.keys()))
            Store.set_display_hook('html+js', LabelledData, pprint_display)
            Store.set_display_hook('png', LabelledData, png_display)
            Store.set_display_hook('svg', LabelledData, svg_display)
            bk_settings.simple_ids.set_value(False)
            notebook_extension._loaded = True

        css = ''
        if p.width is not None:
            css += f'<style>div.container {{ width: {p.width}% }}</style>'
        if p.css:
            css += f'<style>{p.css}</style>'

        if css:
            display(HTML(css))

        resources = list(resources)
        if len(resources) == 0: return

        from panel import config, extension as panel_extension
        if hasattr(config, 'comms') and comms:
            config.comms = comms

        same_cell_execution = published = getattr(self, '_repeat_execution_in_cell', False)
        for r in [r for r in resources if r != 'holoviews']:
            Store.renderers[r].load_nb(inline=p.inline)

        from ..plotting.renderer import Renderer
        Renderer.load_nb(inline=p.inline, reloading=same_cell_execution, enable_mathjax=p.enable_mathjax)

        if not published and hasattr(panel_extension, "_display_globals"):
            panel_extension._display_globals()

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
        """Fixed version of IPyton.completer.completions_sorting_key

        """
        prio1, prio2 = 0, 0
        if word.startswith('__'):  prio1 = 2
        elif word.startswith('_'): prio1 = 1
        if word.endswith('='):     prio1 = -1
        if word.startswith('%%'):
            if '%' not in word[2:]:
                word, prio2 = word[2:], 2
        elif word.startswith('%'):
            if '%' not in word[1:]:
                word, prio2 = word[1:], 1
        return prio1, word, prio2


    def _get_resources(self, args, params):
        """Finds the list of resources from the keyword parameters and pops
        them out of the params dictionary.

        """
        resources = []
        disabled = []
        for resource in ['holoviews', *Store.renderers]:
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
            display(HTML("<b>Warning:</b> Unrecognized resources '{}'".format("', '".join(unmatched_args))))

        resources = [r for r in resources if r not in disabled]
        if ('holoviews' not in disabled) and ('holoviews' not in resources):
            resources = ['holoviews', *resources]
        return resources

    @classmethod
    def load_logo(cls, logo: dict | bool = False, bokeh_logo=False, mpl_logo=False, plotly_logo=False):
        """Allow to display Holoviews' logo and the plotting extensions' logo.

        """
        import jinja2

        templateLoader = jinja2.FileSystemLoader(os.path.dirname(os.path.abspath(__file__)))
        jinjaEnv = jinja2.Environment(loader=templateLoader)
        template = jinjaEnv.get_template('load_notebook.html')
        if isinstance(logo, dict):
            logo_src = logo['logo_src']
            logo_link = logo['logo_link']
            logo_title = logo['logo_title']
        elif not logo:
            logo_src = logo_link = logo_title = ''
        else:
            from .. import __version__

            logo_src = f'data:image/png;base64,{HOLOVIEWS_B64_LOGO}'
            logo_link = 'https://holoviews.org'
            logo_title = f'HoloViews {__version__}'

        bokeh_version = mpl_version = plotly_version = ''
        # Backends are already imported at this stage.
        if bokeh_logo:
            import bokeh
            bokeh_version = bokeh.__version__
        if mpl_logo:
            import matplotlib as mpl
            mpl_version = mpl.__version__
        if plotly_logo:
            import plotly
            plotly_version = plotly.__version__

        html = template.render({'logo':        logo,
                                'logo_src':    logo_src,
                                'logo_link':   logo_link,
                                'logo_title':  logo_title,
                                'bokeh_logo':  bokeh_logo,
                                'mpl_logo':    mpl_logo,
                                'plotly_logo': plotly_logo,
                                'bokeh_version':  bokeh_version,
                                'mpl_version':    mpl_version,
                                'plotly_version': plotly_version})
        publish_display_data(data={'text/html': html})


def _delete_plot(plot_id):
    from ..plotting.renderer import Renderer
    return Renderer._delete_plot(plot_id)

notebook_extension.add_delete_action(_delete_plot)


def load_ipython_extension(ip):
    notebook_extension(ip=ip)

def unload_ipython_extension(ip):
    notebook_extension._loaded = False
