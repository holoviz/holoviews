"""
Implements NotebookArchive used to automatically capture notebook data
and export it to disk via the display hooks.
"""
from __future__ import unicode_literals, division, absolute_import, print_function

import time, sys, os, traceback, io

from IPython import version_info
from IPython.display import Javascript, display

# Import appropriate nbconvert machinery
if version_info[0] >= 4:
    # Jupyter/IPython >=4.0
    from nbformat import reader
    from nbconvert.preprocessors import Preprocessor
    from nbconvert import HTMLExporter

    from nbconvert.preprocessors.clearoutput import ClearOutputPreprocessor
    from nbconvert import NotebookExporter
else:
    # IPython <= 3.0
    from IPython.nbformat import reader
    from IPython.nbconvert.preprocessors import Preprocessor
    from IPython.nbconvert import HTMLExporter

    if version_info[0] == 3:
        # IPython 3
        from IPython.nbconvert.preprocessors.clearoutput import ClearOutputPreprocessor
        from IPython.nbconvert import NotebookExporter
    else:
        # IPython 2
        from IPython.nbformat import current
        NotebookExporter, ClearOutputPreprocessor = None, None

        def v3_strip_output(nb):
            """strip the outputs from a notebook object"""
            nb["nbformat"] = 3
            nb["nbformat_minor"] = 0
            nb.metadata.pop('signature', None)
            for cell in nb.worksheets[0].cells:
                if 'outputs' in cell:
                    cell['outputs'] = []
                if 'prompt_number' in cell:
                    cell['prompt_number'] = None
            return nb

import param
from ..core.io import FileArchive, Pickler
from ..core.options import Store
from ..plotting.renderer import HTML_TAGS

try:
    # Only matplotlib outputs to graphical file formats at this time
    renderers = [Store.renderers['matplotlib'].instance(holomap=None, fig='svg')]
except:
    renderers = []

class NotebookArchive(FileArchive):
    """
    FileArchive that can automatically capture notebook data via the
    display hooks and automatically adds a notebook HTML snapshot to
    the archive upon export.
    """
    exporters = param.List(default=renderers + [Pickler])

    skip_notebook_export = param.Boolean(default=False, doc="""
        Whether to skip JavaScript capture of notebook data which may
        be unreliable. Also disabled automatic capture of notebook
        name.""")

    snapshot_name = param.String('index', doc="""
        The basename of the exported notebook snapshot (html). It may
        optionally use the {timestamp} formatter.""")

    filename_formatter = param.String(default='{dimensions},{obj}', doc="""
        Similar to FileArchive.filename_formatter except with support
        for the notebook name field as {notebook}.""")

    export_name = param.String(default='{notebook}', doc="""
        Similar to FileArchive.filename_formatter except with support
        for the notebook name field as {notebook}.""")


    auto = param.Boolean(False)

    # Used for debugging to view Exceptions raised from Javascript
    traceback = None

    ffields = FileArchive.ffields.union({'notebook'})
    efields = FileArchive.efields.union({'notebook'})

    def __init__(self, **params):
        super(NotebookArchive, self).__init__(**params)
        self.nbversion = None
        self.notebook_name = None
        self.export_success = None

        self._auto = False
        self._replacements = {}
        self._notebook_data = None
        self._timestamp = None
        self._tags = {val[0]:val[1] for val in HTML_TAGS.values()
                      if isinstance(val, tuple) and len(val)==2}

        keywords = ['%s=%s' % (k, v.__class__.__name__) for k,v in self.params().items()]
        self.auto.__func__.__doc__ = 'auto(enabled=Boolean, %s)' % ', '.join(keywords)


    def get_namespace(self):
        """
        Find the name the user is using to access holoviews.
        """
        if 'holoviews' not in sys.modules:
            raise ImportError('HoloViews does not seem to be imported')
        matches = [k for k,v in get_ipython().user_ns.items()
           if not k.startswith('_') and v is sys.modules['holoviews']]
        if len(matches) == 0:
            raise Exception("Could not find holoviews module in namespace")
        return '%s.archive' % matches[0]


    def last_export_status(self):
        "Helper to show the status of the last call to the export method."
        if self.export_success is True:
            print("The last call to holoviews.archive.export was successful.")
            return
        elif self.export_success is None:
            print("Status of the last call to holoviews.archive.export is unknown."
                  "\n(Re-execute this method once kernel status is idle.)")
            return
        print("The last call to holoviews.archive.export was unsuccessful.")
        if self.traceback is None:
            print("\n<No traceback captured>")
        else:
            print("\n"+self.traceback)


    def auto(self, enabled=True, **kwargs):
        """
        Method to enable or disable automatic capture, allowing you to
        simultaneously set the instance parameters.
        """
        self.namespace = self.get_namespace()
        self.notebook_name = "{notebook}"
        self._timestamp = tuple(time.localtime())
        kernel = r'var kernel = IPython.notebook.kernel; '
        nbname = r"var nbname = IPython.notebook.get_notebook_name(); "
        nbcmd = (r"var name_cmd = '%s.notebook_name = \"' + nbname + '\"'; " % self.namespace)
        cmd = (kernel + nbname + nbcmd + "kernel.execute(name_cmd); ")
        display(Javascript(cmd))
        time.sleep(0.5)
        self._auto=enabled
        self.set_param(**kwargs)

        tstamp = time.strftime(" [%Y-%m-%d %H:%M:%S]", self._timestamp)
        print("Automatic capture is now %s.%s"
              % ('enabled' if enabled else 'disabled',
                 tstamp if enabled else ''))

    def export(self, timestamp=None):
        """
        Get the current notebook data and export.
        """
        if self.skip_notebook_export:
            super(NotebookArchive, self).export(timestamp=self._timestamp,
                                                info={'notebook':self.notebook_name})
            return

        self.export_success = None
        self._notebook_data = io.StringIO()
        name = self.get_namespace()
        # Unfortunate javascript hacks to get at notebook data
        capture_cmd = ((r"var capture = '%s._notebook_data.write(r\"\"\"'" % name)
                       + r"+json_string+'\"\"\".decode(\'utf-8\'))'; ")
        cmd = (r'var kernel = IPython.notebook.kernel; '
               + r'var json_data = IPython.notebook.toJSON(); '
               + r'var json_string = JSON.stringify(json_data); '
               + capture_cmd
               + "var pycmd = capture + ';%s._export_with_html()'; " % name
               + r"kernel.execute(pycmd)")

        tstamp = time.strftime(self.timestamp_format, self._timestamp)
        export_name = self._format(self.export_name, {'timestamp':tstamp, 'notebook':self.notebook_name})
        print(('Export name: %r\nDirectory    %r' % (export_name,
                                                     os.path.join(os.path.abspath(self.root))))
               + '\n\nIf no output appears, please check holoviews.archive.last_export_status()')
        display(Javascript(cmd))


    def add(self, obj=None, filename=None, data=None, info={}, html=None):
        "Similar to FileArchive.add but accepts html strings for substitution"
        initial_last_key = list(self._files.keys())[-1] if len(self) else None
        if self._auto:
            super(NotebookArchive, self).add(obj, filename, data,
                                             info=dict(info, notebook=self.notebook_name))
            # Only add substitution if file successfully added to archive.
            new_last_key = list(self._files.keys())[-1] if len(self) else None
            if new_last_key != initial_last_key:
                self._replacements[new_last_key] = html


    # The following methods are executed via JavaScript and so fail
    # to appear in the coverage report even though they are tested.

    def _generate_html(self, node, substitutions):  # pragma: no cover
        exporter = HTMLExporter()
        exporter.register_preprocessor(Substitute(self.nbversion,
                                                  substitutions))
        html,_ = exporter.from_notebook_node(node)
        return html


    def _clear_notebook(self, node):                # pragma: no cover
        if NotebookExporter is not None:
            exporter = NotebookExporter()
            exporter.register_preprocessor(ClearOutputPreprocessor(enabled=True))
            cleared,_ = exporter.from_notebook_node(node)
        else:
            stripped_node = v3_strip_output(node)
            cleared = current.writes(stripped_node, 'ipynb')
        return cleared


    def _export_with_html(self):                    # pragma: no cover
        "Computes substitions before using nbconvert with preprocessors"
        self.export_success = False
        try:
            tstamp = time.strftime(self.timestamp_format, self._timestamp)
            substitutions = {}
            for (basename, ext), entry in self._files.items():
                (_, info) = entry
                html_key = self._replacements.get((basename, ext), None)
                if html_key is None: continue
                filename = self._format(basename, {'timestamp':tstamp,
                                                   'notebook':self.notebook_name})
                fpath = filename+(('.%s' % ext) if ext else '')
                info = {'src':fpath, 'mime_type':info['mime_type']}
                # No mime type
                if 'mime_type' not in info: pass
                # Not displayable in an HTML tag
                elif info['mime_type'] not in self._tags: pass
                else:
                    link_html = self._format(self._tags[info['mime_type']],
                                             {'src':fpath, 'mime_type':info['mime_type']})
                    substitutions[html_key] = (link_html, fpath)

            node = self._get_notebook_node()
            html = self._generate_html(node, substitutions)

            export_filename = self.snapshot_name

            # Add the html snapshot
            super(NotebookArchive, self).add(filename=export_filename,
                                             data=html, info={'file-ext':'html',
                                                              'mime_type':'text/html',
                                                              'notebook':self.notebook_name})
            # Add cleared notebook
            cleared = self._clear_notebook(node)
            super(NotebookArchive, self).add(filename=export_filename,
                                             data=cleared, info={'file-ext':'ipynb',
                                                                 'mime_type':'text/json',
                                                                 'notebook':self.notebook_name})
            # If store cleared_notebook... save here
            super(NotebookArchive, self).export(timestamp=self._timestamp,
                                                info={'notebook':self.notebook_name})
        except:
            self.traceback = traceback.format_exc()
        else:
            self.export_success = True

    def _get_notebook_node(self):                   # pragma: no cover
        "Load captured notebook node"
        self._notebook_data.seek(0, os.SEEK_END)
        size = self._notebook_data.tell()
        if size == 0:
            raise Exception("Captured buffer size for notebook node is zero.")
        self._notebook_data.seek(0)
        node = reader.reads(self._notebook_data.read())
        self.nbversion = reader.get_version(node)
        self._notebook_data.close()
        return node



class Substitute(Preprocessor):
    """
    An nbconvert preprocessor that substitutes one set of HTML data
    output for another, adding annotation to the output as required.

    The constructor accepts the notebook format version and a
    substitutions dictionary:

    {source_html:(target_html, annotation)}

    Where the annotation may be None (i.e. no annotation).
    """
    annotation = '<center><b>%s</b></center>'

    def __init__(self, version, substitutions, **kw):
        self.nbversion = version
        self.substitutions = substitutions
        super(Preprocessor, self).__init__(**kw)

    def __call__(self, nb, resources): # Temporary hack around 'enabled' flag
        return self.preprocess(nb,resources)


    def replace(self, src):
        "Given some source html substitute and annotated as applicable"
        for html in self.substitutions.keys():
            if src == html:
                annotation = self.annotation % self.substitutions[src][1]
                return annotation + self.substitutions[src][0]
        return src


    def preprocess_cell(self, cell, resources, index):
        v4 = (self.nbversion[0] == 4)
        if cell['cell_type'] == 'code':
            for outputs in cell['outputs']:
                output_key = ('execute_result' if v4 else 'pyout')
                if outputs['output_type'] == output_key:
                    # V1-3
                    if not v4 and 'html' in outputs:
                        outputs['html'] = self.replace(outputs['html'])
                    # V4
                    for data in outputs.get('data',[]):
                        if v4 and data == 'text/html':
                            substitution = self.replace(outputs['data']['text/html'])
                            outputs['data']['text/html'] = substitution
        return cell, resources


notebook_archive = NotebookArchive()
