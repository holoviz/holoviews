"""
Implements NotebookArchive used to automatically capture notebook data
and export it to disk via the display hooks.
"""

import time, os, json, traceback
import io

from IPython.nbformat import reader
from IPython.nbformat import convert
from IPython.display import Javascript, display

from IPython.nbconvert.preprocessors import Preprocessor
from IPython.nbconvert import HTMLExporter

import param
from ..core.io import FileArchive
from ..plotting import PlotRenderer, HTML_TAGS

try:  # IPython 3
    from IPython.nbconvert.preprocessors.clearoutput import ClearOutputPreprocessor
    from IPython.nbconvert import NotebookExporter
except:
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


class NotebookArchive(FileArchive):
    """
    FileArchive that can automatically capture notebook data via the
    display hooks and automatically adds a notebook HTML snapshot to
    the archive upon export.
    """
    exporter = param.Callable(default=PlotRenderer.instance(holomap=None))

    namespace = param.String('holoviews.archive', doc="""
        The name of the current in the NotebookArchive instance in the
        IPython namespace that must be available.""")

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
        self._replacements = {}
        self.export_success = False
        self._notebook_data = None
        self._notebook_name = None
        self._timestamp = None
        self._tags = {val[0]:val[1] for val in HTML_TAGS.values()
                      if isinstance(val, tuple) and len(val)==2}


    def export(self, timestamp=None):
        """
        Get the current notebook data and export.
        """
        self._timestamp = timestamp if (timestamp is not None) else tuple(time.localtime())
        self.export_success = False
        self._notebook_data = io.BytesIO()
        name = self.namespace
        # Unfortunate javascript hacks to get at notebook data
        capture_cmd = ((r"var capture = '%s._notebook_data.write(r\"\"\"'" % name)
                       + r"+json_string+'\"\"\".encode(\'utf-8\'))'; ")
        nbname = r"var nbname = IPython.notebook.get_notebook_name(); "
        nbcmd = (r"var nbcmd = '%s._notebook_name = \"' + nbname + '\"'; " % name)
        cmd = (r'var kernel = IPython.notebook.kernel; '
               + nbname + nbcmd + "kernel.execute(nbcmd); "
               + r'var json_data = IPython.notebook.toJSON(); '
               + r'var json_string = JSON.stringify(json_data); '
               + capture_cmd
               + "var pycmd = capture + ';%s._export_with_html()'; " % name
               + r"kernel.execute(pycmd)")

        tstamp = time.strftime(self.timestamp_format, self._timestamp)
        export_name = self._format(self.export_name, {'timestamp':tstamp, 'notebook':'{notebook}'})
        print(('Export name: %r\nDirectory    %r' % (export_name,
                                                     os.path.join(os.path.abspath(self.root))))
               + '\n\nIf no output appears, please check holoviews.archive.traceback')
        display(Javascript(cmd))


    def add(self, obj=None, filename=None, data=None, info={}, html=None):
        "Similar to FileArchive.add but accepts html strings for substitution"
        initial_last_key = self._files.keys()[-1] if len(self) else None
        if self.auto:
            super(NotebookArchive, self).add(obj, filename, data,
                                             info=dict(info, notebook='{notebook}'))
            # Only add substitution if file successfully added to archive.
            new_last_key = self._files.keys()[-1] if len(self) else None
            if new_last_key != initial_last_key:
                self._replacements[new_last_key] = html


    def _generate_html(self, node, substitutions):
        exporter = HTMLExporter()
        exporter.register_preprocessor(Substitute(self.nbversion,
                                                  substitutions))
        html,_ = exporter.from_notebook_node(node)
        return html


    def _clear_notebook(self, node):
        if NotebookExporter is not None:
            exporter = NotebookExporter()
            exporter.register_preprocessor(ClearOutputPreprocessor(enabled=True))
            cleared,_ = exporter.from_notebook_node(node)
        else:
            stripped_node = v3_strip_output(node)
            cleared = current.writes(node, 'ipynb')
        return cleared


    def _export_with_html(self):
        "Computes substitions before using nbconvert with preprocessors"
        try:
            tstamp = time.strftime(self.timestamp_format, self._timestamp)

            substitutions = {}
            for (basename, ext), entry in self._files.items():
                (_, info) = entry
                html_key = self._replacements.get((basename, ext), None)
                if html_key is None: continue
                filename = self._format(basename, {'timestamp':tstamp,
                                                   'notebook':self._notebook_name})
                fpath = filename+(('.%s' % ext) if ext else '')
                msg = "<center><b>%s</b><center/>"
                if 'mime_type' not in info:
                    link_html = msg % "Could not determine data mime-type"
                if info['mime_type'] not in self._tags:
                    link_html = msg % 'Could not determine HTML tag from mime-type'

                info = {'src':fpath, 'mime_type':info['mime_type']}
                link_html = self._format(self._tags[info['mime_type']],
                                         {'src':fpath, 'mime_type':info['mime_type']})
                substitutions[html_key] = (link_html, fpath)

            node = self._get_notebook_node()
            html = self._generate_html(node, substitutions)

            export_filename = self.snapshot_name

            notebook =  self._notebook_name if self._notebook_name else '-no-notebook-name-'
            # Add the html snapshot
            super(NotebookArchive, self).add(filename=export_filename,
                                             data=html, info={'file-ext':'html',
                                                              'mime_type':'text/html',
                                                              'notebook':notebook})
            # Add cleared notebook
            cleared = self._clear_notebook(node)
            super(NotebookArchive, self).add(filename=export_filename,
                                             data=cleared, info={'file-ext':'ipynb',
                                                                 'mime_type':'text/json',
                                                                 'notebook':notebook})
            # If store cleared_notebook... save here
            super(NotebookArchive, self).export(timestamp=self._timestamp,
                                                info={'notebook':notebook})
        except Exception as e:
            self.traceback = traceback.format_exc()
        self.export_success = True

    def _get_notebook_node(self):
        "Load captured notebook node"
        self._notebook_data.seek(0, os.SEEK_END)
        size = self._notebook_data.tell()
        if size == 0:
            raise Exception("Captured buffer size for notebook node is zero.")
        self._notebook_data.seek(0)
        node = reader.reads(self._notebook_data.read().decode('utf-8'))
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
