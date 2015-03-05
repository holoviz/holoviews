import time
import os
import json
import traceback
from io import BytesIO

from IPython.nbformat import reader
from IPython.display import Javascript, display

from IPython.nbconvert.preprocessors import Preprocessor
from IPython.nbconvert import HTMLExporter

import param
from ..core.io import FileArchive
from ..plotting import PlotRenderer, HTML_TAGS


class NotebookArchive(FileArchive):

    exporter = param.Callable(default=PlotRenderer)

    namespace = param.String('holoviews.archive', doc="""
        The name of the current in the NotebookArchive instance in the
        IPython namespace that must be available.""")

    snapshot_name = param.String('snapshot', doc="""
        The basename of the exported notebook snapshot (html)""")

    auto = param.Boolean(False)

    # Used for debugging to view Exceptions raised from Javascript!
    traceback = None

    def __init__(self, **params):
        super(NotebookArchive, self).__init__(**params)
        self.notebook = BytesIO()
        self.nbversion = None
        self._replacements = {}
        self._tags = {val[0]:val[1] for val in HTML_TAGS.values()
                      if isinstance(val, tuple) and len(val)==2}

    def export(self):
        """
        Get the current notebook data and export.
        """
        name = self.namespace
        # Unfortunate javascript hacks to get at notebook data
        cmd = (r'var kernel = IPython.notebook.kernel;'
               r'var json_data = IPython.notebook.toJSON();'
               r'var json_string = JSON.stringify(json_data);'
               + (r"var command = '%s.notebook.write(r\"\"\"'+json_string+'\"\"\".encode(\'utf-8\'))';" % name)
               + "var pycmd = command + ';%s._export_with_html()';" % name
               + r"kernel.execute(pycmd)")
        display(Javascript(cmd))


    def add(self, obj=None, filename=None, data=None, info={}, html=None):
        if self.auto:
            super(NotebookArchive, self).add(obj, filename, data, info)
            last_key = self._files.keys()[-1]
            self._replacements[last_key] = html


    def _export_with_html(self):
        "Use nbconvert with Substitute preprocessor"
        try:
            tval = tuple(time.localtime())
            tstamp = time.strftime(self.timestamp_format, tval)

            substitutions = {}
            for (basename, ext), entry in self._files.items():
                (_, info) = entry
                html_key = self._replacements[(basename, ext)]
                filename = self._format(basename, {'timestamp':tstamp})
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
            exporter = HTMLExporter()
            exporter.register_preprocessor(Substitute(self.nbversion,
                                                      substitutions))
            html,_ = exporter.from_notebook_node(node)
        except Exception as e:
            html = "Exception: " + str(e)
            html += traceback.format_exc()

        try:
            filename = self.snapshot_name if (filename is None) else filename
            super(NotebookArchive, self).add(filename=filename,
                                             data=html, info={'file-ext':'html',
                                                              'mime_type':'text/html'})
            # If store cleared_notebook... save here
            super(NotebookArchive, self).export(timestamp=tval)
        except Exception as e:
            self.traceback = traceback.format_exc()


    def _get_notebook_node(self):
        "Load captured notebook node"
        self.notebook.seek(0, os.SEEK_END)
        size = self.notebook.tell()
        if size == 0:
            raise Exception("Captured buffer size for notebook node is zero.")
        self.notebook.seek(0)
        node = reader.reads(self.notebook.read().decode('utf-8'))
        self.nbversion = reader.get_version(node)
        self.notebook.close()
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
