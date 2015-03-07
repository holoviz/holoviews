"""
Implements NotebookArchive used to automatically capture notebook data
and export it to disk via the display hooks.
"""

import time, os, json, traceback, threading
from io import BytesIO

from IPython.nbformat import reader
from IPython.display import Javascript, display

from IPython.nbconvert.preprocessors import Preprocessor
from IPython.nbconvert import HTMLExporter

import param
from ..core.io import FileArchive
from ..plotting import PlotRenderer, HTML_TAGS



class NotebookArchive(FileArchive):
    """
    FileArchive that can automatically capture notebook data via the
    display hooks and automatically adds a notebook HTML snapshot to
    the archive upon export.
    """

    export_timeout = param.Integer(default=15, doc="""
       A timeout limit (in seconds) between when the export method is
       run and when the export process begins. Note that a few more
       seconds may elapse before the files actually appear on disk.""")

    exporter = param.Callable(default=PlotRenderer.instance(holomap=None))

    namespace = param.String('holoviews.archive', doc="""
        The name of the current in the NotebookArchive instance in the
        IPython namespace that must be available.""")

    snapshot_name = param.String('index', doc="""
        The basename of the exported notebook snapshot (html). It may
        optionally use the {timestamp} formatter.""")

    auto = param.Boolean(False)

    # Used for debugging to view Exceptions raised from Javascript!
    traceback = None

    def __init__(self, **params):
        super(NotebookArchive, self).__init__(**params)
        self.notebook = BytesIO()
        self.nbversion = None
        self._replacements = {}
        self._exported, self._cancel = False, False
        self._timestamp = None
        self._tags = {val[0]:val[1] for val in HTML_TAGS.values()
                      if isinstance(val, tuple) and len(val)==2}

    def _timeout(self, parent, export_name):
        """
        Timeout run in a separate thread to allow the notebook to keep
        running after export is called.
        """
        # If you click 'Run All' in the notebook, all code cells are
        # queued which means export may never actually happen (e.g. if
        # an Exception occurs between the export and the end of the
        # notebook. This method makes sure the user knows if the
        # export has not occured (via an alert) using a timeout.
        timeout = time.time() + (parent.export_timeout if parent.export_timeout else float('inf'))
        while (time.time() < timeout or parent._exported):
            time.sleep(0.5)
        if not parent._exported:
            parent._cancel = True
            msg = (("%s:\n\n" %  self.namespace)
                + "Export \'{export_name}\' cancelled."
                + "\n\nTimeout of {export_timeout} seconds exceeded"
                + "\n\nPlease make sure to call export at the very end of your notebook.")
            msg = msg.format(export_name=export_name, export_timeout=parent.export_timeout)
            display(Javascript("alert({msg});".format(msg=repr(msg))))


    def export(self, export_timeout=None, timestamp=None):
        """
        Get the current notebook data and export.
        """
        self.export_timeout = export_timeout if export_timeout else self.export_timeout
        self._timestamp = timestamp if (timestamp is not None) else tuple(time.localtime())
        self._cancel, self._exported = False, False
        name = self.namespace
        # Unfortunate javascript hacks to get at notebook data
        capture_cmd = ((r"var capture = '%s.notebook.write(r\"\"\"'" % name)
                       + r"+json_string+'\"\"\".encode(\'utf-8\'))';")
        cmd = (r'var kernel = IPython.notebook.kernel;'
               r'var json_data = IPython.notebook.toJSON();'
               r'var json_string = JSON.stringify(json_data);'
               + capture_cmd
               + "var pycmd = capture + ';%s._export_with_html()';" % name
               + r"kernel.execute(pycmd)")

        tstamp = time.strftime(self.timestamp_format, self._timestamp)
        export_name = self._format(self.export_name, {'timestamp':tstamp})
        info = (export_name, os.path.join(os.path.abspath(self.root)), self.export_timeout)
        print('Export name:  %r\nDirectory     %r\nTimeout limit: %d seconds' % info)
        display(Javascript(cmd))
        t = threading.Thread(target=self._timeout, args=(self,export_name))
        t.start()


    def add(self, obj=None, filename=None, data=None, info={}, html=None):
        "Similar to FileArchive.add but accepts html strings for substitution"
        initial_last_key = self._files.keys()[-1] if len(self) else None
        if self.auto:
            super(NotebookArchive, self).add(obj, filename, data, info)
            # Only add substitution if file successfully added to archive.
            new_last_key = self._files.keys()[-1] if len(self) else None
            if new_last_key != initial_last_key:
                self._replacements[new_last_key] = html


    def _export_with_html(self):
        "Computes substitions before using nbconvert with preprocessors"
        self._exported = True
        try:
            tstamp = time.strftime(self.timestamp_format, self._timestamp)

            substitutions = {}
            for (basename, ext), entry in self._files.items():
                (_, info) = entry
                html_key = self._replacements.get((basename, ext), None)
                if html_key is None: continue
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
            if not self._cancel:
                html,_ = exporter.from_notebook_node(node)
        except Exception as e:
            html = "Exception: " + str(e)
            html += traceback.format_exc()

        try:
            html_filename = self.snapshot_name
            super(NotebookArchive, self).add(filename=html_filename,
                                             data=html, info={'file-ext':'html', 'mime_type':'text/html'})
            # If store cleared_notebook... save here
            super(NotebookArchive, self).export(timestamp=self._timestamp)
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
