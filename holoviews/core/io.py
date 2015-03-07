"""
Module defining input/output interfaces to HoloViews.

There are two components for input/output:

Exporters: Process (composite) HoloViews objects one at a time. For
           instance, an exporter may render a HoloViews object as a
           svg or perhaps pickle it.

Archives: A collection of HoloViews objects that are first collected
          then processed together. For instance, collecting HoloViews
          objects for a report then generating a PDF or collecting
          HoloViews objects to dump to HDF5.
"""
from __future__ import absolute_import
import os
import time
import string
import pickle
import zipfile
import tarfile

from io import BytesIO
from hashlib import sha256

import param

from .util import unique_iterator
from .ndmapping import OrderedDict, UniformNdMapping


class Exporter(param.ParameterizedFunction):
    """
    An Exporter is a parameterized function that accepts a HoloViews
    object and converts it to a new some new format. This mechanism is
    designed to be very general so here are a few examples:

    Pickling:   Native Python, supported by HoloViews.
    Rendering:  Currently using matplotlib but could use any plotting backend.
    Storage:    Databases (e.g SQL), HDF5 etc.
    """

    def __call__(self, obj, fmt=None):
        """
        Given a HoloViews object return the raw exported data and
        corresponding metadata as the tuple (data, metadata). The
        metadata should include:

        'file-ext' : The file extension if applicable (else empty string)
        'mime_type': The mime-type of the data.
        'size'     : Size in bytes of the returned data.

        The fmt argument may be used with exporters that support multiple
        output formats. If not supplied, the exporter is to pick an
        appropriate format automatically.
        """
        raise NotImplementedError("Exporter not implemented.")

    def save(self, obj, basename, fmt=None):
        """
        Similar to the call method except saves exporter data to disk
        into a file with specified basename. For exporters that
        support multiple formats, the fmt argument may also be
        supplied (which typically corresponds to the file-extension).
        """
        raise NotImplementedError("Exporter save method not implemented.")


class Pickler(Exporter):
    """
    Simple example of an archiver that simply returns the pickled data.
    """

    protocol = param.Integer(default=2, doc="""
      The pickling protocol where 0 is ASCII, 1 supports old Python
      versions and 2 is efficient for new style classes.""")

    def __call__(self, obj):
        data = pickle.dumps(obj, protocol=self.protocol)
        return data, {'file-ext':'pkl',
                      'size':len(data),
                      'mime_type':'application/python-pickle'}

    def save(self, obj, basename):
        with open(basename+'.pkl', 'w') as f:
            pickle.dump(obj, f, protocol=self.protocol)



class Archive(param.Parameterized):
    """
    An Archive is a means to collect and store a collection of
    HoloViews objects in any number of different ways. Examples of
    possible archives:

    * Generating tar or zip files (compressed or uncompressed).
    * Collating a report or document (e.g. PDF, HTML, LaTex).
    * Storing a collection of HoloViews objects to a database or HDF5.
    """

    exporter= param.ClassSelector(class_=Exporter, doc="""
      The exporter function used to convert HoloViews objects into the
      appropriate format."""  )

    def __init__(self, **params):
        super(Archive, self).__init__(**params)
        keywords = ['%s=%s' % (k, v.__class__.__name__) for k,v in self.params().items()]
        self.__call__.__func__.__doc__ = '__call__(%s)' % ', '.join(keywords)


    def __call__(self, **kwargs):
        """
        Convenience method for setting multiple parameters on an
        Archive instance.
        """
        for k, v in kwargs.items():
            self.set_param(k,v)

    def add(self, obj, *args, **kwargs):
        """
        Add a HoloViews object to the archive.
        """
        raise NotImplementedError

    def export(self,*args, **kwargs):
        """
        Finalize and close the archive.
        """
        raise NotImplementedError



def name_generator(obj):
    return ','.join(obj.traverse(lambda x: (x.group
                                            + ('-'  +x.label if x.label else ''))))

class FileArchive(Archive):
    """
    A file archive stores files on disk, either unpacked in a
    directory or in an archive format (e.g. a zip file).
    """

    exporter= param.Callable(default=Pickler, doc="""
        The exporter function used to convert HoloViews objects into the
        appropriate format.""")

    dimension_formatter = param.String("{name}_{range}", doc="""
        A string formatter for the output file based on the
        supplied HoloViews objects dimension names and values.
        Valid fields are the {name}, {range} and {unit} of the
        dimensions.""")

    object_formatter = param.Callable(default=name_generator, doc="""
        Callable that given an object returns a string suitable for
        inclusion in file and directory names. This is what generates
        the value used in the {obj} field of the filename
        formatter.""")

    filename_formatter = param.String('{dimensions}-{group}-{label}-{obj}', doc="""
        A string formatter for output filename based on the HoloViews
        object that is being rendered to disk.

        The available fields are the {type}, {group}, {label}, {obj}
        of the holoviews object added to the archive as well as
        {timestamp}, {obj} and {SHA}. The {timestamp} is the export
        timestamp using timestamp_format, {obj} is the object
        representation as returned by object_formatter and {SHA} is
        the SHA of the {obj} value used to compress it into a shorter
        string.""")

    timestamp_format = param.String("%Y_%m_%d-%H_%M_%S", doc="""
        The timestamp format that will be substituted for the
        {timestamp} field in the export name.""")

    root = param.String('.', doc="""
        The root directory in which the output directory is
        located. May be an absolute or relative path.""")

    archive_format = param.ObjectSelector('zip', objects=['zip', 'tar'], doc="""
        The archive format to use if there are multiple files and pack
        is set to True """)

    pack = param.Boolean(default=True, doc="""
        Whether or not to pack to contents into the specified archive
        format. If pack is False, the contents will be output to a
        directory.

        Note that if there is only a single file in the archive, no
        packing will occur and no directory is created. Instead, the
        file is treated as a single-file archive. """)

    export_name = param.String(default='{timestamp}', doc="""
        The name assigned to the overall export. If an archive file is
        used, this is the correspond filename (e.g of the exporter zip
        file). Alternatively, if unpack=False, this is the name of the
        output directory. Lastly, for archives of a single file, this
        is the basename of the output file.

        The {timestamp} field is available to include the timestamp at
        the time of export in the chosen timestamp format.""")

    unique_name = param.Boolean(default=True, doc="""
       Whether the export name should be made unique with a numeric
       suffix. If set to False, any existing export of the same name
       will be removed and replaced.""")

    @classmethod
    def parse_fields(cls, formatter):
        "Returns the format fields otherwise raise exception"
        if formatter is None: return []
        try:
            parse = list(string.Formatter().parse(formatter))
            return  set(f for f in list(zip(*parse))[1] if f is not None)
        except:
            raise SyntaxError("Could not parse formatter %r" % formatter)

    # Mime-types that need encoding as utf-8 before archiving.
    _utf8_mime_types = ['image/svg+xml', 'text/html']

    def __init__(self, **params):
        super(FileArchive, self).__init__(**params)
        #  Items with key: (basename,ext) and value: (data, info)
        self._files = OrderedDict()
        self._validate_formatters()


    def _dim_formatter(self, obj):
        if not obj: return ''
        key_dims = obj.traverse(lambda x: x.key_dimensions, [UniformNdMapping])
        constant_dims = obj.traverse(lambda x: x.constant_dimensions)
        dims = []
        map(dims.extend, key_dims + constant_dims)
        dims = unique_iterator(dims)
        dim_strings = []
        for dim in dims:
            lower, upper = obj.range(dim.name)
            lower, upper = (dim.pprint_value(lower),
                            dim.pprint_value(upper))
            if lower == upper:
                range = dim.pprint_value(lower)
            else:
                range = "%s-%s" % (lower, upper)
            formatters = {'name': dim.name, 'range': range,
                          'unit': dim.unit}
            dim_strings.append(self.dimension_formatter.format(**formatters))
        return '_'.join(dim_strings)


    def _validate_formatters(self):
        ffields =   {'type', 'group', 'label', 'obj', 'SHA', 'timestamp', 'dimensions'}
        efields = {'timestamp'}
        if not self.parse_fields(self.filename_formatter).issubset(ffields):
            raise Exception("Valid filename fields are: %s" % ','.join(sorted(ffields)))
        elif not self.parse_fields(self.export_name).issubset(efields):
            raise Exception("Valid export fields are: %s" % ','.join(sorted(efields)))
        try: time.strftime(self.timestamp_format, tuple(time.localtime()))
        except: raise Exception("Timestamp format invalid")


    def add(self, obj=None, filename=None, data=None, info={}):
        """
        If a filename is supplied, it will be used. Otherwise, a
        filename will be generated from the supplied object. Note that
        if the explicit filename uses the {timestamp} field, it will
        be formatted upon export.

        The data to be archived is either supplied explicitly as
        'data' or automatically rendered from the object.
        """
        if [filename, obj] == [None, None]:
            raise Exception("Either filename or a HoloViews object is "
                            "needed to create an entry in the archive.")
        elif obj is None and not self.parse_fields(filename).issubset({'timestamp'}):
            raise Exception("Only the {timestamp} formatter may be used unless an object is supplied.")
        elif [obj, data] == [None, None]:
            raise Exception("Either an object or explicit data must be "
                            "supplied to create an entry in the archive.")
        if data is None:
            rendered = self.exporter(obj)
            if rendered is None: return
            (data, info) = rendered
        self._validate_formatters()

        hashfn = sha256()
        obj_str = 'None' if obj is None else self.object_formatter(obj)
        dimensions = self._dim_formatter(obj)
        dimensions = dimensions if dimensions else 'no-dimensions'

        hashfn.update(obj_str.encode('utf-8'))
        format_values = {'timestamp': '{timestamp}',
                         'dimensions': dimensions,
                         'group':   getattr(obj, 'group', 'no-group'),
                         'label':   getattr(obj, 'label', 'no-label'),
                         'type':    obj.__class__.__name__,
                         'obj':     obj_str,
                         'SHA':     hashfn.hexdigest()}

        if filename is None:
            filename = self._format(self.filename_formatter, format_values)

        ext = info.get('file-ext', '')
        (unique_key, ext) = self._unique_name(filename, ext,
                                              self._files.keys(), force=True)
        self._files[(unique_key, ext)] = (data, info)

    def _encoding(self, entry):
        (data, info) = entry
        if info['mime_type'] in self._utf8_mime_types:
            utf_data = data.encode('utf-8')
            return utf_data
        else:
            return data

    def _zip_archive(self, export_name, files, root):
        archname = '.'.join(self._unique_name(export_name, 'zip', root))
        with zipfile.ZipFile(os.path.join(root, archname), 'w') as zipf:
            for (basename, ext), entry in files:
                filename = self._normalize_name(basename, ext)
                zipf.writestr(('%s/%s' % (export_name, filename)), self._encoding(entry))

    def _tar_archive(self, export_name, files, root):
        archname = '.'.join(self._unique_name(export_name, 'tar', root))
        with tarfile.TarFile(os.path.join(root, archname), 'w') as tarf:
            for (basename, ext), entry in files:
                filename = self._normalize_name(basename, ext)
                tarinfo = tarfile.TarInfo('%s/%s' % (export_name, filename))
                filedata = self._encoding(entry)
                tarinfo.size = len(filedata)
                tarf.addfile(tarinfo, BytesIO(filedata))


    def _unique_name(self, basename, ext, existing, force=False):
        """
        Find a unique basename for a new file/key where existing is
        either a list of (basename, ext) pairs or an absolute path to
        a directory.

        By default, uniqueness is enforced dependning on the state of
        the unique_name parameter (for export names). If force is
        True, this parameter is ignored and uniqueness is guaranteed.
        """
        skip = False if force else (not self.unique_name)
        if skip: return (basename, ext)
        ext = '' if ext is None else ext

        ext = '' if ext is None else ext
        if isinstance(existing, str):
            split = [os.path.splitext(el)
                     for el in os.listdir(os.path.abspath(existing))]
            existing = [(n, ex if not ex else ex[1:]) for (n, ex) in split]
        new_name, counter = basename, 1
        while (new_name, ext) in existing:
            new_name = basename+'-'+str(counter)
            counter += 1
        return (new_name, ext)


    def _normalize_name(self, basename, ext='', tail=10, join='...', maxlen=100):
        max_len = maxlen-len(ext)
        if len(basename) > max_len:
            start = basename[:max_len-(tail + len(join))]
            end = basename[-tail:]
            basename = start + join + end
        filename = '%s.%s' % (basename, ext) if ext else basename
        return filename.replace(' ', '_')


    def export(self, timestamp=None):
        """
        Export the archive, directory or file.
        """
        tval = tuple(time.localtime()) if timestamp is None else timestamp
        tstamp = time.strftime(self.timestamp_format, tval)
        export_name = self._format(self.export_name, {'timestamp':tstamp})
        files = [((self._format(base, {'timestamp':tstamp}), ext), val)
                 for ((base, ext), val) in self._files.items()]
        root = os.path.abspath(self.root)
        # Make directory and populate if multiple files and not packed
        if len(self) > 1 and not self.pack:
            output_dir = os.path.join(root,
                                      self._unique_name(export_name,'', root)[0])
            os.makedirs(output_dir)
            for (basename, ext), entry in files:
                (data, info) = entry

                filename = '%s.%s' % (basename, ext) if ext else basename
                fpath = os.path.join(output_dir, filename)
                with open(fpath, 'w') as f: f.write(data)
        elif len(files) == 1:
            ((_, ext), entry) = files[0]
            (data, info) = entry
            unique_name = self._unique_name(export_name, ext, root)
            filename = self._normalize_name(*unique_name)
            fpath = os.path.join(root, filename)
            with open(fpath, 'w') as f: f.write(data)
        elif self.archive_format == 'zip':
            self._zip_archive(export_name, files, root)
        elif self.archive_format == 'tar':
            self._tar_archive(export_name, files, root)
        self._files = OrderedDict()

    def _format(self, formatter, info):
        filtered = {k:v for k,v in info.items()
                    if k in self.parse_fields(formatter)}
        return formatter.format(**filtered)

    def __len__(self):
        "The number of files currently specified in the archive"
        return len(self._files)

    def __repr__(self):
        return self.pprint()
