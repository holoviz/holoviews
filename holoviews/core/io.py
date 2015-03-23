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

import re, os, time, string, zipfile, tarfile, shutil, itertools

from io import BytesIO
from hashlib import sha256

import param

from .options import Store
from .util import unique_iterator
from .ndmapping import OrderedDict, UniformNdMapping
from .dimension import LabelledData


class Reference(param.Parameterized):
    """
    A Reference allows access to an object to be deferred until it is
    needed in the appropriate context. References are used by
    Collector to capture the state of an object at collection time.

    One particularly important property of references is that they
    should be pickleable. This means that you can pickle Collectors so
    that you can unpickle them in different environments and still
    collect from the required object.

    A Reference only needs to have a resolved_type property and a
    resolve method. The constructor will take some specification of
    where to find the target object (may be the object itself).
    """

    @property
    def resolved_type(self):
        """
        Returns the type of the object resolved by this references. If
        multiple types are possible, the return is a tuple of types.
        """
        raise NotImplementedError


    def resolve(self, container=None):
        """
        Return the referenced object. Optionally, a container may be
        passed in from which the object is to be resolved.
        """
        raise NotImplementedError



class Exporter(param.ParameterizedFunction):
    """
    An Exporter is a parameterized function that accepts a HoloViews
    object and converts it to a new some new format. This mechanism is
    designed to be very general so here are a few examples:

    Pickling:   Native Python, supported by HoloViews.
    Rendering:  Currently using matplotlib but could use any plotting backend.
    Storage:    Databases (e.g SQL), HDF5 etc.
    """

    # Mime-types that need encoding as utf-8 upon export
    utf8_mime_types = ['image/svg+xml', 'text/html']

    metadata_fn = param.Callable(doc="""
      Function that generates additional metadata information from the
      HoloViews object being saved.

      Must return a dictionary containing string keys and simple
      literal values such ints, floats, short strings and booleans. By
      default the metadata is an empty dictionary.""")


    @classmethod
    def encode(cls, entry):
        """
        Classmethod that applies conditional encoding based on
        mime-type. Given an entry as returned by __call__ return the
        data in the appropriate encoding.
        """
        (data, info) = entry
        if info['mime_type'] in cls.utf8_mime_types:
            return data.encode('utf-8')
        else:
            return data


    def __call__(self, obj, fmt=None):
        """
        Given a HoloViews object return the raw exported data and
        corresponding metadata as the tuple (data, metadata). The
        metadata should include:

        'file-ext' : The file extension if applicable (else empty string)
        'mime_type': The mime-type of the data.

        The fmt argument may be used with exporters that support multiple
        output formats. If not supplied, the exporter is to pick an
        appropriate format automatically.
        """
        raise NotImplementedError("Exporter not implemented.")

    def save(self, obj, basename, fmt=None, metadata={}, **kwargs):
        """
        Similar to the call method except saves exporter data to disk
        into a file with specified basename. For exporters that
        support multiple formats, the fmt argument may also be
        supplied (which typically corresponds to the file-extension).

        The supplied metadata dictionary updates the output of the
        metadata_fn (if any) which is then saved if supported.
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
        data = Store.dumps(obj, protocol=self.protocol)
        return data, {'file-ext':'pkl',
                      'mime_type':'application/python-pickle'}

    def save(self, obj, basename):
        with open(basename+'.pkl', 'w') as f:
            Store.dump(obj, f, protocol=self.protocol)



class Archive(param.Parameterized):
    """
    An Archive is a means to collect and store a collection of
    HoloViews objects in any number of different ways. Examples of
    possible archives:

    * Generating tar or zip files (compressed or uncompressed).
    * Collating a report or document (e.g. PDF, HTML, LaTex).
    * Storing a collection of HoloViews objects to a database or HDF5.
    """

    exporters= param.List(default=[], doc="""
        The exporter functions used to convert HoloViews objects into the
        appropriate format(s)."""  )

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



def simple_name_generator(obj):
    """
    Simple name_generator designed for HoloViews objects.

    Objects are labeled with {group}-{label} for each nested
    object, based on a depth-first search.  Adjacent objects with
    identical representations yield only a single copy of the
    representation, to avoid long names for the common case of
    a container whose element(s) share the same group and label.
    """

    if isinstance(obj, LabelledData):
        labels = obj.traverse(lambda x:
                              (x.group + ('-'  +x.label if x.label else '')))
        labels=[l[0] for l in itertools.groupby(labels)]
        obj_str = ','.join(labels)
    else:
        obj_str = repr(obj)
    return obj_str



class FileArchive(Archive):
    """
    A file archive stores files on disk, either unpacked in a
    directory or in an archive format (e.g. a zip file).
    """

    exporters= param.List(default=[Pickler], doc="""
        The exporter functions used to convert HoloViews objects into
        the appropriate format(s).""")

    dimension_formatter = param.String("{name}_{range}", doc="""
        A string formatter for the output file based on the
        supplied HoloViews objects dimension names and values.
        Valid fields are the {name}, {range} and {unit} of the
        dimensions.""")

    object_formatter = param.Callable(default=simple_name_generator, doc="""
        Callable that given an object returns a string suitable for
        inclusion in file and directory names. This is what generates
        the value used in the {obj} field of the filename
        formatter.""")

    filename_formatter = param.String('{dimensions},{obj}', doc="""
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

    pack = param.Boolean(default=False, doc="""
        Whether or not to pack to contents into the specified archive
        format. If pack is False, the contents will be output to a
        directory.

        Note that if there is only a single file in the archive, no
        packing will occur and no directory is created. Instead, the
        file is treated as a single-file archive.""")

    export_name = param.String(default='{timestamp}', doc="""
        The name assigned to the overall export. If an archive file is
        used, this is the correspond filename (e.g of the exporter zip
        file). Alternatively, if unpack=False, this is the name of the
        output directory. Lastly, for archives of a single file, this
        is the basename of the output file.

        The {timestamp} field is available to include the timestamp at
        the time of export in the chosen timestamp format.""")

    unique_name = param.Boolean(default=False, doc="""
       Whether the export name should be made unique with a numeric
       suffix. If set to False, any existing export of the same name
       will be removed and replaced.""")

    max_filename = param.Integer(default=100, bounds=(0,None), doc="""
       Maximum length to enforce on generated filenames.  100 is the
       practical maximum for zip and tar file generation, but you may
       wish to use a lower value to avoid long filenames.""")


    ffields = {'type', 'group', 'label', 'obj', 'SHA', 'timestamp', 'dimensions'}
    efields = {'timestamp'}

    @classmethod
    def parse_fields(cls, formatter):
        "Returns the format fields otherwise raise exception"
        if formatter is None: return []
        try:
            parse = list(string.Formatter().parse(formatter))
            return  set(f for f in list(zip(*parse))[1] if f is not None)
        except:
            raise SyntaxError("Could not parse formatter %r" % formatter)

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
        if not self.parse_fields(self.filename_formatter).issubset(self.ffields):
            raise Exception("Valid filename fields are: %s" % ','.join(sorted(self.ffields)))
        elif not self.parse_fields(self.export_name).issubset(self.efields):
            raise Exception("Valid export fields are: %s" % ','.join(sorted(self.efields)))
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

        self._validate_formatters()

        entries = []
        if data is None:
            for exporter in self.exporters:
                rendered = exporter(obj)
                if rendered is None: continue
                (data, new_info) = rendered
                info = dict(info, **new_info)
                entries.append((data, info))
        else:
            entries.append((data, info))

        for (data, info) in entries:
            self._add_content(obj, data, info, filename=filename)


    def _add_content(self, obj, data, info, filename=None):
        (unique_key, ext) = self._compute_filename(obj, info, filename=filename)
        self._files[(unique_key, ext)] = (data, info)


    def _compute_filename(self, obj, info, filename=None):
        if filename is None:
            hashfn = sha256()
            obj_str = 'None' if obj is None else self.object_formatter(obj)
            dimensions = self._dim_formatter(obj)
            dimensions = dimensions if dimensions else ''

            hashfn.update(obj_str.encode('utf-8'))
            format_values = {'timestamp': '{timestamp}',
                             'dimensions': dimensions,
                             'group':   getattr(obj, 'group', 'no-group'),
                             'label':   getattr(obj, 'label', 'no-label'),
                             'type':    obj.__class__.__name__,
                             'obj':     obj_str,
                             'SHA':     hashfn.hexdigest()}

            filename = self._format(self.filename_formatter,
                                    dict(info, **format_values))

        filename = self._normalize_name(filename)
        ext = info.get('file-ext', '')
        (unique_key, ext) = self._unique_name(filename, ext,
                                              self._files.keys(), force=True)
        return (unique_key, ext)



    def _zip_archive(self, export_name, files, root):
        archname = '.'.join(self._unique_name(export_name, 'zip', root))
        with zipfile.ZipFile(os.path.join(root, archname), 'w') as zipf:
            for (basename, ext), entry in files:
                filename = self._truncate_name(basename, ext)
                zipf.writestr(('%s/%s' % (export_name, filename)),Exporter.encode(entry))

    def _tar_archive(self, export_name, files, root):
        archname = '.'.join(self._unique_name(export_name, 'tar', root))
        with tarfile.TarFile(os.path.join(root, archname), 'w') as tarf:
            for (basename, ext), entry in files:
                filename = self._truncate_name(basename, ext)
                tarinfo = tarfile.TarInfo('%s/%s' % (export_name, filename))
                filedata = Exporter.encode(entry)
                tarinfo.size = len(filedata)
                tarf.addfile(tarinfo, BytesIO(filedata))

    def _single_file_archive(self, export_name, files, root):
        ((_, ext), entry) = files[0]
        (data, info) = entry
        unique_name = self._unique_name(export_name, ext, root)
        filename = self._truncate_name(self._normalize_name(*unique_name))
        fpath = os.path.join(root, filename)
        with open(fpath, 'w') as f:
            f.write(Exporter.encode(entry))

    def _directory_archive(self, export_name, files, root):
        output_dir = os.path.join(root, self._unique_name(export_name,'', root)[0])
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        for (basename, ext), entry in files:
            (data, info) = entry
            filename = self._truncate_name(basename, ext)
            fpath = os.path.join(output_dir, filename)
            with open(fpath, 'w') as f:
                f.write(Exporter.encode(entry))


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


    def _truncate_name(self, basename, ext='', tail=10, join='...', maxlen=None):
        maxlen = self.max_filename if maxlen is None else maxlen
        max_len = maxlen-len(ext)
        if len(basename) > max_len:
            start = basename[:max_len-(tail + len(join))]
            end = basename[-tail:]
            basename = start + join + end
        filename = '%s.%s' % (basename, ext) if ext else basename

        return filename


    def _normalize_name(self, basename):
        basename=re.sub('-+','-',basename)
        basename=re.sub('^[-,_]','',basename)
        return basename.replace(' ', '_')


    def export(self, timestamp=None, info={}):
        """
        Export the archive, directory or file.
        """
        tval = tuple(time.localtime()) if timestamp is None else timestamp
        tstamp = time.strftime(self.timestamp_format, tval)

        info = dict(info, timestamp=tstamp)
        export_name = self._format(self.export_name, info)
        files = [((self._format(base, info), ext), val)
                 for ((base, ext), val) in self._files.items()]
        root = os.path.abspath(self.root)
        # Make directory and populate if multiple files and not packed
        if len(self) > 1 and not self.pack:
            self._directory_archive(export_name, files, root)
        elif len(files) == 1:
            self._single_file_archive(export_name, files, root)
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

    def contents(self, maxlen=70):
        "Print the current (unexported) contents of the archive"
        lines = []
        if len(self._files) == 0:
            print("Empty %s" % self.__class__.__name__)
            return

        fnames = [self._truncate_name(maxlen=maxlen, *k) for k in self._files]
        max_len = max([len(f) for f in fnames])
        for name,v in zip(fnames, self._files.values()):
            mime_type = v[1].get('mime_type', 'no mime type')
            lines.append('%s : %s' % (name.ljust(max_len), mime_type))
        print('\n'.join(lines))
