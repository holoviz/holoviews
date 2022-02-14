import sys, warnings, operator
import json
import time
import types
import numbers
import inspect
import itertools
import string
import unicodedata
import datetime as dt

from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from packaging.version import Version as LooseVersion
from functools import partial
from threading import Thread, Event
from types import FunctionType

import numpy as np
import param


# Python3 compatibility
if sys.version_info.major >= 3:
    import builtins as builtins   # noqa (compatibility)

    if sys.version_info.minor > 3:
        from collections.abc import Iterable # noqa (compatibility)
    else:
        from collections import Iterable # noqa (compatibility)

    basestring = str
    unicode = str
    long = int
    cmp = lambda a, b: (a>b)-(a<b)
    generator_types = (zip, range, types.GeneratorType)
    RecursionError = RecursionError if sys.version_info.minor > 4 else RuntimeError # noqa
    _getargspec = inspect.getfullargspec
    get_keywords = operator.attrgetter('varkw')
else:
    import __builtin__ as builtins # noqa (compatibility)
    from collections import Iterable # noqa (compatibility)

    basestring = basestring
    unicode = unicode
    from itertools import izip
    generator_types = (izip, xrange, types.GeneratorType) # noqa
    RecursionError = RuntimeError
    _getargspec = inspect.getargspec
    get_keywords = operator.attrgetter('keywords')

numpy_version = LooseVersion(np.__version__)
param_version = LooseVersion(param.__version__)

datetime_types = (np.datetime64, dt.datetime, dt.date, dt.time)
timedelta_types = (np.timedelta64, dt.timedelta,)
arraylike_types = (np.ndarray,)
masked_types = ()

try:
    import pandas as pd
except ImportError:
    pd = None

if pd:
    pandas_version = LooseVersion(pd.__version__)
    try:
        if pandas_version >= LooseVersion('1.3.0'):
            from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtypeType
            from pandas.core.dtypes.generic import ABCSeries, ABCIndex as ABCIndexClass
        elif pandas_version >= LooseVersion('0.24.0'):
            from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtypeType
            from pandas.core.dtypes.generic import ABCSeries, ABCIndexClass
        elif pandas_version > LooseVersion('0.20.0'):
            from pandas.core.dtypes.dtypes import DatetimeTZDtypeType
            from pandas.core.dtypes.generic import ABCSeries, ABCIndexClass
        else:
            from pandas.types.dtypes import DatetimeTZDtypeType
            from pandas.types.dtypes.generic import ABCSeries, ABCIndexClass
        pandas_datetime_types = (pd.Timestamp, DatetimeTZDtypeType, pd.Period)
        pandas_timedelta_types = (pd.Timedelta,)
        datetime_types = datetime_types + pandas_datetime_types
        timedelta_types = timedelta_types + pandas_timedelta_types
        arraylike_types = arraylike_types + (ABCSeries, ABCIndexClass)
        if pandas_version > LooseVersion('0.23.0'):
            from pandas.core.dtypes.generic import ABCExtensionArray
            arraylike_types = arraylike_types + (ABCExtensionArray,)
        if pandas_version > LooseVersion('1.0'):
            from pandas.core.arrays.masked import BaseMaskedArray
            masked_types = (BaseMaskedArray,)
    except Exception as e:
        param.main.param.warning('pandas could not register all extension types '
                                 'imports failed with the following error: %s' % e)

try:
    import cftime
    cftime_types = (cftime.datetime,)
    datetime_types += cftime_types
except:
    cftime_types = ()
_STANDARD_CALENDARS = set(['standard', 'gregorian', 'proleptic_gregorian'])


class VersionError(Exception):
    "Raised when there is a library version mismatch."
    def __init__(self, msg, version=None, min_version=None, **kwargs):
        self.version = version
        self.min_version = min_version
        super(VersionError, self).__init__(msg, **kwargs)


class Config(param.ParameterizedFunction):
    """
    Set of boolean configuration values to change HoloViews' global
    behavior. Typically used to control warnings relating to
    deprecations or set global parameter such as style 'themes'.
    """

    future_deprecations = param.Boolean(default=False, doc="""
       Whether to warn about future deprecations""")

    image_rtol = param.Number(default=10e-4, doc="""
      The tolerance used to enforce regular sampling for regular,
      gridded data where regular sampling is expected. Expressed as the
      maximal allowable sampling difference between sample
      locations.""")

    no_padding = param.Boolean(default=False, doc="""
       Disable default padding (introduced in 1.13.0).""")

    warn_options_call = param.Boolean(default=True, doc="""
       Whether to warn when the deprecated __call__ options syntax is
       used (the opts method should now be used instead). It is
       recommended that users switch this on to update any uses of
       __call__ as it will be deprecated in future.""")

    default_cmap = param.String(default='kbc_r', doc="""
       Global default colormap. Prior to HoloViews 1.14.0, the default
       value was 'fire' which can be set for backwards compatibility.""")

    default_gridded_cmap = param.String(default='kbc_r', doc="""
       Global default colormap for gridded elements (i.e. Image, Raster
       and QuadMesh). Can be set to 'fire' to match raster defaults
       prior to HoloViews 1.14.0 while allowing the default_cmap to be
       the value of 'kbc_r' used in HoloViews >= 1.14.0""")

    default_heatmap_cmap = param.String(default='kbc_r', doc="""
       Global default colormap for HeatMap elements. Prior to HoloViews
       1.14.0, the default value was the 'RdYlBu_r' colormap.""")

    raise_deprecated_tilesource_exception = param.Boolean(default=False,
       doc=""" Whether deprecated tile sources should raise a
       deprecation exception instead of issuing warnings.""")

    def __call__(self, **params):
        self.param.set_param(**params)
        return self

config = Config()

class HashableJSON(json.JSONEncoder):
    """
    Extends JSONEncoder to generate a hashable string for as many types
    of object as possible including nested objects and objects that are
    not normally hashable. The purpose of this class is to generate
    unique strings that once hashed are suitable for use in memoization
    and other cases where deep equality must be tested without storing
    the entire object.

    By default JSONEncoder supports booleans, numbers, strings, lists,
    tuples and dictionaries. In order to support other types such as
    sets, datetime objects and mutable objects such as pandas Dataframes
    or numpy arrays, HashableJSON has to convert these types to
    datastructures that can normally be represented as JSON.

    Support for other object types may need to be introduced in
    future. By default, unrecognized object types are represented by
    their id.

    One limitation of this approach is that dictionaries with composite
    keys (e.g. tuples) are not supported due to the JSON spec.
    """
    string_hashable = (dt.datetime,)
    repr_hashable = ()

    def default(self, obj):
        if isinstance(obj, set):
            return hash(frozenset(obj))
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd and isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_csv(header=True).encode('utf-8')
        elif isinstance(obj, self.string_hashable):
            return str(obj)
        elif isinstance(obj, self.repr_hashable):
            return repr(obj)
        try:
            return hash(obj)
        except:
            return id(obj)


def merge_option_dicts(old_opts, new_opts):
    """
    Update the old_opts option dictionary with the options defined in
    new_opts. Instead of a shallow update as would be performed by calling
    old_opts.update(new_opts), this updates the dictionaries of all option
    types separately.

    Given two dictionaries
        old_opts = {'a': {'x': 'old', 'y': 'old'}}
    and
        new_opts = {'a': {'y': 'new', 'z': 'new'}, 'b': {'k': 'new'}}
    this returns a dictionary
        {'a': {'x': 'old', 'y': 'new', 'z': 'new'}, 'b': {'k': 'new'}}
    """
    merged = dict(old_opts)

    for option_type, options in new_opts.items():
        if option_type not in merged:
            merged[option_type] = {}

        merged[option_type].update(options)

    return merged


def merge_options_to_dict(options):
    """
    Given a collection of Option objects or partial option dictionaries,
    merge everything to a single dictionary.
    """
    merged_options = {}
    for obj in options:
        if isinstance(obj,dict):
            new_opts = obj
        else:
            new_opts = {obj.key: obj.kwargs}

        merged_options = merge_option_dicts(merged_options, new_opts)
    return merged_options


def deprecated_opts_signature(args, kwargs):
    """
    Utility to help with the deprecation of the old .opts method signature

    Returns whether opts.apply_groups should be used (as a bool) and the
    corresponding options.
    """
    from .options import Options
    groups = set(Options._option_groups)
    opts = {kw for kw in kwargs if kw != 'clone'}
    apply_groups = False
    options = None
    new_kwargs = {}
    if len(args) > 0 and isinstance(args[0], dict):
        apply_groups = True
        if (not set(args[0]).issubset(groups) and
            all(isinstance(v, dict) and (not set(v).issubset(groups) or not v)
                for v in args[0].values())):
            apply_groups = False
        elif set(args[0].keys()) <= groups:
            new_kwargs = args[0]
        else:
            options = args[0]
    elif opts and opts.issubset(set(groups)):
        apply_groups = True
    elif kwargs.get('options', None) is not None:
        apply_groups = True
    elif not args and not kwargs:
        apply_groups = True

    return apply_groups, options, new_kwargs


class periodic(Thread):
    """
    Run a callback count times with a given period without blocking.

    If count is None, will run till timeout (which may be forever if None).
    """

    def __init__(self, period, count, callback, timeout=None, block=False):

        if isinstance(count, int):
            if count < 0: raise ValueError('Count value must be positive')
        elif not type(count) is type(None):
            raise ValueError('Count value must be a positive integer or None')

        if block is False and count is None and timeout is None:
            raise ValueError('When using a non-blocking thread, please specify '
                             'either a count or a timeout')

        super(periodic, self).__init__()
        self.period = period
        self.callback = callback
        self.count = count
        self.counter = 0
        self.block = block
        self.timeout = timeout
        self._completed = Event()
        self._start_time = None

    @property
    def completed(self):
        return self._completed.is_set()

    def start(self):
        self._start_time = time.time()
        if self.block is False:
            super(periodic,self).start()
        else:
            self.run()

    def stop(self):
        self.timeout = None
        self._completed.set()

    def __repr__(self):
        return 'periodic(%s, %s, %s)' % (self.period,
                                         self.count,
                                         callable_name(self.callback))
    def __str__(self):
        return repr(self)

    def run(self):
        while not self.completed:
            if self.block:
                time.sleep(self.period)
            else:
                self._completed.wait(self.period)
            self.counter += 1
            try:
                self.callback(self.counter)
            except Exception:
                self.stop()

            if self.timeout is not None:
                dt = (time.time() - self._start_time)
                if dt > self.timeout:
                    self.stop()
            if self.counter == self.count:
                self.stop()



def deephash(obj):
    """
    Given an object, return a hash using HashableJSON. This hash is not
    architecture, Python version or platform independent.
    """
    try:
        return hash(json.dumps(obj, cls=HashableJSON, sort_keys=True))
    except:
        return None


def tree_attribute(identifier):
    """
    Predicate that returns True for custom attributes added to AttrTrees
    that are not methods, properties or internal attributes.

    These custom attributes start with a capitalized character when
    applicable (not applicable to underscore or certain unicode characters)
    """
    if identifier == '':
        return True
    if identifier[0].upper().isupper() is False and identifier[0] != '_':
        return True
    else:
        return identifier[0].isupper()

def argspec(callable_obj):
    """
    Returns an ArgSpec object for functions, staticmethods, instance
    methods, classmethods and partials.

    Note that the args list for instance and class methods are those as
    seen by the user. In other words, the first argument which is
    conventionally called 'self' or 'cls' is omitted in these cases.
    """
    if (isinstance(callable_obj, type)
        and issubclass(callable_obj, param.ParameterizedFunction)):
        # Parameterized function.__call__ considered function in py3 but not py2
        spec = _getargspec(callable_obj.__call__)
        args = spec.args[1:]
    elif inspect.isfunction(callable_obj):    # functions and staticmethods
        spec = _getargspec(callable_obj)
        args = spec.args
    elif isinstance(callable_obj, partial): # partials
        arglen = len(callable_obj.args)
        spec =  _getargspec(callable_obj.func)
        args = [arg for arg in spec.args[arglen:] if arg not in callable_obj.keywords]
    elif inspect.ismethod(callable_obj):    # instance and class methods
        spec = _getargspec(callable_obj)
        args = spec.args[1:]
    else:                                   # callable objects
        return argspec(callable_obj.__call__)

    return inspect.ArgSpec(args=args,
                           varargs=spec.varargs,
                           keywords=get_keywords(spec),
                           defaults=spec.defaults)



def validate_dynamic_argspec(callback, kdims, streams):
    """
    Utility used by DynamicMap to ensure the supplied callback has an
    appropriate signature.

    If validation succeeds, returns a list of strings to be zipped with
    the positional arguments, i.e. kdim values. The zipped values can then
    be merged with the stream values to pass everything to the Callable
    as keywords.

    If the callbacks use *args, None is returned to indicate that kdim
    values must be passed to the Callable by position. In this
    situation, Callable passes *args and **kwargs directly to the
    callback.

    If the callback doesn't use **kwargs, the accepted keywords are
    validated against the stream parameter names.
    """
    argspec = callback.argspec
    name = callback.name
    kdims = [kdim.name for kdim in kdims]
    stream_params = stream_parameters(streams)
    defaults = argspec.defaults if argspec.defaults else []
    all_posargs = argspec.args[:-len(defaults)] if defaults else argspec.args
    # Filter out any posargs for streams
    posargs = [arg for arg in all_posargs if arg not in stream_params]
    kwargs = argspec.args[-len(defaults):]

    if argspec.keywords is None:
        unassigned_streams = set(stream_params) - set(argspec.args)
        if unassigned_streams:
            unassigned = ','.join(unassigned_streams)
            raise KeyError('Callable {name!r} missing keywords to '
                           'accept stream parameters: {unassigned}'.format(name=name,
                                                                    unassigned=unassigned))


    if len(posargs) > len(kdims) + len(stream_params):
        raise KeyError('Callable {name!r} accepts more positional arguments than '
                       'there are kdims and stream parameters'.format(name=name))
    if kdims == []:                  # Can be no posargs, stream kwargs already validated
        return []
    if set(kdims) == set(posargs):   # Posargs match exactly, can all be passed as kwargs
        return kdims
    elif len(posargs) == len(kdims): # Posargs match kdims length, supplying names
        if argspec.args[:len(kdims)] != posargs:
            raise KeyError('Unmatched positional kdim arguments only allowed at '
                           'the start of the signature of {name!r}'.format(name=name))

        return posargs
    elif argspec.varargs:            # Posargs missing, passed to Callable directly
        return None
    elif set(posargs) - set(kdims):
        raise KeyError('Callable {name!r} accepts more positional arguments {posargs} '
                       'than there are key dimensions {kdims}'.format(name=name,
                                                                      posargs=posargs,
                                                                      kdims=kdims))
    elif set(kdims).issubset(set(kwargs)): # Key dims can be supplied by keyword
        return kdims
    elif set(kdims).issubset(set(posargs+kwargs)):
        return kdims
    elif argspec.keywords:
        return kdims
    else:
        raise KeyError('Callback {name!r} signature over {names} does not accommodate '
                       'required kdims {kdims}'.format(name=name,
                                                       names=list(set(posargs+kwargs)),
                                                       kdims=kdims))


def callable_name(callable_obj):
    """
    Attempt to return a meaningful name identifying a callable or generator
    """
    try:
        if (isinstance(callable_obj, type)
            and issubclass(callable_obj, param.ParameterizedFunction)):
            return callable_obj.__name__
        elif (isinstance(callable_obj, param.Parameterized)
              and 'operation' in callable_obj.param):
            return callable_obj.operation.__name__
        elif isinstance(callable_obj, partial):
            return str(callable_obj)
        elif inspect.isfunction(callable_obj):  # functions and staticmethods
            return callable_obj.__name__
        elif inspect.ismethod(callable_obj):    # instance and class methods
            meth = callable_obj
            if sys.version_info < (3,0):
                owner =  meth.im_class if meth.im_self is None else meth.im_self
                if meth.__name__ == '__call__':
                    return type(owner).__name__
                return '.'.join([owner.__name__, meth.__name__])
            else:
                return meth.__func__.__qualname__.replace('.__call__', '')
        elif isinstance(callable_obj, types.GeneratorType):
            return callable_obj.__name__
        else:
            return type(callable_obj).__name__
    except Exception:
        return str(callable_obj)


def process_ellipses(obj, key, vdim_selection=False):
    """
    Helper function to pad a __getitem__ key with the right number of
    empty slices (i.e. :) when the key contains an Ellipsis (...).

    If the vdim_selection flag is true, check if the end of the key
    contains strings or Dimension objects in obj. If so, extra padding
    will not be applied for the value dimensions (i.e. the resulting key
    will be exactly one longer than the number of kdims). Note: this
    flag should not be used for composite types.
    """
    if getattr(getattr(key, 'dtype', None), 'kind', None) == 'b':
        return key
    wrapped_key = wrap_tuple(key)
    ellipse_count = sum(1 for k in wrapped_key if k is Ellipsis)
    if ellipse_count == 0:
        return key
    elif ellipse_count != 1:
        raise Exception("Only one ellipsis allowed at a time.")
    dim_count = len(obj.dimensions())
    index = wrapped_key.index(Ellipsis)
    head = wrapped_key[:index]
    tail = wrapped_key[index+1:]

    padlen = dim_count - (len(head) + len(tail))
    if vdim_selection:
        # If the end of the key (i.e. the tail) is in vdims, pad to len(kdims)+1
        if wrapped_key[-1] in obj.vdims:
            padlen = (len(obj.kdims) +1 ) - len(head+tail)
    return head + ((slice(None),) * padlen) + tail


def bytes_to_unicode(value):
    """
    Safely casts bytestring to unicode
    """
    if isinstance(value, bytes):
        return unicode(value.decode('utf-8'))
    return value


def get_method_owner(method):
    """
    Gets the instance that owns the supplied method
    """
    if isinstance(method, partial):
        method = method.func
    return method.__self__ if sys.version_info.major >= 3 else method.im_self


def capitalize_unicode_name(s):
    """
    Turns a string such as 'capital delta' into the shortened,
    capitalized version, in this case simply 'Delta'. Used as a
    transform in sanitize_identifier.
    """
    index = s.find('capital')
    if index == -1: return s
    tail = s[index:].replace('capital', '').strip()
    tail = tail[0].upper() + tail[1:]
    return s[:index] + tail


class sanitize_identifier_fn(param.ParameterizedFunction):
    """
    Sanitizes group/label values for use in AttrTree attribute
    access. Depending on the version parameter, either sanitization
    appropriate for Python 2 (no unicode gn identifiers allowed) or
    Python 3 (some unicode allowed) is used.

    Note that if you are using Python 3, you can switch to version 2
    for compatibility but you cannot enable relaxed sanitization if
    you are using Python 2.

    Special characters are sanitized using their (lowercase) unicode
    name using the unicodedata module. For instance:

    >>> unicodedata.name(u'$').lower()
    'dollar sign'

    As these names are often very long, this parameterized function
    allows filtered, substitutions and transforms to help shorten these
    names appropriately.
    """

    version = param.ObjectSelector(sys.version_info.major, objects=[2,3], doc="""
        The sanitization version. If set to 2, more aggressive
        sanitization appropriate for Python 2 is applied. Otherwise,
        if set to 3, more relaxed, Python 3 sanitization is used.""")

    capitalize = param.Boolean(default=True, doc="""
       Whether the first letter should be converted to
       uppercase. Note, this will only be applied to ASCII characters
       in order to make sure paths aren't confused with method
       names.""")

    eliminations = param.List(['extended', 'accent', 'small', 'letter', 'sign', 'digit',
                               'latin', 'greek', 'arabic-indic', 'with', 'dollar'], doc="""
       Lowercase strings to be eliminated from the unicode names in
       order to shorten the sanitized name ( lowercase). Redundant
       strings should be removed but too much elimination could cause
       two unique strings to map to the same sanitized output.""")

    substitutions = param.Dict(default={'circumflex':'power',
                                        'asterisk':'times',
                                        'solidus':'over'}, doc="""
       Lowercase substitutions of substrings in unicode names. For
       instance the ^ character has the name 'circumflex accent' even
       though it is more typically used for exponentiation. Note that
       substitutions occur after filtering and that there should be no
       ordering dependence between substitutions.""")

    transforms = param.List(default=[capitalize_unicode_name], doc="""
       List of string transformation functions to apply after
       filtering and substitution in order to further compress the
       unicode name. For instance, the default capitalize_unicode_name
       function will turn the string "capital delta" into "Delta".""")

    disallowed = param.List(default=['trait_names', '_ipython_display_',
                                     '_getAttributeNames'], doc="""
       An explicit list of name that should not be allowed as
       attribute names on Tree objects.

       By default, prevents IPython from creating an entry called
       Trait_names due to an inconvenient getattr check (during
       tab-completion).""")

    disable_leading_underscore = param.Boolean(default=False, doc="""
       Whether leading underscores should be allowed to be sanitized
       with the leading prefix.""")

    aliases = param.Dict(default={}, doc="""
       A dictionary of aliases mapping long strings to their short,
       sanitized equivalents""")

    prefix = 'A_'

    _lookup_table = param.Dict(default={}, doc="""
       Cache of previously computed sanitizations""")


    @param.parameterized.bothmethod
    def add_aliases(self_or_cls, **kwargs):
        """
        Conveniently add new aliases as keyword arguments. For instance
        you can add a new alias with add_aliases(short='Longer string')
        """
        self_or_cls.aliases.update({v:k for k,v in kwargs.items()})

    @param.parameterized.bothmethod
    def remove_aliases(self_or_cls, aliases):
        """
        Remove a list of aliases.
        """
        for k,v in self_or_cls.aliases.items():
            if v in aliases:
                self_or_cls.aliases.pop(k)

    @param.parameterized.bothmethod
    def allowable(self_or_cls, name, disable_leading_underscore=None):
       disabled_reprs = ['javascript', 'jpeg', 'json', 'latex',
                         'latex', 'pdf', 'png', 'svg', 'markdown']
       disabled_ = (self_or_cls.disable_leading_underscore
                    if disable_leading_underscore is None
                    else disable_leading_underscore)
       if disabled_ and name.startswith('_'):
          return False
       isrepr = any(('_repr_%s_' % el) == name for el in disabled_reprs)
       return (name not in self_or_cls.disallowed) and not isrepr

    @param.parameterized.bothmethod
    def prefixed(self, identifier, version):
        """
        Whether or not the identifier will be prefixed.
        Strings that require the prefix are generally not recommended.
        """
        invalid_starting = ['Mn', 'Mc', 'Nd', 'Pc']
        if identifier.startswith('_'):  return True
        return((identifier[0] in string.digits) if version==2
               else (unicodedata.category(identifier[0]) in invalid_starting))

    @param.parameterized.bothmethod
    def remove_diacritics(self_or_cls, identifier):
        """
        Remove diacritics and accents from the input leaving other
        unicode characters alone."""
        chars = ''
        for c in identifier:
            replacement = unicodedata.normalize('NFKD', c).encode('ASCII', 'ignore')
            if replacement != '':
                chars += bytes_to_unicode(replacement)
            else:
                chars += c
        return chars

    @param.parameterized.bothmethod
    def shortened_character_name(self_or_cls, c, eliminations=[], substitutions={}, transforms=[]):
        """
        Given a unicode character c, return the shortened unicode name
        (as a list of tokens) by applying the eliminations,
        substitutions and transforms.
        """
        name = unicodedata.name(c).lower()
        # Filtering
        for elim in eliminations:
            name = name.replace(elim, '')
        # Substitution
        for i,o in substitutions.items():
            name = name.replace(i, o)
        for transform in transforms:
            name = transform(name)
        return ' '.join(name.strip().split()).replace(' ','_').replace('-','_')


    def __call__(self, name, escape=True, version=None):
        if name in [None, '']:
           return name
        elif name in self.aliases:
            return self.aliases[name]
        elif name in self._lookup_table:
           return self._lookup_table[name]
        name = bytes_to_unicode(name)
        version = self.version if version is None else version
        if not self.allowable(name):
            raise AttributeError("String %r is in the disallowed list of attribute names: %r" % (name, self.disallowed))

        if version == 2:
            name = self.remove_diacritics(name)
        if self.capitalize and name and name[0] in string.ascii_lowercase:
            name = name[0].upper()+name[1:]

        sanitized = (self.sanitize_py2(name) if version==2 else self.sanitize_py3(name))
        if self.prefixed(name, version):
           sanitized = self.prefix + sanitized
        self._lookup_table[name] = sanitized
        return sanitized


    def _process_underscores(self, tokens):
        "Strip underscores to make sure the number is correct after join"
        groups = [[str(''.join(el))] if b else list(el)
                  for (b,el) in itertools.groupby(tokens, lambda k: k=='_')]
        flattened = [el for group in groups for el in group]
        processed = []
        for token in flattened:
            if token == '_':  continue
            if token.startswith('_'):
                token = str(token[1:])
            if token.endswith('_'):
                token = str(token[:-1])
            processed.append(token)
        return processed

    def sanitize_py2(self, name):
        # This fix works but masks an issue in self.sanitize (py2)
        prefix = '_' if name.startswith('_') else ''
        valid_chars = string.ascii_letters+string.digits+'_'
        return prefix + str('_'.join(self.sanitize(name, lambda c: c in valid_chars)))


    def sanitize_py3(self, name):
        if not name.isidentifier():
            return '_'.join(self.sanitize(name, lambda c: ('_'+c).isidentifier()))
        else:
            return name

    def sanitize(self, name, valid_fn):
        "Accumulate blocks of hex and separate blocks by underscores"
        invalid = {'\a':'a','\b':'b', '\v':'v','\f':'f','\r':'r'}
        for cc in filter(lambda el: el in name, invalid.keys()):
            raise Exception(r"Please use a raw string or escape control code '\%s'"
                            % invalid[cc])
        sanitized, chars = [], ''
        for split in name.split():
            for c in split:
                if valid_fn(c): chars += str(c) if c=='_' else c
                else:
                    short = self.shortened_character_name(c, self.eliminations,
                                                         self.substitutions,
                                                         self.transforms)
                    sanitized.extend([chars] if chars else [])
                    if short != '':
                       sanitized.append(short)
                    chars = ''
            if chars:
                sanitized.extend([chars])
                chars=''
        return self._process_underscores(sanitized + ([chars] if chars else []))

sanitize_identifier = sanitize_identifier_fn.instance()


group_sanitizer = sanitize_identifier_fn.instance()
label_sanitizer = sanitize_identifier_fn.instance()
dimension_sanitizer = sanitize_identifier_fn.instance(capitalize=False)

def isscalar(val):
    """
    Value is scalar or None
    """
    return val is None or np.isscalar(val) or isinstance(val, datetime_types)


def isnumeric(val):
    if isinstance(val, (basestring, bool, np.bool_)):
        return False
    try:
        float(val)
        return True
    except:
        return False


def asarray(arraylike, strict=True):
    """
    Converts arraylike objects to NumPy ndarray types. Errors if
    object is not arraylike and strict option is enabled.
    """
    if isinstance(arraylike, np.ndarray):
        return arraylike
    elif isinstance(arraylike, list):
        return np.asarray(arraylike, dtype=object)
    elif not isinstance(arraylike, np.ndarray) and isinstance(arraylike, arraylike_types):
        return arraylike.values
    elif hasattr(arraylike, '__array__'):
        return np.asarray(arraylike)
    elif strict:
        raise ValueError('Could not convert %s type to array' % type(arraylike))
    return arraylike


nat_as_integer = np.datetime64('NAT').view('i8')

def isnat(val):
    """
    Checks if the value is a NaT. Should only be called on datetimelike objects.
    """
    if (isinstance(val, (np.datetime64, np.timedelta64)) or
        (isinstance(val, np.ndarray) and val.dtype.kind == 'M')):
        if numpy_version >= LooseVersion('1.13'):
            return np.isnat(val)
        else:
            return val.view('i8') == nat_as_integer
    elif pd and val is pd.NaT:
        return True
    elif pd and isinstance(val, pandas_datetime_types+pandas_timedelta_types):
        return pd.isna(val)
    else:
        return False


def isfinite(val):
    """
    Helper function to determine if scalar or array value is finite extending
    np.isfinite with support for None, string, datetime types.
    """
    is_dask = is_dask_array(val)
    if not np.isscalar(val) and not is_dask:
        if isinstance(val, np.ma.core.MaskedArray):
            return ~val.mask & isfinite(val.data)
        elif isinstance(val, masked_types):
            return ~val.isna() & isfinite(val._data)
        val = asarray(val, strict=False)

    if val is None:
        return False
    elif is_dask:
        import dask.array as da
        return da.isfinite(val)
    elif isinstance(val, np.ndarray):
        if val.dtype.kind == 'M':
            return ~isnat(val)
        elif val.dtype.kind == 'O':
            return np.array([isfinite(v) for v in val], dtype=bool)
        elif val.dtype.kind in 'US':
            return ~pd.isna(val) if pd else np.ones_like(val, dtype=bool)
        finite = np.isfinite(val)
        if pd and pandas_version >= LooseVersion('1.0.0'):
            finite &= ~pd.isna(val)
        return finite
    elif isinstance(val, datetime_types+timedelta_types):
        return not isnat(val)
    elif isinstance(val, (basestring, bytes)):
        return True
    finite = np.isfinite(val)
    if pd and pandas_version >= LooseVersion('1.0.0'):
        if finite is pd.NA:
            return False
        return finite & (~pd.isna(val))
    return finite


def isdatetime(value):
    """
    Whether the array or scalar is recognized datetime type.
    """
    if isinstance(value, np.ndarray):
        return (value.dtype.kind == "M" or
                (value.dtype.kind == "O" and len(value) and
                 isinstance(value[0], datetime_types)))
    else:
        return isinstance(value, datetime_types)


def find_minmax(lims, olims):
    """
    Takes (a1, a2) and (b1, b2) as input and returns
    (np.nanmin(a1, b1), np.nanmax(a2, b2)). Used to calculate
    min and max values of a number of items.
    """
    try:
        limzip = zip(list(lims), list(olims), [np.nanmin, np.nanmax])
        limits = tuple([float(fn([l, ol])) for l, ol, fn in limzip])
    except:
        limits = (np.NaN, np.NaN)
    return limits


def find_range(values, soft_range=[]):
    """
    Safely finds either the numerical min and max of
    a set of values, falling back to the first and
    the last value in the sorted list of values.
    """
    try:
        values = np.array(values)
        values = np.squeeze(values) if len(values.shape) > 1 else values
        if len(soft_range):
            values = np.concatenate([values, soft_range])
        if values.dtype.kind == 'M':
            return values.min(), values.max()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
            return np.nanmin(values), np.nanmax(values)
    except:
        try:
            values = sorted(values)
            return (values[0], values[-1])
        except:
            return (None, None)


def max_range(ranges, combined=True):
    """
    Computes the maximal lower and upper bounds from a list bounds.

    Args:
       ranges (list of tuples): A list of range tuples
       combined (boolean, optional): Whether to combine bounds
          Whether range should be computed on lower and upper bound
          independently or both at once

    Returns:
       The maximum range as a single tuple
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
            values = [tuple(np.NaN if v is None else v for v in r) for r in ranges]
            if pd and any(isinstance(v, datetime_types) and not isinstance(v, cftime_types+(dt.time,))
                          for r in values for v in r):
                converted = []
                for l, h in values:
                    if isinstance(l, datetime_types) and isinstance(h, datetime_types):
                        l, h = (pd.Timestamp(l).to_datetime64(),
                                pd.Timestamp(h).to_datetime64())
                    converted.append((l, h))
                values = converted

            arr = np.array(values)
            if not len(arr):
                return np.NaN, np.NaN
            elif arr.dtype.kind in 'OSU':
                arr = list(python2sort([
                    v for r in values for v in r
                    if not is_nan(v) and v is not None]))
                return arr[0], arr[-1]
            elif arr.dtype.kind in 'M':
                drange = ((arr.min(), arr.max()) if combined else
                          (arr[:, 0].min(), arr[:, 1].max()))
                return drange

            if combined:
                return (np.nanmin(arr), np.nanmax(arr))
            else:
                return (np.nanmin(arr[:, 0]), np.nanmax(arr[:, 1]))
    except:
        return (np.NaN, np.NaN)


def range_pad(lower, upper, padding=None, log=False):
    """
    Pads the range by a fraction of the interval
    """
    if padding is not None and not isinstance(padding, tuple):
        padding = (padding, padding)
    if is_number(lower) and is_number(upper) and padding is not None:
        if not isinstance(lower, datetime_types) and log and lower > 0 and upper > 0:
            log_min = np.log(lower) / np.log(10)
            log_max = np.log(upper) / np.log(10)
            lspan = (log_max-log_min)*(1+padding[0]*2)
            uspan = (log_max-log_min)*(1+padding[1]*2)
            center = (log_min+log_max) / 2.0
            start, end = np.power(10, center-lspan/2.), np.power(10, center+uspan/2.)
        else:
            if isinstance(lower, datetime_types) and not isinstance(lower, cftime_types):
                # Ensure timedelta can be safely divided
                lower, upper = np.datetime64(lower), np.datetime64(upper)
                span = (upper-lower).astype('>m8[ns]')
            else:
                span = (upper-lower)
            lpad = span*(padding[0])
            upad = span*(padding[1])
            start, end = lower-lpad, upper+upad
    else:
        start, end = lower, upper

    return start, end


def dimension_range(lower, upper, hard_range, soft_range, padding=None, log=False):
    """
    Computes the range along a dimension by combining the data range
    with the Dimension soft_range and range.
    """
    plower, pupper = range_pad(lower, upper, padding, log)
    if isfinite(soft_range[0]) and soft_range[0] <= lower:
        lower = soft_range[0]
    else:
        lower = max_range([(plower, None), (soft_range[0], None)])[0]
    if isfinite(soft_range[1]) and soft_range[1] >= upper:
        upper = soft_range[1]
    else:
        upper = max_range([(None, pupper), (None, soft_range[1])])[1]
    dmin, dmax = hard_range
    lower = lower if dmin is None or not isfinite(dmin) else dmin
    upper = upper if dmax is None or not isfinite(dmax) else dmax
    return lower, upper


def max_extents(extents, zrange=False):
    """
    Computes the maximal extent in 2D and 3D space from
    list of 4-tuples or 6-tuples. If zrange is enabled
    all extents are converted to 6-tuples to compute
    x-, y- and z-limits.
    """
    if zrange:
        num = 6
        inds = [(0, 3), (1, 4), (2, 5)]
        extents = [e if len(e) == 6 else (e[0], e[1], None,
                                          e[2], e[3], None)
                   for e in extents]
    else:
        num = 4
        inds = [(0, 2), (1, 3)]
    arr = list(zip(*extents)) if extents else []
    extents = [np.NaN] * num
    if len(arr) == 0:
        return extents
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        for lidx, uidx in inds:
            lower = [v for v in arr[lidx] if v is not None and not is_nan(v)]
            upper = [v for v in arr[uidx] if v is not None and not is_nan(v)]
            if lower and isinstance(lower[0], datetime_types):
                extents[lidx] = np.min(lower)
            elif any(isinstance(l, basestring) for l in lower):
                extents[lidx] = np.sort(lower)[0]
            elif lower:
                extents[lidx] = np.nanmin(lower)
            if upper and isinstance(upper[0], datetime_types):
                extents[uidx] = np.max(upper)
            elif any(isinstance(u, basestring) for u in upper):
                extents[uidx] = np.sort(upper)[-1]
            elif upper:
                extents[uidx] = np.nanmax(upper)
    return tuple(extents)


def int_to_alpha(n, upper=True):
    "Generates alphanumeric labels of form A-Z, AA-ZZ etc."
    casenum = 65 if upper else 97
    label = ''
    count= 0
    if n == 0: return str(chr(n + casenum))
    while n >= 0:
        mod, div = n % 26, n
        for _ in range(count):
            div //= 26
        div %= 26
        if count == 0:
            val = mod
        else:
            val = div
        label += str(chr(val + casenum))
        count += 1
        n -= 26**count
    return label[::-1]


def int_to_roman(input):
   if type(input) != type(1):
      raise TypeError("expected integer, got %s" % type(input))
   if not 0 < input < 4000:
      raise ValueError("Argument must be between 1 and 3999")
   ints = (1000, 900,  500, 400, 100,  90, 50,  40, 10,  9,   5,  4,   1)
   nums = ('M',  'CM', 'D', 'CD','C', 'XC','L','XL','X','IX','V','IV','I')
   result = ""
   for i in range(len(ints)):
      count = int(input / ints[i])
      result += nums[i] * count
      input -= ints[i] * count
   return result


def unique_iterator(seq):
    """
    Returns an iterator containing all non-duplicate elements
    in the input sequence.
    """
    seen = set()
    for item in seq:
        if item not in seen:
            seen.add(item)
            yield item


def lzip(*args):
    """
    zip function that returns a list.
    """
    return list(zip(*args))


def unique_zip(*args):
    """
    Returns a unique list of zipped values.
    """
    return list(unique_iterator(zip(*args)))


def unique_array(arr):
    """
    Returns an array of unique values in the input order.

    Args:
       arr (np.ndarray or list): The array to compute unique values on

    Returns:
       A new array of unique values
    """
    if not len(arr):
        return np.asarray(arr)
    elif pd:
        if isinstance(arr, np.ndarray) and arr.dtype.kind not in 'MO':
            # Avoid expensive unpacking if not potentially datetime
            return pd.unique(arr)

        values = []
        for v in arr:
            if (isinstance(v, datetime_types) and
                not isinstance(v, cftime_types)):
                v = pd.Timestamp(v).to_datetime64()
            values.append(v)
        return pd.unique(values)
    else:
        arr = np.asarray(arr)
        _, uniq_inds = np.unique(arr, return_index=True)
        return arr[np.sort(uniq_inds)]


def match_spec(element, specification):
    """
    Matches the group.label specification of the supplied
    element against the supplied specification dictionary
    returning the value of the best match.
    """
    match_tuple = ()
    match = specification.get((), {})
    for spec in [type(element).__name__,
                 group_sanitizer(element.group, escape=False),
                 label_sanitizer(element.label, escape=False)]:
        match_tuple += (spec,)
        if match_tuple in specification:
            match = specification[match_tuple]
    return match


def python2sort(x,key=None):
    if len(x) == 0: return x
    it = iter(x)
    groups = [[next(it)]]
    for item in it:
        for group in groups:
            try:
                item_precedence = item if key is None else key(item)
                group_precedence = group[0] if key is None else key(group[0])
                item_precedence < group_precedence  # exception if not comparable
                group.append(item)
                break
            except TypeError:
                continue
        else:  # did not break, make new group
            groups.append([item])
    return itertools.chain.from_iterable(sorted(group, key=key) for group in groups)


def merge_dimensions(dimensions_list):
    """
    Merges lists of fully or partially overlapping dimensions by
    merging their values.

    >>> from holoviews import Dimension
    >>> dim_list = [[Dimension('A', values=[1, 2, 3]), Dimension('B')],
    ...             [Dimension('A', values=[2, 3, 4])]]
    >>> dimensions = merge_dimensions(dim_list)
    >>> dimensions
    [Dimension('A'), Dimension('B')]
    >>> dimensions[0].values
    [1, 2, 3, 4]
    """
    dvalues = defaultdict(list)
    dimensions = []
    for dims in dimensions_list:
        for d in dims:
            dvalues[d.name].append(d.values)
            if d not in dimensions:
                dimensions.append(d)
    dvalues = {k: list(unique_iterator(itertools.chain(*vals)))
               for k, vals in dvalues.items()}
    return [d.clone(values=dvalues.get(d.name, [])) for d in dimensions]


def dimension_sort(odict, kdims, vdims, key_index):
    """
    Sorts data by key using usual Python tuple sorting semantics
    or sorts in categorical order for any categorical Dimensions.
    """
    sortkws = {}
    ndims = len(kdims)
    dimensions = kdims+vdims
    indexes = [(dimensions[i], int(i not in range(ndims)),
                    i if i in range(ndims) else i-ndims)
                for i in key_index]
    cached_values = {d.name: [None]+list(d.values) for d in dimensions}

    if len(set(key_index)) != len(key_index):
        raise ValueError("Cannot sort on duplicated dimensions")
    else:
       sortkws['key'] = lambda x: tuple(cached_values[dim.name].index(x[t][d])
                                        if dim.values else x[t][d]
                                        for i, (dim, t, d) in enumerate(indexes))
    if sys.version_info.major == 3:
        return python2sort(odict.items(), **sortkws)
    else:
        return sorted(odict.items(), **sortkws)


# Copied from param should make param version public
def is_number(obj):
    if isinstance(obj, numbers.Number): return True
    elif isinstance(obj, (np.str_, np.unicode_)): return False
    # The extra check is for classes that behave like numbers, such as those
    # found in numpy, gmpy, etc.
    elif (hasattr(obj, '__int__') and hasattr(obj, '__add__')): return True
    # This is for older versions of gmpy
    elif hasattr(obj, 'qdiv'): return True
    else: return False


def is_int(obj, int_like=False):
    """
    Checks for int types including the native Python type and NumPy-like objects

    Args:
        obj: Object to check for integer type
        int_like (boolean): Check for float types with integer value

    Returns:
        Boolean indicating whether the supplied value is of integer type.
    """
    real_int = isinstance(obj, int) or getattr(getattr(obj, 'dtype', None), 'kind', 'o') in 'ui'
    if real_int or (int_like and hasattr(obj, 'is_integer') and obj.is_integer()):
        return True
    return False


class ProgressIndicator(param.Parameterized):
    """
    Baseclass for any ProgressIndicator that indicates progress
    as a completion percentage.
    """

    percent_range = param.NumericTuple(default=(0.0, 100.0), doc="""
        The total percentage spanned by the progress bar when called
        with a value between 0% and 100%. This allows an overall
        completion in percent to be broken down into smaller sub-tasks
        that individually complete to 100 percent.""")

    label = param.String(default='Progress', allow_None=True, doc="""
        The label of the current progress bar.""")

    def __call__(self, completion):
        raise NotImplementedError


def sort_topologically(graph):
    """
    Stackless topological sorting.

    graph = {
        3: [1],
        5: [3],
        4: [2],
        6: [4],
    }

    sort_topologically(graph)
    [[1, 2], [3, 4], [5, 6]]
    """
    levels_by_name = {}
    names_by_level = defaultdict(list)

    def add_level_to_name(name, level):
        levels_by_name[name] = level
        names_by_level[level].append(name)


    def walk_depth_first(name):
        stack = [name]
        while(stack):
            name = stack.pop()
            if name in levels_by_name:
                continue

            if name not in graph or not graph[name]:
                level = 0
                add_level_to_name(name, level)
                continue

            children = graph[name]

            children_not_calculated = [child for child in children if child not in levels_by_name]
            if children_not_calculated:
                stack.append(name)
                stack.extend(children_not_calculated)
                continue

            level = 1 + max(levels_by_name[lname] for lname in children)
            add_level_to_name(name, level)

    for name in graph:
        walk_depth_first(name)

    return list(itertools.takewhile(lambda x: x is not None,
                                    (names_by_level.get(i, None)
                                     for i in itertools.count())))


def is_cyclic(graph):
    """
    Return True if the directed graph g has a cycle. The directed graph
    should be represented as a dictionary mapping of edges for each node.
    """
    path = set()

    def visit(vertex):
        path.add(vertex)
        for neighbour in graph.get(vertex, ()):
            if neighbour in path or visit(neighbour):
                return True
        path.remove(vertex)
        return False

    return any(visit(v) for v in graph)


def one_to_one(graph, nodes):
    """
    Return True if graph contains only one to one mappings. The
    directed graph should be represented as a dictionary mapping of
    edges for each node. Nodes should be passed a simple list.
    """
    edges = itertools.chain.from_iterable(graph.values())
    return len(graph) == len(nodes) and len(set(edges)) == len(nodes)


def get_overlay_spec(o, k, v):
    """
    Gets the type.group.label + key spec from an Element in an Overlay.
    """
    k = wrap_tuple(k)
    return ((type(v).__name__, v.group, v.label) + k if len(o.kdims) else
            (type(v).__name__,) + k)


def layer_sort(hmap):
   """
   Find a global ordering for layers in a HoloMap of CompositeOverlay
   types.
   """
   orderings = {}
   for o in hmap:
      okeys = [get_overlay_spec(o, k, v) for k, v in o.data.items()]
      if len(okeys) == 1 and not okeys[0] in orderings:
         orderings[okeys[0]] = []
      else:
         orderings.update({k: [] if k == v else [v] for k, v in zip(okeys[1:], okeys)})
   return [i for g in sort_topologically(orderings) for i in sorted(g)]


def layer_groups(ordering, length=2):
   """
   Splits a global ordering of Layers into groups based on a slice of
   the spec.  The grouping behavior can be modified by changing the
   length of spec the entries are grouped by.
   """
   group_orderings = defaultdict(list)
   for el in ordering:
      group_orderings[el[:length]].append(el)
   return group_orderings


def group_select(selects, length=None, depth=None):
    """
    Given a list of key tuples to select, groups them into sensible
    chunks to avoid duplicating indexing operations.
    """
    if length == None and depth == None:
        length = depth = len(selects[0])
    getter = operator.itemgetter(depth-length)
    if length > 1:
        selects = sorted(selects, key=getter)
        grouped_selects = defaultdict(dict)
        for k, v in itertools.groupby(selects, getter):
            grouped_selects[k] = group_select(list(v), length-1, depth)
        return grouped_selects
    else:
        return list(selects)


def iterative_select(obj, dimensions, selects, depth=None):
    """
    Takes the output of group_select selecting subgroups iteratively,
    avoiding duplicating select operations.
    """
    ndims = len(dimensions)
    depth = depth if depth is not None else ndims
    items = []
    if isinstance(selects, dict):
        for k, v in selects.items():
            items += iterative_select(obj.select(**{dimensions[ndims-depth]: k}),
                                      dimensions, v, depth-1)
    else:
        for s in selects:
            items.append((s, obj.select(**{dimensions[-1]: s[-1]})))
    return items


def get_spec(obj):
   """
   Gets the spec from any labeled data object.
   """
   return (obj.__class__.__name__,
           obj.group, obj.label)


def is_dataframe(data):
    """
    Checks whether the supplied data is of DataFrame type.
    """
    dd = None
    if 'dask.dataframe' in sys.modules and 'pandas' in sys.modules:
        import dask.dataframe as dd
    return((pd is not None and isinstance(data, pd.DataFrame)) or
          (dd is not None and isinstance(data, dd.DataFrame)))


def is_series(data):
    """
    Checks whether the supplied data is of Series type.
    """
    dd = None
    if 'dask.dataframe' in sys.modules:
        import dask.dataframe as dd
    return((pd is not None and isinstance(data, pd.Series)) or
          (dd is not None and isinstance(data, dd.Series)))


def is_dask_array(data):
    da = None
    if 'dask.array' in sys.modules:
        import dask.array as da
    return (da is not None and isinstance(data, da.Array))


def is_cupy_array(data):
    if 'cupy' in sys.modules:
        import cupy
        return isinstance(data, cupy.ndarray)
    return False


def is_ibis_expr(data):
    if 'ibis' in sys.modules:
        import ibis
        return isinstance(data, ibis.expr.types.ColumnExpr)
    return False


def get_param_values(data):
    params = dict(kdims=data.kdims, vdims=data.vdims,
                  label=data.label)
    if (data.group != data.param.objects(False)['group'].default and not
        isinstance(type(data).group, property)):
        params['group'] = data.group
    return params


def is_param_method(obj, has_deps=False):
    """Whether the object is a method on a parameterized object.

    Args:
       obj: Object to check
       has_deps (boolean, optional): Check for dependencies
          Whether to also check whether the method has been annotated
          with param.depends

    Returns:
       A boolean value indicating whether the object is a method
       on a Parameterized object and if enabled whether it has any
       dependencies
    """
    parameterized = (inspect.ismethod(obj) and
                     isinstance(get_method_owner(obj), param.Parameterized))
    if parameterized and has_deps:
        return getattr(obj, "_dinfo", {}).get('dependencies')
    return parameterized


def resolve_dependent_value(value):
    """Resolves parameter dependencies on the supplied value

    Resolves parameter values, Parameterized instance methods,
    parameterized functions with dependencies on the supplied value,
    including such parameters embedded in a list, tuple, or dictionary.

    Args:
       value: A value which will be resolved

    Returns:
       A new value where any parameter dependencies have been
       resolved.
    """
    range_widget = False
    if isinstance(value, list):
        value = [resolve_dependent_value(v) for v in value]
    elif isinstance(value, tuple):
        value = tuple(resolve_dependent_value(v) for v in value)
    elif isinstance(value, dict):
        value = {
            resolve_dependent_value(k): resolve_dependent_value(v) for k, v in value.items()
        }

    if 'panel' in sys.modules:
        from panel.widgets import RangeSlider, Widget
        range_widget = isinstance(value, RangeSlider)
        try:
            from panel.depends import param_value_if_widget
            value = param_value_if_widget(value)
        except Exception:
            if isinstance(value, Widget):
                value = value.param.value
    if is_param_method(value, has_deps=True):
        value = value()
    elif isinstance(value, param.Parameter) and isinstance(value.owner, param.Parameterized):
        value = getattr(value.owner, value.name)
    elif isinstance(value, FunctionType) and hasattr(value, '_dinfo'):
        deps = value._dinfo
        args = (getattr(p.owner, p.name) for p in deps.get('dependencies', []))
        kwargs = {k: getattr(p.owner, p.name) for k, p in deps.get('kw', {}).items()}
        value = value(*args, **kwargs)
    if isinstance(value, tuple) and range_widget:
        value = slice(*value)
    return value


def resolve_dependent_kwargs(kwargs):
    """Resolves parameter dependencies in the supplied dictionary

    Resolves parameter values, Parameterized instance methods and
    parameterized functions with dependencies in the supplied
    dictionary.

    Args:
       kwargs (dict): A dictionary of keyword arguments

    Returns:
       A new dictionary where any parameter dependencies have been
       resolved.
    """
    return {k: resolve_dependent_value(v) for k, v in kwargs.items()}


@contextmanager
def disable_constant(parameterized):
    """
    Temporarily set parameters on Parameterized object to
    constant=False.
    """
    params = parameterized.param.objects('existing').values()
    constants = [p.constant for p in params]
    for p in params:
        p.constant = False
    try:
        yield
    except:
        raise
    finally:
        for (p, const) in zip(params, constants):
            p.constant = const


def get_ndmapping_label(ndmapping, attr):
    """
    Function to get the first non-auxiliary object
    label attribute from an NdMapping.
    """
    label = None
    els = itervalues(ndmapping.data)
    while label is None:
        try:
            el = next(els)
        except StopIteration:
            return None
        if not getattr(el, '_auxiliary_component', True):
            label = getattr(el, attr)
    if attr == 'group':
        tp = type(el).__name__
        if tp == label:
            return None
    return label


def wrap_tuple(unwrapped):
    """ Wraps any non-tuple types in a tuple """
    return (unwrapped if isinstance(unwrapped, tuple) else (unwrapped,))


def stream_name_mapping(stream, exclude_params=['name'], reverse=False):
    """
    Return a complete dictionary mapping between stream parameter names
    to their applicable renames, excluding parameters listed in
    exclude_params.

    If reverse is True, the mapping is from the renamed strings to the
    original stream parameter names.
    """
    from ..streams import Params
    if isinstance(stream, Params):
        mapping = {}
        for p in stream.parameters:
            if isinstance(p, str):
                mapping[p] = stream._rename.get(p, p)
            else:
                mapping[p.name] = stream._rename.get((p.owner, p.name), p.name)
    else:
        filtered = [k for k in stream.param if k not in exclude_params]
        mapping = {k: stream._rename.get(k, k) for k in filtered}
    if reverse:
        return {v: k for k,v in mapping.items()}
    else:
        return mapping

def rename_stream_kwargs(stream, kwargs, reverse=False):
    """
    Given a stream and a kwargs dictionary of parameter values, map to
    the corresponding dictionary where the keys are substituted with the
    appropriately renamed string.

    If reverse, the output will be a dictionary using the original
    parameter names given a dictionary using the renamed equivalents.
    """
    mapped_kwargs = {}
    mapping = stream_name_mapping(stream, reverse=reverse)
    for k,v in kwargs.items():
        if k not in mapping:
            msg = 'Could not map key {key} {direction} renamed equivalent'
            direction = 'from' if reverse else 'to'
            raise KeyError(msg.format(key=repr(k), direction=direction))
        mapped_kwargs[mapping[k]] = v
    return mapped_kwargs


def stream_parameters(streams, no_duplicates=True, exclude=['name', '_memoize_key']):
    """
    Given a list of streams, return a flat list of parameter name,
    excluding those listed in the exclude list.

    If no_duplicates is enabled, a KeyError will be raised if there are
    parameter name clashes across the streams.
    """
    from ..streams import Params
    param_groups = {}
    for s in streams:
        if not s.contents and isinstance(s.hashkey, dict):
            param_groups[s] = list(s.hashkey)
        else:
            param_groups[s] = list(s.contents)

    if no_duplicates:
        seen, clashes = {}, []
        clash_streams = []
        for s in streams:
            if isinstance(s, Params):
                continue
            for c in param_groups[s]:
                if c in seen:
                    clashes.append(c)
                    if seen[c] not in clash_streams:
                        clash_streams.append(seen[c])
                    clash_streams.append(s)
                else:
                    seen[c] = s
        clashes = sorted(clashes)
        if clashes:
            clashing = ', '.join([repr(c) for c in clash_streams[:-1]])
            raise Exception('The supplied stream objects %s and %s '
                            'clash on the following parameters: %r'
                            % (clashing, clash_streams[-1], clashes))
    return [name for group in param_groups.values() for name in group
            if name not in exclude]


def dimensionless_contents(streams, kdims, no_duplicates=True):
    """
    Return a list of stream parameters that have not been associated
    with any of the key dimensions.
    """
    names = stream_parameters(streams, no_duplicates)
    return [name for name in names if name not in kdims]


def unbound_dimensions(streams, kdims, no_duplicates=True):
    """
    Return a list of dimensions that have not been associated with
    any streams.
    """
    params = stream_parameters(streams, no_duplicates)
    return [d for d in kdims if d not in params]


def wrap_tuple_streams(unwrapped, kdims, streams):
    """
    Fills in tuple keys with dimensioned stream values as appropriate.
    """
    param_groups = [(s.contents.keys(), s) for s in streams]
    pairs = [(name,s)  for (group, s) in param_groups for name in group]
    substituted = []
    for pos,el in enumerate(wrap_tuple(unwrapped)):
        if el is None and pos < len(kdims):
            matches = [(name,s) for (name,s) in pairs if name==kdims[pos].name]
            if len(matches) == 1:
                (name, stream) = matches[0]
                el = stream.contents[name]
        substituted.append(el)
    return tuple(substituted)


def drop_streams(streams, kdims, keys):
    """
    Drop any dimensioned streams from the keys and kdims.
    """
    stream_params = stream_parameters(streams)
    inds, dims = zip(*[(ind, kdim) for ind, kdim in enumerate(kdims)
                       if kdim not in stream_params])
    get = operator.itemgetter(*inds) # itemgetter used for performance
    keys = (get(k) for k in keys)
    return dims, ([wrap_tuple(k) for k in keys] if len(inds) == 1 else list(keys))


def itervalues(obj):
    "Get value iterator from dictionary for Python 2 and 3"
    return iter(obj.values()) if sys.version_info.major == 3 else obj.itervalues()


def iterkeys(obj):
    "Get key iterator from dictionary for Python 2 and 3"
    return iter(obj.keys()) if sys.version_info.major == 3 else obj.iterkeys()


def get_unique_keys(ndmapping, dimensions):
    inds = [ndmapping.get_dimension_index(dim) for dim in dimensions]
    getter = operator.itemgetter(*inds)
    return unique_iterator(getter(key) if len(inds) > 1 else (key[inds[0]],)
                           for key in ndmapping.data.keys())


def unpack_group(group, getter):
    for k, v in group.iterrows():
        obj = v.values[0]
        key = getter(k)
        if hasattr(obj, 'kdims'):
            yield (key, obj)
        else:
            yield (wrap_tuple(key), obj)


def capitalize(string):
    """
    Capitalizes the first letter of a string.
    """
    if string:
        return string[0].upper() + string[1:]
    else:
        return string


def get_path(item):
    """
    Gets a path from an Labelled object or from a tuple of an existing
    path and a labelled object. The path strings are sanitized and
    capitalized.
    """
    sanitizers = [group_sanitizer, label_sanitizer]
    if isinstance(item, tuple):
        path, item = item
        if item.label:
            if len(path) > 1 and item.label == path[1]:
                path = path[:2]
            else:
                path = path[:1] + (item.label,)
        else:
            path = path[:1]
    else:
        path = (item.group, item.label) if item.label else (item.group,)
    return tuple(capitalize(fn(p)) for (p, fn) in zip(path, sanitizers))


def make_path_unique(path, counts, new):
    """
    Given a path, a list of existing paths and counts for each of the
    existing paths.
    """
    added = False
    while any(path == c[:i] for c in counts for i in range(1, len(c)+1)):
        count = counts[path]
        counts[path] += 1
        if (not new and len(path) > 1) or added:
            path = path[:-1]
        else:
            added = True
        path = path + (int_to_roman(count),)
    if len(path) == 1:
        path = path + (int_to_roman(counts.get(path, 1)),)
    if path not in counts:
        counts[path] = 1
    return path


class ndmapping_groupby(param.ParameterizedFunction):
    """
    Apply a groupby operation to an NdMapping, using pandas to improve
    performance (if available).
    """

    sort = param.Boolean(default=False, doc='Whether to apply a sorted groupby')

    def __call__(self, ndmapping, dimensions, container_type,
                 group_type, sort=False, **kwargs):
        try:
            import pandas # noqa (optional import)
            groupby = self.groupby_pandas
        except:
            groupby = self.groupby_python
        return groupby(ndmapping, dimensions, container_type,
                       group_type, sort=sort, **kwargs)

    @param.parameterized.bothmethod
    def groupby_pandas(self_or_cls, ndmapping, dimensions, container_type,
                       group_type, sort=False, **kwargs):
        if 'kdims' in kwargs:
            idims = [ndmapping.get_dimension(d) for d in kwargs['kdims']]
        else:
            idims = [dim for dim in ndmapping.kdims if dim not in dimensions]

        all_dims = [d.name for d in ndmapping.kdims]
        inds = [ndmapping.get_dimension_index(dim) for dim in idims]
        getter = operator.itemgetter(*inds) if inds else lambda x: tuple()

        multi_index = pd.MultiIndex.from_tuples(ndmapping.keys(), names=all_dims)
        df = pd.DataFrame(list(map(wrap_tuple, ndmapping.values())), index=multi_index)

        # TODO: Look at sort here
        kwargs = dict(dict(get_param_values(ndmapping), kdims=idims), sort=sort, **kwargs)
        groups = ((wrap_tuple(k), group_type(OrderedDict(unpack_group(group, getter)), **kwargs))
                   for k, group in df.groupby(level=[d.name for d in dimensions], sort=sort))

        if sort:
            selects = list(get_unique_keys(ndmapping, dimensions))
            groups = sorted(groups, key=lambda x: selects.index(x[0]))

        return container_type(groups, kdims=dimensions, sort=sort)

    @param.parameterized.bothmethod
    def groupby_python(self_or_cls, ndmapping, dimensions, container_type,
                       group_type, sort=False, **kwargs):
        idims = [dim for dim in ndmapping.kdims if dim not in dimensions]
        dim_names = [dim.name for dim in dimensions]
        selects = get_unique_keys(ndmapping, dimensions)
        selects = group_select(list(selects))
        groups = [(k, group_type((v.reindex(idims) if hasattr(v, 'kdims')
                                  else [((), v)]), **kwargs))
                  for k, v in iterative_select(ndmapping, dim_names, selects)]
        return container_type(groups, kdims=dimensions)


def cartesian_product(arrays, flat=True, copy=False):
    """
    Efficient cartesian product of a list of 1D arrays returning the
    expanded array views for each dimensions. By default arrays are
    flattened, which may be controlled with the flat flag. The array
    views can be turned into regular arrays with the copy flag.
    """
    arrays = np.broadcast_arrays(*np.ix_(*arrays))
    if flat:
        return tuple(arr.flatten() if copy else arr.flat for arr in arrays)
    return tuple(arr.copy() if copy else arr for arr in arrays)


def cross_index(values, index):
    """
    Allows efficiently indexing into a cartesian product without
    expanding it. The values should be defined as a list of iterables
    making up the cartesian product and a linear index, returning
    the cross product of the values at the supplied index.
    """
    lengths = [len(v) for v in values]
    length = np.product(lengths)
    if index >= length:
        raise IndexError('Index %d out of bounds for cross-product of size %d'
                         % (index, length))
    indexes = []
    for i in range(1, len(values))[::-1]:
        p = np.product(lengths[-i:])
        indexes.append(index//p)
        index -= indexes[-1] * p
    indexes.append(index)
    return tuple(v[i] for v, i in zip(values, indexes))


def arglexsort(arrays):
    """
    Returns the indices of the lexicographical sorting
    order of the supplied arrays.
    """
    dtypes = ','.join(array.dtype.str for array in arrays)
    recarray = np.empty(len(arrays[0]), dtype=dtypes)
    for i, array in enumerate(arrays):
        recarray['f%s' % i] = array
    return recarray.argsort()


def dimensioned_streams(dmap):
    """
    Given a DynamicMap return all streams that have any dimensioned
    parameters, i.e. parameters also listed in the key dimensions.
    """
    dimensioned = []
    for stream in dmap.streams:
        stream_params = stream_parameters([stream])
        if set([str(k) for k in dmap.kdims]) & set(stream_params):
            dimensioned.append(stream)
    return dimensioned


def expand_grid_coords(dataset, dim):
    """
    Expand the coordinates along a dimension of the gridded
    dataset into an ND-array matching the dimensionality of
    the dataset.
    """
    irregular = [d.name for d in dataset.kdims
                 if d is not dim and dataset.interface.irregular(dataset, d)]
    if irregular:
        array = dataset.interface.coords(dataset, dim, True)
        example = dataset.interface.values(dataset, irregular[0], True, False)
        return array * np.ones_like(example)
    else:
        arrays = [dataset.interface.coords(dataset, d.name, True)
                  for d in dataset.kdims]
        idx = dataset.get_dimension_index(dim)
        return cartesian_product(arrays, flat=False)[idx].T


def dt64_to_dt(dt64):
    """
    Safely converts NumPy datetime64 to a datetime object.
    """
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
    return dt.datetime(1970,1,1,0,0,0) + dt.timedelta(seconds=ts)


def is_nan(x):
    """
    Checks whether value is NaN on arbitrary types
    """
    try:
        return np.isnan(x)
    except:
        return False


def bound_range(vals, density, time_unit='us'):
    """
    Computes a bounding range and density from a number of samples
    assumed to be evenly spaced. Density is rounded to machine precision
    using significant digits reported by sys.float_info.dig.
    """
    if not len(vals):
        return(np.nan, np.nan, density, False)
    low, high = vals.min(), vals.max()
    invert = False
    if len(vals) > 1 and vals[0] > vals[1]:
        invert = True
    if not density:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered in double_scalars')
            full_precision_density = compute_density(low, high, len(vals)-1)
            with np.errstate(over='ignore'):
                density = round(full_precision_density, sys.float_info.dig)
        if density == 0 or density == np.inf:
            density = full_precision_density
    if density == 0:
        raise ValueError('Could not determine Image density, ensure it has a non-zero range.')
    halfd = 0.5/density
    if isinstance(low, datetime_types):
        halfd = np.timedelta64(int(round(halfd)), time_unit)
    return low-halfd, high+halfd, density, invert


def validate_regular_sampling(values, rtol=10e-6):
    """
    Validates regular sampling of a 1D array ensuring that the difference
    in sampling steps is at most rtol times the smallest sampling step.
    Returns a boolean indicating whether the sampling is regular.
    """
    diffs = np.diff(values)
    return (len(diffs) < 1) or abs(diffs.min()-diffs.max()) < abs(diffs.min()*rtol)


def compute_density(start, end, length, time_unit='us'):
    """
    Computes a grid density given the edges and number of samples.
    Handles datetime grids correctly by computing timedeltas and
    computing a density for the given time_unit.
    """
    if isinstance(start, int): start = float(start)
    if isinstance(end, int): end = float(end)
    diff = end-start
    if isinstance(diff, timedelta_types):
        if isinstance(diff, np.timedelta64):
            diff = np.timedelta64(diff, time_unit).tolist()
        tscale = 1./np.timedelta64(1, time_unit).tolist().total_seconds()
        return (length/(diff.total_seconds()*tscale))
    else:
        return length/diff


def date_range(start, end, length, time_unit='us'):
    """
    Computes a date range given a start date, end date and the number
    of samples.
    """
    step = (1./compute_density(start, end, length, time_unit))
    if pd and isinstance(start, pd.Timestamp):
        start = start.to_datetime64()
    step = np.timedelta64(int(round(step)), time_unit)
    return start+step/2.+np.arange(length)*step


def parse_datetime(date):
    """
    Parses dates specified as string or integer or pandas Timestamp
    """
    if pd is None:
        raise ImportError('Parsing dates from strings requires pandas')
    return pd.to_datetime(date).to_datetime64()


def parse_datetime_selection(sel):
    """
    Parses string selection specs as datetimes.
    """
    if isinstance(sel, basestring) or isdatetime(sel):
        sel = parse_datetime(sel)
    if isinstance(sel, slice):
        if isinstance(sel.start, basestring) or isdatetime(sel.start):
            sel = slice(parse_datetime(sel.start), sel.stop)
        if isinstance(sel.stop, basestring) or isdatetime(sel.stop):
            sel = slice(sel.start, parse_datetime(sel.stop))
    if isinstance(sel, (set, list)):
        sel = [parse_datetime(v) if isinstance(v, basestring) else v for v in sel]
    return sel


def dt_to_int(value, time_unit='us'):
    """
    Converts a datetime type to an integer with the supplied time unit.
    """
    if pd:
        if isinstance(value, pd.Period):
            value = value.to_timestamp()
        if isinstance(value, pd.Timestamp):
            try:
                value = value.to_datetime64()
            except Exception:
                value = np.datetime64(value.to_pydatetime())
    elif isinstance(value, cftime_types):
        return cftime_to_timestamp(value, time_unit)

    # date class is a parent for datetime class
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        value = dt.datetime(*value.timetuple()[:6])

    # Handle datetime64 separately
    if isinstance(value, np.datetime64):
        try:
            value = np.datetime64(value, 'ns')
            tscale = (np.timedelta64(1, time_unit)/np.timedelta64(1, 'ns'))
            return value.tolist()/tscale
        except Exception:
            # If it can't handle ns precision fall back to datetime
            value = value.tolist()

    if time_unit == 'ns':
        tscale = 1e9
    else:
        tscale = 1./np.timedelta64(1, time_unit).tolist().total_seconds()

    try:
        # Handle python3
        if value.tzinfo is None:
            _epoch = dt.datetime(1970, 1, 1)
        else:
            _epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
        return int((value - _epoch).total_seconds() * tscale)
    except Exception:
        # Handle python2
        return (time.mktime(value.timetuple()) + value.microsecond / 1e6) * tscale


def cftime_to_timestamp(date, time_unit='us'):
    """Converts cftime to timestamp since epoch in milliseconds

    Non-standard calendars (e.g. Julian or no leap calendars)
    are converted to standard Gregorian calendar. This can cause
    extra space to be added for dates that don't exist in the original
    calendar. In order to handle these dates correctly a custom bokeh
    model with support for other calendars would have to be defined.

    Args:
        date: cftime datetime object (or array)

    Returns:
        time_unit since 1970-01-01 00:00:00
    """
    import cftime
    if time_unit == 'us':
        tscale = 1
    else:
        tscale = (np.timedelta64(1, 'us')/np.timedelta64(1, time_unit))

    return cftime.date2num(date,'microseconds since 1970-01-01 00:00:00',
                           calendar='standard')*tscale

def search_indices(values, source):
    """
    Given a set of values returns the indices of each of those values
    in the source array.
    """
    orig_indices = source.argsort()
    return orig_indices[np.searchsorted(source[orig_indices], values)]


def compute_edges(edges):
    """
    Computes edges as midpoints of the bin centers.  The first and
    last boundaries are equidistant from the first and last midpoints
    respectively.
    """
    edges = np.asarray(edges)
    if edges.dtype.kind == 'i':
        edges = edges.astype('f')
    midpoints = (edges[:-1] + edges[1:])/2.0
    boundaries = (2*edges[0] - midpoints[0], 2*edges[-1] - midpoints[-1])
    return np.concatenate([boundaries[:1], midpoints, boundaries[-1:]])


def mimebundle_to_html(bundle):
    """
    Converts a MIME bundle into HTML.
    """
    if isinstance(bundle, tuple):
        data, metadata = bundle
    else:
        data = bundle
    html = data.get('text/html', '')
    if 'application/javascript' in data:
        js = data['application/javascript']
        html += '\n<script type="application/javascript">{js}</script>'.format(js=js)
    return html


def numpy_scalar_to_python(scalar):
    """
    Converts a NumPy scalar to a regular python type.
    """
    scalar_type = type(scalar)
    if np.issubclass_(scalar_type, np.float_):
        return float(scalar)
    elif np.issubclass_(scalar_type, np.int_):
        return int(scalar)
    return scalar


def closest_match(match, specs, depth=0):
    """
    Recursively iterates over type, group, label and overlay key,
    finding the closest matching spec.
    """
    new_specs = []
    match_lengths = []
    for i, spec in specs:
        if spec[0] == match[0]:
            new_specs.append((i, spec[1:]))
        else:
            if all(isinstance(s[0], basestring) for s in [spec, match]):
                match_length = max(i for i in range(len(match[0]))
                                   if match[0].startswith(spec[0][:i]))
            elif is_number(match[0]) and is_number(spec[0]):
                m = bool(match[0]) if isinstance(match[0], np.bool_) else match[0]
                s = bool(spec[0]) if isinstance(spec[0], np.bool_) else spec[0]
                match_length = -abs(m-s)
            else:
                match_length = 0
            match_lengths.append((i, match_length, spec[0]))

    if len(new_specs) == 1:
        return new_specs[0][0]
    elif new_specs:
        depth = depth+1
        return closest_match(match[1:], new_specs, depth)
    else:
        if depth == 0 or not match_lengths:
            return None
        else:
            return sorted(match_lengths, key=lambda x: -x[1])[0][0]


def cast_array_to_int64(array):
    """
    Convert a numpy array  to `int64`. Suppress the following warning
    emitted by Numpy, which as of 12/2021 has been extensively discussed
    (https://github.com/pandas-dev/pandas/issues/22384)
    and whose fate (possible revert) has not yet been settled:

        FutureWarning: casting datetime64[ns] values to int64 with .astype(...)
        is deprecated and will raise in a future version. Use .view(...) instead.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore',
            message='casting datetime64',
            category=FutureWarning,
        )
        return array.astype('int64')


def flatten(line):
    """
    Flatten an arbitrarily nested sequence.

    Inspired by: pd.core.common.flatten

    Parameters
    ----------
    line : sequence
        The sequence to flatten

    Notes
    -----
    This only flattens list, tuple, and dict sequences.

    Returns
    -------
    flattened : generator
    """

    for element in line:
        if any(isinstance(element, tp) for tp in (list, tuple, dict)):
            yield from flatten(element)
        else:
            yield element
