import os, sys, warnings, operator
import numbers
import itertools
import string, fnmatch
import unicodedata
from collections import defaultdict

import numpy as np
import param

try:
    from cyordereddict import OrderedDict
except:
    from collections import OrderedDict

try:
    import pandas as pd # noqa (optional import)
except ImportError:
    pd = None

try:
    import dask.dataframe as dd
except ImportError:
    dd = None

# Python3 compatibility
import types
if sys.version_info.major == 3:
    basestring = str
    unicode = str
    generator_types = (zip, range, types.GeneratorType)
else:
    basestring = basestring
    unicode = unicode
    from itertools import izip
    generator_types = (izip, xrange, types.GeneratorType)


def process_ellipses(obj, key, vdim_selection=False):
    """
    Helper function to pad a __getitem__ key with the right number of
    empty slices (i.e :) when the key contains an Ellipsis (...).

    If the vdim_selection flag is true, check if the end of the key
    contains strings or Dimension objects in obj. If so, extra padding
    will not be applied for the value dimensions (i.e the resulting key
    will be exactly one longer than the number of kdims). Note: this
    flag should not be used for composite types.
    """
    if isinstance(key, np.ndarray) and key.dtype.kind == 'b':
        return key
    wrapped_key = wrap_tuple(key)
    if wrapped_key.count(Ellipsis)== 0:
        return key
    if wrapped_key.count(Ellipsis)!=1:
        raise Exception("Only one ellipsis allowed at a time.")
    dim_count = len(obj.dimensions())
    index = wrapped_key.index(Ellipsis)
    head = wrapped_key[:index]
    tail = wrapped_key[index+1:]

    padlen = dim_count - (len(head) + len(tail))
    if vdim_selection:
        # If the end of the key (i.e the tail) is in vdims, pad to len(kdims)+1
        if wrapped_key[-1] in obj.vdims:
            padlen = (len(obj.kdims) +1 ) - len(head+tail)
    return head + ((slice(None),) * padlen) + tail


def safe_unicode(value):
   if sys.version_info.major == 3 or not isinstance(value, str): return value
   else: return unicode(value.decode('utf-8'))



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


class Aliases(object):
    """
    Helper class useful for defining a set of alias tuples on a single object.

    For instance, when defining a group or label with an alias, instead
    of setting tuples in the constructor, you could use
    ``aliases.water`` if you first define:

    >>> aliases = Aliases(water='H_2O', glucose='C_6H_{12}O_6')
    >>> aliases.water
    ('water', 'H_2O')

    This may be used to conveniently define aliases for groups, labels
    or dimension names.
    """
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, (k,v))



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
    allows filtered, substitions and transforms to help shorten these
    names appropriately.
    """

    version = param.ObjectSelector(sys.version_info.major, objects=[2,3], doc="""
        The sanitization version. If set to 2, more aggresive
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
       unicode name. For instance, the defaultcapitalize_unicode_name
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
                chars += safe_unicode(replacement)
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
        # Substitition
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
        name = safe_unicode(name)
        version = self.version if version is None else version
        if not self.allowable(name):
            raise AttributeError("String %r is in the disallowed list of attribute names: %r" % self.disallowed)

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
            raise Exception("Please use a raw string or escape control code '\%s'"
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


def isnumeric(val):
    if isinstance(val, (basestring, bool, np.bool_)):
        return False
    try:
        float(val)
        return True
    except:
        return False


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
        return np.nanmin(values), np.nanmax(values)
    except:
        try:
            values = sorted(values)
            return (values[0], values[-1])
        except:
            return (None, None)


def max_range(ranges):
    """
    Computes the maximal lower and upper bounds from a list bounds.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
            values = [r for r in ranges for v in r if v is not None]
            if pd and all(isinstance(v, pd.tslib.Timestamp) for r in values for v in r):
                values = [(v1.to_datetime64(), v2.to_datetime64()) for v1, v2 in values]
            arr = np.array(values)
            if arr.dtype.kind in 'M':
                return arr[:, 0].min(), arr[:, 1].max()
            return (np.nanmin(arr[:, 0]), np.nanmax(arr[:, 1]))
    except:
        return (np.NaN, np.NaN)


def max_extents(extents, zrange=False):
    """
    Computes the maximal extent in 2D and 3D space from
    list of 4-tuples or 6-tuples. If zrange is enabled
    all extents are converted to 6-tuples to comput
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
            lower = [v for v in arr[lidx] if v is not None]
            upper = [v for v in arr[uidx] if v is not None]
            if lower and isinstance(lower[0], np.datetime64):
                extents[lidx] = np.min(lower)
            elif lower:
                extents[lidx] = np.nanmin(lower)
            if upper and isinstance(upper[0], np.datetime64):
                extents[uidx] = np.max(upper)
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


def unique_array(arr):
    """
    Returns an array of unique values in the input order
    """
    if pd:
        return pd.unique(arr)
    else:
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


def dimension_sort(odict, kdims, vdims, categorical, key_index, cached_values):
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

    if len(set(key_index)) != len(key_index):
        raise ValueError("Cannot sort on duplicated dimensions")
    elif categorical:
       sortkws['key'] = lambda x: tuple(cached_values[dim.name].index(x[t][d])
                                        if dim.values else x[t][d]
                                        for i, (dim, t, d) in enumerate(indexes))
    elif key_index != list(range(len(kdims+vdims))):
        sortkws['key'] = lambda x: tuple(x[t][d] for _, t, d in indexes)
    if sys.version_info.major == 3:
        return python2sort(odict.items(), **sortkws)
    else:
        return sorted(odict.items(), **sortkws)


# Copied from param should make param version public
def is_number(obj):
    if isinstance(obj, numbers.Number): return True
    # The extra check is for classes that behave like numbers, such as those
    # found in numpy, gmpy, etc.
    elif (hasattr(obj, '__int__') and hasattr(obj, '__add__')): return True
    # This is for older versions of gmpy
    elif hasattr(obj, 'qdiv'): return True
    else: return False


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
    [set([1, 2]), set([3, 4]), set([5, 6])]
    """
    levels_by_name = {}
    names_by_level = defaultdict(set)

    def add_level_to_name(name, level):
        levels_by_name[name] = level
        names_by_level[level].add(name)


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


def find_file(folder, filename):
    """
    Find a file given folder and filename.
    """
    matches = []
    for root, _, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, filename):
            matches.append(os.path.join(root, filename))
    return matches[-1]


def is_dataframe(data):
    """
    Checks whether the supplied data is DatFrame type.
    """
    return((pd is not None and isinstance(data, pd.DataFrame)) or
          (dd is not None and isinstance(data, dd.DataFrame)))


def get_param_values(data):
    params = dict(kdims=data.kdims, vdims=data.vdims,
                  label=data.label)
    if data.group != data.params()['group'].default:
        params['group'] = data.group
    return params


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
        if not el._auxiliary_component:
            label = getattr(el, attr)
    if attr == 'group':
        tp = type(el).__name__
        if tp == label:
            return None
    return label


def wrap_tuple(unwrapped):
    """ Wraps any non-tuple types in a tuple """
    return (unwrapped if isinstance(unwrapped, tuple) else (unwrapped,))


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
            obj = tuple(v)
            yield (wrap_tuple(key), obj)




class ndmapping_groupby(param.ParameterizedFunction):
    """
    Apply a groupby operation to an NdMapping, using pandas to improve
    performance (if available).
    """

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

        kwargs = dict(dict(get_param_values(ndmapping), kdims=idims), **kwargs)
        groups = ((wrap_tuple(k), group_type(OrderedDict(unpack_group(group, getter)), **kwargs))
                   for k, group in df.groupby(level=[d.name for d in dimensions]))

        if sort:
            selects = list(get_unique_keys(ndmapping, dimensions))
            groups = sorted(groups, key=lambda x: selects.index(x[0]))
        return container_type(groups, kdims=dimensions)

    @param.parameterized.bothmethod
    def groupby_python(self_or_cls, ndmapping, dimensions, container_type,
                       group_type, sort=False, **kwargs):
        idims = [dim for dim in ndmapping.kdims if dim not in dimensions]
        dim_names = [dim.name for dim in dimensions]
        selects = get_unique_keys(ndmapping, dimensions)
        selects = group_select(list(selects))
        groups = [(k, group_type((v.reindex(idims) if hasattr(v, 'kdims')
                                  else [((), (v,))]), **kwargs))
                  for k, v in iterative_select(ndmapping, dim_names, selects)]
        return container_type(groups, kdims=dimensions)


def cartesian_product(arrays):
    """
    Computes the cartesian product of a list of 1D arrays
    returning arrays matching the shape defined by all
    supplied dimensions.
    """
    return np.broadcast_arrays(*np.ix_(*arrays))


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


def get_dynamic_item(map_obj, dimensions, key):
    """
    Looks up an item in a DynamicMap given a list of dimensions
    and a corresponding key. The dimensions must be a subset
    of the map_obj key dimensions.
    """
    if isinstance(key, tuple):
        dims = {d.name: k for d, k in zip(dimensions, key)
                if d in map_obj.kdims}
        key = tuple(dims.get(d.name) for d in map_obj.kdims)
        el = map_obj.select([lambda x: type(x).__name__ == 'DynamicMap'],
                            **dims)
    elif key < map_obj.counter:
        key_offset = max([key-map_obj.cache_size, 0])
        key = map_obj.keys()[min([key-key_offset,
                                  len(map_obj)-1])]
        el = map_obj[key]
    elif key >= map_obj.counter:
        el = next(map_obj)
        key = list(map_obj.keys())[-1]
    else:
        el = None
    return key, el
