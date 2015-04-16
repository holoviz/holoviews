import sys, warnings
import numbers
import itertools
import string
import unicodedata

import numpy as np
import param

# Python3 compatibility
basestring = str if sys.version_info.major == 3 else basestring


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


class sanitize_identifier(param.ParameterizedFunction):
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
                               'latin', 'greek', 'arabic-indic', 'with'], doc="""
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

    prefix = 'A_'

    @param.parameterized.bothmethod
    def allowable(self_or_cls, name, disable_leading_underscore=None):
       disabled_reprs = ['javascript', 'jpeg', 'json', 'latex',
                         'latex', 'pdf', 'png', 'svg']
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
            replacement = replacement.decode('unicode_escape')
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
        if name in [None, '']: return name
        name = safe_unicode(name)
        version = self.version if version is None else version
        if not self.allowable(name):
            raise AttributeError("String %r is in the disallowed list of attribute names: %r" % self.disallowed)

        if version == 2:
            name = self.remove_diacritics(name)
        if self.capitalize and name and name[0] in string.ascii_lowercase:
            name = name[0].upper()+name[1:]

        sanitized = (self.sanitize_py2(name) if version==2 else self.sanitize_py3(name))
        if not self.prefixed(name, version): return sanitized
        else:                                return self.prefix + sanitized


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
        sanitized, chars = [], ''
        for split in name.split():
            for c in split:
                if valid_fn(c): chars += str(c) if c=='_' else c
                else:
                    short = self.shortened_character_name(c, self.eliminations,
                                                         self.substitutions,
                                                         self.transforms)
                    sanitized.extend([chars] if chars else [])
                    sanitized.append(short)
                    chars = ''
            if chars:
                sanitized.extend([chars])
                chars=''
        return self._process_underscores(sanitized + ([chars] if chars else []))


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


def max_range(ranges):
   """
   Computes the maximal lower and upper bounds from a list bounds.
   """
   try:
      with warnings.catch_warnings():
         warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
         arr = np.array(ranges, dtype=np.float)
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
      inds = [(0, 2), (1, 3)]
      extents = [e if len(e) == 6 else (e[0], e[1], None,
                                        e[2], e[3], None)
                 for e in extents]
   else:
      num = 4
      inds = [(0, 2), (1, 3)]
   arr = np.array(extents, dtype=np.float, ndmin=2)
   extents = [np.NaN] * num
   if 0 in arr.shape:
      return extents
   with warnings.catch_warnings():
      warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
      for lower, upper in inds:
         extents[lower] = np.nanmin(arr[:, lower])
         extents[upper] = np.nanmax(arr[:, upper])
   return tuple(extents)


def int_to_alpha(n, upper=True):
    "Generates alphanumeric labels of form A-Z, AA-ZZ etc."
    casenum = 65 if upper else 97
    label = ''
    count= 0
    if n == 0: return str(chr(n + casenum))
    while n >= 0:
        mod, div = n % 26, n
        for i in range(count):
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


def match_spec(element, specification):
    """
    Matches the group.label specification of the supplied
    element against the supplied specification dictionary
    returning the value of the best match.
    """
    match_tuple = ()
    match = specification.get((), {})
    for spec in [type(element).__name__,
                 sanitize_identifier(element.group, escape=False),
                 sanitize_identifier(element.label, escape=False)]:
        match_tuple += (spec,)
        if match_tuple in specification:
            match = specification[match_tuple]
    return match


def python2sort(x,key=None):
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


def dimension_sort(odict, dimensions, categorical, cached_values):
    """
    Sorts data by key using usual Python tuple sorting semantics
    or sorts in categorical order for any categorical Dimensions.
    """
    sortkws = {}
    if categorical:
        sortkws['key'] = lambda x: tuple(cached_values[d.name].index(x[0][i])
                                         if d.values else x[0][i]
                                         for i, d in enumerate(dimensions))
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
