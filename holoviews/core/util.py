import string
import numpy as np

import param

def find_minmax(lims, olims):
    """
    Takes (a1, a2) and (b1, b2) as input and returns
    (np.min(a1, b1), np.max(a2, b2)). Used to calculate
    min and max values of a number of items.
    """

    try:
        limzip = zip(list(lims), list(olims), [np.min, np.max])
        limits = tuple([float(fn([l, ol])) for l, ol, fn in limzip])
    except:
        limits = (np.NaN, np.NaN)
    return limits


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


UNDERSCORE_TOKEN = 'UNDERSCORE'

def sanitize_identifier(name, escape=True):
    """
    Sanitizes group/label values for use in AttrTree
    attribute access
    """
    if name is None: return ''
    valid_chars = string.ascii_letters+string.digits+'_'
    name = name.replace(' ', '_')
    chars = []
    if name and name[0].islower():
        name = string.upper(name[0])+name[1:]
    for c in name:
        if c not in valid_chars:
            chars.append('_%s_' % hex(ord(c)))
        else:
            chars.append(c)
    if escape and len(chars) and chars[0][0] == '_':
        chars[0] = UNDERSCORE_TOKEN + chars[0][1:]
    return ''.join(chars)


def unescape_identifier(identifier):
    return identifier.replace(UNDERSCORE_TOKEN, '_')


def allowable(name):
    valid_second_chars= string.ascii_letters+string.digits
    if len(name) >= 2 and name.startswith('_') and (name[1] not in valid_second_chars):
        return False
    return True


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
