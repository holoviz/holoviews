import numpy as np

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


def valid_identifier(identifier):
    """
    Replace spaces with underscores and returns value after
    checking validity.
    """
    if not identifier: return
    identifier = identifier.replace(' ', '_')
    invalid_chars = any(not el.isalnum() and el!='_' for el in identifier)
    valid_first_char = identifier[0].isalpha() or identifier[0]=='_'
    if invalid_chars or not valid_first_char:
        raise SyntaxError("Invalid Python identifier: %r" % identifier)
    return identifier
